import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from Algorithms import Algorithm2D


class PotentialFieldAlgorithm(Algorithm2D):
    """Classic artificial potential field (APF) planner — 2D baseline.

    The potential is the textbook formulation:

    * attractive:  U_att = 0.5 · k_att · d(goal)²
    * repulsive:   U_rep = 0.5 · k_rep · (1/d − 1/q*)²  for d < q*, else 0,
      where d is the Euclidean distance to the *nearest* obstacle
      (computed exactly with a distance transform).

    Gradient descent follows the steepest descent of U_att + U_rep from start
    to goal. When ``bilinear_interpolation`` is True (default) the field is
    sampled at continuous positions via bilinear interpolation — the same
    option the Frontier Voltage Boost Laplace planner exposes; when False the
    gradient uses nearest-cell central differences. The returned path keeps
    the raw decimal (x, y) positions — round only for pixel-level
    visualization, never for metrics.

    Like any APF planner it can stall in local minima (a balance point between
    attraction and repulsion makes the descent oscillate in place). This is
    detected — no progress toward the goal for ``stall_window`` consecutive
    steps — and reported honestly by returning ``None`` (map unsolved), as is
    leaving the grid or entering an obstacle cell.
    """

    OBSTACLE_POTENTIAL = 1e6

    def __init__(
        self,
        map,
        q_star: float = 30.0,
        k_att: float = 1.0,
        k_rep: float = 1e4,
        step_size: float = 1.0,
        max_iters: int = 10000,
        stall_window: int = 100,
        bilinear_interpolation: bool = True,
        use_gpu: bool = False,
    ):
        super().__init__(map)
        self.q_star = q_star
        self.k_att = k_att
        self.k_rep = k_rep
        self.step_size = step_size
        self.max_iters = max_iters
        self.stall_window = stall_window
        self.bilinear_interpolation = bilinear_interpolation
        self.use_gpu = use_gpu   # assemble the potential with PyTorch (CUDA when available)
        self._path = None
        self._potential = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid
        start = self.map.start  # (x, y)
        end = self.map.end      # (x, y)

        if self.use_gpu:
            self._potential = self._compute_potential_torch(grid, end)
        else:
            self._potential = self._compute_potential(grid, end)
        self._path = self._gradient_descent(self._potential, grid, start, end)

        if visualize:
            self._visualize()

        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    # ------------------------------------------------------------------
    # Potential field
    # ------------------------------------------------------------------

    def _compute_potential(self, grid, goal):
        gx, gy = goal
        rows, cols = grid.shape
        yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

        u_att = 0.5 * self.k_att * ((yy - gy) ** 2 + (xx - gx) ** 2).astype(float)

        # Euclidean distance from every free cell to its nearest obstacle cell.
        obstacle = grid == 1
        dist = distance_transform_edt(~obstacle)

        u_rep = np.zeros_like(u_att)
        near = (dist < self.q_star) & (dist > 0)
        u_rep[near] = 0.5 * self.k_rep * (1.0 / dist[near] - 1.0 / self.q_star) ** 2
        u_rep[obstacle] = self.OBSTACLE_POTENTIAL

        return u_att + u_rep

    def _compute_potential_torch(self, grid, goal):
        """Same potential, elementwise math on a torch device (CUDA if present).
        The distance transform itself stays on the CPU (SciPy)."""
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gx, gy = goal
        rows, cols = grid.shape

        ys = torch.arange(rows, dtype=torch.double, device=device).view(-1, 1)
        xs = torch.arange(cols, dtype=torch.double, device=device).view(1, -1)
        u_att = 0.5 * self.k_att * ((ys - gy) ** 2 + (xs - gx) ** 2)

        obstacle_np = grid == 1
        dist = torch.from_numpy(
            np.asarray(distance_transform_edt(~obstacle_np), dtype=np.float64)).to(device)
        obstacle = torch.from_numpy(obstacle_np).to(device)

        u_rep = torch.zeros_like(u_att)
        near = (dist < self.q_star) & (dist > 0)
        u_rep[near] = 0.5 * self.k_rep * (1.0 / dist[near] - 1.0 / self.q_star) ** 2
        u_rep[obstacle] = self.OBSTACLE_POTENTIAL

        return (u_att + u_rep).cpu().numpy()

    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def _gradient_descent(self, potential, grid, start, end):
        sx, sy = start
        ex, ey = end

        posx = float(sx)
        posy = float(sy)

        # Raw continuous positions — round only for visualization.
        path = [(posx, posy)]

        # Local-minimum detection: give up when the distance to the goal has
        # not improved for `stall_window` consecutive steps (oscillation at a
        # balance point between attraction and repulsion).
        best_goal_dist = math.hypot(ex - posx, ey - posy)
        steps_since_best = 0

        for _ in range(self.max_iters):
            if not (1 <= posy < potential.shape[0] - 1 and
                    1 <= posx < potential.shape[1] - 1):
                return None

            if self.bilinear_interpolation:
                gradx, grady = self._get_grad_bilinear(potential, posx, posy)
            else:
                y = math.floor(posy)
                x = math.floor(posx)
                grady = potential[y + 1, x] - potential[y - 1, x]
                gradx = potential[y, x + 1] - potential[y, x - 1]
            mag = math.hypot(gradx, grady)
            if mag == 0:
                return None   # exact local minimum

            posx -= (self.step_size / mag) * gradx
            posy -= (self.step_size / mag) * grady

            if not (0 <= posy < potential.shape[0] and 0 <= posx < potential.shape[1]):
                return None
            if grid[int(posy), int(posx)] == 1:
                return None   # descended into an obstacle cell

            path.append((posx, posy))

            if abs(ex - posx) <= 1 and abs(ey - posy) <= 1:
                path.append((float(ex), float(ey)))
                return path

            goal_dist = math.hypot(ex - posx, ey - posy)
            if goal_dist < best_goal_dist - 1e-6:
                best_goal_dist = goal_dist
                steps_since_best = 0
            else:
                steps_since_best += 1
                if steps_since_best >= self.stall_window:
                    return None   # oscillating in a local minimum

        return None

    def _get_grad_bilinear(self, potential, posx, posy):
        """Central-difference gradient of the bilinearly interpolated potential."""
        gradx = (self._bilinear_interp(potential, posx + 1, posy) -
                 self._bilinear_interp(potential, posx - 1, posy))
        grady = (self._bilinear_interp(potential, posx, posy + 1) -
                 self._bilinear_interp(potential, posx, posy - 1))
        return gradx, grady

    @staticmethod
    def _bilinear_interp(field, x, y):
        """Sample field at continuous (x, y); field is indexed field[y, x]."""
        y0, x0 = math.floor(y), math.floor(x)
        y1, x1 = y0 + 1, x0 + 1
        y0 = max(0, min(y0, field.shape[0] - 1))
        y1 = max(0, min(y1, field.shape[0] - 1))
        x0 = max(0, min(x0, field.shape[1] - 1))
        x1 = max(0, min(x1, field.shape[1] - 1))
        dy = y - math.floor(y)
        dx = x - math.floor(x)
        return (  (1 - dy) * (1 - dx) * field[y0, x0]
                + (1 - dy) *      dx  * field[y0, x1]
                +      dy  * (1 - dx) * field[y1, x0]
                +      dy  *      dx  * field[y1, x1])

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _visualize(self):
        from PIL import Image

        grey = self.map.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()

        if self._path:
            # Path coords are continuous — round only for pixel painting.
            for x, y in self._path:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < rgb.shape[0] and 0 <= xi < rgb.shape[1]:
                    rgb[yi, xi] = [128, 128, 255]

        sx, sy = self.map.start
        ex, ey = self.map.end
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]

        Image.fromarray(rgb, mode='RGB').save('potential_field_result.png')
