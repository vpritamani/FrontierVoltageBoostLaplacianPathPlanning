import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from Algorithms.algorithm import BaseAlgorithm


class PotentialFieldAlgorithmND(BaseAlgorithm):
    """Artificial potential field planner for arbitrary dimensionality.

    Same formulation as the 2D baseline, generalised:

    * attractive:  U_att = 0.5 · k_att · Σ_d (x_d − goal_d)²
    * repulsive:   U_rep = 0.5 · k_rep · (1/d − 1/q*)²  for d < q*, else 0,
      where d is the Euclidean distance to the nearest obstacle
      (``distance_transform_edt`` is dimension-agnostic).

    Gradient descent samples the field at continuous positions with N-linear
    interpolation when ``bilinear_interpolation`` is True (bilinear in 2D,
    trilinear in 3D, ...), or nearest-cell central differences when False.
    The returned path keeps raw decimal (x, y, ...) positions.

    ``use_gpu`` assembles the potential with PyTorch on CUDA when available
    (falling back to PyTorch-CPU otherwise); the distance transform itself
    stays on the CPU (SciPy). The default pure-NumPy path is unchanged.

    Local minima are detected via ``stall_window`` (no progress toward the
    goal) and reported honestly as unsolved.

    Points follow the user (x, y, ...) convention; the grid is indexed with
    reversed coordinates, matching the other N-D algorithms in this repo.
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
        self.use_gpu = use_gpu
        self._path = None
        self._potential = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid
        start = self.map.start   # (x, y, ...)
        end = self.map.end

        if self.use_gpu:
            self._potential = self._compute_potential_torch(grid, end)
        else:
            self._potential = self._compute_potential(grid, end)
        self._path = self._gradient_descent(self._potential, grid, start, end)

        return self._path

    def result(self, visualize=False):
        return self._path

    # ------------------------------------------------------------------
    # Potential field
    # ------------------------------------------------------------------

    def _compute_potential(self, grid, goal):
        goal_idx = tuple(reversed(goal))              # grid-order goal
        coords = np.indices(grid.shape, dtype=float)  # grid-order axes

        u_att = np.zeros(grid.shape, dtype=float)
        for d in range(grid.ndim):
            u_att += (coords[d] - goal_idx[d]) ** 2
        u_att *= 0.5 * self.k_att

        obstacle = grid == 1
        dist = distance_transform_edt(~obstacle)

        u_rep = np.zeros_like(u_att)
        near = (dist < self.q_star) & (dist > 0)
        u_rep[near] = 0.5 * self.k_rep * (1.0 / dist[near] - 1.0 / self.q_star) ** 2
        u_rep[obstacle] = self.OBSTACLE_POTENTIAL

        return u_att + u_rep

    def _compute_potential_torch(self, grid, goal):
        """Same potential, elementwise math on a torch device (CUDA if present)."""
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        goal_idx = tuple(reversed(goal))
        u_att = torch.zeros(grid.shape, dtype=torch.double, device=device)
        for d in range(grid.ndim):
            axis = torch.arange(grid.shape[d], dtype=torch.double, device=device)
            shape = [1] * grid.ndim
            shape[d] = grid.shape[d]
            u_att += (axis.view(shape) - goal_idx[d]) ** 2
        u_att *= 0.5 * self.k_att

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
    # Gradient descent (grid-order internally, user-order in the path)
    # ------------------------------------------------------------------

    def _gradient_descent(self, potential, grid, start, end):
        ndim = potential.ndim
        end_g = [float(c) for c in reversed(end)]
        pos = [float(c) for c in reversed(start)]

        # Raw continuous positions in user (x, y, ...) order.
        path = [tuple(reversed(pos))]

        best_goal_dist = math.sqrt(sum((e - p) ** 2 for e, p in zip(end_g, pos)))
        steps_since_best = 0

        for _ in range(self.max_iters):
            if any(not (1 <= pos[d] < potential.shape[d] - 1) for d in range(ndim)):
                return None

            if self.bilinear_interpolation:
                grad = []
                for d in range(ndim):
                    fwd = list(pos); fwd[d] += 1
                    bwd = list(pos); bwd[d] -= 1
                    grad.append(self._nlinear_interp(potential, fwd) -
                                self._nlinear_interp(potential, bwd))
            else:
                idx = [math.floor(p) for p in pos]
                grad = []
                for d in range(ndim):
                    fwd = idx.copy(); fwd[d] += 1
                    bwd = idx.copy(); bwd[d] -= 1
                    grad.append(potential[tuple(fwd)] - potential[tuple(bwd)])

            mag = math.sqrt(sum(g * g for g in grad))
            if mag == 0:
                return None   # exact local minimum

            for d in range(ndim):
                pos[d] -= (self.step_size / mag) * grad[d]

            if any(not (0 <= pos[d] < potential.shape[d]) for d in range(ndim)):
                return None
            if grid[tuple(int(p) for p in pos)] == 1:
                return None   # descended into an obstacle cell

            path.append(tuple(reversed(pos)))

            if all(abs(end_g[d] - pos[d]) <= 1 for d in range(ndim)):
                path.append(tuple(float(c) for c in end))
                return path

            goal_dist = math.sqrt(sum((e - p) ** 2 for e, p in zip(end_g, pos)))
            if goal_dist < best_goal_dist - 1e-6:
                best_goal_dist = goal_dist
                steps_since_best = 0
            else:
                steps_since_best += 1
                if steps_since_best >= self.stall_window:
                    return None   # oscillating in a local minimum

        return None

    @staticmethod
    def _nlinear_interp(field, pos):
        """N-linear interpolation at continuous grid-order position ``pos``."""
        ndim = len(pos)
        floors = [math.floor(p) for p in pos]
        fracs = [p - math.floor(p) for p in pos]
        result = 0.0
        for corner in range(1 << ndim):
            idx = []
            weight = 1.0
            for d in range(ndim):
                if corner & (1 << d):
                    idx.append(max(0, min(floors[d] + 1, field.shape[d] - 1)))
                    weight *= fracs[d]
                else:
                    idx.append(max(0, min(floors[d], field.shape[d] - 1)))
                    weight *= (1 - fracs[d])
            result += weight * field[tuple(idx)]
        return result
