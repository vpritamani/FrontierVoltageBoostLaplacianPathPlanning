import math

import numpy as np
from scipy.signal import convolve2d

from Algorithms import Algorithm2D

_KERNEL_CACHE = {}

# Default algorithm parameters — override via constructor or CLI.
DEFAULT_Q_STAR = 30
DEFAULT_K_ATT = 1.0
DEFAULT_K_REP = 1e4
DEFAULT_STEP_SIZE = 1.0
DEFAULT_MAX_ITERS = 10000


class PotentialFieldAlgorithm(Algorithm2D):
    """Artificial potential field planner with attractive and repulsive terms."""

    OBSTACLE_POTENTIAL = 1e6

    def __init__(
        self,
        map,
        q_star: int = DEFAULT_Q_STAR,
        k_att: float = DEFAULT_K_ATT,
        k_rep: float = DEFAULT_K_REP,
        step_size: float = DEFAULT_STEP_SIZE,
        max_iters: int = DEFAULT_MAX_ITERS,
    ):
        super().__init__(map)
        self.q_star = q_star
        self.k_att = k_att
        self.k_rep = k_rep
        self.step_size = step_size
        self.max_iters = max_iters
        self._path = None
        self._potential = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid
        start = self.map.start  # (x, y)
        end = self.map.end      # (x, y)

        self._potential = self._compute_potential(grid, end[1], end[0])
        self._path = self._gradient_descent(self._potential, start, end)

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

    def _compute_potential(self, grid, goal_y, goal_x):
        rows, cols = grid.shape
        gy, gx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        u_att = 0.5 * self.k_att * (
            (gy - goal_y) ** 2 + (gx - goal_x) ** 2
        ).astype(float)

        obs_map = (grid == 1).astype(float)
        kernel = self._repulsive_kernel(self.q_star, self.k_rep)
        u_rep = convolve2d(obs_map, kernel, mode='same', boundary='fill', fillvalue=0)
        u_rep[obs_map > 0.5] = self.OBSTACLE_POTENTIAL

        return u_att + u_rep

    @classmethod
    def _repulsive_kernel(cls, q_star, k_rep):
        key = (int(q_star), float(k_rep))
        kernel = _KERNEL_CACHE.get(key)
        if kernel is None:
            ys, xs = np.mgrid[-q_star:q_star + 1, -q_star:q_star + 1]
            d = np.sqrt(xs ** 2 + ys ** 2)
            kernel = np.zeros_like(d, dtype=float)
            mask = (d < q_star) & (d >= 1e-6)
            inv_d = np.zeros_like(d, dtype=float)
            inv_d[mask] = 1.0 / d[mask]
            kernel[mask] = 0.5 * k_rep * (inv_d[mask] - 1.0 / q_star) ** 2
            _KERNEL_CACHE[key] = kernel
        return kernel

    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def _gradient_descent(self, potential, start, end):
        sx, sy = start
        ex, ey = end

        posr = float(sy)
        posc = float(sx)
        path = [(round(posc), round(posr))]

        for _ in range(self.max_iters):
            y = int(posr)
            x = int(posc)

            if not (1 <= y < potential.shape[0] - 1 and 1 <= x < potential.shape[1] - 1):
                return None

            point_voltage = potential[y, x]
            gradr = (
                self._fv(point_voltage, potential[y + 1, x])
                - self._fv(point_voltage, potential[y - 1, x])
            )
            gradc = (
                self._fv(point_voltage, potential[y, x + 1])
                - self._fv(point_voltage, potential[y, x - 1])
            )
            maggrad = math.hypot(gradr, gradc)

            if maggrad == 0:
                return None

            posr -= (self.step_size / maggrad) * gradr
            posc -= (self.step_size / maggrad) * gradc

            if not (0 <= posr < potential.shape[0] and 0 <= posc < potential.shape[1]):
                return None

            if self._bilinear_interp(potential, posr, posc) >= self.OBSTACLE_POTENTIAL:
                return None

            path.append((round(posc), round(posr)))

            if abs(ey - posr) <= 1 and abs(ex - posc) <= 1:
                path.append((ex, ey))
                return path

        return None

    @staticmethod
    def _fv(point_voltage, neighbor_voltage):
        return min(neighbor_voltage, math.ceil(point_voltage))

    @staticmethod
    def _bilinear_interp(grid, y, x):
        y0 = int(math.floor(y))
        x0 = int(math.floor(x))
        y0 = max(0, min(grid.shape[0] - 1, y0))
        x0 = max(0, min(grid.shape[1] - 1, x0))
        y1 = min(y0 + 1, grid.shape[0] - 1)
        x1 = min(x0 + 1, grid.shape[1] - 1)

        dy = y - y0
        dx = x - x0

        v00 = grid[y0, x0]
        v10 = grid[y1, x0]
        v01 = grid[y0, x1]
        v11 = grid[y1, x1]

        return (
            v00 * (1 - dy) * (1 - dx)
            + v10 * dy * (1 - dx)
            + v01 * (1 - dy) * dx
            + v11 * dy * dx
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _visualize(self):
        from PIL import Image

        grey = self.map.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()

        if self._path:
            for x, y in self._path:
                if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    rgb[y, x] = [128, 128, 255]

        sx, sy = self.map.start
        ex, ey = self.map.end
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]

        Image.fromarray(rgb, mode='RGB').save('potential_field_result.png')
