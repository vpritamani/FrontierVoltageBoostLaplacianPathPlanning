import math
import numpy as np
from Algorithms import Algorithm2D


class FrontierVoltageBoostLaplace(Algorithm2D):

    def __init__(self, map, n_l: int, epsilon: float, step_size: float = 1.0):
        super().__init__(map)
        self.n_l = n_l
        self.epsilon = epsilon
        self.step_size = step_size
        self._path = None
        self._phi = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid
        start = self.map.start  # (x, y)
        end = self.map.end      # (x, y)
        sx, sy = start
        ex, ey = end

        v_max = 1.0
        phi = np.ones(grid.shape, dtype=np.float64)
        phi[ey, ex] = 0.0

        obstacle, solved, free = self._build_masks(grid, ey, ex)
        phi[obstacle] = v_max

        while not solved[sy, sx]:
            for _ in range(self.n_l):
                self._laplace_step(phi, free, ey, ex)

            s_new = free & (phi <= v_max - self.epsilon)
            solved |= s_new
            free &= ~s_new

            v_max += 1.0
            phi[free | obstacle] = v_max

        self._phi = phi
        self._path = self._gradient_descent(phi, start, end)

        if visualize:
            self._visualize()

        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_masks(self, grid, ey, ex):
        obstacle = grid == 1
        # Treat map boundary as obstacles — same as original implementation
        obstacle[0, :] = True
        obstacle[-1, :] = True
        obstacle[:, 0] = True
        obstacle[:, -1] = True

        solved = np.zeros(grid.shape, dtype=bool)
        solved[ey, ex] = True

        free = ~obstacle & ~solved
        return obstacle, solved, free

    def _laplace_step(self, phi, free, ey, ex):
        """4-neighbour Laplace average applied only to free cells. Mutates phi in-place."""
        padded = np.pad(phi, 1, mode='edge')
        new_phi = (
            padded[:-2, 1:-1] +  # up
            padded[2:,  1:-1] +  # down
            padded[1:-1, :-2] +  # left
            padded[1:-1, 2:]     # right
        ) * 0.25
        phi[free] = new_phi[free]
        phi[ey, ex] = 0.0  # pin goal throughout

    def _gradient_descent(self, phi, start, end):
        """
        Follow steepest descent of phi from start to end.
        floor_voltage caps each neighbour's contribution to ceil(current voltage),
        preventing the path from skipping over wavefront boundaries.
        """
        sx, sy = start
        ex, ey = end

        posr = float(sy)  # current row  (y)
        posc = float(sx)  # current col  (x)

        path = [(round(posc), round(posr))]

        for _ in range(10000):
            r = math.floor(posr)
            c = math.floor(posc)

            if not (0 < r < phi.shape[0] - 1 and 0 < c < phi.shape[1] - 1):
                return None

            point_v = phi[r, c]

            def floor_v(nv, pv=point_v):
                return min(nv, math.ceil(pv))

            gradr = floor_v(phi[r + 1, c]) - floor_v(phi[r - 1, c])
            gradc = floor_v(phi[r, c + 1]) - floor_v(phi[r, c - 1])

            mag = math.sqrt(gradr ** 2 + gradc ** 2)
            if mag == 0:
                return None

            step = self.step_size / mag
            posr -= step * gradr
            posc -= step * gradc

            path.append((math.floor(posc), math.floor(posr)))

            if posc - 1 <= ex <= posc + 1 and ey - 1 <= posr <= ey + 1:
                path.append((ex, ey))
                return path

        return None

    def _visualize(self):
        from PIL import Image
        grey = self.map.to_image()                          # (H, W) uint8
        rgb = np.stack([grey, grey, grey], axis=2).copy()  # (H, W, 3)

        if self._path:
            for x, y in self._path:
                if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    rgb[y, x] = [128, 128, 255]

        sx, sy = self.map.start
        ex, ey = self.map.end
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]

        Image.fromarray(rgb, mode='RGB').save('frontier_voltage_boost_result.png')
