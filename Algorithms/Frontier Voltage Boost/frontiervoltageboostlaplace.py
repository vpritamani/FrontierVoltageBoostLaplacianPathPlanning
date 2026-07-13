import math
import numpy as np
from Algorithms import Algorithm2D


class FrontierVoltageBoostLaplace(Algorithm2D):

    def __init__(self, map, n_l: int, epsilon: float, step_size: float = 1.0,
                 bilinear_interpolation: bool = True):
        super().__init__(map)
        self.n_l = n_l
        self.epsilon = epsilon
        self.step_size = step_size
        self.bilinear_interpolation = bilinear_interpolation
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
        When bilinear_interpolation is True, phi is sampled at continuous positions
        using bilinear interpolation instead of nearest-cell indexing.

        The returned path keeps the raw continuous (x, y) positions — this is
        where the smoothness advantage of the method lives. Round only for
        pixel-level visualization, never for metrics or reporting.
        """
        sx, sy = start
        ex, ey = end

        posx = float(sx)
        posy = float(sy)

        path = [(posx, posy)]

        for _ in range(10000):
            if not (0 < posy < phi.shape[0] - 1 and 0 < posx < phi.shape[1] - 1):
                return None

            if self.bilinear_interpolation:
                gradx, grady = self._get_grad_bilinear(phi, posx, posy)
            else:
                y = math.floor(posy)
                x = math.floor(posx)
                point_v = phi[y, x]
                grady = self._fv(phi[y + 1, x], point_v) - self._fv(phi[y - 1, x], point_v)
                gradx = self._fv(phi[y, x + 1], point_v) - self._fv(phi[y, x - 1], point_v)

            mag = math.sqrt(gradx ** 2 + grady ** 2)
            if mag == 0:
                return None

            step = self.step_size / mag
            posx -= step * gradx
            posy -= step * grady

            path.append((posx, posy))

            if posx - 1 <= ex <= posx + 1 and ey - 1 <= posy <= ey + 1:
                path.append((float(ex), float(ey)))
                return path

        return None

    @staticmethod
    def _bilinear_interp(phi, x, y):
        """Sample phi at continuous (x, y); grid is indexed phi[y, x]."""
        y0, x0 = math.floor(y), math.floor(x)
        y1, x1 = y0 + 1, x0 + 1
        y0 = max(0, min(y0, phi.shape[0] - 1))
        y1 = max(0, min(y1, phi.shape[0] - 1))
        x0 = max(0, min(x0, phi.shape[1] - 1))
        x1 = max(0, min(x1, phi.shape[1] - 1))
        dy = y - math.floor(y)
        dx = x - math.floor(x)
        return (  (1 - dy) * (1 - dx) * phi[y0, x0]
                + (1 - dy) *      dx  * phi[y0, x1]
                +      dy  * (1 - dx) * phi[y1, x0]
                +      dy  *      dx  * phi[y1, x1])

    @staticmethod
    def _fv(nv, pv):
        """Cap neighbour value to ceil(current value) to stay within the active wavefront."""
        return min(nv, math.ceil(pv))

    def _get_grad_bilinear(self, phi, posx, posy):
        point_v = self._bilinear_interp(phi, posx, posy)
        gradx = (self._fv(self._bilinear_interp(phi, posx + 1, posy), point_v) -
                 self._fv(self._bilinear_interp(phi, posx - 1, posy), point_v))
        grady = (self._fv(self._bilinear_interp(phi, posx, posy + 1), point_v) -
                 self._fv(self._bilinear_interp(phi, posx, posy - 1), point_v))
        return gradx, grady

    def _visualize(self):
        from PIL import Image
        grey = self.map.to_image()                          # (H, W) uint8
        rgb = np.stack([grey, grey, grey], axis=2).copy()  # (H, W, 3)

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

        Image.fromarray(rgb, mode='RGB').save('frontier_voltage_boost_result.png')
