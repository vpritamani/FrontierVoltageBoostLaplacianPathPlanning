import math
import numpy as np
from Algorithms.algorithm import BaseAlgorithm


class FrontierVoltageBoostLaplaceND(BaseAlgorithm):
    """
    Dimension-agnostic Frontier Voltage Boost Laplace.

    Works identically to the 2D/3D variants but generalises:
      - Laplace averaging to 2*ndim face-neighbours, weighted 1/(2*ndim)
      - Gradient interpolation to N-linear (bilinear in 2D, trilinear in 3D, ...)
        using all 2^ndim surrounding corners

    start/end follow the same (x, y, ...) user convention as Map2D/Map3D.
    Internally all grid operations use reversed (last-axis-first) indexing.
    """

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
        start = self.map.start   # (x, y, ...) user coords
        end   = self.map.end     # (x, y, ...) user coords

        end_idx   = tuple(reversed(end))    # grid index for end
        start_idx = tuple(reversed(start))  # grid index for start

        v_max = 1.0
        phi = np.ones(grid.shape, dtype=np.float64)
        phi[end_idx] = 0.0

        obstacle, solved, free = self._build_masks(grid, end_idx)
        phi[obstacle] = v_max

        while not solved[start_idx]:
            for _ in range(self.n_l):
                self._laplace_step(phi, free, end_idx)

            s_new = free & (phi <= v_max - self.epsilon)
            solved |= s_new
            free   &= ~s_new

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

    def _build_masks(self, grid, end_idx):
        obstacle = grid == 1
        for d in range(grid.ndim):
            sl_lo = [slice(None)] * grid.ndim; sl_lo[d] = 0
            sl_hi = [slice(None)] * grid.ndim; sl_hi[d] = -1
            obstacle[tuple(sl_lo)] = True
            obstacle[tuple(sl_hi)] = True

        solved = np.zeros(grid.shape, dtype=bool)
        solved[end_idx] = True

        free = ~obstacle & ~solved
        return obstacle, solved, free

    def _laplace_step(self, phi, free, end_idx):
        """
        Face-neighbour Laplace average for arbitrary ndim.
        Each free cell becomes the mean of its 2*ndim axis-aligned neighbours.
        """
        ndim = phi.ndim
        padded = np.pad(phi, 1, mode='edge')
        new_phi = np.zeros_like(phi)
        for d in range(ndim):
            for direction in (slice(None, -2), slice(2, None)):
                sl = [slice(1, -1)] * ndim
                sl[d] = direction
                new_phi += padded[tuple(sl)]
        new_phi *= 1.0 / (2 * ndim)
        phi[free]    = new_phi[free]
        phi[end_idx] = 0.0  # pin goal throughout

    def _gradient_descent(self, phi, start, end):
        """
        Steepest descent in internal (grid-index) coordinates.
        pos is stored as a list in grid order: [last_user_axis, ..., first_user_axis].
        Path is emitted in user order: (x, y, ...).
        """
        ndim = phi.ndim
        end_g = list(reversed(end))    # grid-order end coords
        pos   = [float(c) for c in reversed(start)]  # grid-order position

        # Raw continuous positions in user (x, y, ...) order — round only for
        # visualization, never for smoothness metrics or reporting.
        path = [tuple(reversed(pos))]

        for _ in range(10000):
            if any(not (0 < pos[d] < phi.shape[d] - 1) for d in range(ndim)):
                return None

            if self.bilinear_interpolation:
                grad = self._get_grad_nlinear(phi, pos)
            else:
                idx = [math.floor(p) for p in pos]
                point_v = phi[tuple(idx)]
                grad = []
                for d in range(ndim):
                    idx_fwd = idx.copy(); idx_fwd[d] += 1
                    idx_bwd = idx.copy(); idx_bwd[d] -= 1
                    grad.append(
                        self._fv(phi[tuple(idx_fwd)], point_v) -
                        self._fv(phi[tuple(idx_bwd)], point_v)
                    )

            mag = math.sqrt(sum(g * g for g in grad))
            if mag == 0:
                return None

            step = self.step_size / mag
            for d in range(ndim):
                pos[d] -= step * grad[d]

            path.append(tuple(reversed(pos)))

            if all(pos[d] - 1 <= end_g[d] <= pos[d] + 1 for d in range(ndim)):
                path.append(tuple(float(c) for c in end))
                return path

        return None

    @staticmethod
    def _fv(nv, pv):
        """Cap neighbour value to ceil(current value) — keeps descent within the active wavefront."""
        return min(nv, math.ceil(pv))

    @staticmethod
    def _nlinear_interp(phi, pos):
        """
        N-linear interpolation at continuous grid position pos.
        Enumerates all 2^ndim corners via bit mask; generalises bilinear (2D)
        and trilinear (3D) to arbitrary dimensionality.
        """
        ndim = len(pos)
        floors = [math.floor(p) for p in pos]
        fracs  = [p - math.floor(p) for p in pos]
        result = 0.0
        for corner in range(1 << ndim):
            idx    = []
            weight = 1.0
            for d in range(ndim):
                if corner & (1 << d):
                    idx.append(max(0, min(floors[d] + 1, phi.shape[d] - 1)))
                    weight *= fracs[d]
                else:
                    idx.append(max(0, min(floors[d], phi.shape[d] - 1)))
                    weight *= (1 - fracs[d])
            result += weight * phi[tuple(idx)]
        return result

    def _get_grad_nlinear(self, phi, pos):
        point_v = self._nlinear_interp(phi, pos)
        grad = []
        for d in range(len(pos)):
            pos_fwd = list(pos); pos_fwd[d] += 1
            pos_bwd = list(pos); pos_bwd[d] -= 1
            grad.append(
                self._fv(self._nlinear_interp(phi, pos_fwd), point_v) -
                self._fv(self._nlinear_interp(phi, pos_bwd), point_v)
            )
        return grad

    # ------------------------------------------------------------------
    # Visualize (2D and 3D only; higher dims are skipped)
    # ------------------------------------------------------------------

    def _visualize(self):
        ndim = self.map.grid.ndim
        if ndim == 2:
            self._visualize_2d()
        elif ndim == 3:
            self._visualize_3d()

    def _visualize_2d(self):
        from PIL import Image
        grey = self.map.to_image()
        rgb  = np.stack([grey, grey, grey], axis=2).copy()
        if self._path:
            # Continuous coords — round only for pixel painting.
            for pt in self._path:
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    rgb[y, x] = [128, 128, 255]
        sx, sy = self.map.start[0], self.map.start[1]
        ex, ey = self.map.end[0],   self.map.end[1]
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]
        Image.fromarray(rgb, mode='RGB').save('fvb_nd_result.png')

    def _visualize_3d(self):
        import os
        from PIL import Image
        path_set = set(tuple(int(round(c)) for c in pt) for pt in self._path) if self._path else set()
        sx, sy, sz = self.map.start[0], self.map.start[1], self.map.start[2]
        ex, ey, ez = self.map.end[0],   self.map.end[1],   self.map.end[2]
        out_dir = 'fvb_nd_3d_result'
        os.makedirs(out_dir, exist_ok=True)
        for z in range(self.map.depth):
            grey = self.map.to_image_slice(z)
            rgb  = np.stack([grey, grey, grey], axis=2).copy()
            for pt in path_set:
                x, y, pz = pt[0], pt[1], pt[2]
                if pz == z and 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    rgb[y, x] = [128, 128, 255]
            if sz == z:
                rgb[sy, sx] = [0, 255, 0]
            if ez == z:
                rgb[ey, ex] = [255, 0, 0]
            Image.fromarray(rgb, mode='RGB').save(
                os.path.join(out_dir, f'slice_{z:04d}.png')
            )
