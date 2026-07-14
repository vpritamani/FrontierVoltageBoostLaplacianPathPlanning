import math
import os
import numpy as np
from Algorithms import Algorithm3D


class FrontierVoltageBoostLaplace3D(Algorithm3D):

    def __init__(self, map, n_l: int, epsilon: float, step_size: float = 1.0,
                 bilinear_interpolation: bool = True, use_gpu: bool = False):
        super().__init__(map)
        self.n_l = n_l
        self.epsilon = epsilon
        self.step_size = step_size
        self.bilinear_interpolation = bilinear_interpolation
        self.use_gpu = use_gpu   # PyTorch wavefront solve (CUDA when available)
        self._path = None
        self._phi = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid        # shape (D, H, W) = (Z, Y, X)
        start = self.map.start      # (x, y, z)
        end = self.map.end          # (x, y, z)

        if self.use_gpu:
            phi = self._solve_phi_torch(grid, start, end)
        else:
            phi = self._solve_phi(grid, start, end)

        self._phi = phi
        self._path = self._gradient_descent(phi, start, end)

        if visualize:
            self._visualize()

        return self._path

    # ------------------------------------------------------------------
    # Wavefront potential solve — CPU (NumPy, default)
    # ------------------------------------------------------------------

    def _solve_phi(self, grid, start, end):
        sx, sy, sz = start
        ex, ey, ez = end

        v_max = 1.0
        phi = np.ones(grid.shape, dtype=np.float64)
        phi[ez, ey, ex] = 0.0

        obstacle, solved, free = self._build_masks(grid, ez, ey, ex)
        phi[obstacle] = v_max

        while not solved[sz, sy, sx]:
            for _ in range(self.n_l):
                self._laplace_step(phi, free, ez, ey, ex)

            s_new = free & (phi <= v_max - self.epsilon)
            solved |= s_new
            free &= ~s_new

            v_max += 1.0
            phi[free | obstacle] = v_max

        return phi

    # ------------------------------------------------------------------
    # Wavefront potential solve — PyTorch (CUDA when available)
    # ------------------------------------------------------------------

    def _solve_phi_torch(self, grid, start, end):
        """Identical wavefront/boost logic with the 6-neighbour Laplace
        average expressed as a conv3d so it runs on the GPU."""
        import torch
        import torch.nn.functional as tF

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sx, sy, sz = start
        ex, ey, ez = end

        kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.double, device=device)
        w = 1.0 / 6.0
        kernel[0, 0, 0, 1, 1] = w
        kernel[0, 0, 2, 1, 1] = w
        kernel[0, 0, 1, 0, 1] = w
        kernel[0, 0, 1, 2, 1] = w
        kernel[0, 0, 1, 1, 0] = w
        kernel[0, 0, 1, 1, 2] = w

        obstacle_np, solved_np, free_np = self._build_masks(grid, ez, ey, ex)
        obstacle = torch.from_numpy(obstacle_np).to(device)
        solved = torch.from_numpy(solved_np).to(device)
        free = torch.from_numpy(free_np).to(device)

        v_max = 1.0
        phi = torch.ones(grid.shape, dtype=torch.double, device=device)
        phi[ez, ey, ex] = 0.0
        phi[obstacle] = v_max

        while not bool(solved[sz, sy, sx]):
            for _ in range(self.n_l):
                padded = tF.pad(phi.view(1, 1, *phi.shape), (1, 1, 1, 1, 1, 1), mode='replicate')
                new_phi = tF.conv3d(padded, kernel).view(*phi.shape)
                phi = torch.where(free, new_phi, phi)
                phi[ez, ey, ex] = 0.0

            s_new = free & (phi <= v_max - self.epsilon)
            solved |= s_new
            free &= ~s_new

            v_max += 1.0
            phi = torch.where(free | obstacle, torch.full_like(phi, v_max), phi)

        return phi.cpu().numpy()

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_masks(self, grid, ez, ey, ex):
        obstacle = grid == 1
        obstacle[0, :, :] = True
        obstacle[-1, :, :] = True
        obstacle[:, 0, :] = True
        obstacle[:, -1, :] = True
        obstacle[:, :, 0] = True
        obstacle[:, :, -1] = True

        solved = np.zeros(grid.shape, dtype=bool)
        solved[ez, ey, ex] = True

        free = ~obstacle & ~solved
        return obstacle, solved, free

    def _laplace_step(self, phi, free, ez, ey, ex):
        """6-neighbour Laplace average applied only to free cells. Mutates phi in-place."""
        padded = np.pad(phi, 1, mode='edge')
        new_phi = (
            padded[:-2, 1:-1, 1:-1] +  # z-1
            padded[2:,  1:-1, 1:-1] +  # z+1
            padded[1:-1, :-2, 1:-1] +  # y-1
            padded[1:-1, 2:,  1:-1] +  # y+1
            padded[1:-1, 1:-1, :-2] +  # x-1
            padded[1:-1, 1:-1, 2:]     # x+1
        ) * (1.0 / 6.0)
        phi[free] = new_phi[free]
        phi[ez, ey, ex] = 0.0  # pin goal throughout

    def _gradient_descent(self, phi, start, end):
        sx, sy, sz = start
        ex, ey, ez = end

        posz = float(sz)
        posy = float(sy)
        posx = float(sx)

        # Raw continuous positions — round only for visualization, never for
        # smoothness metrics or reporting.
        path = [(posx, posy, posz)]

        for _ in range(10000):
            if not (0 < posz < phi.shape[0] - 1 and
                    0 < posy < phi.shape[1] - 1 and
                    0 < posx < phi.shape[2] - 1):
                return None

            if self.bilinear_interpolation:
                gradz, grady, gradx = self._get_grad_trilinear(phi, posz, posy, posx)
            else:
                z = math.floor(posz)
                y = math.floor(posy)
                x = math.floor(posx)
                point_v = phi[z, y, x]
                gradz = self._fv(phi[z+1, y, x], point_v) - self._fv(phi[z-1, y, x], point_v)
                grady = self._fv(phi[z, y+1, x], point_v) - self._fv(phi[z, y-1, x], point_v)
                gradx = self._fv(phi[z, y, x+1], point_v) - self._fv(phi[z, y, x-1], point_v)

            mag = math.sqrt(gradz**2 + grady**2 + gradx**2)
            if mag == 0:
                return None

            step = self.step_size / mag
            posz -= step * gradz
            posy -= step * grady
            posx -= step * gradx

            path.append((posx, posy, posz))

            if (posx - 1 <= ex <= posx + 1 and
                    posy - 1 <= ey <= posy + 1 and
                    posz - 1 <= ez <= posz + 1):
                path.append((float(ex), float(ey), float(ez)))
                return path

        return None

    @staticmethod
    def _fv(nv, pv):
        """Cap neighbour value to ceil(current value) to stay within the active wavefront."""
        return min(nv, math.ceil(pv))

    @staticmethod
    def _trilinear_interp(phi, z, y, x):
        z0, y0, x0 = math.floor(z), math.floor(y), math.floor(x)
        z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1
        z0 = max(0, min(z0, phi.shape[0] - 1))
        z1 = max(0, min(z1, phi.shape[0] - 1))
        y0 = max(0, min(y0, phi.shape[1] - 1))
        y1 = max(0, min(y1, phi.shape[1] - 1))
        x0 = max(0, min(x0, phi.shape[2] - 1))
        x1 = max(0, min(x1, phi.shape[2] - 1))
        dz = z - math.floor(z)
        dy = y - math.floor(y)
        dx = x - math.floor(x)
        return (  (1-dz)*(1-dy)*(1-dx)*phi[z0, y0, x0]
                + (1-dz)*(1-dy)*   dx *phi[z0, y0, x1]
                + (1-dz)*   dy *(1-dx)*phi[z0, y1, x0]
                + (1-dz)*   dy *   dx *phi[z0, y1, x1]
                +    dz *(1-dy)*(1-dx)*phi[z1, y0, x0]
                +    dz *(1-dy)*   dx *phi[z1, y0, x1]
                +    dz *   dy *(1-dx)*phi[z1, y1, x0]
                +    dz *   dy *   dx *phi[z1, y1, x1])

    def _get_grad_trilinear(self, phi, posz, posy, posx):
        point_v = self._trilinear_interp(phi, posz, posy, posx)
        gradz = (self._fv(self._trilinear_interp(phi, posz + 1, posy, posx), point_v) -
                 self._fv(self._trilinear_interp(phi, posz - 1, posy, posx), point_v))
        grady = (self._fv(self._trilinear_interp(phi, posz, posy + 1, posx), point_v) -
                 self._fv(self._trilinear_interp(phi, posz, posy - 1, posx), point_v))
        gradx = (self._fv(self._trilinear_interp(phi, posz, posy, posx + 1), point_v) -
                 self._fv(self._trilinear_interp(phi, posz, posy, posx - 1), point_v))
        return gradz, grady, gradx

    def _visualize(self):
        from PIL import Image
        path_set = set()
        if self._path:
            # Continuous coords — round only for pixel painting.
            for x, y, z in self._path:
                path_set.add((int(round(x)), int(round(y)), int(round(z))))

        sx, sy, sz = self.map.start
        ex, ey, ez = self.map.end

        out_dir = 'frontier_voltage_boost_3d_result'
        os.makedirs(out_dir, exist_ok=True)

        for z in range(self.map.depth):
            grey = self.map.to_image_slice(z)
            rgb = np.stack([grey, grey, grey], axis=2).copy()
            for x, y, pz in path_set:
                if pz == z and 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    rgb[y, x] = [128, 128, 255]
            if sz == z:
                rgb[sy, sx] = [0, 255, 0]
            if ez == z:
                rgb[ey, ex] = [255, 0, 0]
            Image.fromarray(rgb, mode='RGB').save(
                os.path.join(out_dir, f'slice_{z:04d}.png')
            )
