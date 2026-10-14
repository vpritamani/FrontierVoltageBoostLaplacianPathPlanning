import importlib.util
import itertools
import math
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

from Algorithms.algorithm import BaseAlgorithm


def _load_common():
    name = 'external_algorithms_common'
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'externalcommon.py')
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_common = _load_common()


class FastMarchingAlgorithm(BaseAlgorithm):
    """Fast Marching Method planner (any dimensionality, 2–6D).

    Wraps the eikonal solver of thinks/fast-marching-method (pinned in
    ``third_party/``; compiled with ``python third_party/setup_external.py fmm``).
    The planner solves |∇T| = 1/F on the grid with the goal as the source,
    then descends T from the start — the arrival-time field has no local
    minima other than the source, so descent reaches the goal whenever the
    front got there through free space.

    Speed field F:

    * ``clearance_saturation == 0`` — plain FMM: F = 1 in free space, so T is
      the geodesic distance and the path is (near-)shortest, hugging corners.
    * ``clearance_saturation > 0`` — Fast Marching Square (Garrido et al.):
      F = min(1, d / clearance_saturation) with d the Euclidean distance to the
      nearest obstacle, which trades length for clearance and smoothness.

    The solver needs F > 0 everywhere, so obstacle cells get
    ``obstacle_speed`` (crossing one costs ~1/obstacle_speed cells — far more
    than any detour). Following the upstream README, the goal's free
    neighbours are seeded with exact distances so the high-accuracy
    (second-order) solver starts from a dilated boundary.

    Path extraction is the repo's standard normalised N-linear gradient
    descent (the same as the potential-field baselines; shared
    ``descend_field``) — the upstream library computes T only. Returned
    positions are continuous (x, y, ...) coordinates.
    """

    def __init__(self, map, clearance_saturation: float = 0.0, obstacle_speed: float = 1e-6,
                 high_accuracy: bool = True, step_size: float = 1.0,
                 bilinear_interpolation: bool = True, max_iters: int = 20000,
                 stall_window: int = 200):
        super().__init__(map)
        self.clearance_saturation = clearance_saturation
        self.obstacle_speed = obstacle_speed
        self.high_accuracy = high_accuracy
        self.step_size = step_size
        self.bilinear_interpolation = bilinear_interpolation
        self.max_iters = max_iters
        self.stall_window = stall_window
        self._fmm = _common.import_built('external_fmm', 'fmm')
        self._path = None
        self.failure_reason = None   # why the last solve found no path
        self._arrival = None

    def solve(self, visualize=False):
        grid = self.map.grid
        if not 2 <= grid.ndim <= self._fmm.MAX_DIMS:
            raise _common.UnsupportedMapError(
                f'fast marching binding supports 2-{self._fmm.MAX_DIMS}D grids, got {grid.ndim}D')

        self._arrival = self._arrival_times(grid, self.map.end)
        diag = {}
        self._path = _common.descend_field(
            self._arrival, grid, self.map.start, self.map.end,
            step_size=self.step_size, max_iters=self.max_iters,
            stall_window=self.stall_window, interpolate=self.bilinear_interpolation,
            diag=diag)
        self.failure_reason = diag.get('reason')

        if visualize:
            self._visualize()
        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    def _arrival_times(self, grid, end):
        free = grid == 0
        if self.clearance_saturation > 0:
            clearance = distance_transform_edt(free)
            speed = np.minimum(1.0, clearance / self.clearance_saturation)
        else:
            speed = np.ones(grid.shape)
        speed[~free] = self.obstacle_speed
        speed = np.maximum(speed, self.obstacle_speed)

        # Boundary: the goal plus its free neighbours (3^N block) at their
        # exact arrival times, user (x, y, ...) index order.
        end = tuple(int(c) for c in end)
        indices, times = [], []
        for delta in itertools.product((-1, 0, 1), repeat=grid.ndim):
            cell = tuple(c + d for c, d in zip(end, delta))
            g_idx = tuple(reversed(cell))
            if all(0 <= g_idx[i] < grid.shape[i] for i in range(grid.ndim)) and free[g_idx]:
                indices.append(cell)
                times.append(math.sqrt(sum(d * d for d in delta)) / speed[g_idx])
        return self._fmm.arrival_time(
            speed, np.array(indices, dtype=np.int32), np.array(times),
            self.high_accuracy)

    def _visualize(self):
        if self.map.grid.ndim != 2:
            return
        from PIL import Image
        grey = self.map.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()
        for x, y in self._path or []:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < rgb.shape[0] and 0 <= xi < rgb.shape[1]:
                rgb[yi, xi] = [128, 128, 255]
        sx, sy = self.map.start
        ex, ey = self.map.end
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]
        Image.fromarray(rgb, mode='RGB').save('fast_marching_result.png')
