import heapq
import math
import time
import numpy as np

from Algorithms.algorithm import BaseAlgorithm


class AStarAlgorithm(BaseAlgorithm):
    # Move tables in internal (reversed) coordinate order.
    # 2D entries: (dr, dc, cost)  —  internal coords are (row, col)
    # 3D entries: (dz, dy, dx, cost)  —  internal coords are (z, y, x)
    _MOVES = {
        2: [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),  (1,  1, math.sqrt(2)),
        ],
        3: [
            (dz, dy, dx, math.sqrt(dx * dx + dy * dy + dz * dz))
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ],
    }

    def __init__(self, map, timeout: float = 30.0):
        super().__init__(map)
        self.timeout = timeout
        self._path = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        self._path = self._astar(
            self.map.grid, self.map.start, self.map.end, self.timeout
        )
        if visualize:
            self._visualize()
        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    @staticmethod
    def is_solvable(grid: np.ndarray, start: tuple, end: tuple, timeout: float = 30.0) -> bool:
        """Standalone solvability check that does not require a Map object."""
        return AStarAlgorithm._astar(grid, start, end, timeout) is not None

    # ------------------------------------------------------------------
    # Core search (classmethod so is_solvable can call it without an instance)
    # ------------------------------------------------------------------

    @classmethod
    def _astar(cls, grid: np.ndarray, start: tuple, end: tuple, timeout: float):
        ndim = grid.ndim
        if ndim not in cls._MOVES:
            raise ValueError(f"Unsupported grid dimensionality: {ndim}")

        s = cls._to_internal(start)
        e = cls._to_internal(end)

        if grid[s] != 0 or grid[e] != 0:
            return None

        t0 = time.perf_counter()
        moves = cls._MOVES[ndim]

        pq = [(cls._h(s, e), 0.0, s)]
        g = {s: 0.0}
        prev = {}

        while pq:
            if time.perf_counter() - t0 > timeout:
                return None

            _, cost, cur = heapq.heappop(pq)

            if cur == e:
                return cls._reconstruct(prev, s, e)

            if cost > g.get(cur, float('inf')):
                continue

            for *deltas, w in moves:
                nb = tuple(c + d for c, d in zip(cur, deltas))
                if all(0 <= nb[i] < grid.shape[i] for i in range(ndim)) and grid[nb] == 0:
                    ng = cost + w
                    if ng < g.get(nb, float('inf')):
                        g[nb] = ng
                        prev[nb] = cur
                        heapq.heappush(pq, (ng + cls._h(nb, e), ng, nb))

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_internal(point: tuple) -> tuple:
        """(x, y, ...) → (..., y, x) to match numpy grid indexing."""
        return tuple(reversed(point))

    @staticmethod
    def _from_internal(point: tuple) -> tuple:
        """(..., y, x) → (x, y, ...) back to user-facing convention."""
        return tuple(reversed(point))

    @staticmethod
    def _h(a: tuple, b: tuple) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    @classmethod
    def _reconstruct(cls, prev: dict, start: tuple, end: tuple) -> list:
        path, cur = [], end
        while cur in prev:
            path.append(cls._from_internal(cur))
            cur = prev[cur]
        path.append(cls._from_internal(start))
        return path[::-1]

    def _visualize(self):
        ndim = self.map.grid.ndim
        if ndim == 2:
            self._visualize_2d()
        elif ndim == 3:
            self._visualize_3d()

    def _visualize_2d(self):
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

        Image.fromarray(rgb, mode='RGB').save('astar_result.png')

    def _visualize_3d(self):
        self.map.save_image_slices('astar_result_3d')
