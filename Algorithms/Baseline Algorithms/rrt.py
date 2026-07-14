import math
import random

import numpy as np

from Algorithms.algorithm import BaseAlgorithm


class RRTAlgorithm(BaseAlgorithm):
    """Rapidly-exploring Random Tree — dimension-agnostic baseline.

    Works on any N-dimensional grid map (reads ``grid.ndim`` at solve time,
    same as A*). Points follow the user (x, y, ...) convention; the grid is
    indexed with reversed coordinates.

    Standard RRT loop:

    1. Sample a random free-space point (with probability ``goal_sample_rate``
       the goal itself is sampled, biasing growth toward it).
    2. Find the nearest tree node (vectorized Euclidean search).
    3. Steer from it toward the sample by at most ``step_size`` — node
       positions are continuous, so the tree (and the returned path) naturally
       lives at decimal coordinates.
    4. Add the new node if the connecting segment is collision-free
       (sampled at sub-cell resolution).
    5. When a node lands within ``goal_tolerance`` of the goal and the final
       segment is free, walk parent pointers back to the start.

    Returns the path start → end as float tuples, or ``None`` after
    ``max_iterations`` samples without reaching the goal.
    """

    _COLLISION_RESOLUTION = 0.25   # sampling step (cells) along tested segments

    def __init__(
        self,
        map,
        max_iterations: int = 5000,
        step_size: float = 3.0,
        goal_sample_rate: float = 0.05,
        goal_tolerance: float = 1.5,
        seed: int = -1,
    ):
        super().__init__(map)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        self.seed = seed               # -1 = unseeded (fresh randomness per solve)
        self._path = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid
        start = np.asarray(self.map.start, dtype=float)
        goal = np.asarray(self.map.end, dtype=float)
        ndim = grid.ndim
        sizes = grid.shape[::-1]       # user-order (x, y, ...) sizes

        rng = random.Random(self.seed) if self.seed >= 0 else random.Random()

        if self._cell_blocked(grid, start) or self._cell_blocked(grid, goal):
            self._path = None
            return None

        nodes = np.empty((self.max_iterations + 1, ndim), dtype=float)
        nodes[0] = start
        parents = [-1]
        count = 1

        for _ in range(self.max_iterations):
            if rng.random() < self.goal_sample_rate:
                sample = goal
            else:
                sample = np.array([rng.uniform(0, s - 1) for s in sizes])

            # Nearest existing node (vectorized over the filled prefix).
            diffs = nodes[:count] - sample
            nearest_idx = int(np.argmin(np.einsum('ij,ij->i', diffs, diffs)))
            nearest = nodes[nearest_idx]

            direction = sample - nearest
            dist = float(np.linalg.norm(direction))
            if dist < 1e-9:
                continue
            new = nearest + direction * min(1.0, self.step_size / dist)

            if not self._segment_free(grid, nearest, new):
                continue

            nodes[count] = new
            parents.append(nearest_idx)
            new_idx = count
            count += 1

            if (np.linalg.norm(new - goal) <= self.goal_tolerance
                    and self._segment_free(grid, new, goal)):
                path = [tuple(float(c) for c in goal)]
                idx = new_idx
                while idx != -1:
                    path.append(tuple(float(c) for c in nodes[idx]))
                    idx = parents[idx]
                self._path = path[::-1]
                if visualize:
                    self._visualize()
                return self._path

        self._path = None
        return None

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    @staticmethod
    def _cell_blocked(grid, point) -> bool:
        """True if the cell containing user-order ``point`` is out of bounds or an obstacle."""
        idx = tuple(int(math.floor(c)) for c in reversed(tuple(point)))
        if any(not (0 <= idx[d] < grid.shape[d]) for d in range(grid.ndim)):
            return True
        return grid[idx] != 0

    def _segment_free(self, grid, a, b) -> bool:
        """Sample the segment a→b at sub-cell resolution; every cell must be free."""
        dist = float(np.linalg.norm(b - a))
        steps = max(1, int(math.ceil(dist / self._COLLISION_RESOLUTION)))
        for i in range(steps + 1):
            p = a + (b - a) * (i / steps)
            if self._cell_blocked(grid, p):
                return False
        return True

    # ------------------------------------------------------------------
    # Visualization (2D only; higher dims use the app's viewers)
    # ------------------------------------------------------------------

    def _visualize(self):
        if self.map.grid.ndim != 2:
            return
        from PIL import Image
        grey = ((1 - self.map.grid) * 255).astype(np.uint8)
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
        Image.fromarray(rgb, mode='RGB').save('rrt_result.png')
