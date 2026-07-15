import numpy as np
import random
import os
import csv
import importlib.util
from datetime import datetime

from map_generation import Map2D

_astar_path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Algorithms', 'Baseline Algorithms', 'astar.py')
)
_astar_spec = importlib.util.spec_from_file_location('astar', _astar_path)
_astar_mod = importlib.util.module_from_spec(_astar_spec)
_astar_spec.loader.exec_module(_astar_mod)
AStarAlgorithm = _astar_mod.AStarAlgorithm

_MAX_RETRIES = 100


class MapGenerator2D:
    def __init__(
        self,
        width: int,
        height: int,
        num_obstacles_range: tuple = (5, 30),
        obstacle_size_range: tuple = None,
        solvability_timeout: float = 30.0,
    ):
        self.width = width
        self.height = height
        self.num_obstacles_range = num_obstacles_range
        self.solvability_timeout = solvability_timeout

        if obstacle_size_range is None:
            min_dim = min(width, height)
            self.obstacle_size_range = (
                max(1, min_dim // 20),
                max(2, min_dim // 8),
            )
        else:
            self.obstacle_size_range = obstacle_size_range

    def generate(self):
        for _ in range(_MAX_RETRIES):
            grid = np.zeros((self.height, self.width), dtype=np.uint8)
            for _ in range(random.randint(*self.num_obstacles_range)):
                self._place_obstacle(grid)

            start = self._find_free_point(grid)
            end = self._find_free_point(grid)

            if AStarAlgorithm.is_solvable(grid, start, end, self.solvability_timeout):
                return Map2D(grid, start, end)

        raise RuntimeError(f"Could not generate a solvable 2D map after {_MAX_RETRIES} attempts")

    def generate_many(self, num_maps: int, output_dir: str = None) -> list:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = output_dir or f"2dblock_map_{timestamp}"
        os.makedirs(folder, exist_ok=True)

        csv_path = os.path.join(folder, "start_end_points.csv")
        maps = []

        with open(csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["map_name", "start_x", "start_y", "end_x", "end_y"])

            for i in range(num_maps):
                m = self.generate()
                map_name = f"block_map_{i}.npy"
                np.save(os.path.join(folder, map_name), m.grid)
                writer.writerow([map_name, m.start[0], m.start[1], m.end[0], m.end[1]])
                maps.append(m)

        print(f"Generated {num_maps} maps in '{folder}'")
        return maps

    def _place_obstacle(self, grid: np.ndarray):
        size = min(random.randint(*self.obstacle_size_range), self.width, self.height)
        x = random.randint(0, self.width - size)
        y = random.randint(0, self.height - size)
        grid[y:y + size, x:x + size] = 1

    def _find_free_point(self, grid: np.ndarray) -> tuple:
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if grid[y, x] == 0:
                return (x, y)
