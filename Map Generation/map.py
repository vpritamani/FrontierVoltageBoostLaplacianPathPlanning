import numpy as np


class Map:
    """
    Base map class. Grid convention: 0 = free space, 1 = obstacle.
    """

    def __init__(self, grid: np.ndarray, start: tuple, end: tuple):
        self.grid = grid
        self.start = start
        self.end = end
