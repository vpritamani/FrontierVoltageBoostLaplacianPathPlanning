import numpy as np
from map import Map


class Map2D(Map):
    """
    2D grid map. Grid shape: (height, width).
    Values: 0 = free space, 1 = obstacle.
    """

    def __init__(self, grid: np.ndarray, start: tuple, end: tuple):
        assert grid.ndim == 2, "Map2D requires a 2D grid"
        super().__init__(grid, start, end)

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    def to_image(self) -> np.ndarray:
        """Returns a (height, width) uint8 array: free=255 (white), obstacle=0 (black)."""
        return ((1 - self.grid) * 255).astype(np.uint8)

    def save_image(self, path: str):
        from PIL import Image
        Image.fromarray(self.to_image(), mode='L').save(path)
