import os
import numpy as np
from map import Map


class Map3D(Map):
    """
    3D voxel map. Grid shape: (depth, height, width).
    Values: 0 = free space, 1 = obstacle.
    """

    def __init__(self, grid: np.ndarray, start: tuple, end: tuple):
        assert grid.ndim == 3, "Map3D requires a 3D grid"
        super().__init__(grid, start, end)

    @property
    def depth(self) -> int:
        return self.grid.shape[0]

    @property
    def height(self) -> int:
        return self.grid.shape[1]

    @property
    def width(self) -> int:
        return self.grid.shape[2]

    def to_image_slice(self, z: int) -> np.ndarray:
        """Returns a (height, width) uint8 array for slice z: free=255 (white), obstacle=0 (black)."""
        return ((1 - self.grid[z]) * 255).astype(np.uint8)

    def save_image_slices(self, output_dir: str):
        """Saves each z-slice as a PNG image."""
        from PIL import Image
        os.makedirs(output_dir, exist_ok=True)
        for z in range(self.depth):
            Image.fromarray(self.to_image_slice(z), mode='L').save(
                os.path.join(output_dir, f"slice_{z:04d}.png")
            )
