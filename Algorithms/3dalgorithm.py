from Algorithms.algorithm import BaseAlgorithm
from map_generation import Map3D


class Algorithm3D(BaseAlgorithm):
    def __init__(self, map: Map3D):
        super().__init__(map)

    @property
    def depth(self) -> int:
        return self.map.depth

    @property
    def height(self) -> int:
        return self.map.height

    @property
    def width(self) -> int:
        return self.map.width
