from Algorithms.algorithm import BaseAlgorithm
from map_generation import Map2D


class Algorithm2D(BaseAlgorithm):
    def __init__(self, map: Map2D):
        super().__init__(map)

    @property
    def height(self) -> int:
        return self.map.height

    @property
    def width(self) -> int:
        return self.map.width
