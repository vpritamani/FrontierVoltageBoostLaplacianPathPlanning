from abc import ABC, abstractmethod
from map_generation import Map


class BaseAlgorithm(ABC):
    def __init__(self, map: Map):
        self.map = map

    def setup(self):
        pass

    @abstractmethod
    def solve(self, visualize=False): ...

    @abstractmethod
    def result(self, visualize=False): ...
