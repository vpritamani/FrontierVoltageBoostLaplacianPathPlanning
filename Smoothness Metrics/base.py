from abc import ABC, abstractmethod


class BaseSmoothnessMetric(ABC):
    """Abstract base for all path smoothness metrics.

    All concrete metrics accept a path as an ordered list of (x, y) tuples
    and return either a scalar float or a dict of named floats.
    Lower values indicate smoother paths for scalar metrics.
    """

    @abstractmethod
    def compute(self, path: list[tuple[float, float]]) -> "float | dict":
        """Compute the metric for *path*.

        Args:
            path: Ordered list of ``(x, y)`` coordinate tuples from start to end.

        Returns:
            A scalar ``float``, or a ``dict[str, float]`` for multi-value metrics.
        """

    def __call__(self, path: list[tuple[float, float]]) -> "float | dict":
        return self.compute(path)
