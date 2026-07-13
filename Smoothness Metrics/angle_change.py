import math

from .base import BaseSmoothnessMetric


class AngleChangeSmoothness(BaseSmoothnessMetric):
    """Total turning angle normalized by arc length.

    Uses the law of cosines to recover the exterior turning angle at each
    interior waypoint, accumulates the total, and divides by the path length
    actually traversed while computing those angles.  Paths with many sharp
    turns over a short distance score higher.

    Returns:
        float: ``total_angle_change / total_arc_length``.  Lower is smoother.
    """

    @staticmethod
    def _dist(p1: tuple, p2: tuple) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def compute(self, path: list[tuple[float, float]]) -> float:
        if len(path) < 3:
            return 0.0

        total_angle = 0.0
        total_length = 0.0
        a = self._dist(path[0], path[1])

        for i in range(2, len(path)):
            b = self._dist(path[i - 1], path[i])
            c = self._dist(path[i - 2], path[i])
            if a != 0 and b != 0:
                cos_val = (a * a + b * b - c * c) / (2.0 * a * b)
                if -1.0 <= cos_val <= 1.0:
                    angle = math.pi - math.acos(cos_val)
                    total_angle += abs(angle)
                    total_length += a + b
            a = b

        return total_angle / total_length if total_length > 0 else 0.0
