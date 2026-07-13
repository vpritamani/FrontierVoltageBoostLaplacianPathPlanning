import math

from .base import BaseSmoothnessMetric


class DiscreteSmoothness(BaseSmoothnessMetric):
    """Discrete curvature-squared metric for grid-based paths.

    Rounds all coordinates to integers before computing distances, matching
    the discrete cell structure of grid maps.  At each interior point the
    curvature ``k = 2·angle / (a + b)`` is computed via the law of cosines;
    the sum of ``k²`` is averaged over segments.

    Suitable when the path lives on integer grid coordinates and per-cell
    curvature intensity matters more than total accumulated angle.

    Returns:
        float: mean squared discrete curvature.  Lower is smoother.
    """

    @staticmethod
    def _dist(p1: tuple, p2: tuple) -> float:
        return math.sqrt(sum((int(a) - int(b)) ** 2 for a, b in zip(p1, p2)))

    def compute(self, path: list[tuple[float, float]]) -> float:
        if len(path) < 3:
            return 0.0

        s = 0.0
        line_segs = 0
        a = self._dist(path[0], path[1])

        for i in range(2, len(path)):
            b = self._dist(path[i - 1], path[i])
            c = self._dist(path[i - 2], path[i])
            if a != 0 and b != 0:
                cos_val = (a * a + b * b - c * c) / (2.0 * a * b)
                if -1.0 <= cos_val <= 1.0:
                    angle = math.pi - math.acos(cos_val)
                    k = 2.0 * angle / (a + b)
                    s += k * k
                    line_segs += 1
            a = b

        return s / line_segs if line_segs > 0 else 0.0
