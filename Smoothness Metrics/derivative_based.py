import math

import numpy as np

from .base import BaseSmoothnessMetric


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dist(p1: tuple, p2: tuple) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def _direction(p1: tuple, p2: tuple) -> np.ndarray:
    v = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros_like(v)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)))


def _angle_derivatives(path: list[tuple]) -> tuple[list, list, list]:
    """Return (angle_deltas, delta_of_deltas, delta_of_delta_of_deltas).

    Uses the angle between consecutive direction vectors, which generalises to
    any number of spatial dimensions.  Angles are always in [0, π].
    """
    dirs = [_direction(path[i], path[i + 1]) for i in range(len(path) - 1)]
    ad = [_angle_between(dirs[i - 1], dirs[i]) for i in range(1, len(dirs))]
    dod = [ad[i] - ad[i - 1] for i in range(1, len(ad))]
    dodod = [dod[i] - dod[i - 1] for i in range(1, len(dod))]
    return ad, dod, dodod


_ZERO_RESULT = {
    "curvature": 0.0,
    "second_derivative": 0.0,
    "third_derivative": 0.0,
    "total": 0.0,
}


# ---------------------------------------------------------------------------
# Metric classes
# ---------------------------------------------------------------------------

class DerivativeSmoothnessV1(BaseSmoothnessMetric):
    """Derivative-based smoothness normalized by total path length.

    Computes three quantities from the heading sequence:

    * **curvature** — sum of |heading deltas| (1st derivative of heading)
    * **second_derivative** — sum of |delta-of-deltas|
    * **third_derivative** — sum of |delta-of-delta-of-deltas|

    Each is divided by the total Euclidean path length, making the score
    length-independent.  Captures oscillatory turning at all orders.

    Returns:
        dict with keys ``curvature``, ``second_derivative``, ``third_derivative``,
        ``total`` (sum of all three).  Lower is smoother.
    """

    def compute(self, path: list[tuple[float, float]]) -> dict:
        if len(path) < 4:
            return dict(_ZERO_RESULT)

        path_length = sum(_dist(path[i], path[i + 1]) for i in range(len(path) - 1))
        if path_length == 0:
            return dict(_ZERO_RESULT)

        ad, dod, dodod = _angle_derivatives(path)
        c = sum(abs(d) for d in ad)
        cr = sum(abs(d) for d in dod)
        ca = sum(abs(d) for d in dodod)

        return {
            "curvature": c / path_length,
            "second_derivative": cr / path_length,
            "third_derivative": ca / path_length,
            "total": (c + cr + ca) / path_length,
        }


class DerivativeSmoothnessV2(BaseSmoothnessMetric):
    """Derivative-based smoothness normalized by per-order segment count.

    Identical computation to :class:`DerivativeSmoothnessV1` but divides each
    order by its own number of terms rather than by path length.  Useful when
    comparing paths of very different lengths where per-step average matters
    more than overall intensity.

    Returns:
        dict with keys ``curvature``, ``second_derivative``, ``third_derivative``,
        ``total``.  Lower is smoother.
    """

    def compute(self, path: list[tuple[float, float]]) -> dict:
        if len(path) < 4:
            return dict(_ZERO_RESULT)

        ad, dod, dodod = _angle_derivatives(path)
        if not ad:
            return dict(_ZERO_RESULT)

        c = sum(abs(d) for d in ad)
        cr = sum(abs(d) for d in dod) if dod else 0.0
        ca = sum(abs(d) for d in dodod) if dodod else 0.0

        total_count = len(ad) + len(dod) + len(dodod)

        return {
            "curvature": c / len(ad),
            "second_derivative": cr / len(dod) if dod else 0.0,
            "third_derivative": ca / len(dodod) if dodod else 0.0,
            "total": (c + cr + ca) / total_count if total_count else 0.0,
        }
