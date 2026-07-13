import math

import numpy as np

from .base import BaseSmoothnessMetric


class SteeringPenaltySmoothness(BaseSmoothnessMetric):
    """Steering penalty metric — the recommended metric for path evaluation.

    For every waypoint *i* and every symmetric offset ``1 … sweep_range``,
    the steering angle formed by ``(path[i-offset], path[i], path[i+offset])``
    is computed.  Only angles that exceed *theta_degrees* contribute; the
    excess ``(angle - theta)`` is accumulated and returned in degrees.

    Small incidental bends are ignored; only genuinely sharp turns are penalised.
    Validated empirically to best distinguish path quality in robot path planning.

    Args:
        theta_degrees: Penalty threshold in degrees (default ``30``).  Steering
            angles below this are considered acceptable and incur no penalty.
        sweep_range: Number of symmetric neighbors to check at each waypoint
            (default ``5``).  Higher values capture longer-range curvature.

    Returns:
        float: Total excess steering angle in degrees.  Lower is smoother.
    """

    def __init__(self, theta_degrees: float = 30.0, sweep_range: int = 5):
        self.theta_degrees = theta_degrees
        self.sweep_range = sweep_range

    @staticmethod
    def _steering_angle(p1: tuple, p2: tuple, p3: tuple) -> float:
        # Incoming direction p1→p2, outgoing direction p2→p3.
        # Straight path → parallel → angle 0; sharp turn → larger angle.
        v1 = np.array(p2, dtype=float) - np.array(p1, dtype=float)
        v2 = np.array(p3, dtype=float) - np.array(p2, dtype=float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)))

    def compute(self, path: list[tuple[float, float]]) -> float:
        """Return total excess steering penalty in degrees."""
        if len(path) < 3:
            return 0.0

        theta_rad = math.radians(self.theta_degrees)
        penalty = 0.0

        for i in range(len(path)):
            for offset in range(1, self.sweep_range + 1):
                prev_idx = i - offset
                next_idx = i + offset
                if 0 <= prev_idx and next_idx < len(path):
                    angle = self._steering_angle(path[prev_idx], path[i], path[next_idx])
                    if angle > theta_rad:
                        penalty += angle - theta_rad

        return math.degrees(penalty)

    def compute_at_thresholds(
        self,
        path: list[tuple[float, float]],
        thresholds: list[float] | None = None,
    ) -> dict[float, float]:
        """Compute penalty at multiple angle thresholds.

        Args:
            path: Ordered ``(x, y)`` tuple list.
            thresholds: Threshold angles in degrees.  Defaults to
                ``[10, 15, 20, 25, 30, 35, 40, 45]``.

        Returns:
            ``{threshold_degrees: penalty_score}`` for each threshold.
        """
        if thresholds is None:
            thresholds = [10, 15, 20, 25, 30, 35, 40, 45]
        return {
            t: SteeringPenaltySmoothness(
                theta_degrees=t, sweep_range=self.sweep_range
            ).compute(path)
            for t in thresholds
        }
