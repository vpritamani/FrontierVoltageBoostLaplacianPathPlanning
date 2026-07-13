"""
SmoothnessMetrics — aggregator for all path smoothness metrics.

Quick start::

    from smoothness_metrics import SmoothnessMetrics

    path = [(0, 0), (1, 1), (2, 1), (3, 2), ...]   # list of (x, y) tuples
    sm = SmoothnessMetrics(path)

    score = sm.recommended()         # steering penalty @ theta=30 (recommended)
    all_scores = sm.compute_all()    # dict of every metric
"""

from .angle_change import AngleChangeSmoothness
from .derivative_based import DerivativeSmoothnessV1, DerivativeSmoothnessV2
from .discrete import DiscreteSmoothness
from .pathbench import PathBenchSmoothness
from .steering_penalty import SteeringPenaltySmoothness


class SmoothnessMetrics:
    """Run all smoothness metrics on a path in one call.

    The path must be an ordered ``list[tuple[float, float]]`` of ``(x, y)``
    (or ``(x, y, z)`` for 3-D) coordinates from start to end, matching the
    output format of all algorithms in this project.

    The **recommended primary metric** is :meth:`recommended`, which uses
    :class:`~smoothness_metrics.SteeringPenaltySmoothness` with
    ``theta=30°, sweep_range=5``.  This was validated empirically to best
    distinguish path quality: small incidental bends are ignored, only sharp
    turns accumulate penalty.

    ``METRIC_KEYS`` lists the keys returned by :meth:`compute_all` in order,
    useful for building CSV headers without running a dummy computation.

    Args:
        path: Ordered list of coordinate tuples.
        steering_theta_degrees: Threshold for the steering penalty metric
            (default ``30``). Angles below it incur no penalty.
        steering_sweep_range: Symmetric neighbor offsets checked per waypoint
            for the steering penalty metric (default ``5``).
    """

    METRIC_KEYS: list[str] = [
        "pathbench",
        "angle_change",
        "deriv_v1_curvature",
        "deriv_v1_second",
        "deriv_v1_third",
        "deriv_v1_total",
        "deriv_v2_curvature",
        "deriv_v2_second",
        "deriv_v2_third",
        "deriv_v2_total",
        "discrete",
        "steering_penalty",
    ]

    def __init__(self, path: list[tuple[float, float]],
                 steering_theta_degrees: float = 30.0,
                 steering_sweep_range: int = 5):
        self.path = path
        self._pathbench = PathBenchSmoothness()
        self._angle_change = AngleChangeSmoothness()
        self._deriv_v1 = DerivativeSmoothnessV1()
        self._deriv_v2 = DerivativeSmoothnessV2()
        self._discrete = DiscreteSmoothness()
        self._steering = SteeringPenaltySmoothness(
            theta_degrees=steering_theta_degrees,
            sweep_range=steering_sweep_range,
        )

    def recommended(self) -> float:
        """Steering penalty metric at the configured theta/sweep (recommended).

        Returns:
            float: Total excess steering angle in degrees.  Lower is smoother.
        """
        return self._steering.compute(self.path)

    def compute_all(self) -> dict[str, float]:
        """Compute every available smoothness metric.

        Returns:
            Flat ``dict[str, float]`` with one entry per scalar metric and
            flattened entries for multi-value metrics.
        """
        dv1 = self._deriv_v1.compute(self.path)
        dv2 = self._deriv_v2.compute(self.path)
        return {
            "pathbench":          self._pathbench.compute(self.path),
            "angle_change":       self._angle_change.compute(self.path),
            "deriv_v1_curvature": dv1["curvature"],
            "deriv_v1_second":    dv1["second_derivative"],
            "deriv_v1_third":     dv1["third_derivative"],
            "deriv_v1_total":     dv1["total"],
            "deriv_v2_curvature": dv2["curvature"],
            "deriv_v2_second":    dv2["second_derivative"],
            "deriv_v2_third":     dv2["third_derivative"],
            "deriv_v2_total":     dv2["total"],
            "discrete":           self._discrete.compute(self.path),
            "steering_penalty":   self._steering.compute(self.path),
        }
