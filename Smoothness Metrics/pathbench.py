import numpy as np

from .base import BaseSmoothnessMetric


class PathBenchSmoothness(BaseSmoothnessMetric):
    """PathBench smoothness: mean absolute change in successive turning angles.

    Computes the heading angle at each consecutive segment pair, then measures
    how much that angle changes from one pair to the next.  A perfectly straight
    path scores 0; erratic direction changes produce higher scores.

    Returns:
        float: ``sum(|Δangle|) / (n - 2)``.  Lower is smoother.
    """

    def compute(self, path: list[tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0

        angle_diff_sum = 0.0
        prev_angle = None

        for i in range(1, len(path)):
            p1 = np.array(path[i - 1], dtype=float)
            p2 = np.array(path[i], dtype=float)
            dir1 = p2 - p1
            n1 = np.linalg.norm(dir1)
            if n1 == 0:
                continue
            dir1 /= n1

            if i + 1 < len(path):
                p3 = np.array(path[i + 1], dtype=float)
                dir2 = p3 - p2
                n2 = np.linalg.norm(dir2)
                if n2 == 0:
                    continue
                dir2 /= n2

                angle = np.arccos(np.clip(np.dot(dir1, dir2), -1.0, 1.0))

                if prev_angle is None:
                    prev_angle = angle
                    continue

                angle_diff_sum += abs(prev_angle - angle)
                prev_angle = angle

        return angle_diff_sum / (len(path) - 2 if len(path) > 2 else 1)
