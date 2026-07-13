# Smoothness Metrics

A collection of path smoothness metrics for evaluating robot/agent paths produced by planning algorithms.  All metrics operate on paths represented as ordered `list[tuple[float, float]]` of `(x, y)` coordinates, matching the output format of every algorithm in this project.

## Recommended metric

**`SteeringPenaltySmoothness` (theta=30°, sweep=5)** — validated empirically to best distinguish path quality.  Small incidental bends are below the threshold and incur no penalty; only genuinely sharp turns accumulate.  Access it via `SmoothnessMetrics.recommended()`.

## Quick start

```python
from smoothness_metrics import SmoothnessMetrics, SteeringPenaltySmoothness

path = [(0, 0), (1, 0), (2, 1), (3, 3), ...]   # (x, y) tuples

# Aggregator — run everything at once
sm = SmoothnessMetrics(path)
print(sm.recommended())      # recommended scalar score
print(sm.compute_all())      # dict of all metrics

# Use a single metric directly
metric = SteeringPenaltySmoothness(theta_degrees=20, sweep_range=3)
print(metric(path))

# Sweep across thresholds
print(metric.compute_at_thresholds(path))
```

## Available metrics

| Class | File | Returns | Notes |
|---|---|---|---|
| `PathBenchSmoothness` | `pathbench.py` | `float` | Mean absolute change in successive turning angles |
| `AngleChangeSmoothness` | `angle_change.py` | `float` | Total turning angle / arc length (law of cosines) |
| `DerivativeSmoothnessV1` | `derivative_based.py` | `dict` | Heading derivatives normalized by path length |
| `DerivativeSmoothnessV2` | `derivative_based.py` | `dict` | Heading derivatives normalized by per-order count |
| `DiscreteSmoothness` | `discrete.py` | `float` | Mean squared discrete curvature (integer grid) |
| `SteeringPenaltySmoothness` | `steering_penalty.py` | `float` | **Recommended.** Sum of excess steering angles above threshold |

All classes inherit from `BaseSmoothnessMetric` (`base.py`) and are callable directly.

## Configurable steering parameters

`SmoothnessMetrics` accepts the steering penalty knobs directly:

```python
sm = SmoothnessMetrics(path, steering_theta_degrees=20, steering_sweep_range=3)
sm.compute_all()   # steering_penalty now uses θ=20°, sweep=3
```

## Integration with the benchmark app

The benchmark app (`python run_app.py`) computes all metrics for every solved path, stores them with each run, and exports them as CSV. Because solved paths are stored too, the app's **Metrics lab** can recompute every metric at a different θ / sweep range on any past run — no planner re-run needed.
