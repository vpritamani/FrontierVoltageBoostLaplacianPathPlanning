"""
run_test_suite_frontier_voltage_boost_laplace.py

Runs the Frontier Voltage Boost Laplace algorithm against a folder of maps
produced by init_test_suite.py (or any generate_many call).

Usage (from project root):
    python "Test Suite/run_test_suite_frontier_voltage_boost_laplace.py" \\
        --maps_dir "Test Suite/output/2d_maps" \\
        --laplace_iters 10 \\
        --epsilon 0.5

    # With optional args and selective output:
    python "Test Suite/run_test_suite_frontier_voltage_boost_laplace.py" \\
        --maps_dir "Test Suite/output/2d_maps" \\
        --laplace_iters 20 \\
        --epsilon 0.3 \\
        --step_size 0.5 \\
        --output paths phi

Output:
    fvb_results/smoothness_metrics.csv  — one row per map with all metrics,
                                          hyperparameters, and run metadata.
"""

import argparse
import csv
import os
import sys
import importlib.util
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------------

from map_generation import Map2D  # also adds Map Generation/ to sys.path

_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)
from smoothness_metrics import SmoothnessMetrics

_here = os.path.dirname(os.path.abspath(__file__))

_fvb_path = os.path.normpath(
    os.path.join(_here, '..', 'Algorithms', 'Frontier Voltage Boost', 'frontiervoltageboostlaplace.py')
)
_fvb_spec = importlib.util.spec_from_file_location('frontiervoltageboostlaplace', _fvb_path)
_fvb_mod = importlib.util.module_from_spec(_fvb_spec)
_fvb_spec.loader.exec_module(_fvb_mod)
FrontierVoltageBoostLaplace = _fvb_mod.FrontierVoltageBoostLaplace

ALGORITHM = 'FrontierVoltageBoostLaplace'

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Run Frontier Voltage Boost Laplace on a folder of maps.'
)
parser.add_argument('--maps_dir',      required=True,
                    help='Folder containing start_end_points.csv and .npy map files.')
parser.add_argument('--laplace_iters', required=True, type=int,
                    help='Number of Laplace iterations per wavefront step (n_l).')
parser.add_argument('--epsilon',       required=True, type=float,
                    help='Convergence threshold — cell is solved when φ ≤ v_max - epsilon.')
parser.add_argument('--step_size',     type=float, default=1.0,
                    help='Gradient descent step size (default: 1.0).')
parser.add_argument('--no_bilinear',   action='store_true',
                    help='Disable bilinear interpolation for gradient descent (default: on).')
parser.add_argument('--output',        nargs='+', default=['maps', 'paths', 'phi'],
                    choices=['maps', 'paths', 'phi'],
                    help='Output types: maps (raw PNG), paths (path overlay PNG), '
                         'phi (potential field PNG). Default: all three.')
args = parser.parse_args()

save_maps  = 'maps'  in args.output
save_paths = 'paths' in args.output
save_phi   = 'phi'   in args.output

maps_dir = os.path.abspath(args.maps_dir)
csv_path = os.path.join(maps_dir, 'start_end_points.csv')

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"No start_end_points.csv found in {maps_dir}")

output_dir = os.path.join(maps_dir, 'fvb_results')
os.makedirs(output_dir, exist_ok=True)

print(f"Maps dir:      {maps_dir}")
print(f"Algorithm:     {ALGORITHM}")
print(f"laplace_iters: {args.laplace_iters}")
print(f"epsilon:       {args.epsilon}")
print(f"step_size:     {args.step_size}")
print(f"bilinear:      {not args.no_bilinear}")
print(f"Output:        {', '.join(args.output)}\n")

# ---------------------------------------------------------------------------
# CSV schema — defined upfront so every row (solved or not) has the same keys
# ---------------------------------------------------------------------------

_RUN_TIMESTAMP = datetime.now().isoformat(timespec='seconds')

_META_FIELDS = [
    'run_timestamp', 'algorithm',
    'laplace_iters', 'epsilon', 'step_size', 'bilinear',
]
_MAP_FIELDS = [
    'map_name',
    'start_x', 'start_y',
    'end_x',   'end_y',
]
_RESULT_FIELDS = [
    'solved',
    'path_steps',          # number of waypoints in path
    'path_length_euclidean',  # total Euclidean distance
]
_FIELDNAMES = _META_FIELDS + _MAP_FIELDS + _RESULT_FIELDS + SmoothnessMetrics.METRIC_KEYS

_EMPTY_METRICS = {k: '' for k in SmoothnessMetrics.METRIC_KEYS}


def _euclidean_length(path: list[tuple]) -> float:
    pts = np.array(path, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _base_row(map_name, start, end) -> dict:
    return {
        'run_timestamp': _RUN_TIMESTAMP,
        'algorithm':     ALGORITHM,
        'laplace_iters': args.laplace_iters,
        'epsilon':       args.epsilon,
        'step_size':     args.step_size,
        'bilinear':      not args.no_bilinear,
        'map_name':      map_name,
        'start_x':       start[0], 'start_y': start[1],
        'end_x':         end[0],   'end_y':   end[1],
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

metrics_rows: list[dict] = []

for row in rows:
    map_name = row['map_name']
    start = (int(row['start_x']), int(row['start_y']))
    end   = (int(row['end_x']),   int(row['end_y']))
    stem  = os.path.splitext(map_name)[0]

    grid = np.load(os.path.join(maps_dir, map_name))
    m = Map2D(grid, start, end)

    print(f"{map_name}  start={start}  end={end}")

    algo = FrontierVoltageBoostLaplace(
        m,
        n_l=args.laplace_iters,
        epsilon=args.epsilon,
        step_size=args.step_size,
        bilinear_interpolation=not args.no_bilinear,
    )
    path = algo.solve()

    if path is None:
        print(f"  WARNING: no path found\n")
        metrics_rows.append({
            **_base_row(map_name, start, end),
            'solved':               False,
            'path_steps':           '',
            'path_length_euclidean': '',
            **_EMPTY_METRICS,
        })
        continue

    euc_len = _euclidean_length(path)
    print(f"  Path steps:  {len(path)}")
    print(f"  Path length: {euc_len:.2f} (Euclidean)")

    sm = SmoothnessMetrics(path)
    metrics = sm.compute_all()
    print(f"  Smoothness (recommended — steering penalty @ 30°): {metrics['steering_penalty']:.4f}")
    print(f"  Smoothness (all metrics):")
    for name, val in metrics.items():
        print(f"    {name:<25} {val:.6f}")

    metrics_rows.append({
        **_base_row(map_name, start, end),
        'solved':                True,
        'path_steps':            len(path),
        'path_length_euclidean': round(euc_len, 6),
        **{k: float(v) for k, v in metrics.items()},
    })

    if save_maps:
        from PIL import Image
        Image.fromarray(m.to_image(), mode='L').save(
            os.path.join(output_dir, f'{stem}_map.png')
        )
        print(f"  Saved map  → fvb_results/{stem}_map.png")

    if save_paths:
        from PIL import Image
        grey = m.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()
        for x, y in path:
            if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                rgb[y, x] = [128, 128, 255]
        rgb[start[1], start[0]] = [0, 255, 0]
        rgb[end[1],   end[0]]   = [255, 0, 0]
        Image.fromarray(rgb, mode='RGB').save(
            os.path.join(output_dir, f'{stem}_path.png')
        )
        print(f"  Saved path → fvb_results/{stem}_path.png")

    if save_phi and algo._phi is not None:
        from PIL import Image
        phi = algo._phi
        phi_norm = (phi - phi.min()) / (phi.max() - phi.min() + 1e-9)
        phi_img = (phi_norm * 255).astype(np.uint8)
        Image.fromarray(phi_img, mode='L').save(
            os.path.join(output_dir, f'{stem}_phi.png')
        )
        print(f"  Saved phi  → fvb_results/{stem}_phi.png")

    print()

# ---------------------------------------------------------------------------
# Write smoothness metrics CSV
# ---------------------------------------------------------------------------

if metrics_rows:
    metrics_csv_path = os.path.join(output_dir, 'smoothness_metrics.csv')
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"Smoothness metrics saved → fvb_results/smoothness_metrics.csv")

print("Done.")
