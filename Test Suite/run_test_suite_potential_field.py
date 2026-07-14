"""
run_test_suite_potential_field.py

Runs the artificial potential field algorithm against a folder of maps
produced by init_test_suite.py (or any generate_many call).

Usage (from project root):
    python "Test Suite/run_test_suite_potential_field.py" \\
        --maps_dir "Test Suite/output/2d_maps"

    # Override any parameter; omitted flags use algorithm defaults:
    python "Test Suite/run_test_suite_potential_field.py" \\
        --maps_dir "Test Suite/output/2d_maps" \\
        --q_star 30 \\
        --k_att 1.0 \\
        --k_rep 10000 \\
        --step_size 1.0 \\
        --max_iters 10000 \\
        --output paths phi
"""

import argparse
import csv
import os
import importlib.util
import numpy as np

# ---------------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------------

from map_generation import Map2D  # also adds Map Generation/ to sys.path

_here = os.path.dirname(os.path.abspath(__file__))

_pf_path = os.path.normpath(
    os.path.join(_here, '..', 'Algorithms', 'Baseline Algorithms', 'potentialfield2d.py')
)
_pf_spec = importlib.util.spec_from_file_location('potentialfield2d', _pf_path)
_pf_mod = importlib.util.module_from_spec(_pf_spec)
_pf_spec.loader.exec_module(_pf_mod)
PotentialFieldAlgorithm = _pf_mod.PotentialFieldAlgorithm
DEFAULT_Q_STAR = _pf_mod.DEFAULT_Q_STAR
DEFAULT_K_ATT = _pf_mod.DEFAULT_K_ATT
DEFAULT_K_REP = _pf_mod.DEFAULT_K_REP
DEFAULT_STEP_SIZE = _pf_mod.DEFAULT_STEP_SIZE
DEFAULT_MAX_ITERS = _pf_mod.DEFAULT_MAX_ITERS

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Run artificial potential field planning on a folder of maps.'
)
parser.add_argument('--maps_dir', required=True,
                    help='Folder containing start_end_points.csv and .npy map files.')
parser.add_argument('--q_star', type=int, default=DEFAULT_Q_STAR,
                    help=f'Influence radius for repulsive obstacles (default: {DEFAULT_Q_STAR}).')
parser.add_argument('--k_att', type=float, default=DEFAULT_K_ATT,
                    help=f'Attractive potential gain (default: {DEFAULT_K_ATT}).')
parser.add_argument('--k_rep', type=float, default=DEFAULT_K_REP,
                    help=f'Repulsive potential gain (default: {DEFAULT_K_REP:g}).')
parser.add_argument('--step_size', type=float, default=DEFAULT_STEP_SIZE,
                    help=f'Gradient descent step size (default: {DEFAULT_STEP_SIZE}).')
parser.add_argument('--max_iters', type=int, default=DEFAULT_MAX_ITERS,
                    help=f'Max gradient-descent iterations (default: {DEFAULT_MAX_ITERS}).')
parser.add_argument('--output', nargs='+', default=['maps', 'paths', 'phi'],
                    choices=['maps', 'paths', 'phi'],
                    help='Output types: maps (raw PNG), paths (path overlay PNG), '
                         'phi (potential field PNG). Default: all three.')
args = parser.parse_args()

save_maps = 'maps' in args.output
save_paths = 'paths' in args.output
save_phi = 'phi' in args.output

maps_dir = os.path.abspath(args.maps_dir)
csv_path = os.path.join(maps_dir, 'start_end_points.csv')

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"No start_end_points.csv found in {maps_dir}")

output_dir = os.path.join(maps_dir, 'pf_results')
os.makedirs(output_dir, exist_ok=True)

print(f"Maps dir:   {maps_dir}")
print(f"q_star:     {args.q_star}")
print(f"k_att:      {args.k_att}")
print(f"k_rep:      {args.k_rep}")
print(f"step_size:  {args.step_size}")
print(f"max_iters:  {args.max_iters}")
print(f"Output:     {', '.join(args.output)}\n")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for row in rows:
    map_name = row['map_name']
    start = (int(row['start_x']), int(row['start_y']))
    end = (int(row['end_x']), int(row['end_y']))
    stem = os.path.splitext(map_name)[0]

    grid = np.load(os.path.join(maps_dir, map_name))
    m = Map2D(grid, start, end)

    print(f"{map_name}  start={start}  end={end}")

    algo = PotentialFieldAlgorithm(
        m,
        q_star=args.q_star,
        k_att=args.k_att,
        k_rep=args.k_rep,
        step_size=args.step_size,
        max_iters=args.max_iters,
    )
    path = algo.solve()

    if path is None:
        print("  WARNING: no path found\n")
        continue

    print(f"  Path length: {len(path)} steps")

    if save_maps:
        from PIL import Image
        Image.fromarray(m.to_image(), mode='L').save(
            os.path.join(output_dir, f'{stem}_map.png')
        )
        print(f"  Saved map  → pf_results/{stem}_map.png")

    if save_paths:
        from PIL import Image
        grey = m.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()
        for x, y in path:
            if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                rgb[y, x] = [128, 128, 255]
        rgb[start[1], start[0]] = [0, 255, 0]
        rgb[end[1], end[0]] = [255, 0, 0]
        Image.fromarray(rgb, mode='RGB').save(
            os.path.join(output_dir, f'{stem}_path.png')
        )
        print(f"  Saved path → pf_results/{stem}_path.png")

    if save_phi and algo._potential is not None:
        from PIL import Image
        potential = algo._potential
        pot_norm = (potential - potential.min()) / (potential.max() - potential.min() + 1e-9)
        pot_img = (pot_norm * 255).astype(np.uint8)
        Image.fromarray(pot_img, mode='L').save(
            os.path.join(output_dir, f'{stem}_phi.png')
        )
        print(f"  Saved phi  → pf_results/{stem}_phi.png")

    print()

print("Done.")
