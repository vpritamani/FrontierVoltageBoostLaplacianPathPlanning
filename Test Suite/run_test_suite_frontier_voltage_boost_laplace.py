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

_fvb_path = os.path.normpath(
    os.path.join(_here, '..', 'Algorithms', 'Frontier Voltage Boost', 'frontiervoltageboostlaplace.py')
)
_fvb_spec = importlib.util.spec_from_file_location('frontiervoltageboostlaplace', _fvb_path)
_fvb_mod = importlib.util.module_from_spec(_fvb_spec)
_fvb_spec.loader.exec_module(_fvb_mod)
FrontierVoltageBoostLaplace = _fvb_mod.FrontierVoltageBoostLaplace

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
print(f"laplace_iters: {args.laplace_iters}")
print(f"epsilon:       {args.epsilon}")
print(f"step_size:     {args.step_size}")
print(f"bilinear:      {not args.no_bilinear}")
print(f"Output:        {', '.join(args.output)}\n")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

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
        continue

    print(f"  Path length: {len(path)} steps")

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

print("Done.")
