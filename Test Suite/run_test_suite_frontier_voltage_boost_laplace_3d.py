"""
run_test_suite_frontier_voltage_boost_laplace_3d.py

Runs the 3D Frontier Voltage Boost Laplace algorithm against a folder of maps
produced by init_test_suite.py (with -dims 3).

Usage (from project root):
    python "Test Suite/run_test_suite_frontier_voltage_boost_laplace_3d.py" `
        --maps_dir "Test Suite/output/3d_maps" `
        --laplace_iters 200 `
        --epsilon 0.0001

    # With optional args and selective output:
    python "Test Suite/run_test_suite_frontier_voltage_boost_laplace_3d.py" `
        --maps_dir "Test Suite/output/3d_maps" `
        --laplace_iters 200 `
        --epsilon 0.0001 `
        --step_size 0.5 `
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

from map_generation import Map3D

_here = os.path.dirname(os.path.abspath(__file__))

_fvb_path = os.path.normpath(
    os.path.join(_here, '..', 'Algorithms', 'Frontier Voltage Boost', '3dfrontiervoltageboostlaplace.py')
)
_fvb_spec = importlib.util.spec_from_file_location('fvb3d', _fvb_path)
_fvb_mod = importlib.util.module_from_spec(_fvb_spec)
_fvb_spec.loader.exec_module(_fvb_mod)
FrontierVoltageBoostLaplace3D = _fvb_mod.FrontierVoltageBoostLaplace3D

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Run 3D Frontier Voltage Boost Laplace on a folder of maps.'
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
                    help='Disable trilinear interpolation for gradient descent (default: on).')
parser.add_argument('--output',        nargs='+', default=['maps', 'paths', 'phi'],
                    choices=['maps', 'paths', 'phi'],
                    help='Output types: maps (slice PNGs), paths (path overlay slice PNGs), '
                         'phi (potential field slice PNGs). Default: all three.')
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
print(f"trilinear:     {not args.no_bilinear}")
print(f"Output:        {', '.join(args.output)}\n")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for row in rows:
    map_name = row['map_name']
    start = (int(row['start_x']), int(row['start_y']), int(row['start_z']))
    end   = (int(row['end_x']),   int(row['end_y']),   int(row['end_z']))
    stem  = os.path.splitext(map_name)[0]

    grid = np.load(os.path.join(maps_dir, map_name))
    m = Map3D(grid, start, end)

    print(f"{map_name}  start={start}  end={end}")

    algo = FrontierVoltageBoostLaplace3D(
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

    map_out = os.path.join(output_dir, stem)
    os.makedirs(map_out, exist_ok=True)

    if save_maps:
        from PIL import Image
        for z in range(m.depth):
            Image.fromarray(m.to_image_slice(z), mode='L').save(
                os.path.join(map_out, f'map_slice_{z:04d}.png')
            )
        print(f"  Saved map slices → fvb_results/{stem}/map_slice_*.png")

    if save_paths:
        from PIL import Image
        # Path coords are continuous — round only for pixel painting.
        path_set = {(int(round(x)), int(round(y)), int(round(z))) for x, y, z in path}
        sx, sy, sz = start
        ex, ey, ez = end
        for z in range(m.depth):
            grey = m.to_image_slice(z)
            rgb = np.stack([grey, grey, grey], axis=2).copy()
            for x, y, pz in path_set:
                if pz == z and 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    rgb[y, x] = [128, 128, 255]
            if sz == z:
                rgb[sy, sx] = [0, 255, 0]
            if ez == z:
                rgb[ey, ex] = [255, 0, 0]
            Image.fromarray(rgb, mode='RGB').save(
                os.path.join(map_out, f'path_slice_{z:04d}.png')
            )
        print(f"  Saved path slices → fvb_results/{stem}/path_slice_*.png")

    if save_phi and algo._phi is not None:
        from PIL import Image
        phi = algo._phi
        phi_min, phi_max = phi.min(), phi.max()
        phi_norm = (phi - phi_min) / (phi_max - phi_min + 1e-9)
        phi_img = (phi_norm * 255).astype(np.uint8)
        for z in range(m.depth):
            Image.fromarray(phi_img[z], mode='L').save(
                os.path.join(map_out, f'phi_slice_{z:04d}.png')
            )
        print(f"  Saved phi slices  → fvb_results/{stem}/phi_slice_*.png")

    print()

print("Done.")
