"""
init_test_suite.py

Generates maps, optionally runs A*, and saves configurable output.

Usage (from project root):
    python "Test Suite/init_test_suite.py" -dims 2 -num_maps 5 -xdim 100 -ydim 100
    python "Test Suite/init_test_suite.py" -dims 3 -num_maps 3 -xdim 30 -ydim 30 -zdim 30 --output npy
    python "Test Suite/init_test_suite.py" -dims 2 -num_maps 10 -xdim 200 -ydim 200 --output maps paths
"""

import argparse
import csv
import os
import importlib
import importlib.util
import numpy as np

# ---------------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------------

from map_generation import Map2D, Map3D  # also adds Map Generation/ to sys.path

MapGenerator2D = importlib.import_module('2dmapgenerator').MapGenerator2D
MapGenerator3D = importlib.import_module('3dmapgenerator').MapGenerator3D

_here = os.path.dirname(os.path.abspath(__file__))
_astar_path = os.path.normpath(
    os.path.join(_here, '..', 'Algorithms', 'Baseline Algorithms', 'astar.py')
)
_astar_spec = importlib.util.spec_from_file_location('astar', _astar_path)
_astar_mod = importlib.util.module_from_spec(_astar_spec)
_astar_spec.loader.exec_module(_astar_mod)
AStarAlgorithm = _astar_mod.AStarAlgorithm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Generate and verify maps.')
parser.add_argument('-dims',     type=int, required=True, choices=[2, 3],
                    help='Dimensionality of maps to generate (2 or 3)')
parser.add_argument('-num_maps', type=int, default=1,
                    help='Number of maps to generate (default: 1)')
parser.add_argument('-xdim',     type=int, default=100, help='Width  (default: 100)')
parser.add_argument('-ydim',     type=int, default=100, help='Height (default: 100)')
parser.add_argument('-zdim',     type=int, default=30,  help='Depth  (default: 30, 3D only)')
parser.add_argument('--output',  nargs='+', default=['maps', 'paths', 'npy'],
                    choices=['maps', 'paths', 'npy'],
                    help='Output types: maps (PNG), paths (A* overlay PNG), npy (grid + CSV). '
                         'Default: all three.')
args = parser.parse_args()

save_maps  = 'maps'  in args.output
save_paths = 'paths' in args.output
save_npy   = 'npy'   in args.output

OUTPUT_DIR = os.path.join(_here, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

dims_label = f"{args.xdim}x{args.ydim}" if args.dims == 2 else f"{args.xdim}x{args.ydim}x{args.zdim}"
print(f"Generating {args.num_maps} x {args.dims}D map(s) — {dims_label}")
print(f"Output:     {', '.join(args.output)}\n")

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

if args.dims == 2:
    gen = MapGenerator2D(width=args.xdim, height=args.ydim)
else:
    gen = MapGenerator3D(width=args.xdim, height=args.ydim, depth=args.zdim)

maps = [gen.generate() for _ in range(args.num_maps)]

# ---------------------------------------------------------------------------
# Save .npy + CSV
# ---------------------------------------------------------------------------

if save_npy:
    npy_dir = os.path.join(OUTPUT_DIR, f'{args.dims}d_maps')
    os.makedirs(npy_dir, exist_ok=True)
    csv_path = os.path.join(npy_dir, 'start_end_points.csv')

    if args.dims == 2:
        fieldnames = ['map_name', 'start_x', 'start_y', 'end_x', 'end_y']
    else:
        fieldnames = ['map_name', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, m in enumerate(maps):
            map_name = f'map_{i}.npy'
            np.save(os.path.join(npy_dir, map_name), m.grid)
            row = {'map_name': map_name,
                   'start_x': m.start[0], 'start_y': m.start[1],
                   'end_x':   m.end[0],   'end_y':   m.end[1]}
            if args.dims == 3:
                row['start_z'] = m.start[2]
                row['end_z']   = m.end[2]
            writer.writerow(row)

    print(f"Saved {args.num_maps} .npy file(s) + CSV → output/{args.dims}d_maps/")

# ---------------------------------------------------------------------------
# Per-map output
# ---------------------------------------------------------------------------

for i, m in enumerate(maps):
    print(f"Map {i}  start={m.start}  end={m.end}")

    if save_maps:
        if args.dims == 2:
            img_path = os.path.join(OUTPUT_DIR, f'map_2d_{i}.png')
            m.save_image(img_path)
            print(f"  Saved map image  → output/map_2d_{i}.png")
        else:
            slices_dir = os.path.join(OUTPUT_DIR, f'map_3d_{i}_slices')
            m.save_image_slices(slices_dir)
            print(f"  Saved {m.depth} slice(s) → output/map_3d_{i}_slices/")

    if save_paths:
        algo = AStarAlgorithm(m)
        path = algo.solve()
        print(f"  Path length: {len(path)} steps")

        if args.dims == 2:
            algo.result(visualize=True)
            _default = 'astar_result.png'
            _dest = os.path.join(OUTPUT_DIR, f'astar_result_2d_{i}.png')
            if os.path.exists(_default):
                os.replace(_default, _dest)
            print(f"  Saved path image → output/astar_result_2d_{i}.png")

print(f"\nDone.")
