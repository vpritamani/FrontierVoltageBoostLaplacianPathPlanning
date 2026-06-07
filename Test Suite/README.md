# Test Suite

Scripts for verifying setup and benchmarking algorithms against generated maps.

---

## Setup

Install the project from the root directory first:

```bash
pip install -e .
pip install numpy pillow
```

---

## init_test_suite.py

Verifies the full pipeline end-to-end: map generation, solvability checking, A* solve, and visualisation output.

```bash
# Generate 5 solvable 2D maps at 100x100
python "Test Suite/init_test_suite.py" -dims 2 -num_maps 5 -xdim 100 -ydim 100

# Generate 3 solvable 3D maps at 30x30x30
python "Test Suite/init_test_suite.py" -dims 3 -num_maps 3 -xdim 30 -ydim 30 -zdim 30
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-dims` | yes | — | `2` or `3` |
| `-num_maps` | no | `1` | Number of maps to generate |
| `-xdim` | no | `100` | Width |
| `-ydim` | no | `100` | Height |
| `-zdim` | no | `30` | Depth (3D only) |
| `--output` | no | all | Space-separated list of `maps`, `paths`, `npy` |

**`--output` values:**

| Value | Saves |
|-------|-------|
| `npy` | `.npy` grid file + `start_end_points.csv` per map |
| `maps` | PNG image of the raw map (2D) or per-slice PNGs (3D) |
| `paths` | A* path overlay PNG (2D) / path length printed (3D) |

```bash
# Only .npy files, no images
python "Test Suite/init_test_suite.py" -dims 2 -num_maps 10 -xdim 200 -ydim 200 --output npy

# Only map images, skip A*
python "Test Suite/init_test_suite.py" -dims 2 -num_maps 5 --output maps

# Path overlays only (no raw map PNGs, no .npy)
python "Test Suite/init_test_suite.py" -dims 2 -num_maps 5 --output paths
```

For each map the script:
1. Generates a solvable map (A* verified internally by the generator)
2. Saves the raw map image
3. Runs A* and saves the path overlay (2D) or prints path length (3D)

All output is written to `Test Suite/output/`.

---

## Output images

| File | Description |
|------|-------------|
| `output/map_2d.png` | Raw 2D map — white = free, black = obstacle |
| `output/astar_result_2d.png` | Path overlaid — blue = path, green = start, red = end |
| `output/map_3d_slices/slice_ZZZZ.png` | One grayscale image per Z-slice of the 3D map |
