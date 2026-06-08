# Test Suite

Scripts for generating maps and benchmarking algorithms.

---

## Setup

Install the project from the root directory first (one-time):

```bash
pip install -e .
pip install numpy pillow
```

---

## init_test_suite.py — Map Generation

Generates solvable maps and saves them as `.npy` files. Each map is verified by A* before being saved; unsolvable candidates are silently discarded and regenerated.

```bash
# Minimal — 1 map at default size (100×100)
python "Test Suite/init_test_suite.py" -dims 2

# 10 maps at 200×200, .npy files only (fastest, no images)
python "Test Suite/init_test_suite.py" -dims 2 -num_maps 10 -xdim 200 -ydim 200 --output npy

# 5 maps at 100×100, save .npy + map images
python "Test Suite/init_test_suite.py" -dims 2 -num_maps 5 -xdim 100 -ydim 100 --output npy maps

# 3D maps
python "Test Suite/init_test_suite.py" -dims 3 -num_maps 3 -xdim 30 -ydim 30 -zdim 30 --output npy
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-dims` | yes | — | `2` or `3` |
| `-num_maps` | no | `1` | Number of maps to generate |
| `-xdim` | no | `100` | Width |
| `-ydim` | no | `100` | Height |
| `-zdim` | no | `30` | Depth (3D only) |
| `--output` | no | all | Space-separated: `npy`, `maps`, `paths` |

**`--output` values:**

| Value | Saves |
|-------|-------|
| `npy` | `.npy` grid file + `start_end_points.csv` |
| `maps` | Raw map PNG — white = free, black = obstacle |
| `paths` | A* path overlay PNG (2D) or path length printed (3D) |

Output lands in `Test Suite/output/2d_maps/` or `Test Suite/output/3d_maps/`.

---

## run_test_suite_frontier_voltage_boost_laplace.py

Runs Frontier Voltage Boost Laplace against a map folder produced by `init_test_suite.py`.

```bash
# Basic run — all output types
python "Test Suite/run_test_suite_frontier_voltage_boost_laplace.py" `
    --maps_dir "Test Suite/output/2d_maps" `
    --laplace_iters 200 `
    --epsilon 0.0001

# Custom step size, selective output
python "Test Suite/run_test_suite_frontier_voltage_boost_laplace.py" `
    --maps_dir "Test Suite/output/2d_maps" `
    --laplace_iters 200 `
    --epsilon 0.0001 `
    --step_size 0.5 `
    --output paths phi
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--maps_dir` | yes | — | Folder with `.npy` files and `start_end_points.csv` |
| `--laplace_iters` | yes | — | Laplace iterations per wavefront step (`n_l`) |
| `--epsilon` | yes | — | Solved threshold: cell solved when `φ ≤ v_max − ε` |
| `--step_size` | no | `1.0` | Gradient descent step size |
| `--no_bilinear` | no | off | Disable bilinear interpolation for gradient sampling (on by default) |
| `--output` | no | all | Space-separated: `maps`, `paths`, `phi` |

**`--output` values:**

| Value | Saves |
|-------|-------|
| `maps` | Raw map PNG |
| `paths` | Path overlay — blue path, green start, red end |
| `phi` | Potential field φ as grayscale heatmap — darker = closer to goal |

Results are written to `<maps_dir>/fvb_results/`, one file per map per output type:

```
fvb_results/
├── map_0_map.png
├── map_0_path.png
├── map_0_phi.png
├── map_1_map.png
└── ...
```
