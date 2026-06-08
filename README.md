# FrontierVoltageBoostLaplacianPathPlanning

This repository contains implementations for Frontier Voltage Boost Laplacian Path Planning, along with the incremental algorithms that led to it.

---

## 1. Installation

**Requirements:** Python 3.10+, pip

From the project root, run these two commands once:

```bash
pip install -e .
pip install numpy pillow
```

`pip install -e .` registers `Algorithms` and `map_generation` on your Python path so all imports work without any manual path setup. You only need to do this once per environment.

---

## 2. Generate Maps

Maps are generated with `init_test_suite.py`. Each generated map is guaranteed solvable — the generator internally runs A* and discards any map with no valid path.

```bash
# 10 solvable 2D maps at 200×200 — saves .npy files and a start_end_points.csv
python "Test Suite/init_test_suite.py" -dims 2 -num_maps 10 -xdim 200 -ydim 200 --output npy
```

This writes to `Test Suite/output/2d_maps/`:
```
Test Suite/output/2d_maps/
├── map_0.npy
├── map_1.npy
├── ...
└── start_end_points.csv      ← start/end coordinates for every map
```

Other useful generation commands:

```bash
# Generate maps and also save PNG images of each one
python "Test Suite/init_test_suite.py" -dims 2 -num_maps 5 -xdim 100 -ydim 100 --output npy maps

# Generate 3D maps
python "Test Suite/init_test_suite.py" -dims 3 -num_maps 3 -xdim 30 -ydim 30 -zdim 30 --output npy
```

Full argument reference: see [Test Suite/README.md](Test%20Suite/README.md).

---

## 3. Run Frontier Voltage Boost Laplace

Point the runner at the folder produced in step 2 and supply the two required algorithm parameters:

```bash
python "Test Suite/run_test_suite_frontier_voltage_boost_laplace.py" `
    --maps_dir "Test Suite/output/2d_maps" `
    --laplace_iters 200 `
    --epsilon 0.0001
```

Results are written to `Test Suite/output/2d_maps/fvb_results/`.

**Algorithm parameters:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--maps_dir` | yes | — | Folder containing `.npy` files and `start_end_points.csv` |
| `--laplace_iters` | yes | — | Laplace iterations per wavefront step (`n_l`) |
| `--epsilon` | yes | — | Solved threshold: cell is solved when `φ ≤ v_max − ε` |
| `--step_size` | no | `1.0` | Gradient descent step size |
| `--no_bilinear` | no | off | Disable bilinear interpolation for gradient sampling (on by default) |
| `--output` | no | all | Space-separated: `maps`, `paths`, `phi` |

**`--output` values:**

| Value | Saves |
|-------|-------|
| `maps` | Raw map PNG — white = free, black = obstacle |
| `paths` | Path overlay PNG — blue = path, green = start, red = end |
| `phi` | Potential field φ as a grayscale heatmap — dark = near goal |

```bash
# Only save path overlays and potential field
python "Test Suite/run_test_suite_frontier_voltage_boost_laplace.py" `
    --maps_dir "Test Suite/output/2d_maps" `
    --laplace_iters 200 `
    --epsilon 0.0001 `
    --step_size 0.5 `
    --output paths phi
```

---

## Project Structure

```
FrontierVoltageBoostLaplacianPathPlanning/
├── Algorithms/
│   ├── algorithm.py              # BaseAlgorithm (abstract base class)
│   ├── 2dalgorithm.py            # Algorithm2D  (2D-specific base)
│   ├── 3dalgorithm.py            # Algorithm3D  (3D-specific base)
│   ├── Baseline Algorithms/
│   │   ├── astar.py              # AStarAlgorithm (dimension-agnostic)
│   │   └── potentialfield.py
│   ├── Frontier Voltage Boost/
│   │   ├── frontiervoltageboostlaplace.py
│   │   └── 3dfrontiervoltageboostlaplace.py
│   └── Incremental Algorithms/
│       ├── rrtlaplacefrontier.py
│       └── rrtlaplacerandomsampling.py
├── Map Generation/
│   ├── map.py                    # Map base class
│   ├── 2dmap.py                  # Map2D
│   ├── 3dmap.py                  # Map3D
│   ├── 2dmapgenerator.py         # MapGenerator2D
│   └── 3dmapgenerator.py         # MapGenerator3D
├── Smoothness Metrics/
│   └── smoothnessmetrics.py
├── Test Suite/
│   ├── init_test_suite.py        # Map generation script
│   ├── run_test_suite_frontier_voltage_boost_laplace.py
│   └── main.py                   # Full benchmark runner
├── pyproject.toml
└── README.md
```

---

## Algorithm Class Hierarchy

```
BaseAlgorithm  (algorithm.py)
├── Algorithm2D  (2dalgorithm.py)        — for algorithms that only work in 2D
│   └── FrontierVoltageBoostLaplace
├── Algorithm3D  (3dalgorithm.py)        — for algorithms that only work in 3D
└── AStarAlgorithm  (astar.py)           — dimension-agnostic, reads grid.ndim at solve time
```

New algorithms extend `Algorithm2D`, `Algorithm3D`, or `BaseAlgorithm` directly depending on whether they are dimension-specific.

---

## RRT – Laplace Hybridization Algorithm (Pure Random Sampling)

## Frontier RRT – Laplace Hybridization Algorithm (Random Sampling on Frontier)

## Frontier Voltage Boost Laplacian Path Planning (Final Algorithm)

## Smoothness Metrics
