# FrontierVoltageBoostLaplacianPathPlanning

This repository contains implementations for Frontier Voltage Boost Laplacian Path Planning, along with the incremental algorithms that led to it.

---

## 1. Installation

**Requirements:** Python 3.10+, pip

From the project root, run these two commands once:

```bash
pip install -e .
pip install numpy pillow flask
```

`pip install -e .` registers `Algorithms`, `map_generation`, and `smoothness_metrics` on your Python path so all imports work without any manual path setup. You only need to do this once per environment.

---

## 2. Benchmark App (recommended)

The easiest way to use everything in this repo is the local web app:

```bash
python run_app.py
```

This starts a local server at `http://127.0.0.1:8177` and opens your browser. From the UI you can:

- **Generate map sets** (2D or 3D, any size/count, obstacle range, optional seed) — every map is verified solvable with A* before being accepted — or **import** existing `.npy` map folders
- **Run one or more algorithms** with full parameter control on new maps, an existing map set, or any subset of maps — and **cancel** an in-flight run at any point (partial results are kept)
- **Compare results** — per-instance summaries labeled with their hyperparameters (the same algorithm can be added to a run any number of times with different parameters), a metric comparison view, the full per-map results table with all smoothness metrics, and side-by-side path images
- **Compare tab** — cross-run comparison with any number of steering-penalty configs at once (e.g. @20° and @30° side by side or as X/Y axes), a metric graph (2D/3D scatter or lines vs map size/dimensionality with one line per algorithm configuration), PNG export of the graph, and a combined raw-data CSV for all selected runs
- **Metrics lab** — solved paths are stored with each run (FVB paths keep their continuous decimal coordinates), so you can recompute all smoothness metrics with a different steering θ / sweep range instantly, preview the effect, and optionally save — without re-running any planner
- **3D / ND viewer** — for 3D runs, a slice-by-slice layer view plus a rotatable, zoomable 3D view; for higher-dimensional runs, coordinate traces (each axis vs. path step) plus a rotatable 3D projection of any three chosen dimensions
- **Keep everything** — every run and map set is stored under `Benchmark Data/` with a unique ID (identical runs never overwrite each other) and is available again on the next launch
- **Export** any run as CSV
- **Delete** runs or map sets with explicit warnings about what will be lost

Options: `python run_app.py --port 9000 --no-browser`

To register a new algorithm in the app, add one `AlgorithmSpec` entry to `benchmark_app/registry.py` — the UI picks up its label, supported dimensions, and parameter form automatically.

Each solve executes in a watchdog subprocess with a configurable timeout, so an algorithm that hangs or crashes cannot take down the app or stall a benchmark run.

---

## 3. Importing maps from the old CLI scripts

The old CLI scripts (`init_test_suite.py`, `run_test_suite_frontier_voltage_boost_laplace.py`) are retired — the app covers everything they did. Folders they produced (e.g. `Test Suite/output/2d_maps` with `.npy` files + `start_end_points.csv`) can be pulled into the app via **Map Sets → Import from folder…**; the files are copied, the source folder is untouched.

---

## 4. Run 3D Frontier Voltage Boost Laplace

Generate 3D maps first, then point the 3D runner at them:

```bash
# Generate 3D maps
python "Test Suite/init_test_suite.py" -dims 3 -num_maps 3 -xdim 30 -ydim 30 -zdim 30 --output npy

# Run the 3D algorithm
python "Test Suite/run_test_suite_frontier_voltage_boost_laplace_3d.py" `
    --maps_dir "Test Suite/output/3d_maps" `
    --laplace_iters 200 `
    --epsilon 0.0001
```

Results are written to `Test Suite/output/3d_maps/fvb_results/<map_name>/` as per-z-slice PNGs.

Full argument reference: see [Test Suite/README.md](Test%20Suite/README.md).

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
├── Smoothness Metrics/           # path smoothness metrics package (see its README)
│   ├── smoothnessmetrics.py      # SmoothnessMetrics aggregator
│   ├── steering_penalty.py       # recommended metric
│   └── ...                       # one file per metric
├── Test Suite/
│   ├── init_test_suite.py        # Map generation script
│   ├── run_test_suite_frontier_voltage_boost_laplace.py
│   ├── run_test_suite_frontier_voltage_boost_laplace_3d.py
│   └── main.py                   # Full benchmark runner
│   └── output/                   # legacy CLI map output (importable in the app)
├── benchmark_app/                # local web app (Flask + vanilla JS)
│   ├── registry.py               # ← add new algorithms here
│   ├── server.py                 # HTTP API
│   ├── runner.py                 # background job queue
│   ├── solver_worker.py          # per-solve subprocess watchdog
│   ├── mapgen.py                 # map set generation service
│   ├── storage.py                # Benchmark Data/ persistence
│   └── static/                   # UI (index.html, app.js, style.css)
├── Benchmark Data/               # generated map sets + run results (app-managed)
├── run_app.py                    # ← one command to launch the app
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
│   └── FrontierVoltageBoostLaplace3D
└── AStarAlgorithm  (astar.py)           — dimension-agnostic, reads grid.ndim at solve time
```

New algorithms extend `Algorithm2D`, `Algorithm3D`, or `BaseAlgorithm` directly depending on whether they are dimension-specific.

---

## RRT – Laplace Hybridization Algorithm (Pure Random Sampling)

## Frontier RRT – Laplace Hybridization Algorithm (Random Sampling on Frontier)

## Frontier Voltage Boost Laplacian Path Planning (Final Algorithm)

## Smoothness Metrics

Six path smoothness metrics (all N-dimensional), each its own class, plus a `SmoothnessMetrics` aggregator. The empirically preferred metric is the **steering penalty** (sum of steering angles above a 30° threshold). See [Smoothness Metrics/README.md](Smoothness%20Metrics/README.md) for usage and the full metric table.
