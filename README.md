# FrontierVoltageBoostLaplacianPathPlanning

This repository contains implementations for Frontier Voltage Boost Laplacian Path Planning, along with the incremental algorithms that led to it.

---

## 1. Installation

**Requirements:** Python 3.10+, pip

From the project root, run these two commands once:

```bash
pip install -e .
pip install numpy scipy pillow flask
pip install torch        # optional — enables the "Use GPU (PyTorch)" algorithm variants
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
- **Parallelize** a run across CPU cores ("Parallel workers" on the Execute card) — (map, algorithm) cells run concurrently, each still in its own watchdog subprocess; use 1 worker for the fairest timing comparisons
- **GPU acceleration** — every algorithm except A* has a "Use GPU (PyTorch)" toggle (FVB's Laplace step as a torch convolution, potential-field assembly and RRT nearest-neighbour/collision checks as tensor ops). CUDA is used when available, with a CPU-PyTorch fallback otherwise; the Execute card shows what your machine supports. A* stays CPU-only — its sequential priority-queue search has no meaningful GPU mapping
- **Compare results** — per-instance summaries labeled with their hyperparameters (the same algorithm can be added to a run any number of times with different parameters), a metric comparison view, the full per-map results table with all smoothness metrics, and side-by-side path images
- **Compare tab** — cross-run comparison with any number of steering-penalty configs at once (e.g. @20° and @30° side by side or as X/Y axes), a metric graph (2D/3D scatter or lines vs map size/dimensionality with one line per algorithm configuration), a **hyperparameter sweep** chart (same algorithm; X = any numeric parameter, optionally a 2nd parameter as a third dimension with the metric on the vertical axis; one series/color per configuration of the rest), and a combined raw-data CSV for all selected runs
- **Interactive graphs** — editable X/Y/Z axis limits, wheel/button zoom, drag-rotation in 3D, a reset-view button, and PNG export of the exact current view at 2× resolution
- **Chart labels** — every algorithm instance can carry a custom graph label (set when adding it to a run, editable later from Compare); labels are cosmetic — the full hyperparameters stay stored and visible alongside
- **Light/dark theme** toggle (top right), persisted across sessions
- **Metrics lab** — solved paths are stored with each run (FVB paths keep their continuous decimal coordinates), so you can recompute all smoothness metrics with a different steering θ / sweep range instantly, preview the effect, and optionally save — without re-running any planner
- **3D / ND viewer** — for 3D runs, a slice-by-slice layer view plus a rotatable, zoomable 3D view; for higher-dimensional runs, coordinate traces (each axis vs. path step) plus a rotatable 3D projection of any three chosen dimensions
- **Keep everything** — every run and map set is stored under `Benchmark Data/` with a unique ID (identical runs never overwrite each other) and is available again on the next launch
- **Export** any run as CSV
- **Delete** runs or map sets with explicit warnings about what will be lost

Options: `python run_app.py --port 9000 --no-browser`

To register a new algorithm in the app, add one `AlgorithmSpec` entry to `benchmark_app/registry.py` — the UI picks up its label, supported dimensions, and parameter form automatically.

Each solve executes in a watchdog subprocess with a configurable timeout, so an algorithm that hangs or crashes cannot take down the app or stall a benchmark run.

All state (map sets and run results) is written under `Benchmark Data/` in the project root and reloaded on the next launch, so nothing is lost between sessions.

---

## 3. Running on a remote VM

The app is a normal local web server, so a VM works the same as your laptop — you just need a way to reach the port. Clone and install as in step 1 (on a headless VM add `--no-browser` so it doesn't try to open a browser there).

### Option A — SSH port forwarding (recommended)

Keeps the app private to your machine; nothing is exposed on the VM's network.

```bash
# On the VM: run bound to localhost (the default)
python run_app.py --no-browser        # serves on 127.0.0.1:8177 on the VM

# On your laptop: forward a local port to the VM's port over SSH
ssh -L 8177:localhost:8177 user@your-vm-host
```

Then open `http://localhost:8177` in your local browser — the traffic is tunneled to the VM. Use a different left-hand number (e.g. `-L 9000:localhost:8177`) if 8177 is taken locally.

### Option B — bind to all interfaces

Expose the app on the VM's own address. Only do this on a trusted network — the built-in server is a single-user dev server with no authentication.

```bash
# On the VM
python run_app.py --host 0.0.0.0 --port 8177 --no-browser
```

Then browse to `http://<vm-ip>:8177`. On a cloud VM you must also allow inbound TCP on that port in the firewall / security group (AWS security group, GCP firewall rule, Azure NSG, or `ufw allow 8177`). If you can SSH in, **Option A needs none of this** and is the safer default.

### Notes

- `--port N` changes the port on both sides; keep the VM-side port consistent with your tunnel/firewall rule.
- `Benchmark Data/` lives on the VM, so generated maps and results persist there across restarts and disconnects — a run keeps going even if you close the SSH session, as long as the `run_app.py` process stays alive (use `tmux`/`screen` or `nohup … &` to detach it).

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
│   │   ├── potentialfield2d.py   # PotentialFieldAlgorithm (2D)
│   │   ├── potentialfieldnd.py   # PotentialFieldAlgorithmND (any dimensionality)
│   │   └── rrt.py                # RRTAlgorithm (dimension-agnostic)
│   ├── Frontier Voltage Boost/
│   │   ├── frontiervoltageboostlaplace.py
│   │   ├── 3dfrontiervoltageboostlaplace.py
│   │   └── ndfrontiervoltageboostlaplace.py
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
│   ├── FrontierVoltageBoostLaplace
│   └── PotentialFieldAlgorithm  (potentialfield2d.py)
├── Algorithm3D  (3dalgorithm.py)        — for algorithms that only work in 3D
│   └── FrontierVoltageBoostLaplace3D
├── FrontierVoltageBoostLaplaceND  (ndfrontiervoltageboostlaplace.py) — any dimensionality
├── PotentialFieldAlgorithmND  (potentialfieldnd.py) — any dimensionality
├── AStarAlgorithm  (astar.py)           — dimension-agnostic, reads grid.ndim at solve time
└── RRTAlgorithm  (rrt.py)               — dimension-agnostic, continuous (decimal) node positions
```

All algorithms except A* accept `use_gpu=True` for a PyTorch implementation (CUDA when available, CPU-PyTorch fallback otherwise); the default NumPy paths are unchanged and produce numerically identical results.

New algorithms extend `Algorithm2D`, `Algorithm3D`, or `BaseAlgorithm` directly depending on whether they are dimension-specific.

---

## RRT – Laplace Hybridization Algorithm (Pure Random Sampling)

## Frontier RRT – Laplace Hybridization Algorithm (Random Sampling on Frontier)

## Frontier Voltage Boost Laplacian Path Planning (Final Algorithm)

## Smoothness Metrics

Six path smoothness metrics (all N-dimensional), each its own class, plus a `SmoothnessMetrics` aggregator. The empirically preferred metric is the **steering penalty** (sum of steering angles above a 30° threshold). See [Smoothness Metrics/README.md](Smoothness%20Metrics/README.md) for usage and the full metric table.
