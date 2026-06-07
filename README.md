# FrontierVoltageBoostLaplacianPathPlanning

This repository contains implementations for Frontier Voltage Boost Laplacian Path Planning, along with the incremental algorithms that led to it.

---

## Setup

**Requirements:** Python 3.10+, pip

From the project root, install the project as an editable package once:

```bash
pip install -e .
```

This registers `Algorithms` and `map_generation` on your Python path so all imports work without manual path setup.

**Dependencies** (install separately if not already present):

```bash
pip install numpy pillow
```

---

## Quick Start

Generate maps and run A* to verify the full pipeline:

```bash
python "Test Suite/init_test_suite.py"
```

Output images are written to `Test Suite/output/`.

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
│   ├── init_test_suite.py        # Setup verification script
│   └── main.py                   # Full benchmark runner
├── pyproject.toml
└── README.md
```

---

## Algorithm Class Hierarchy

```
BaseAlgorithm  (algorithm.py)
├── Algorithm2D  (2dalgorithm.py)   — for algorithms that only work in 2D
├── Algorithm3D  (3dalgorithm.py)   — for algorithms that only work in 3D
└── AStarAlgorithm  (astar.py)      — dimension-agnostic, reads grid.ndim at solve time
```

New algorithms extend `Algorithm2D`, `Algorithm3D`, or `BaseAlgorithm` directly depending on whether they are dimension-specific.

---

## RRT – Laplace Hybridization Algorithm (Pure Random Sampling)

## Frontier RRT – Laplace Hybridization Algorithm (Random Sampling on Frontier)

## Frontier Voltage Boost Laplacian Path Planning (Final Algorithm)

## Smoothness Metrics
