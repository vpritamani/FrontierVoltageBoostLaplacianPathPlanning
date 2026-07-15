# Algorithms

All algorithms extend a common base class hierarchy. The base classes handle map storage and enforce a consistent interface; concrete algorithms implement `solve()` and `result()`.

---

## Class Hierarchy

```
BaseAlgorithm          (algorithm.py)
├── Algorithm2D        (2dalgorithm.py)   — adds width/height helpers
├── Algorithm3D        (3dalgorithm.py)   — adds width/height/depth helpers
└── AStarAlgorithm     (Baseline Algorithms/astar.py)   — dimension-agnostic
```

Use `Algorithm2D` or `Algorithm3D` as the base when your algorithm only makes sense in one dimensionality. Use `BaseAlgorithm` directly when it is dimension-agnostic (like A*).

---

## Interface

Every algorithm exposes:

| Method | Description |
|--------|-------------|
| `setup()` | Optional pre-solve initialisation. No-op by default. |
| `solve(visualize=False)` | Run the algorithm. Returns the result. |
| `result(visualize=False)` | Return the already-computed result. |

`solve()` and `result()` are abstract in `BaseAlgorithm` — subclasses must implement both.

---

## Implementing a New Algorithm

**2D-only example:**

```python
import importlib
Algorithm2D = importlib.import_module('Algorithms.2dalgorithm').Algorithm2D

class MyAlgorithm2D(Algorithm2D):
    def solve(self, visualize=False):
        grid  = self.map.grid   # (height, width) numpy array, 0=free 1=obstacle
        start = self.map.start  # (x, y)
        end   = self.map.end    # (x, y)
        # ... your logic ...
        self._result = path
        return self._result

    def result(self, visualize=False):
        return self._result
```

**Dimension-agnostic example:**

```python
from Algorithms.algorithm import BaseAlgorithm

class MyGenericAlgorithm(BaseAlgorithm):
    def solve(self, visualize=False):
        ndim = self.map.grid.ndim   # 2 or 3
        ...
```

---

## Baseline Algorithms

### `AStarAlgorithm` (Baseline Algorithms/astar.py)

A* search that works on both 2D and 3D maps. Reads `grid.ndim` at solve time.

```python
import importlib.util, os

_spec = importlib.util.spec_from_file_location(
    'astar',
    os.path.join('<project_root>', 'Algorithms', 'Baseline Algorithms', 'astar.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AStarAlgorithm = _mod.AStarAlgorithm

from map_generation import Map2D
import importlib
m = importlib.import_module('2dmapgenerator').MapGenerator2D(100, 100).generate()

algo = AStarAlgorithm(m, timeout=30.0)
path = algo.solve(visualize=True)   # saves astar_result.png
path = algo.result()                # list of (x, y) or (x, y, z) tuples
```

**Static solvability check** (used by generators, no Map object needed):

```python
AStarAlgorithm.is_solvable(grid, start, end, timeout=30.0)  # -> bool
```
