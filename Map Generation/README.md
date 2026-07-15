# Map Generation

Maps are generated as numpy arrays and wrapped in typed `Map` objects that carry the grid, start point, and end point together.

---

## Grid Convention

| Value | Meaning   |
|-------|-----------|
| `0`   | Free space |
| `1`   | Obstacle   |

- **2D grid shape:** `(height, width)` — indexed as `grid[y, x]`
- **3D grid shape:** `(depth, height, width)` — indexed as `grid[z, y, x]`
- **Start / end points:** `(x, y)` for 2D, `(x, y, z)` for 3D

---

## Classes

### `Map` (map.py)
Base class. Holds `grid`, `start`, `end`.

### `Map2D` (2dmap.py)
Extends `Map`. Adds `width`, `height` properties and image export.

```python
from map_generation import Map2D

m: Map2D
m.width        # int
m.height       # int
m.to_image()   # (height, width) uint8 array — free=255, obstacle=0
m.save_image('map.png')
```

### `Map3D` (3dmap.py)
Extends `Map`. Adds `width`, `height`, `depth` properties and slice export.

```python
from map_generation import Map3D

m: Map3D
m.width                          # int
m.height                         # int
m.depth                          # int
m.to_image_slice(z)              # (height, width) uint8 array for slice z
m.save_image_slices('slices/')   # saves one PNG per z-slice
```

---

## Generators

Both generators guarantee the returned map is solvable — they run A* internally and regenerate up to 100 times if a map has no valid path.

### `MapGenerator2D` (2dmapgenerator.py)

```python
import importlib
MapGenerator2D = importlib.import_module('2dmapgenerator').MapGenerator2D

gen = MapGenerator2D(
    width=200,
    height=200,
    num_obstacles_range=(5, 30),   # random number of obstacles per map
    obstacle_size_range=(2, 25),   # random obstacle side length (auto if None)
    solvability_timeout=30.0,      # seconds before a candidate is discarded
)

m = gen.generate()                      # one solvable Map2D
maps = gen.generate_many(num_maps=10)   # list of Map2D, also writes .npy + CSV
```

### `MapGenerator3D` (3dmapgenerator.py)

```python
import importlib
MapGenerator3D = importlib.import_module('3dmapgenerator').MapGenerator3D

gen = MapGenerator3D(
    width=50, height=50, depth=50,
    num_obstacles_range=(5, 20),
    solvability_timeout=30.0,
)

m = gen.generate()
maps = gen.generate_many(num_maps=5, output_dir='my_maps/')
```

`generate_many` writes each grid as a `.npy` file and a `start_end_points.csv` into the output directory.
