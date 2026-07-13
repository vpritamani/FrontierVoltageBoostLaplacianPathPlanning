"""
Map-set generation service.

Wraps the project's MapGenerator2D/3D with two additions:

* **Interior start/end points.** FrontierVoltageBoostLaplace treats the map
  boundary as an obstacle, so a start/end on row/col 0 or the last row/col
  would be unsolvable for it (while still passing the A* solvability check).
  To keep benchmarks fair across algorithms, start/end are sampled strictly
  inside the boundary.
* **Optional seed** for reproducible sets.

Maps are persisted through :mod:`benchmark_app.storage`; each map set gets a
manifest recording generation parameters and per-map start/end points.
"""

import importlib
import os
import random
import sys
from datetime import datetime

import numpy as np

from . import storage

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_MAP_GEN_DIR = os.path.join(_PROJECT_ROOT, 'Map Generation')
if _MAP_GEN_DIR not in sys.path:
    sys.path.insert(0, _MAP_GEN_DIR)

import map_generation  # noqa: F401  (registers map/2dmap/3dmap modules)

MapGenerator2D = importlib.import_module('2dmapgenerator').MapGenerator2D
MapGenerator3D = importlib.import_module('3dmapgenerator').MapGenerator3D


class _InteriorMixin:
    """Overrides _find_free_point to avoid boundary cells (see module docstring)."""

    def _find_free_point(self, grid: np.ndarray) -> tuple:
        # grid shape is (h, w) for 2D or (d, h, w) for 3D; points are (x, y[, z]).
        shape = grid.shape[::-1]  # (w, h[, d]) to match point order
        for _ in range(100000):
            point = tuple(random.randint(1, s - 2) for s in shape)
            if grid[tuple(reversed(point))] == 0:
                return point
        raise RuntimeError('Could not find a free interior cell')


class InteriorMapGenerator2D(_InteriorMixin, MapGenerator2D):
    pass


class InteriorMapGenerator3D(_InteriorMixin, MapGenerator3D):
    pass


def _save_thumb(ms_dir: str, stem: str, grid: np.ndarray, start: tuple, end: tuple):
    """2D thumbnail PNG: white free, black obstacle, green start, red end."""
    from PIL import Image
    grey = ((1 - grid) * 255).astype(np.uint8)
    rgb = np.stack([grey, grey, grey], axis=2)
    rgb[start[1], start[0]] = [0, 200, 0]
    rgb[end[1], end[0]] = [220, 0, 0]
    thumbs = os.path.join(ms_dir, 'thumbs')
    os.makedirs(thumbs, exist_ok=True)
    Image.fromarray(rgb, mode='RGB').save(os.path.join(thumbs, f'{stem}.png'))


# ---------------------------------------------------------------------------
# N-dimensional generation (dims > 3)
# ---------------------------------------------------------------------------

def _bfs_solvable(grid: np.ndarray, start: tuple, end: tuple) -> bool:
    """Face-neighbour BFS reachability — dimension-agnostic solvability check.

    Stricter than the diagonal-move A* used for 2D/3D (a face-connected path
    implies solvability under any richer neighbourhood).
    Points are in user (x, y, ...) order; grid is indexed reversed.
    """
    from collections import deque
    s = tuple(reversed(start))
    e = tuple(reversed(end))
    if grid[s] != 0 or grid[e] != 0:
        return False
    seen = np.zeros(grid.shape, dtype=bool)
    seen[s] = True
    dq = deque([s])
    while dq:
        cur = dq.popleft()
        if cur == e:
            return True
        for d in range(grid.ndim):
            for delta in (-1, 1):
                nb = list(cur)
                nb[d] += delta
                if 0 <= nb[d] < grid.shape[d]:
                    nb = tuple(nb)
                    if not seen[nb] and grid[nb] == 0:
                        seen[nb] = True
                        dq.append(nb)
    return False


class NDMapGenerator:
    """Random hyper-rectangle obstacle maps for arbitrary dimensionality.

    ``sizes`` is in user order (x, y, z, w, ...); the grid is allocated with
    reversed shape so ``grid[reversed(point)]`` indexes a user-order point —
    the same convention as Map2D/Map3D.
    """

    _MAX_RETRIES = 100

    def __init__(self, sizes: list, num_obstacles_range=(5, 30)):
        self.sizes = [int(s) for s in sizes]
        self.num_obstacles_range = num_obstacles_range
        min_side = min(self.sizes)
        self.obstacle_size_range = (max(1, min_side // 20), max(2, min_side // 8))

    def generate(self):
        from map_generation import Map
        shape = tuple(reversed(self.sizes))
        for _ in range(self._MAX_RETRIES):
            grid = np.zeros(shape, dtype=np.uint8)
            for _ in range(random.randint(*self.num_obstacles_range)):
                self._place_obstacle(grid)
            start = self._find_free_interior(grid)
            end = self._find_free_interior(grid)
            if _bfs_solvable(grid, start, end):
                return Map(grid, start, end)
        raise RuntimeError(f'Could not generate a solvable {len(self.sizes)}D map '
                           f'after {self._MAX_RETRIES} attempts')

    def _place_obstacle(self, grid: np.ndarray):
        size = min(random.randint(*self.obstacle_size_range), *self.sizes)
        corner = [random.randint(0, s - size) for s in self.sizes]
        # slices in grid (reversed) order
        sl = tuple(slice(c, c + size) for c in reversed(corner))
        grid[sl] = 1

    def _find_free_interior(self, grid: np.ndarray) -> tuple:
        for _ in range(100000):
            point = tuple(random.randint(1, s - 2) for s in self.sizes)
            if grid[tuple(reversed(point))] == 0:
                return point
        raise RuntimeError('Could not find a free interior cell')


def generate_map_set(
    name: str,
    dims: int,
    width: int = 0,
    height: int = 0,
    depth: int = 30,
    num_maps: int = 5,
    obstacles_min: int = 5,
    obstacles_max: int = 30,
    seed: int | None = None,
    shape: list | None = None,
    progress_cb=None,
    id_holder: dict | None = None,
) -> dict:
    """Generate a solvable map set and persist it. Returns the manifest.

    2D/3D use the project generators (A*-validated); for dims > 3 pass
    ``shape`` — a user-order size list like ``[12, 12, 12, 8]`` — and maps are
    validated with face-neighbour BFS instead.

    ``id_holder`` (if given) receives the new set's id under key ``ms_id`` as
    soon as it is allocated, so a caller cancelling mid-generation can clean up
    the partial directory.
    """
    if shape:
        sizes = [int(s) for s in shape]
        dims = len(sizes)
    elif dims == 2:
        sizes = [width, height]
    elif dims == 3:
        sizes = [width, height, depth]
    else:
        raise ValueError('for dims > 3 provide a shape list, e.g. [12, 12, 12, 8]')

    if dims < 2:
        raise ValueError('dims must be >= 2')
    if num_maps < 1:
        raise ValueError('num_maps must be >= 1')
    min_side = 4  # need at least 2 interior cells per axis
    if any(s < min_side for s in sizes):
        raise ValueError(f'each map side must be >= {min_side}')

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if dims == 2:
        gen = InteriorMapGenerator2D(width=sizes[0], height=sizes[1],
                                     num_obstacles_range=(obstacles_min, obstacles_max))
    elif dims == 3:
        gen = InteriorMapGenerator3D(width=sizes[0], height=sizes[1], depth=sizes[2],
                                     num_obstacles_range=(obstacles_min, obstacles_max))
    else:
        gen = NDMapGenerator(sizes, num_obstacles_range=(obstacles_min, obstacles_max))

    ms_id = storage.new_id('ms')
    if id_holder is not None:
        id_holder['ms_id'] = ms_id
    ms_dir = storage.map_set_dir(ms_id)
    os.makedirs(ms_dir, exist_ok=True)

    manifest = {
        'id': ms_id,
        'name': name or ms_id,
        'created': datetime.now().isoformat(timespec='seconds'),
        'status': 'generating',
        'dims': dims,
        'shape': sizes,                                # user-order (x, y, ...)
        'width': sizes[0] if dims <= 3 else None,
        'height': sizes[1] if dims <= 3 else None,
        'depth': sizes[2] if dims == 3 else None,
        'num_maps': num_maps,
        'gen_params': {
            'obstacles_min': obstacles_min,
            'obstacles_max': obstacles_max,
            'seed': seed,
        },
        'maps': [],
    }
    storage.save_map_set_manifest(manifest)

    for i in range(num_maps):
        if progress_cb:
            progress_cb(i, num_maps, f'generating map {i + 1}/{num_maps}')
        m = gen.generate()
        stem = f'map_{i:03d}'
        np.save(os.path.join(ms_dir, f'{stem}.npy'), m.grid)
        if dims == 2:
            _save_thumb(ms_dir, stem, m.grid, m.start, m.end)
        manifest['maps'].append({
            'name': f'{stem}.npy',
            'stem': stem,
            'start': list(m.start),
            'end': list(m.end),
        })
        storage.save_map_set_manifest(manifest)  # persist progress incrementally

    manifest['status'] = 'ready'
    storage.save_map_set_manifest(manifest)
    if progress_cb:
        progress_cb(num_maps, num_maps, 'done')
    return manifest


def import_map_set(
    folder: str,
    name: str | None = None,
    progress_cb=None,
    id_holder: dict | None = None,
) -> dict:
    """Import an existing folder of ``.npy`` maps + ``start_end_points.csv``
    (the format written by the retired CLI scripts) as a new map set.

    The folder itself is not modified — files are copied into
    ``Benchmark Data/map_sets/<new id>/``. Maps are not re-validated for
    solvability (the original generators validated them at creation).
    """
    import csv as _csv

    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise ValueError(f'Folder not found: {folder}')
    csv_path = os.path.join(folder, 'start_end_points.csv')
    if not os.path.isfile(csv_path):
        raise ValueError(f'No start_end_points.csv in {folder} — cannot import '
                         'maps without start/end points.')

    with open(csv_path, newline='') as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError('start_end_points.csv is empty.')

    has_z = 'start_z' in rows[0] and rows[0].get('start_z') not in (None, '')
    dims = 3 if has_z else 2

    ms_id = storage.new_id('ms')
    if id_holder is not None:
        id_holder['ms_id'] = ms_id
    ms_dir = storage.map_set_dir(ms_id)
    os.makedirs(ms_dir, exist_ok=True)

    manifest = {
        'id': ms_id,
        'name': name or f'imported: {os.path.basename(folder)}',
        'created': datetime.now().isoformat(timespec='seconds'),
        'status': 'generating',
        'dims': dims,
        'width': None, 'height': None, 'depth': None,
        'num_maps': len(rows),
        'gen_params': {'imported_from': folder},
        'maps': [],
    }
    storage.save_map_set_manifest(manifest)

    skipped = []
    for i, row in enumerate(rows):
        if progress_cb:
            progress_cb(i, len(rows), f"importing {row['map_name']}")
        src = os.path.join(folder, row['map_name'])
        if not os.path.isfile(src):
            skipped.append(row['map_name'])   # partially cleaned legacy folder
            continue
        grid = np.load(src)
        if grid.ndim != dims:
            skipped.append(row['map_name'])
            continue

        if dims == 2:
            start = (int(row['start_x']), int(row['start_y']))
            end = (int(row['end_x']), int(row['end_y']))
            h, w = grid.shape
            manifest['width'], manifest['height'] = w, h
        else:
            start = (int(row['start_x']), int(row['start_y']), int(row['start_z']))
            end = (int(row['end_x']), int(row['end_y']), int(row['end_z']))
            d, h, w = grid.shape
            manifest['width'], manifest['height'], manifest['depth'] = w, h, d

        stem = f'map_{len(manifest["maps"]):03d}'
        np.save(os.path.join(ms_dir, f'{stem}.npy'), grid.astype(np.uint8))
        if dims == 2:
            _save_thumb(ms_dir, stem, grid, start, end)
        manifest['maps'].append({
            'name': f'{stem}.npy',
            'stem': stem,
            'source_name': row['map_name'],
            'start': list(start),
            'end': list(end),
        })
        storage.save_map_set_manifest(manifest)

    if not manifest['maps']:
        raise ValueError('No importable maps found (all CSV entries missing or wrong dimensionality).')

    manifest['shape'] = ([manifest['width'], manifest['height']] +
                         ([manifest['depth']] if dims == 3 else []))
    manifest['num_maps'] = len(manifest['maps'])
    if skipped:
        manifest['gen_params']['skipped'] = skipped
    manifest['status'] = 'ready'
    storage.save_map_set_manifest(manifest)
    if progress_cb:
        progress_cb(len(rows), len(rows), 'done')
    return manifest
