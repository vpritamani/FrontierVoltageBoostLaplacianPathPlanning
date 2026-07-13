"""
Local persistence for the benchmark app.

Everything lives under ``Benchmark Data/`` at the project root:

    Benchmark Data/
    ├── map_sets/
    │   └── ms_20260712-141530_ab12/
    │       ├── manifest.json          # dims, size, gen params, maps list
    │       ├── map_000.npy ...
    │       └── thumbs/map_000.png     # 2D only, start/end marked
    └── runs/
        └── run_20260712-142001_cd34/
            ├── manifest.json          # config, status, map set ref
            ├── results.json           # one record per (map, algorithm)
            ├── results.csv            # flattened export
            ├── paths/<algo>/<stem>.json
            └── images/<algo>/<stem>_path.png   # 2D only

IDs embed a timestamp plus random suffix, so re-running an identical
configuration never overwrites earlier data.
"""

import csv
import json
import os
import shutil
import time
import uuid
from datetime import datetime

import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(_PROJECT_ROOT, 'Benchmark Data')
MAP_SETS_DIR = os.path.join(DATA_DIR, 'map_sets')
RUNS_DIR = os.path.join(DATA_DIR, 'runs')


def ensure_dirs():
    os.makedirs(MAP_SETS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)


def new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"


def clean_json(obj):
    """Recursively convert numpy scalars / non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _read_json(path: str):
    # Windows: reading while a writer is mid-replace can raise PermissionError;
    # writers retry too, so a couple of short retries always converge.
    for attempt in range(5):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except PermissionError:
            time.sleep(0.03 * (attempt + 1))
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _write_json(path: str, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(clean_json(obj), f, indent=2)
    # Windows denies os.replace while another thread has the target open for
    # reading (e.g. a UI poll hitting the manifest). Readers are short-lived,
    # so brief retries make the swap reliable.
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.04 * (attempt + 1))
    os.replace(tmp, path)   # last attempt — surface the error if still locked


# ---------------------------------------------------------------------------
# Map sets
# ---------------------------------------------------------------------------

def map_set_dir(ms_id: str) -> str:
    return os.path.join(MAP_SETS_DIR, ms_id)


def list_map_sets() -> list:
    ensure_dirs()
    out = []
    for name in sorted(os.listdir(MAP_SETS_DIR), reverse=True):
        d = os.path.join(MAP_SETS_DIR, name)
        mf = os.path.join(d, 'manifest.json')
        if os.path.isfile(mf) and not _is_tombstoned(d):
            try:
                out.append(_read_json(mf))
            except (json.JSONDecodeError, OSError):
                continue
    return out


def get_map_set(ms_id: str) -> dict:
    return _read_json(os.path.join(map_set_dir(ms_id), 'manifest.json'))


def save_map_set_manifest(manifest: dict):
    _write_json(os.path.join(map_set_dir(manifest['id']), 'manifest.json'), manifest)


def load_map_grid(ms_id: str, map_name: str) -> np.ndarray:
    return np.load(os.path.join(map_set_dir(ms_id), map_name))


_DELETED_MARKER = '.deleted'


def _is_tombstoned(path: str) -> bool:
    return os.path.isfile(os.path.join(path, _DELETED_MARKER))


def _rmtree(path: str):
    """Delete a data directory, robust to Windows file locks.

    OneDrive / AV / indexers can hold handles on fresh files for a long time,
    making rmtree fail with PermissionError. After brief retries we fall back
    to a tombstone: remove whatever is removable, write a `.deleted` marker,
    and hide the directory from all listings. Startup sweeps tombstones away
    once the locks are gone.
    """
    for attempt in range(6):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except (PermissionError, OSError):
            time.sleep(0.15 * (attempt + 1))
    shutil.rmtree(path, ignore_errors=True)
    if os.path.isdir(path):
        try:
            with open(os.path.join(path, _DELETED_MARKER), 'w', encoding='utf-8') as f:
                f.write('pending cleanup — removed on next launch')
        except OSError:
            pass


def sweep_tombstones():
    """Remove directories tombstoned by earlier failed deletes (call at startup)."""
    for root in (MAP_SETS_DIR, RUNS_DIR):
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p) and _is_tombstoned(p):
                shutil.rmtree(p, ignore_errors=True)


def map_set_exists(ms_id: str) -> bool:
    p = map_set_dir(ms_id)
    return os.path.isdir(p) and not _is_tombstoned(p)


def delete_map_set(ms_id: str):
    _rmtree(map_set_dir(ms_id))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def run_dir(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id)


def list_runs() -> list:
    ensure_dirs()
    out = []
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        d = os.path.join(RUNS_DIR, name)
        mf = os.path.join(d, 'manifest.json')
        if os.path.isfile(mf) and not _is_tombstoned(d):
            try:
                out.append(_read_json(mf))
            except (json.JSONDecodeError, OSError):
                continue
    return out


def get_run(run_id: str) -> dict:
    return _read_json(os.path.join(run_dir(run_id), 'manifest.json'))


def save_run_manifest(manifest: dict):
    os.makedirs(run_dir(manifest['id']), exist_ok=True)
    _write_json(os.path.join(run_dir(manifest['id']), 'manifest.json'), manifest)


def get_run_results(run_id: str) -> list:
    path = os.path.join(run_dir(run_id), 'results.json')
    if not os.path.isfile(path):
        return []
    return _read_json(path)


def save_run_results(run_id: str, results: list):
    _write_json(os.path.join(run_dir(run_id), 'results.json'), results)


def save_path_json(run_id: str, algo_id: str, map_stem: str, path):
    d = os.path.join(run_dir(run_id), 'paths', algo_id)
    os.makedirs(d, exist_ok=True)
    _write_json(os.path.join(d, f'{map_stem}.json'), path)


def get_path_json(run_id: str, algo_id: str, map_stem: str):
    """Stored solved path for one (algorithm, map), or None if absent."""
    p = os.path.join(run_dir(run_id), 'paths', algo_id, f'{map_stem}.json')
    if not os.path.isfile(p):
        return None
    return _read_json(p)


def image_path(run_id: str, algo_id: str, map_stem: str) -> str:
    d = os.path.join(run_dir(run_id), 'images', algo_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{map_stem}_path.png')


def delete_run(run_id: str):
    _rmtree(run_dir(run_id))


def dependent_runs(ms_id: str) -> list:
    """Runs that reference the given map set."""
    return [r for r in list_runs() if r.get('map_set_id') == ms_id]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_results_csv(run_id: str, results: list, metric_keys: list):
    """Flatten results.json records into results.csv inside the run dir."""
    fields = [
        'run_id', 'run_name', 'timestamp', 'map_set_id', 'map_name', 'dims',
        'algorithm_id', 'algorithm_label', 'params',
        'solved', 'error', 'time_s', 'path_steps', 'path_length_euclidean',
    ] + metric_keys
    out = os.path.join(run_dir(run_id), 'results.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for rec in results:
            row = {k: rec.get(k, '') for k in fields}
            row['params'] = json.dumps(rec.get('params', {}))
            metrics = rec.get('metrics') or {}
            for mk in metric_keys:
                row[mk] = metrics.get(mk, '')
            writer.writerow(row)
    return out


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------

def recover_interrupted():
    """Startup recovery: sweep tombstones, mark non-terminal states interrupted."""
    sweep_tombstones()
    for manifest in list_runs():
        if manifest.get('status') in ('pending', 'running', 'cancelling'):
            manifest['status'] = 'interrupted'
            save_run_manifest(manifest)
    for manifest in list_map_sets():
        if manifest.get('status') in ('pending', 'generating'):
            manifest['status'] = 'interrupted'
            save_map_set_manifest(manifest)
