"""
Background job queue for the benchmark app.

A single worker thread processes jobs FIFO. Two job types:

* ``generate`` — create or import a map set
* ``run``      — execute one or more algorithms over a map set

A ``run`` job may reference the map set produced by an earlier ``generate``
job (``map_set_from_job``), enabling the "generate maps and run on them" flow
in one click. Because the queue is FIFO, the generate job always finishes
before the dependent run job starts.

Cancellation: ``request_cancel_run(run_id)`` flags both the run job and (if
still in flight) its chained generation. The current solve subprocess is
terminated, partial results are kept, and the run is marked ``cancelled``.

Metrics: paths are persisted per (algorithm, map), so
``recompute_metrics(run_id, theta, sweep, save)`` can re-derive every
smoothness metric from stored paths without re-running any planner.

Job state lives in memory (the UI polls it); durable state (manifests,
results) is written through :mod:`benchmark_app.storage` as work progresses,
so completed data survives restarts even if the app is killed mid-run.
"""

import os
import queue
import sys
import threading
import traceback
from datetime import datetime

import numpy as np

from . import mapgen, storage
from .registry import get_spec
from .solver_worker import solve_with_timeout

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from smoothness_metrics import SmoothnessMetrics  # noqa: E402

DEFAULT_METRIC_PARAMS = {'steering_theta_degrees': 30.0, 'steering_sweep_range': 5}


# ---------------------------------------------------------------------------
# Job bookkeeping
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()
_worker_started = False

# Cancellation: run_id -> Event (runs), job_id -> Event (generate jobs).
_run_cancel: dict[str, threading.Event] = {}
_gen_cancel: dict[str, threading.Event] = {}


class _Cancelled(Exception):
    pass


def _new_job(job_type: str, payload: dict, result: dict | None = None) -> dict:
    job = {
        'id': storage.new_id('job'),
        'type': job_type,
        'status': 'queued',
        'created': datetime.now().isoformat(timespec='seconds'),
        'progress': {'current': 0, 'total': 0, 'message': 'queued'},
        'payload': payload,
        'result': result or {},
        'error': None,
    }
    with _lock:
        _jobs[job['id']] = job
    _queue.put(job['id'])
    _ensure_worker()
    return job


def get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def jobs_snapshot() -> list:
    with _lock:
        return [
            {k: v for k, v in j.items() if k != 'payload'}
            for j in _jobs.values()
        ]


def _set_progress(job: dict, current: int, total: int, message: str):
    job['progress'] = {'current': current, 'total': total, 'message': message}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue_generate(gen_params: dict) -> dict:
    job = _new_job('generate', {'gen_params': gen_params})
    _gen_cancel[job['id']] = threading.Event()
    return job


def enqueue_run(config: dict, map_set_from_job: str | None = None) -> dict:
    """
    config: {
        name, map_set_id?, map_names (list or None for all),
        algorithms: [{id, params}], solve_timeout_s,
        metric_params?: {steering_theta_degrees, steering_sweep_range}
    }
    """
    mp_raw = config.get('metric_params') or {}
    metric_params = {
        'steering_theta_degrees': float(mp_raw.get('steering_theta_degrees',
                                                   DEFAULT_METRIC_PARAMS['steering_theta_degrees'])),
        'steering_sweep_range': int(mp_raw.get('steering_sweep_range',
                                               DEFAULT_METRIC_PARAMS['steering_sweep_range'])),
    }

    run_id = storage.new_id('run')
    manifest = {
        'id': run_id,
        'name': config.get('name') or run_id,
        'created': datetime.now().isoformat(timespec='seconds'),
        'status': 'pending',
        'map_set_id': config.get('map_set_id'),
        'map_set_name': None,
        'dims': None,
        'map_names': config.get('map_names'),
        # Each entry is an *instance*: the same algorithm may appear several
        # times with different params, so every instance gets a unique key
        # (used for result grouping and paths/images storage dirs).
        'algorithms': [
            {'key': f"{a['id']}-{i + 1}",
             'id': a['id'],
             'label': a.get('label') or get_spec(a['id']).label,
             'params': get_spec(a['id']).coerce_params(a.get('params'))}
            for i, a in enumerate(config['algorithms'])
        ],
        'solve_timeout_s': float(config.get('solve_timeout_s') or 120.0),
        'metric_params': metric_params,
        'gen_job_id': map_set_from_job,
        'summary': None,
        'error': None,
    }
    storage.save_run_manifest(manifest)
    _run_cancel[run_id] = threading.Event()
    return _new_job('run', {
        'run_id': run_id,
        'map_set_from_job': map_set_from_job,
    }, result={'run_id': run_id})


def request_cancel_run(run_id: str) -> dict:
    """Cancel a pending/running run (and its in-flight map generation)."""
    manifest = storage.get_run(run_id)
    if manifest.get('status') not in ('pending', 'running'):
        raise ValueError(f"run is not active (status: {manifest.get('status')})")

    ev = _run_cancel.get(run_id)
    if ev is None:
        # Active per manifest but unknown to this process (stale from a crash).
        manifest['status'] = 'interrupted'
        storage.save_run_manifest(manifest)
        return manifest
    ev.set()

    gen_job_id = manifest.get('gen_job_id')
    if gen_job_id and gen_job_id in _gen_cancel:
        gen_job = get_job(gen_job_id)
        if gen_job and gen_job['status'] in ('queued', 'running'):
            _gen_cancel[gen_job_id].set()

    # A queued (not yet started) run won't reach its cancel check until the
    # worker picks it up; mark intent in the manifest now for instant UI feedback.
    manifest['status'] = 'cancelling'
    storage.save_run_manifest(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Metrics recomputation from stored paths (no planner re-run)
# ---------------------------------------------------------------------------

def recompute_metrics(run_id: str, theta_degrees: float, sweep_range: int,
                      save: bool = False) -> dict:
    """Recompute all smoothness metrics for a run from its stored paths.

    Returns {'metric_params': ..., 'results': [...], 'summary': [...], 'saved': bool}.
    With ``save=True`` the run's results.json, results.csv, summary and
    manifest.metric_params are updated in place.
    """
    manifest = storage.get_run(run_id)
    results = storage.get_run_results(run_id)
    metric_params = {
        'steering_theta_degrees': float(theta_degrees),
        'steering_sweep_range': int(sweep_range),
    }

    for rec in results:
        if not rec.get('solved'):
            continue
        path = storage.get_path_json(run_id, _rec_key(rec), rec['map_stem'])
        if path is None:
            continue
        sm = SmoothnessMetrics([tuple(p) for p in path], **metric_params)
        rec['metrics'] = sm.compute_all()

    summary = _summarize(manifest.get('algorithms') or [], results)

    if save:
        storage.save_run_results(run_id, results)
        storage.write_results_csv(run_id, results, SmoothnessMetrics.METRIC_KEYS)
        manifest['metric_params'] = metric_params
        manifest['summary'] = summary
        storage.save_run_manifest(manifest)

    return {'metric_params': metric_params, 'results': storage.clean_json(results),
            'summary': storage.clean_json(summary), 'saved': bool(save)}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _ensure_worker():
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
    t = threading.Thread(target=_worker_loop, name='benchmark-worker', daemon=True)
    t.start()


def _worker_loop():
    while True:
        job_id = _queue.get()
        job = get_job(job_id)
        if job is None:
            continue
        job['status'] = 'running'
        try:
            if job['type'] == 'generate':
                _execute_generate(job)
            elif job['type'] == 'run':
                _execute_run(job)
            if job['status'] == 'running':
                job['status'] = 'done'
        except _Cancelled:
            job['status'] = 'cancelled'
        except Exception:
            job['status'] = 'error'
            job['error'] = traceback.format_exc(limit=8)
            run_id = job.get('payload', {}).get('run_id')
            if run_id:
                try:
                    manifest = storage.get_run(run_id)
                    manifest['status'] = 'error'
                    manifest['error'] = job['error']
                    storage.save_run_manifest(manifest)
                except OSError:
                    pass


def _execute_generate(job: dict):
    gp = job['payload']['gen_params']
    cancel_ev = _gen_cancel.get(job['id'])
    holder = {'ms_id': None}

    def cb(cur, total, msg):
        _set_progress(job, cur, total, msg)
        if cancel_ev is not None and cancel_ev.is_set():
            raise _Cancelled()

    try:
        if gp.get('import_path'):
            manifest = mapgen.import_map_set(
                folder=gp['import_path'],
                name=gp.get('name'),
                progress_cb=cb,
                id_holder=holder,
            )
        else:
            manifest = mapgen.generate_map_set(
                name=gp.get('name'),
                dims=int(gp.get('dims') or 2),
                width=int(gp.get('width') or 0),
                height=int(gp.get('height') or 0),
                depth=int(gp.get('depth') or 30),
                num_maps=int(gp.get('num_maps') or 5),
                obstacles_min=int(gp.get('obstacles_min') or 5),
                obstacles_max=int(gp.get('obstacles_max') or 30),
                seed=int(gp['seed']) if gp.get('seed') not in (None, '') else None,
                shape=gp.get('shape') or None,
                progress_cb=cb,
                id_holder=holder,
            )
    except BaseException:
        # Cancelled or failed — remove the partial map set so no half-built
        # set lingers in 'generating' state.
        if holder['ms_id']:
            try:
                storage.delete_map_set(holder['ms_id'])
            except OSError:
                pass
        raise

    job['result']['map_set_id'] = manifest['id']


def _render_path_image(out_path: str, grid: np.ndarray, path, start, end):
    from PIL import Image
    grey = ((1 - grid) * 255).astype(np.uint8)
    rgb = np.stack([grey, grey, grey], axis=2)
    if path:
        # Path coords are continuous — round only for pixel painting.
        for x, y in path:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < rgb.shape[0] and 0 <= xi < rgb.shape[1]:
                rgb[yi, xi] = [110, 110, 255]
    rgb[start[1], start[0]] = [0, 200, 0]
    rgb[end[1], end[0]] = [220, 0, 0]
    Image.fromarray(rgb, mode='RGB').save(out_path)


def _euclidean_length(path) -> float:
    pts = np.array(path, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _algo_key(entry: dict) -> str:
    """Instance key with fallback for runs created before instance support."""
    return entry.get('key') or entry.get('algorithm_key') or entry.get('algorithm_id') or entry['id']


def _rec_key(rec: dict) -> str:
    return rec.get('algorithm_key') or rec.get('algorithm_id')


def _summarize(algorithms: list, results: list) -> list:
    summary = []
    for algo in algorithms:
        key = _algo_key(algo)
        recs = [r for r in results if _rec_key(r) == key]
        solved = [r for r in recs if r['solved']]
        times = [r['time_s'] for r in solved if r['time_s'] is not None]
        pens = [r['metrics']['steering_penalty'] for r in solved if r.get('metrics')]
        summary.append({
            'algorithm_key': key,
            'algorithm_id': algo['id'],
            'label': algo['label'],
            'params': algo.get('params'),
            'total': len(recs),
            'solved': len(solved),
            'mean_time_s': float(np.mean(times)) if times else None,
            'mean_steering_penalty': float(np.mean(pens)) if pens else None,
        })
    return summary


def _execute_run(job: dict):
    run_id = job['payload']['run_id']
    manifest = storage.get_run(run_id)
    cancel_ev = _run_cancel.get(run_id)

    def check_cancel():
        if cancel_ev is not None and cancel_ev.is_set():
            raise _Cancelled()

    def finish_cancelled(results):
        manifest['status'] = 'cancelled'
        manifest['summary'] = _summarize(manifest['algorithms'], results)
        storage.save_run_manifest(manifest)
        if results:
            storage.write_results_csv(run_id, results, SmoothnessMetrics.METRIC_KEYS)

    try:
        check_cancel()

        # Resolve map set (possibly produced by an earlier generate job).
        from_job = job['payload'].get('map_set_from_job')
        if from_job:
            gen_job = get_job(from_job)
            if gen_job and gen_job['status'] == 'cancelled':
                raise _Cancelled()
            if not gen_job or gen_job['status'] != 'done':
                raise RuntimeError('Map generation failed — run aborted.')
            manifest['map_set_id'] = gen_job['result']['map_set_id']

        ms = storage.get_map_set(manifest['map_set_id'])
        if ms.get('status') != 'ready':
            raise RuntimeError(f"Map set {ms['id']} is not ready (status: {ms.get('status')})")

        manifest['map_set_name'] = ms['name']
        manifest['dims'] = ms['dims']

        # Every algorithm instance must support this dimensionality.
        for algo in manifest['algorithms']:
            if not get_spec(algo['id']).supports_dims(ms['dims']):
                raise RuntimeError(
                    f"{algo['label']} does not support {ms['dims']}D maps")

        wanted = manifest.get('map_names')
        maps = [m for m in ms['maps'] if not wanted or m['name'] in wanted]
        if not maps:
            raise RuntimeError('No maps selected (names did not match the map set).')
        manifest['map_names'] = [m['name'] for m in maps]

        manifest['status'] = 'running'
        storage.save_run_manifest(manifest)

        total = len(maps) * len(manifest['algorithms'])
        done = 0
        results = []
        timeout_s = manifest['solve_timeout_s']
        metric_params = manifest.get('metric_params') or DEFAULT_METRIC_PARAMS

        for m in maps:
            grid = storage.load_map_grid(ms['id'], m['name'])
            start, end = tuple(m['start']), tuple(m['end'])

            for algo in manifest['algorithms']:
                check_cancel()
                key = _algo_key(algo)
                _set_progress(job, done, total, f"{m['name']} · {algo['label']}")

                outcome = solve_with_timeout(
                    algo['id'], algo['params'], grid, start, end, timeout_s,
                    cancel_event=cancel_ev,
                )
                if outcome.get('cancelled'):
                    finish_cancelled(results)
                    raise _Cancelled()

                record = {
                    'run_id': run_id,
                    'run_name': manifest['name'],
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'map_set_id': ms['id'],
                    'map_name': m['name'],
                    'map_stem': m['stem'],
                    'dims': ms['dims'],
                    'algorithm_key': key,
                    'algorithm_id': algo['id'],
                    'algorithm_label': algo['label'],
                    'params': algo['params'],
                    'solved': outcome['solved'],
                    'error': outcome['error'],
                    'time_s': outcome['time_s'],
                    'path_steps': None,
                    'path_length_euclidean': None,
                    'metrics': None,
                    'has_image': False,
                }

                if outcome['solved']:
                    path = outcome['path']
                    record['path_steps'] = len(path)
                    record['path_length_euclidean'] = _euclidean_length(path)
                    record['metrics'] = SmoothnessMetrics(path, **metric_params).compute_all()
                    storage.save_path_json(run_id, key, m['stem'], path)
                    if ms['dims'] == 2:
                        _render_path_image(
                            storage.image_path(run_id, key, m['stem']),
                            grid, path, start, end,
                        )
                        record['has_image'] = True

                results.append(record)
                done += 1
                _set_progress(job, done, total, f"{m['name']} · {algo['label']}")
                storage.save_run_results(run_id, results)  # persist incrementally

        storage.write_results_csv(run_id, results, SmoothnessMetrics.METRIC_KEYS)
        manifest['status'] = 'done'
        manifest['summary'] = _summarize(manifest['algorithms'], results)
        storage.save_run_manifest(manifest)
        job['result']['run_id'] = run_id

    except _Cancelled:
        # Manifest may already be finalized by finish_cancelled; if not, do it.
        current = storage.get_run(run_id)
        if current.get('status') not in ('cancelled',):
            current['status'] = 'cancelled'
            storage.save_run_manifest(current)
        raise
