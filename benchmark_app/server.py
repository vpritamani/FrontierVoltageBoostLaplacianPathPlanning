"""
Flask server for the benchmark app.

Serves the single-page UI from ``static/`` and a JSON API:

    GET    /api/registry                         algorithm specs (+ param schemas)
    GET    /api/map_sets                         all map set manifests
    GET    /api/map_sets/<id>                    one manifest + dependent runs
    GET    /api/map_sets/<id>/thumb/<stem>.png   2D thumbnail
    DELETE /api/map_sets/<id>?cascade=0|1        delete (optionally + dependent runs)
    POST   /api/map_sets                         generate a new set (background job)
    GET    /api/runs                             run manifests (newest first)
    POST   /api/runs                             start a run (existing or new maps)
    GET    /api/runs/<id>                        manifest + full results
    GET    /api/runs/<id>/image/<algo>/<file>    path overlay PNG
    GET    /api/runs/<id>/results.csv            CSV download
    DELETE /api/runs/<id>                        delete run
    GET    /api/jobs                             background job progress
"""

import os
import re
import threading
import webbrowser

from flask import Flask, Response, jsonify, request, send_from_directory

from . import runner, storage
from .registry import registry_json

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app = Flask(__name__, static_folder=_STATIC, static_url_path='/static')

_ID_RE = re.compile(r'^[A-Za-z0-9_-]+$')
_FILE_RE = re.compile(r'^[A-Za-z0-9_.-]+$')


def _check(value: str, pattern=_ID_RE) -> str:
    if not pattern.match(value):
        raise ValueError(f'invalid identifier: {value!r}')
    return value


@app.errorhandler(Exception)
def _handle_error(exc):
    import traceback
    traceback.print_exc()
    code = 404 if isinstance(exc, (FileNotFoundError, KeyError)) else 400
    return jsonify({'error': str(exc)}), code


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get('/')
def index():
    return send_from_directory(_STATIC, 'index.html')

@app.get('/favicon.ico')
def favicon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
           '<rect width="16" height="16" rx="3" fill="#2a78d6"/>'
           '<path d="M3 13 L7 6 L10 9 L13 3" stroke="#fff" stroke-width="2" fill="none"/></svg>')
    return Response(svg, mimetype='image/svg+xml')


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@app.get('/api/registry')
def api_registry():
    return jsonify(registry_json())


_system_info = None

@app.get('/api/system')
def api_system():
    """Host capabilities for the UI: CPU cores, PyTorch/CUDA availability."""
    global _system_info
    if _system_info is None:
        info = {'cpu_count': os.cpu_count() or 1,
                'torch': False, 'cuda': False, 'cuda_device': None}
        try:
            import torch
            info['torch'] = True
            if torch.cuda.is_available():
                info['cuda'] = True
                info['cuda_device'] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        _system_info = info
    return jsonify(_system_info)


# ---------------------------------------------------------------------------
# Map sets
# ---------------------------------------------------------------------------

@app.get('/api/map_sets')
def api_map_sets():
    return jsonify(storage.list_map_sets())


@app.get('/api/map_sets/<ms_id>')
def api_map_set(ms_id):
    _check(ms_id)
    manifest = storage.get_map_set(ms_id)
    deps = storage.dependent_runs(ms_id)
    manifest['dependent_runs'] = [
        {'id': r['id'], 'name': r['name'], 'status': r['status']} for r in deps
    ]
    return jsonify(manifest)


@app.get('/api/map_sets/<ms_id>/thumb/<name>')
def api_map_set_thumb(ms_id, name):
    _check(ms_id)
    _check(name, _FILE_RE)
    return send_from_directory(os.path.join(storage.map_set_dir(ms_id), 'thumbs'), name)


@app.post('/api/map_sets')
def api_create_map_set():
    body = request.get_json(force=True)
    job = runner.enqueue_generate(body)
    return jsonify({'job_id': job['id']})


@app.post('/api/map_sets/import')
def api_import_map_set():
    """Import a folder of .npy maps + start_end_points.csv as a new map set."""
    body = request.get_json(force=True)
    if not body.get('path'):
        raise ValueError('provide the folder path to import')
    job = runner.enqueue_generate({'import_path': body['path'],
                                   'name': body.get('name')})
    return jsonify({'job_id': job['id']})


@app.get('/api/map_sets/<ms_id>/grid/<name>')
def api_map_set_grid(ms_id, name):
    """Raw grid voxels for client-side 3D/slice rendering (base64 uint8)."""
    import base64
    _check(ms_id)
    _check(name, _FILE_RE)
    grid = storage.load_map_grid(ms_id, name)
    return jsonify({
        'shape': list(grid.shape),               # (h, w) or (d, h, w)
        'data_b64': base64.b64encode(grid.astype('uint8').tobytes()).decode('ascii'),
    })


@app.delete('/api/map_sets/<ms_id>')
def api_delete_map_set(ms_id):
    _check(ms_id)
    cascade = request.args.get('cascade', '0') == '1'
    deps = storage.dependent_runs(ms_id)
    deleted_runs = []
    if cascade:
        for r in deps:
            storage.delete_run(r['id'])
            deleted_runs.append(r['id'])
    storage.delete_map_set(ms_id)
    return jsonify({'deleted': ms_id, 'deleted_runs': deleted_runs})


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@app.get('/api/runs')
def api_runs():
    return jsonify(storage.list_runs())


@app.post('/api/runs')
def api_create_run():
    """
    Body: {
        name, algorithms:[{id, params}], solve_timeout_s,
        map_set_id?, map_names?,          # run on existing maps
        new_map_set?: {gen params}        # or generate fresh maps first
    }
    """
    body = request.get_json(force=True)
    if not body.get('algorithms'):
        raise ValueError('select at least one algorithm')

    gen_job_id = None
    if body.get('new_map_set'):
        gen_job = runner.enqueue_generate(body['new_map_set'])
        gen_job_id = gen_job['id']
    elif not body.get('map_set_id'):
        raise ValueError('provide map_set_id or new_map_set')

    run_job = runner.enqueue_run(body, map_set_from_job=gen_job_id)
    return jsonify({'job_id': run_job['id'],
                    'run_id': run_job['payload']['run_id'],
                    'generate_job_id': gen_job_id})


@app.get('/api/runs/<run_id>')
def api_run(run_id):
    _check(run_id)
    manifest = storage.get_run(run_id)
    manifest['results'] = storage.get_run_results(run_id)
    manifest['map_set_exists'] = (storage.map_set_exists(manifest['map_set_id'])
                                  if manifest.get('map_set_id') else False)
    return jsonify(manifest)


@app.post('/api/runs/<run_id>/cancel')
def api_cancel_run(run_id):
    _check(run_id)
    manifest = runner.request_cancel_run(run_id)
    return jsonify({'status': manifest['status']})


@app.post('/api/runs/<run_id>/metrics')
def api_recompute_metrics(run_id):
    """Recompute smoothness metrics from stored paths — no planner re-run.

    Body: {theta_degrees, sweep_range, save}
    """
    _check(run_id)
    body = request.get_json(force=True)
    out = runner.recompute_metrics(
        run_id,
        theta_degrees=float(body.get('theta_degrees', 30.0)),
        sweep_range=int(body.get('sweep_range', 5)),
        save=bool(body.get('save', False)),
    )
    return jsonify(out)


@app.get('/api/runs/<run_id>/path/<algo_id>/<stem>')
def api_run_path(run_id, algo_id, stem):
    """Stored solved path for one (algorithm, map)."""
    _check(run_id)
    _check(algo_id)
    _check(stem, _FILE_RE)
    path = storage.get_path_json(run_id, algo_id, stem)
    if path is None:
        raise FileNotFoundError('no stored path for this map/algorithm')
    return jsonify(path)


@app.get('/api/runs/<run_id>/image/<algo_id>/<name>')
def api_run_image(run_id, algo_id, name):
    _check(run_id)
    _check(algo_id)
    _check(name, _FILE_RE)
    return send_from_directory(
        os.path.join(storage.run_dir(run_id), 'images', algo_id), name)


@app.get('/api/compare/results.csv')
def api_compare_csv():
    """Combined raw per-record CSV for several runs: ?runs=id1,id2,…"""
    ids = [i for i in (request.args.get('runs') or '').split(',') if i]
    if not ids:
        raise ValueError('pass ?runs=<id>,<id>,…')
    from smoothness_metrics import SmoothnessMetrics
    all_results = []
    for run_id in ids:
        _check(run_id)
        all_results.extend(storage.get_run_results(run_id))
    text = storage.results_csv_text(all_results, SmoothnessMetrics.METRIC_KEYS)
    return Response(text, mimetype='text/csv', headers={
        'Content-Disposition': 'attachment; filename=compare_results.csv'})


@app.get('/api/runs/<run_id>/results.csv')
def api_run_csv(run_id):
    _check(run_id)
    return send_from_directory(storage.run_dir(run_id), 'results.csv',
                               as_attachment=True,
                               download_name=f'{run_id}_results.csv')


@app.delete('/api/runs/<run_id>')
def api_delete_run(run_id):
    _check(run_id)
    manifest = storage.get_run(run_id)
    if manifest.get('status') in ('pending', 'running', 'cancelling'):
        raise ValueError('cannot delete a run that is still executing')
    storage.delete_run(run_id)
    return jsonify({'deleted': run_id})


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.get('/api/jobs')
def api_jobs():
    return jsonify(runner.jobs_snapshot())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host='127.0.0.1', port=8177, open_browser=True):
    storage.ensure_dirs()
    storage.recover_interrupted()
    url = f'http://{host}:{port}'
    print(f'\n  Path Planning Benchmark — {url}\n')
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, threaded=True)
