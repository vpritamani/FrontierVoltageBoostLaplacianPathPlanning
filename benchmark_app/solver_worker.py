"""
Run a single (algorithm, map) solve in a child process with a hard timeout.

Rationale: some algorithms have no internal timeout (FrontierVoltageBoostLaplace
loops until the start cell is solved, which never happens if the start is
unreachable under its rules). Python threads cannot be killed, but a child
process can be terminated — so each solve runs in its own process. This also
isolates the app from crashes inside algorithm code.

The child measures wall-clock time around ``solve()`` itself, so process
startup overhead is not counted in the reported timing.
"""

import multiprocessing as mp
import time
import traceback

import numpy as np


def _worker(conn, algo_id: str, params: dict, grid: np.ndarray, start: tuple, end: tuple):
    """Child-process entry: build map + algorithm, solve, send result back."""
    try:
        from benchmark_app.registry import get_spec
        from map_generation import Map, Map2D, Map3D

        if grid.ndim == 2:
            map_cls = Map2D
        elif grid.ndim == 3:
            map_cls = Map3D
        else:
            map_cls = Map   # N-dimensional grids use the base Map
        map_obj = map_cls(grid, tuple(start), tuple(end))

        spec = get_spec(algo_id)
        algo = spec.build(map_obj, params)

        t0 = time.perf_counter()
        path = algo.solve()
        elapsed = time.perf_counter() - t0

        if path is not None:
            # Preserve continuous coordinates — algorithms like FVB return
            # decimal positions and metrics must be computed on the raw path.
            path = [tuple(float(c) for c in p) for p in path]

        conn.send({'ok': True, 'path': path, 'time_s': elapsed})
    except Exception:
        conn.send({'ok': False, 'error': traceback.format_exc(limit=5)})
    finally:
        conn.close()


def solve_with_timeout(algo_id: str, params: dict, grid: np.ndarray,
                       start: tuple, end: tuple, timeout_s: float = 120.0,
                       cancel_event=None) -> dict:
    """Solve in a subprocess. Returns {solved, path, time_s, error, cancelled?}.

    ``cancel_event`` (a ``threading.Event``) is polled while waiting; setting
    it terminates the child immediately and returns a cancelled outcome.
    """
    ctx = mp.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker,
                       args=(child_conn, algo_id, params, grid, start, end))
    proc.start()
    child_conn.close()  # parent keeps only the read end

    # Wait in short slices so cancellation is responsive.
    result = None
    cancelled = False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        if parent_conn.poll(min(0.2, max(0.0, deadline - time.monotonic()))):
            try:
                result = parent_conn.recv()
            except EOFError:
                result = None
            break

    if cancelled:
        proc.terminate()
        proc.join()
        parent_conn.close()
        return {'solved': False, 'path': None, 'time_s': None,
                'error': 'cancelled', 'cancelled': True}

    proc.join(timeout=5)
    timed_out = proc.is_alive()
    if timed_out:
        proc.terminate()
        proc.join()

    parent_conn.close()

    if result is None:
        if timed_out:
            return {'solved': False, 'path': None, 'time_s': timeout_s,
                    'error': f'timed out after {timeout_s:.0f}s (process terminated)'}
        return {'solved': False, 'path': None, 'time_s': None,
                'error': f'solver process died (exit code {proc.exitcode})'}

    if not result.get('ok'):
        return {'solved': False, 'path': None, 'time_s': None,
                'error': result.get('error', 'unknown error')}

    path = result['path']
    return {
        'solved': path is not None and len(path) > 0,
        'path': path,
        'time_s': result['time_s'],
        'error': None if path else 'algorithm returned no path',
    }
