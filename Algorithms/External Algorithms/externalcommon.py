"""
Shared plumbing for the external comparison planners.

The upstream implementations live in ``third_party/`` (pinned checkouts in
``third_party/src``, compiled modules in ``third_party/build``; run
``python third_party/setup_external.py`` once). This module provides:

* loaders for those checkouts / modules with a clear error when missing
* ``UnsupportedMapError`` — raised when a map falls outside the class of
  worlds a method is defined for (reported as an unsolved map with the reason)
* ``descend_field`` — steepest descent over a scalar grid field, shared by the
  field-based wrappers so they differ only in how the field is built
* grid → geometry conversion for the planar continuous-geometry methods:
  free-space boundary loops (hntf2d) and rectangle chains (CustFields)

Geometry convention: cell (x, y) is the unit square centred on (x, y), i.e.
[x − ½, x + ½] × [y − ½, y + ½]. Points use the repo's user (x, y, ...) order;
grids are indexed reversed (grid[y, x]).
"""

import importlib
import math
import os
import sys

import numpy as np
from scipy import ndimage

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
THIRD_PARTY = os.path.join(PROJECT_ROOT, 'third_party')
BUILD_DIR = os.path.join(THIRD_PARTY, 'build')
SRC_DIR = os.path.join(THIRD_PARTY, 'src')

_SETUP_HINT = 'run `python third_party/setup_external.py {component}` first'


class UnsupportedMapError(RuntimeError):
    """The map is outside the class of worlds the method is defined for."""


def import_built(module: str, component: str):
    """Import a compiled extension from third_party/build."""
    if BUILD_DIR not in sys.path:
        sys.path.insert(0, BUILD_DIR)
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f'{module} is not built — {_SETUP_HINT.format(component=component)} '
                          f'({exc})') from exc


def source_dir(name: str, component: str) -> str:
    """Path to a pinned upstream checkout in third_party/src."""
    path = os.path.join(SRC_DIR, name)
    if not os.path.isdir(path):
        raise ImportError(f'{name} is not fetched — {_SETUP_HINT.format(component=component)}')
    return path


# ---------------------------------------------------------------------------
# Field descent
# ---------------------------------------------------------------------------

def in_free_cell(grid: np.ndarray, pos) -> bool:
    """True when grid-order position ``pos`` lies inside a free cell."""
    idx = []
    for d, p in enumerate(pos):
        i = math.floor(p + 0.5)
        if not 0 <= i < grid.shape[d]:
            return False
        idx.append(i)
    return grid[tuple(idx)] == 0


def descend_field(field: np.ndarray, grid: np.ndarray, start: tuple, end: tuple,
                  step_size: float = 1.0, max_iters: int = 10000,
                  stall_window: int = 200, interpolate: bool = True,
                  diag: dict = None):
    """Normalised steepest descent of ``field`` from ``start`` to ``end``.

    Mirrors the repo's potential-field planners: each step moves
    ``step_size`` cells along the negated central-difference gradient (±1
    cell) of the N-linearly interpolated field (nearest-cell differences when
    ``interpolate`` is False), and the goal is reached once every coordinate
    is within one cell. The path fails (returns ``None``) when it leaves the
    grid, enters an obstacle cell, hits a zero gradient, or the field value
    stops decreasing for ``stall_window`` steps (a local minimum). Progress is
    judged on the field value rather than distance to the goal, so long
    detours around obstacles are not mistaken for stalls.

    Returns raw continuous positions in user (x, y, ...) order. On failure
    the reason is stored in ``diag['reason']`` when a dict is passed.
    """
    def fail(reason):
        if diag is not None:
            diag['reason'] = reason
        return None

    def at(p):
        return '(' + ', '.join(f'{c:.1f}' for c in p[::-1]) + ')'

    ndim = field.ndim
    end_g = np.array([float(c) for c in reversed(end)])
    pos = np.array([float(c) for c in reversed(start)])
    path = [tuple(float(c) for c in start)]

    # Sample pos, then pos ± e_d for every axis, in one interpolation call.
    offsets = np.zeros((2 * ndim + 1, ndim))
    for d in range(ndim):
        offsets[1 + 2 * d, d] = 1.0
        offsets[2 + 2 * d, d] = -1.0
    upper = np.array(field.shape) - 1

    best = math.inf
    since_best = 0
    for _ in range(max_iters):
        pts = pos + offsets
        if interpolate:
            vals = ndimage.map_coordinates(field, pts.T, order=1, mode='nearest')
        else:
            idx = np.clip(np.floor(pts + 0.5).astype(int), 0, upper)
            vals = field[tuple(idx.T)]
        grad = vals[1::2] - vals[2::2]
        mag = float(np.sqrt(grad @ grad))
        if not math.isfinite(mag) or mag == 0.0:
            return fail(f'zero gradient at {at(pos)}')

        value = float(vals[0])
        if not math.isfinite(best) or value < best - 1e-12 * (1.0 + abs(best)):
            best = value
            since_best = 0
        else:
            since_best += 1
            if since_best >= stall_window:
                return fail(f'local minimum near {at(pos)} (field stopped decreasing '
                            f'for {stall_window} steps)')

        pos = pos - (step_size / mag) * grad
        if not in_free_cell(grid, pos):
            return fail(f'descent left free space at {at(pos)}')
        path.append(tuple(float(c) for c in pos[::-1]))

        if np.all(np.abs(end_g - pos) <= 1.0):
            path.append(tuple(float(c) for c in end))
            return path
    return fail(f'goal not reached within {max_iters} steps')


def path_is_free(grid: np.ndarray, path) -> bool:
    """Every vertex of a user-order path lies in a free cell."""
    return all(in_free_cell(grid, tuple(reversed(p))) for p in path)


# ---------------------------------------------------------------------------
# Planar geometry from occupancy grids
# ---------------------------------------------------------------------------

def free_component(grid: np.ndarray, start: tuple) -> np.ndarray:
    """Face-connected free region containing ``start`` (boolean mask).

    Continuous-geometry planners model cells as solid squares, so two obstacle
    cells touching at a corner form a single obstacle and the zero-width gap
    between them is not traversable.
    """
    labels, _ = ndimage.label(grid == 0)
    lab = labels[tuple(reversed(start))]
    if lab == 0:
        raise UnsupportedMapError('start lies in an obstacle cell')
    return labels == lab


_LEFT = {(1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0)}


def domain_boundaries(mask: np.ndarray, pinch_offset: float = 0.05) -> list:
    """Boundary loops of a face-connected region of cells.

    Returns closed polygons (K × 2 arrays in (x, y) coordinates, first vertex
    repeated last) oriented with the region on the left: the outer boundary is
    counter-clockwise and every hole clockwise. Vertices sit only at corners.

    Where the region touches itself diagonally (a "pinch" vertex shared by
    two passes of the boundary) each pass is nudged ``pinch_offset`` towards
    the cell it wraps, so every loop is a simple Jordan curve — the obstacle
    cells meeting at that corner become joined by a thin bridge.
    """
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    inner = pad[1:-1, 1:-1]
    # Corner (cx, cy) is the point (cx − ½, cy − ½). Each blocked side of a
    # region cell contributes one directed unit edge with the region on its left.
    sides = (
        (inner & ~pad[:-2, 1:-1], (0, 0), (1, 0)),    # below blocked: +x along y − ½
        (inner & ~pad[1:-1, 2:], (1, 0), (0, 1)),     # right blocked: +y along x + ½
        (inner & ~pad[2:, 1:-1], (1, 1), (-1, 0)),    # above blocked: −x along y + ½
        (inner & ~pad[1:-1, :-2], (0, 1), (0, -1)),   # left blocked: −y along x − ½
    )
    outgoing: dict = {}
    for side_mask, (ox, oy), direction in sides:
        ys, xs = np.nonzero(side_mask)
        for x, y in zip(xs.tolist(), ys.tolist()):
            outgoing.setdefault((x + ox, y + oy), []).append(direction)
    # Pinch corners are the only ones with two outgoing edges.
    pinch = {v for v, dirs in outgoing.items() if len(dirs) == 2}

    loops = []
    while outgoing:
        # Start every loop at an ordinary corner so it is visited exactly once.
        v0 = next(v for v in outgoing if v not in pinch)
        walk = [(v0, outgoing.pop(v0)[0])]
        v = (v0[0] + walk[0][1][0], v0[1] + walk[0][1][1])
        while v != v0:
            dirs = outgoing[v]
            # At a pinch, turn left so the loop hugs the cell it wraps.
            d = dirs[0] if len(dirs) == 1 else _LEFT[walk[-1][1]]
            dirs.remove(d)
            if not dirs:
                del outgoing[v]
            walk.append((v, d))
            v = (v[0] + d[0], v[1] + d[1])

        verts = []
        for i, (corner, d_out) in enumerate(walk):
            d_in = walk[i - 1][1]
            if d_in == d_out:
                continue   # straight through: not a polygon vertex
            pt = [corner[0] - 0.5, corner[1] - 0.5]
            if corner in pinch:
                pt[0] += pinch_offset * (d_out[0] - d_in[0])
                pt[1] += pinch_offset * (d_out[1] - d_in[1])
            verts.append(pt)
        loops.append(np.array(verts + [verts[0]], dtype=float))
    return loops


def polygon_area(poly: np.ndarray) -> float:
    """Signed shoelace area (positive for counter-clockwise loops)."""
    x, y = poly[:-1, 0], poly[:-1, 1]
    x2, y2 = poly[1:, 0], poly[1:, 1]
    return 0.5 * float(np.sum(x * y2 - x2 * y))


def resample_polygon(poly: np.ndarray, max_len: float) -> np.ndarray:
    """Split every edge of a closed polygon into equal panels ≤ ``max_len``."""
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, math.ceil(float(np.linalg.norm(b - a)) / max_len - 1e-9))
        t = np.arange(n)[:, None] / n
        out.append(a + t * (b - a))
    out.append(poly[:1])
    return np.vstack(out)


def maximal_rectangles(comp: np.ndarray) -> list:
    """All maximal axis-aligned rectangles of cells inside ``comp``.

    Rectangles are (x0, y0, x1, y1) with inclusive cell bounds. Brute force
    over row spans — fine for the obstacle-sized components it is used on.
    """
    ys, xs = np.nonzero(comp)
    if len(ys) == 0:
        return []
    y_lo, y_hi, x_lo, x_hi = ys.min(), ys.max(), xs.min(), xs.max()
    sub = comp[y_lo:y_hi + 1, x_lo:x_hi + 1]
    rects = set()
    for r0 in range(sub.shape[0]):
        run = np.ones(sub.shape[1], dtype=bool)
        for r1 in range(r0, sub.shape[0]):
            run &= sub[r1]
            if not run.any():
                break
            # maximal column runs of the AND over rows r0..r1
            padded = np.concatenate([[False], run, [False]])
            starts = np.nonzero(padded[1:] & ~padded[:-1])[0]
            stops = np.nonzero(~padded[1:] & padded[:-1])[0] - 1
            for c0, c1 in zip(starts.tolist(), stops.tolist()):
                rects.add((c0 + x_lo, r0 + y_lo, c1 + x_lo, r1 + y_lo))
    # keep only rectangles not contained in another
    rects = sorted(rects, key=lambda r: -(r[2] - r[0] + 1) * (r[3] - r[1] + 1))
    maximal = []
    for r in rects:
        if not any(m[0] <= r[0] and m[1] <= r[1] and r[2] <= m[2] and r[3] <= m[3]
                   for m in maximal):
            maximal.append(r)
    return maximal


def rectangles_overlap(a: tuple, b: tuple) -> bool:
    """Inclusive-cell rectangles share at least one cell."""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def rectangle_chain(comp: np.ndarray):
    """Cover a component with overlapping maximal rectangles forming a chain.

    Greedy cover that, after the first pick, only accepts rectangles
    overlapping the ones already chosen (so the cover stays connected through
    shared cells). Returns the rectangles ordered along the chain, or ``None``
    when the overlap graph is not a simple path (branching or cyclic unions).
    """
    cands = maximal_rectangles(comp)
    uncovered = comp.copy()
    chosen = []
    while uncovered.any():
        best, best_gain = None, 0
        for r in cands:
            if chosen and not any(rectangles_overlap(r, c) for c in chosen):
                continue
            gain = int(uncovered[r[1]:r[3] + 1, r[0]:r[2] + 1].sum())
            if gain > best_gain:
                best, best_gain = r, gain
        if best is None:
            return None
        chosen.append(best)
        uncovered[best[1]:best[3] + 1, best[0]:best[2] + 1] = False

    def overlaps(rects):
        return {i: [j for j in range(len(rects))
                    if j != i and rectangles_overlap(rects[i], rects[j])]
                for i in range(len(rects))}

    def covers_connected(rects):
        cover = np.zeros_like(comp)
        for r in rects:
            cover[r[1]:r[3] + 1, r[0]:r[2] + 1] = True
        if not np.array_equal(cover, comp):
            return False
        adj_r, seen, stack = overlaps(rects), {0}, [0]
        while stack:
            for j in adj_r[stack.pop()]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return len(seen) == len(rects)

    # Drop rectangles made redundant by later picks (hubs first), e.g. the
    # bridging strip between two overlapping squares.
    pruned = True
    while pruned and len(chosen) > 1:
        pruned = False
        adj = overlaps(chosen)
        for i in sorted(range(len(chosen)), key=lambda i: -len(adj[i])):
            rest = chosen[:i] + chosen[i + 1:]
            if covers_connected(rest):
                chosen, pruned = rest, True
                break

    n = len(chosen)
    adj = overlaps(chosen)
    if n == 1:
        return chosen
    if any(len(v) > 2 for v in adj.values()) or sum(len(v) for v in adj.values()) != 2 * (n - 1):
        return None
    ends = [i for i in range(n) if len(adj[i]) == 1]
    if len(ends) != 2:
        return None
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    order = [max(ends, key=lambda i: area(chosen[i]))]   # larger end is the root
    while len(order) < n:
        nxt = [j for j in adj[order[-1]] if j not in order]
        order.append(nxt[0])
    return [chosen[i] for i in order]
