import importlib.util
import os
import sys

import numpy as np
from scipy import ndimage

from Algorithms import Algorithm2D


def _load_common():
    name = 'external_algorithms_common'
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'externalcommon.py')
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_common = _load_common()

# Rectangular/Workspace add this much to every width and height internally.
_CF_PADDING = 0.1
_MAX_TREE_DEPTH = 5


class CustomHarmonicFields2D(Algorithm2D):
    """Harmonic potential fields on forest worlds (Wang et al., "Customize
    Harmonic Potential Fields via Hybrid Optimization over Homotopic Paths").

    Calls Shuaikang-Wang/cust_fields (pinned in ``third_party/``; fetched with
    ``python third_party/setup_external.py cust_fields``) directly:

    * ``NF.transformation`` — forest → star → sphere → unbounded point-world
      diffeomorphism (each tree of overlapping star obstacles collapses onto a
      single point P_i; the outer boundary goes to infinity),
    * ``TOPO.integrate_forest`` — the negative-gradient flow of the harmonic
      potential  phi_P(q) = w_g ln|q − q_g|² − Σ_i w_i ln|q − P_i|²  pulled
      back through the map's Jacobian. Weights satisfying Lemma 5
      (w_g ≥ Σ w_i + 1) give collision-free convergence; the defaults are the
      upstream demo's projected initial weights.

    The homotopy *customization* (D-signature target + weight optimization) is
    the paper's addition on top of this field; it needs a target homotopy
    class, which a plain start→goal benchmark does not define, so it is not
    run here.

    Grid → forest world: the map is scaled so its longer side spans
    ``world_size`` units (upstream constants are tuned for ~5-unit worlds).
    Each obstacle (corner-connected component of blocked cells) is covered by
    overlapping maximal rectangles that must form a chain — upstream purges
    each tree as a parent→child chain — and becomes a StarTree of
    ``Rectangular`` squircles (``squircle_s``) with the method's own +0.1
    padding. The workspace squircle is sized to the map exactly (its padding
    is cancelled so paths cannot leave the grid). Maps outside this class —
    obstacles touching the border (upstream supports a single wall-attached
    chain per world — a second workspace tree's root would be read as an
    obstacle — so the wrapper does not build them), branching or cyclic
    obstacle unions, trees deeper than 5, padded obstacles overlapping each
    other, or start/goal inside the padding — raise ``UnsupportedMapError``.

    Upstream bug fixes applied at checkout: ``third_party/patches/cust_fields.patch``.
    """

    def __init__(self, map, world_size: float = 5.0, nf_lambda: float = 1e5,
                 squircle_s: float = 0.99, max_iters: int = 3000):
        super().__init__(map)
        self.world_size = world_size
        self.nf_lambda = nf_lambda
        self.squircle_s = squircle_s
        self.max_iters = max_iters
        self._cf = self._import_upstream()
        self._path = None
        self.world = None
        self.failure_reason = None   # why the last solve found no path

    @staticmethod
    def _import_upstream():
        root = _common.source_dir('cust_fields', 'cust_fields')
        if root not in sys.path:
            sys.path.append(root)
        import NF.geometry
        import NF.transformation
        import TOPO
        return {'geometry': NF.geometry, 'transformation': NF.transformation, 'TOPO': TOPO}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid
        start, end = self.map.start, self.map.end
        h, w = grid.shape
        scale = self.world_size / max(h, w)

        domain = _common.free_component(grid, start)
        if not domain[end[1], end[0]]:
            raise _common.UnsupportedMapError('start and goal are not face-connected')

        trees = self._obstacle_trees(~domain)
        self.world = self._build_world(trees, w, h, scale)

        to_world = lambda p: (np.asarray(p, dtype=float) + 0.5) * scale
        start_w, goal_w = to_world(start), to_world(end)
        for name, pt in (('start', start_w), ('goal', goal_w)):
            if not self.world.check_point_in_free_space(pt, threshold=0.0):
                raise _common.UnsupportedMapError(
                    f'{name} lies inside an obstacle squircle (padding) or outside the '
                    'squircle workspace')

        TOPO = self._cf['TOPO']
        pmap = _PointMap(self._cf['transformation'], self.world, goal_w, self.nf_lambda,
                         [10.0 ** (10 - 2 * d) for d in range(_MAX_TREE_DEPTH)])
        roots = [np.array(group[0].center, dtype=float) for group in self.world.obstacles]
        n_obs = len(roots)
        weights = np.array([max(6.0, n_obs + 1.1)] + [1.0] * n_obs)   # upstream projected init
        # upstream integrator defaults (dt 0.03, goal_tol 0.05, safe 0.05 world units)
        cfg = TOPO.IntegrConfig(max_steps=self.max_iters)
        fpath, reached = TOPO.integrate_forest(self.world, pmap, roots, weights,
                                               start_w, goal_w, cfg)
        last = fpath[-1] / scale - 0.5
        if not reached:
            self._path = None
            if len(fpath) - 1 >= self.max_iters:
                self.failure_reason = f'upstream integrator hit its step limit ({self.max_iters})'
            else:
                self.failure_reason = (f'upstream integrator stalled at ({last[0]:.1f}, {last[1]:.1f}) — '
                                       'could not advance without entering an obstacle squircle')
        else:
            path = [tuple(float(c) for c in pt / scale - 0.5) for pt in fpath]
            path.append((float(end[0]), float(end[1])))
            self._path = path if _common.path_is_free(grid, path) else None
            if self._path is None:
                self.failure_reason = 'integrated path crosses an obstacle cell'

        if visualize:
            self._visualize()
        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    # ------------------------------------------------------------------
    # Grid -> forest world
    # ------------------------------------------------------------------

    @staticmethod
    def _obstacle_trees(blocked):
        labels, n = ndimage.label(blocked, structure=np.ones((3, 3), dtype=bool))
        border = set(np.unique(np.concatenate(
            [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
        if border:
            raise _common.UnsupportedMapError(
                f'{len(border)} obstacle(s) touch the map border; cust_fields only '
                'supports free-standing obstacle trees')
        trees = []
        for lab in range(1, n + 1):
            chain = _common.rectangle_chain(labels == lab)
            if chain is None:
                raise _common.UnsupportedMapError(
                    'an obstacle is not a chain of overlapping rectangles (branching '
                    'or cyclic union) — not a tree of stars as upstream purges them')
            if len(chain) - 1 > _MAX_TREE_DEPTH:
                raise _common.UnsupportedMapError(
                    f'an obstacle tree has depth {len(chain) - 1} > {_MAX_TREE_DEPTH}')
            trees.append(chain)
        return trees

    def _build_world(self, trees, w, h, scale):
        geometry = self._cf['geometry']

        def star(rect):
            x0, y0, x1, y1 = rect
            return {'type': 'Rectangular',
                    'center': [((x0 + x1) / 2 + 0.5) * scale, ((y0 + y1) / 2 + 0.5) * scale],
                    'width': (x1 - x0 + 1) * scale, 'height': (y1 - y0 + 1) * scale,
                    'theta': 0.0, 's': self.squircle_s}

        # Padded boxes of different trees must stay disjoint (forest world).
        pad = _CF_PADDING / 2
        boxes = []
        for t, chain in enumerate(trees):
            for x0, y0, x1, y1 in chain:
                boxes.append((t, (x0 - 0.5) * scale - pad, (y0 - 0.5) * scale - pad,
                              (x1 + 0.5) * scale + pad, (y1 + 0.5) * scale + pad))
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                if a[0] != b[0] and a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                    raise _common.UnsupportedMapError(
                        'two obstacles are closer than the method\'s squircle padding')

        config = {
            'obstacles': [{'shape': 'StarTree', 'stars': [star(r) for r in chain]}
                          for chain in trees],
            'workspace': [{'shape': 'StarTree', 'stars': [{
                'type': 'Workspace', 'center': [w * scale / 2, h * scale / 2],
                'width': w * scale - _CF_PADDING, 'height': h * scale - _CF_PADDING,
                'theta': 0.0, 's': 0.9999}]}],
        }

        class ForestWorld(geometry.World):
            # upstream World reads a YAML path; take the config dict directly
            def load_world_config(self, cfg):
                self._config = cfg

        return ForestWorld(config)

    def _visualize(self):
        from PIL import Image
        grey = self.map.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()
        for x, y in self._path or []:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < rgb.shape[0] and 0 <= xi < rgb.shape[1]:
                rgb[yi, xi] = [128, 128, 255]
        sx, sy = self.map.start
        ex, ey = self.map.end
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]
        Image.fromarray(rgb, mode='RGB').save('custom_fields_result.png')


class _PointMap:
    """Forest → point-world map and its finite-difference Jacobian — the
    composition upstream's ``test_cust_nf.PointMap`` builds (that script also
    imports matplotlib at module level, so it is mirrored here)."""

    def __init__(self, transformation, world, goal, nf_lambda, nf_mu):
        g = np.array(goal[:2], dtype=float)
        self.f2s = transformation.ForestToStar(world, g, nf_mu)
        self.s2s = transformation.StarToSphere(world, g, nf_lambda)
        self.s2p = transformation.SphereToPoint(world, g)

    def __call__(self, q):
        q = np.array(q, dtype=float)
        f = self.f2s.compute_f(q)
        h = self.s2s.compute_h_lambda(f)
        t = self.s2p.compute_T_q(h)
        return np.array(self.s2p.bounded_pw_to_unbounded(t), dtype=float)

    def jacobian(self, q, p0=None, h=1e-4):
        if p0 is None:
            p0 = self(q)
        px = (self(q + np.array([h, 0.0])) - p0) / h
        py = (self(q + np.array([0.0, h])) - p0) / h
        return np.column_stack([px, py])
