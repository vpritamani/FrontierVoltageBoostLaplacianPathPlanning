import importlib.util
import math
import os
import sys

import numpy as np

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


def _sigma_p(x):
    """Eq. (7): x²(3 − 2x) for x ≤ 1, else 1."""
    return np.where(x <= 1.0, x * x * (3.0 - 2.0 * x), 1.0)


def _sigma_v(x):
    """Eq. (8): x² for x ≥ 0, else 0."""
    return np.where(x >= 0.0, x * x, 0.0)


def _xi_1(x, eps):
    """ξ1(x; ε) = 1 − σp(x / ε)."""
    return 1.0 - _sigma_p(x / eps)


def _xi_2(x, eps):
    """Smooth step from 0 (x < ε) to 1 (x > 1)."""
    t = np.clip((x - eps) / (1.0 - eps), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class HarmonicMapNavigation2D(Algorithm2D):
    """Harmonic-map navigation (Vlantis, Bechlioulis & Kyriakopoulos), 2D.

    Two stages, as in the paper (Sensors 2023, 23(9):4464; ICRA 2018):

    1. **Harmonic map** — maxchaos/hntf2d (C++, pinned in ``third_party/``,
       built with ``python third_party/setup_external.py hntf2d``) maps the
       free workspace onto the unit disk: the outer boundary goes to the unit
       circle and every obstacle collapses onto a distinct interior point
       ("puncture"). The domain is the face-connected free region of the
       start; its boundary is traced from the cell squares and split into
       panels of at most ``panel_length`` cells for the panel method.
    2. **Control law** — the paper's adaptive harmonic potential on the disk,
       pulled back through the map's Jacobian (eqs. 6–16; the controller is
       not part of the upstream repository, so it is implemented here):

           phi = k_d ln(|q − q_d|/2) − Σ_i k_i ln(|q − q_i|/2)
           psi = (1 + tanh(phi / w_phi)) / 2
           u   = −K_u s(q, k) J⁻¹(p) ∇psi

       with the adaptive laws for k_d and k_i. Defaults are the paper's
       Section 6.1 values.

    Integration follows the paper: explicit Euler on (p, k) with a 10 ms
    step (``dt``). The only change is that a step which would travel more
    than ``step_size`` cells is taken as a shorter Euler step instead (same
    scheme, finer time step), since at grid scale a 10 ms step can jump
    across several cells. Nothing else alters the trajectory: there is no
    wall handling beyond the controller itself, and leaving the free cells
    counts as a failure.

    Maps where start and goal are not face-connected (only a diagonal pinch
    joins them) are outside the method's continuous-domain model and are
    rejected with ``UnsupportedMapError``.
    """

    def __init__(self, map, panel_length: float = 1.0, step_size: float = 1.0,
                 k_d: float = 20.0, k_i: float = 1.0, k_bar: float = 20.0,
                 w_phi: float = 20.0, K_u: float = 100.0, K_k: float = 100.0,
                 gamma: float = 0.7, alpha: float = 1.0, eps_p: float = 0.025,
                 eps_v: float = 0.1, eps_1: float = 0.01, eps_2: float = 0.1,
                 eps_3: float = 0.1, m: int = -2, dt: float = 0.01,
                 max_iters: int = 20000, stall_window: int = 2000):
        super().__init__(map)
        self.panel_length = panel_length
        self.step_size = step_size
        self.k_d0, self.k_i0, self.k_bar = k_d, k_i, k_bar
        self.w_phi, self.K_u, self.K_k = w_phi, K_u, K_k
        self.gamma, self.alpha = gamma, alpha
        self.eps_p, self.eps_v = eps_p, eps_v
        self.eps_1, self.eps_2, self.eps_3 = eps_1, eps_2, eps_3
        self.m = m
        self.dt = dt
        self.max_iters = max_iters
        self.stall_window = stall_window
        self._ext = _common.import_built('external_hntf2d', 'hntf2d')
        self._poly2circle = self._load_poly2circle()
        self._path = None
        self.hmap = None
        self.failure_reason = None   # why the last solve found no path

    @staticmethod
    def _load_poly2circle():
        path = os.path.join(_common.source_dir('hntf2d', 'hntf2d'), 'cy', 'hntf2d', 'utils.py')
        spec = importlib.util.spec_from_file_location('hntf2d_upstream_utils', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.poly2circle

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid
        start, end = self.map.start, self.map.end
        domain = _common.free_component(grid, start)
        if not domain[end[1], end[0]]:
            raise _common.UnsupportedMapError(
                'start and goal are not face-connected — only a diagonal gap joins '
                'them, which has zero width in the harmonic map\'s continuous domain')

        self.hmap = self._build_map(domain)
        self._path = self._navigate(domain, start, end)
        if visualize:
            self._visualize()
        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    # ------------------------------------------------------------------
    # Stage 1: harmonic map (hntf2d)
    # ------------------------------------------------------------------

    def _build_map(self, domain):
        loops = _common.domain_boundaries(domain)
        outer = [l for l in loops if _common.polygon_area(l) > 0]
        holes = [l for l in loops if _common.polygon_area(l) < 0]
        assert len(outer) == 1, 'a face-connected region has exactly one outer boundary'

        hm = self._ext.HarmonicMap2D()
        outer_panels = _common.resample_polygon(outer[0], self.panel_length)
        hm.boundary_append(outer_panels)
        for hole in holes:
            hm.boundary_append(_common.resample_polygon(hole, self.panel_length))
        uo, vo = self._poly2circle(outer_panels).T
        hm.solve(uo, vo)
        return hm

    # ------------------------------------------------------------------
    # Stage 2: adaptive harmonic potential + Jacobian pull-back
    # ------------------------------------------------------------------

    def _navigate(self, domain, start, end):
        def at(pt):
            return f'({pt[0]:.1f}, {pt[1]:.1f})'

        hm = self.hmap
        punct = hm.puncts()
        P = np.zeros((0, 2)) if punct is None else np.asarray(punct, dtype=float).reshape(-1, 2)
        q_d = np.asarray(hm.map(*end), dtype=float)

        k_d = float(self.k_d0)
        k = np.full(len(P), float(self.k_i0))
        k_bar = np.full(len(P), float(self.k_bar))

        grid = domain.astype(np.uint8) ^ 1   # cells outside the domain are blocked
        p = np.array(start, dtype=float)
        path = [tuple(p)]
        anchor, since_anchor = p.copy(), 0

        for _ in range(self.max_iters):
            q = np.asarray(hm.map(*p), dtype=float)
            J = np.asarray(hm.jacob(*p), dtype=float)
            terms = self._terms(q, q_d, P, k_d, k, k_bar)
            if terms is None:
                return self._fail(f'hit a puncture/goal singularity of the disk at {at(p)}')
            try:
                u = -self.K_u * terms['s'] * np.linalg.solve(J, terms['grad_psi'])
            except np.linalg.LinAlgError:
                return self._fail(f'singular harmonic-map Jacobian at {at(p)}')
            speed = float(np.hypot(*u))
            if not math.isfinite(speed):
                return self._fail(f'non-finite control at {at(p)}')
            kd_rate, A, B = self._gain_rates(q, q_d, P, k_d, k, k_bar, terms)

            # Explicit Euler (paper: 10 ms), with a shorter step only when the
            # 10 ms step would travel more than step_size cells.
            h = min(self.dt, self.step_size / speed) if speed > 0 else self.dt
            p = p + h * u
            k_d += h * kd_rate
            k = k + h * (A - B * k)

            if not _common.in_free_cell(grid, (p[1], p[0])):
                return self._fail(f'left free space at {at(p)}')
            path.append((float(p[0]), float(p[1])))

            if abs(end[0] - p[0]) <= 1 and abs(end[1] - p[1]) <= 1:
                path.append((float(end[0]), float(end[1])))
                return path

            # Stuck at a saddle (or crawling against the outer boundary).
            if np.hypot(*(p - anchor)) >= 1.0:
                anchor, since_anchor = p.copy(), 0
            else:
                since_anchor += 1
                if since_anchor >= self.stall_window:
                    return self._fail(f'stuck near {at(p)} for {self.stall_window} steps')
        return self._fail(f'goal not reached within {self.max_iters} steps')

    def _fail(self, reason):
        self.failure_reason = reason
        return None

    def _terms(self, q, q_d, P, k_d, k, k_bar):
        """Potential, gain function s and adaptive-law ingredients at q."""
        dq_d = q - q_d
        r_d2 = float(dq_d @ dq_d)
        dQ = q - P
        r_i2 = np.einsum('ij,ij->i', dQ, dQ)
        if r_d2 == 0.0 or np.any(r_i2 == 0.0):
            return None

        # (9)-(11): ln(|x|/2) = ½ ln(|x|² / 4);  ∇ ln|x| = x / |x|²
        phi = 0.5 * k_d * math.log(r_d2 / 4.0) - 0.5 * float(k @ np.log(r_i2 / 4.0))
        grad_phi = k_d * dq_d / r_d2 - (k / r_i2) @ dQ
        th = math.tanh(phi / self.w_phi)
        grad_psi = (1.0 - th * th) / (2.0 * self.w_phi) * grad_phi

        # (7): s vanishes at the outer boundary / when moving away from the centre
        nq = float(np.hypot(*q))
        ng = float(np.hypot(*grad_psi))
        s = (self.gamma * float(_sigma_p((1.0 - nq) / self.eps_p))
             + (1.0 - self.gamma) * float(_sigma_v((grad_psi @ q) / (self.eps_v + ng * nq))))

        return {'phi': phi, 'grad_phi': grad_phi, 'grad_psi': grad_psi, 'th': th,
                's': s, 'nq': nq, 'ng': ng, 'dq_d': dq_d, 'r_d2': r_d2, 'dQ': dQ, 'r_i2': r_i2}

    def _gain_rates(self, q, q_d, P, k_d, k, k_bar, t):
        """Eq. (12): returns (dk_d/dt, A, B) with dk_i/dt = A_i − B_i k_i."""
        dq_d, r_d2, dQ, r_i2 = t['dq_d'], t['r_d2'], t['dQ'], t['r_i2']
        grad_phi, grad_psi, s, nq, ng = t['grad_phi'], t['grad_psi'], t['s'], t['nq'], t['ng']

        # λ: non-negative eigenvalue of ∇²φ (traceless: eigenvalues ±λ).
        # ∇² ln|x| = (I |x|² − 2 x xᵀ) / |x|⁴
        H = k_d * (np.eye(2) * r_d2 - 2.0 * np.outer(dq_d, dq_d)) / (r_d2 * r_d2)
        H -= np.eye(2) * float(np.sum(k / r_i2))
        H += 2.0 * np.einsum('i,ij,ik->jk', k / (r_i2 * r_i2), dQ, dQ)
        lam = math.hypot(H[0, 0], H[0, 1])
        kd_rate = float(_xi_1(lam + math.hypot(*grad_phi), self.eps_1))

        if len(P) == 0:
            return kd_rate, np.zeros(0), np.zeros(0)

        # (13)-(14)
        ell = -self.K_u * s / np.log(np.sqrt(r_i2) / 2.0)
        g0 = float(_sigma_v(0.25 * (self.alpha * ng * math.sqrt(r_d2) - grad_psi @ dq_d)))
        gi = _sigma_v(0.5 * (dQ @ grad_psi))

        # (15)-(16): weights from r_0 = (1 − |q|)², r_i = |q − q_i|²
        # r̄_i = (Σ_{j≠i} r_j^m)^(1/m), a smooth under-approximation of min_{j≠i} r_j
        r = np.concatenate([[(1.0 - nq) ** 2], r_i2])
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            rm = r ** self.m
            others = np.where(np.eye(len(r), dtype=bool), 0.0, rm[None, :]).sum(axis=1)
            r_bar = others ** (1.0 / self.m)
            w_bar = r_bar / (r + r_bar)
        w_bar = np.nan_to_num(w_bar, nan=1.0)
        denom = w_bar[0] + float(k_bar @ w_bar[1:])
        w0 = float(_xi_2(w_bar[0], self.eps_3)) / denom
        wi = w_bar[1:] / denom

        # h_i: speed up the decay of the k_i pushing towards the outer boundary
        e_d = -dq_d / r_d2
        e_i = -dQ / r_i2[:, None]
        h_bar = k * (1.0 - t['th'] ** 2) ** 2 * (e_i @ e_d)
        sv = _sigma_v(h_bar)
        h = 1.0 + sv / (1.0 + sv.sum())

        a = wi * ell * gi
        A = k_bar * a
        B = a + self.K_k * h * w0 * (g0 + float(_xi_1(s, self.eps_2)))
        return kd_rate, A, B

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
        Image.fromarray(rgb, mode='RGB').save('harmonic_map_result.png')
