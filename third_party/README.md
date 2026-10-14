# third_party — external comparison planners

Upstream implementations of the planners we compare Frontier Voltage Boost
against, wrapped as normal algorithm classes in
[`Algorithms/External Algorithms/`](../Algorithms/External%20Algorithms) and
registered in the benchmark app.

```bash
pip install pybind11 pyyaml                     # build/runtime deps (once)
python third_party/setup_external.py            # fetch pinned sources + build (~30 s)
python third_party/setup_external.py --selftest # solve a small map with every planner
```

The benchmark app runs the setup automatically before the first run that
uses an external planner (and again if the app's Python version changes,
since compiled modules are per-interpreter). If it cannot build — no git, no
C++ compiler, or pybind11/pyyaml missing — the run stops with the reason.
Unsolved maps report why (e.g. `no path: local minimum near (41.0, 17.3)` or
`unsupported map: 2 obstacle(s) touch the map border …`).

`setup_external.py` checks each upstream repo out at a **pinned commit** into
`third_party/src/` and compiles the native parts into `third_party/build/`.
Both directories are git-ignored: compiled modules are platform-specific (a
macOS arm64 build will not load on a Linux VM), so each machine builds its
own from the same pinned sources. Upstream code is used as checked out —
only the small pybind11 bindings in `bindings/` are ours — except for the
cust_fields bug fixes in `patches/` (see the end of this file). Needs git and a
C++17 compiler; no system libraries (Eigen and tinyxml2 are fetched too).

| Component | Upstream (pinned) | License | How it runs |
|---|---|---|---|
| `fmm` | [thinks/fast-marching-method](https://github.com/thinks/fast-marching-method) @ `b20f0b2` | MIT | C++ eikonal solver via `bindings/fmm_module.cpp` (2–6D) |
| `hntf2d` | [maxchaos/hntf2d](https://github.com/maxchaos/hntf2d) @ `583527e` (+ Eigen 3.4.0, tinyxml2 10.0.0) | GPL-3.0 | C++ harmonic map via `bindings/hntf2d_module.cpp`; upstream `poly2circle` used as-is |
| `cust_fields` | [Shuaikang-Wang/cust_fields](https://github.com/Shuaikang-Wang/cust_fields) @ `5ca178e` | none stated | Pure Python, imported directly (`NF/`, `TOPO/`) |
| — | [johnyf/nfsim](https://github.com/johnyf/nfsim) @ `21bf45d` | BSD-2 | MATLAB — not fetched; the KR navigation function is ported (see below) |

**Licensing note.** hntf2d (GPL-3.0) and cust_fields (no license file, so
all rights reserved by default) are fetched on the user's machine at build
time, not committed — except that `patches/cust_fields.patch` necessarily
quotes a few lines of cust_fields as diff context. Keep it that way unless
the repo's own license is chosen with that in mind.

## The four wrappers

Principle: **run each algorithm as published.** The wrappers call the upstream
code directly wherever it exists and add only what is unavoidable to run it
on our occupancy-grid maps (listed below). Nothing is tuned to make a
baseline succeed more often. Clear upstream bugs are fixed by a small patch
applied at checkout (`patches/`, listed at the end); everything else found
upstream is only reported.

All return continuous `(x, y, ...)` paths like the other planners, run in the
app's watchdog subprocess, and report why a map was not solved: either
`unsupported map: …` (outside the method's world class) or `no path: …`.

| Registry id | Upstream code called | Written here (and why) |
|---|---|---|
| `fast_marching` | the C++ eikonal solver (`HighAccuracyVaryingSpeedEikonalSolver` + `SignedArrivalTime`) | path extraction — the library computes the arrival-time field only. Uses the repo's standard descent (same as the potential-field baselines). |
| `navigation_function` | — (nfsim is MATLAB source; no runtime here) | port of `krf.m` / `bi2b.m`, evaluated on the grid, then the repo's standard descent. |
| `harmonic_map` | the C++ `HarmonicMap2D` (panel-method map, `map`, `jacob`, `puncts`) and upstream's `poly2circle` | the controller (paper eqs. 6–16) — it is not in the upstream repo. |
| `custom_fields` | `NF.geometry.World`, `ForestToStar`, `StarToSphere`, `SphereToPoint`, `TOPO.integrate_forest` (upstream defaults) | grid → forest-world conversion, and the 3-line map composition from upstream's `test_cust_nf.PointMap` (that script imports matplotlib at module level). |

### MATLAB → Python conversion (nfsim)

nfsim is MATLAB-only, so the Koditschek–Rimon navigation function is a
Python translation, not a call into nfsim. What was translated
(nfsim @ `21bf45d`):

| nfsim (MATLAB) | Python (`navigationfunction.py`, `_krf`) |
|---|---|
| [`krf.m` L44](https://github.com/johnyf/nfsim/blob/21bf45daf78823bdf56dcf03ac91f65647585181/navigation_function/krnf/field/krf.m#L44): `b(b < 0) = 0` | `phi[~free] = 1.0` (obstacle cells, where b = 0) |
| [`krf.m` L48–49](https://github.com/johnyf/nfsim/blob/21bf45daf78823bdf56dcf03ac91f65647585181/navigation_function/krnf/field/krf.m#L48-L49): `nf = gd ./ (gd.^k + b).^(1/k)` | same formula, evaluated as `exp(log gd − logaddexp(k·log gd, log b)/k)` so `gd^k` cannot overflow |
| [`bi2b.m` L24](https://github.com/johnyf/nfsim/blob/21bf45daf78823bdf56dcf03ac91f65647585181/navigation_function/krnf/obstacles/bi2b.m#L24): `b = prod(bi, 1)` | `log b = Σ log beta_i` |
| `beta_*.m` implicit obstacle functions (spheres, ellipsoids, …) | distance to each obstacle component of the grid (a grid has no analytic shapes) |
| [`init_sphere_world.m` L55](https://github.com/johnyf/nfsim/blob/21bf45daf78823bdf56dcf03ac91f65647585181/core_engine/init/init_sphere_world.m#L55): demo world radius `r0 = 5` | map scaled so its longer side is 10 units |

Not translated: `krf.m`'s `k ≥ 100` branch (returns ≈1 to dodge overflow;
the log-space evaluation does not need it), `grad_krf.m` (the path uses the
repo's standard finite-difference descent on φ, like the other field
planners), and nfsim's k-tuning/integration GUI code.

### Choices needed to run on grid maps

- **Geometry.** Cell (x, y) is the unit square around (x, y). The continuous
  methods use the start's face-connected free region; obstacle cells touching
  at a corner count as one obstacle (a zero-width gap is not traversable).
  For hntf2d's Jordan-curve boundaries, a boundary passing twice through such
  a corner is separated by 0.05 cells.
- **Units.** Maps are scaled into each method's native world size where its
  behaviour depends on units: nfsim demo worlds (radius 5, so the map's
  longer side = 10 units) for the NF, `world_size` (default 5, upstream's
  example world) for cust_fields. hntf2d's paths do not depend on scale.
- **Obstacles for the NF.** nfsim's implicit obstacle functions β_i become
  the distance to each obstacle component; the grid border is obstacle 0.
- **FMM speed.** The solver requires speed > 0, so obstacle cells get
  `obstacle_speed` (1e-6). The goal's free neighbours are seeded with exact
  distances, as the upstream README recommends for the high-accuracy solver.
  `clearance_saturation > 0` switches to the Fast Marching Square speed
  field — a separate, published variant; off by default.
- **hntf2d integration.** The paper simulates with explicit Euler at 10 ms
  (`dt`) and §6.1 gains. At grid scale a 10 ms step can travel several
  cells, so such a step is split into shorter Euler steps of at most
  `step_size` cells — same scheme, finer step, no other change.
- **Termination.** Goal reached within 1 cell; failure on leaving free
  space, a stalled field (no decrease for `stall_window` steps) or the
  iteration limit — the same rules for every wrapper.
- **cust_fields workspace.** Upstream's `Workspace` adds +0.1 to its size,
  which would let paths leave the map; that padding is cancelled so the
  workspace equals the map. Obstacle padding is left as upstream has it.
- **Not run:** cust_fields' homotopy *customization* — it needs a target
  homotopy class, which a start→goal benchmark does not define.

### Reading the results

- **Timing is not like-for-like across languages:** the FMM solve and the
  hntf2d map are compiled C++; FVB, the NF port and the hntf2d controller
  are NumPy/Python; cust_fields is pure Python.
- **With the repo's standard descent, plain FMM and the NF zig-zag along
  walls** (the ±1-cell stencil samples obstacle values) and plain FMM can
  bounce at the entrance of 1-cell corridors, which shortest paths use.
- **KR navigation functions** only guarantee a unique minimum on sphere-like
  worlds with large enough k; on cluttered maps expect local minima and flat
  regions (φ ≈ 1), reported as unsolved. `k = 0` uses #obstacles + 1.
- **cust_fields** handles only forest worlds: free-standing obstacles that are
  chains of overlapping rectangles, off the border. Its star→sphere λ is
  map-dependent (upstream tunes it per example); most default-generator maps
  are rejected.

### Upstream bugs fixed by `patches/cust_fields.patch`

`setup_external.py` applies the patch right after checking out cust_fields
@ `5ca178e` (and re-fetches older checkouts that lack it). The patch changes
10 lines; none of the fixes changed success rates on our map sets — the
crash path is not reached (the wrapper rejects border obstacles) and the
other two only act within 0.01 of an obstacle.

1. **`compute_virtual_ws` crashes** —
   [`NF/transformation.py` L106](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/transformation.py#L106) called
   `Rectangular(_type, center, width - 0.16, height + 0.16)`, but
   [`Rectangular.__init__`](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/geometry.py#L147) requires `theta` and
   `s`. Reached from [L295](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/transformation.py#L295) /
   [L336](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/transformation.py#L336) for any obstacle tree attached to
   the workspace boundary, it always raised `TypeError`. **Fix:** pass
   `theta = 0.0, s = 0.99`, matching upstream's other virtual shapes
   (`compute_virtual_length_ray` hard-codes the same values).
2. **`SphereToPoint.compute_T_q` mutates its input** —
   [L461–463](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/transformation.py#L461-L463): `T = q` then `T += …`
   changed `q` in place, so each later obstacle in the loop was evaluated
   at the already-shifted point (reproduced: `[3.054, 3.0]` became
   `[3.022, 3.0]` next to an obstacle). **Fix:** `T = np.array(q, dtype=float)`.
3. **0/0 at a star's centre** —
   [`NF/utils.py` L29](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/utils.py#L29) and
   [L40](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/utils.py#L40) divided by |q − centre|, giving NaN (and
   RuntimeWarnings) at the centre. **Fix:** return 0 there; every caller
   multiplies the ray length by q − centre = 0, so the result is the centre
   either way. Verified: no floating-point errors with `np.seterr(all='raise')`.

### Other problems found upstream (reported, not changed)

- **hntf2d README example** builds its "inner box" from `linspace(-1, 1)`
  instead of `linspace(-0.1, 0.1)`, so the example boundary self-intersects
  and its printed output is meaningless. `HarmonicMap2D` does not validate
  input: a repeated vertex (zero-length segment) silently yields NaNs.
- **hntf2d computes in single precision** (`Eigen::MatrixXf`), so the map and
  Jacobian lose accuracy within about a panel of the boundary.
- **cust_fields purges each obstacle tree as a parent→child chain** —
  [L116](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/transformation.py#L116) takes the leaf as `obs[depth]` and
  [L236](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/transformation.py#L236) its parent as `obs[depth - 1]` — but
  its own [`CONFIG/example_world.yaml` L2–26](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/CONFIG/example_world.yaml#L2-L26)
  has a three-star tree whose 2nd star (x 2.45–2.75) and 3rd star
  (x 0.85–1.15) do not overlap. (Example data; not used here.)
- **cust_fields uses two different squircle formulas** for one obstacle:
  [`Rectangular.potential` L182–191](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/geometry.py#L182-L191) (used by
  the transformation; mixes length² and length⁴ terms, so its shape depends
  on the world's scale) and
  [`check_point_inside` L232–245](https://github.com/Shuaikang-Wang/cust_fields/blob/5ca178e588b4b766cbc537af4a2c4e8ebb53895a/NF/geometry.py#L232-L245) (the
  integrator's collision check). A design choice rather than a clear bug —
  changing it would mean guessing which shape was intended.
- **cust_fields supports one wall-attached obstacle chain per world**:
  `StarToSphere.compute_all_root_obs` treats the first entry of every further
  workspace tree as an ordinary obstacle root.
