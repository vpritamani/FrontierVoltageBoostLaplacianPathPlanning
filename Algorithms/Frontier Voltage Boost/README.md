# Frontier Voltage Boost Laplace

Harmonic-potential path planning. A scalar field `phi` is built with the goal
pinned to `0` and obstacles/boundaries held high; the path is steepest descent
of `phi` from start to goal. Because a converged harmonic field obeys the
maximum principle (no interior minima), descent is guaranteed to reach the goal.

Variants: `frontiervoltageboostlaplace.py` (2D), `3d…` (3D), `nd…`
(dimension-agnostic). All three share the same algorithm.

## The frontier voltage boost

Solving Laplace's equation to convergence is slow. Instead, `solve()` grows a
wavefront out from the goal: it relaxes only the current frontier cells for
`n_l` iterations, freezes each cell the moment its voltage crosses
`v_max - epsilon`, raises the ceiling, and repeats until the start is reached.
This is fast, but the resulting field is a **coarse, under-relaxed banding**,
not a true harmonic function — so it can contain flat plateaus that a converged
field would not.

## Interleaved relaxation during descent

A flat plateau gives a zero gradient, which would otherwise dead-end the
descent (it looks like a local minimum but is only an under-relaxation
artifact). `_gradient_descent` handles this the way the method is meant to work:
on a stall it **keeps relaxing** — running more Laplace averaging sweeps over
the whole interior (unfreezing the solved cells so the field converges toward
harmonic) and re-tracing from the start on the refined field. It gives up only
if the relaxation budget is exhausted.

Easy maps never stall, so they never enter this path and pay nothing. Only maps
that would previously fail take the extra relaxation cost.

### Tunable budget (class constants)

| Constant        | Default | Meaning                                            |
|-----------------|---------|----------------------------------------------------|
| `_RELAX_BATCH`  | 500     | Laplace sweeps run per stall                        |
| `_MAX_RELAX`    | 60000   | Total relaxation sweeps before giving up (`None`)   |
| `_MAX_STEPS`    | 50000   | Descent moves per trace attempt                     |
| `_STALL_WINDOW` | 2000    | Descent moves without progress toward goal → relax  |

Lower `_MAX_RELAX` to cap worst-case latency on hard maps; raise it to solve
more pathological ones.

## Note on the voltage cap (`_fv`)

`_fv(nv, pv) = min(nv, ceil(pv))` caps each neighbour's contribution so descent
follows the wavefront bands instead of leaping across them (e.g. being yanked by
an obstacle cell held at `v_max`). It is **not** a local-minimum guard —
`round`/`floor` caps stall *sooner*, and removing the cap makes descent wander.
The cure for stalls is relaxation, above.
