"""
Algorithm registry for the benchmark app.

Each entry declares everything the UI and runner need to know about an
algorithm: an id, a display label, which dimensionalities it supports, a
parameter schema (rendered as a form in the UI), and how to construct it.

To add a new algorithm, append an ``AlgorithmSpec`` to ``REGISTRY`` below.
The ``module_file`` path is relative to the project root, so files living in
directories with spaces (e.g. "Baseline Algorithms") load fine.
"""

import importlib.util
import os
import sys
from dataclasses import dataclass, field

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _gpu_param() -> "ParamSpec":
    """Shared 'Use GPU' toggle: PyTorch implementation, CUDA when available."""
    return ParamSpec('use_gpu', 'Use GPU (PyTorch)', 'bool', False,
                     help='Run the PyTorch implementation — on CUDA when a GPU '
                          'is available, otherwise on CPU via PyTorch. '
                          'Requires PyTorch. Note: pays off on large maps with '
                          'a real GPU; on small maps or without CUDA the NumPy '
                          'default is faster.')


def _load_class(module_file: str, class_name: str):
    """Import ``class_name`` from a file path relative to the project root."""
    path = os.path.join(_PROJECT_ROOT, module_file)
    mod_name = 'benchmark_loaded_' + os.path.splitext(os.path.basename(path))[0]
    if mod_name in sys.modules:
        return getattr(sys.modules[mod_name], class_name)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


@dataclass
class ParamSpec:
    name: str            # constructor kwarg name
    label: str           # UI label
    type: str            # 'int' | 'float' | 'bool'
    default: object
    min: object = None
    max: object = None
    step: object = None
    help: str = ''

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ''}

    def coerce(self, value):
        """Coerce a JSON value to the declared type, falling back to default."""
        if value is None:
            return self.default
        if self.type == 'int':
            return int(value)
        if self.type == 'float':
            return float(value)
        if self.type == 'bool':
            return bool(value)
        return value


@dataclass
class AlgorithmSpec:
    id: str
    label: str
    dims: "list | None"   # supported dimensionalities, e.g. [2] or [2, 3]; None = any
    module_file: str      # path relative to project root
    class_name: str
    params: list = field(default_factory=list)   # list[ParamSpec]
    external: "str | None" = None   # third_party/setup_external.py component it needs

    def supports_dims(self, n: int) -> bool:
        return self.dims is None or n in self.dims

    def to_json(self) -> dict:
        return {
            'id': self.id,
            'label': self.label,
            'dims': self.dims,        # null = any dimensionality
            'params': [p.to_json() for p in self.params],
        }

    def coerce_params(self, raw: dict) -> dict:
        """Validate/coerce raw UI params into constructor kwargs."""
        raw = raw or {}
        return {p.name: p.coerce(raw.get(p.name)) for p in self.params}

    def build(self, map_obj, params: dict):
        cls = _load_class(self.module_file, self.class_name)
        return cls(map_obj, **self.coerce_params(params))


# ---------------------------------------------------------------------------
# The registry. Add new algorithms here.
# ---------------------------------------------------------------------------

REGISTRY: list[AlgorithmSpec] = [
    AlgorithmSpec(
        id='astar',
        label='A* (baseline)',
        dims=[2, 3],
        module_file=os.path.join('Algorithms', 'Baseline Algorithms', 'astar.py'),
        class_name='AStarAlgorithm',
        params=[
            ParamSpec('timeout', 'Search timeout (s)', 'float', 60.0, min=1.0,
                      help='A* gives up after this many seconds.'),
        ],
    ),
    AlgorithmSpec(
        id='potential_field',
        label='Potential Field (baseline)',
        dims=[2],
        module_file=os.path.join('Algorithms', 'Baseline Algorithms', 'potentialfield2d.py'),
        class_name='PotentialFieldAlgorithm',
        params=[
            ParamSpec('q_star', 'Influence radius (q*)', 'float', 30.0, min=1.0, step=1.0,
                      help='Obstacles repel within this distance (in cells). Scale with map size.'),
            ParamSpec('k_att', 'Attractive gain', 'float', 1.0, min=0.0, step=0.1,
                      help='Weight of the goal attraction term.'),
            ParamSpec('k_rep', 'Repulsive gain', 'float', 10000.0, min=0.0, step=100.0,
                      help='Weight of the obstacle repulsion term.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('max_iters', 'Max iterations', 'int', 10000, min=100,
                      help='Descent steps before giving up.'),
            ParamSpec('stall_window', 'Stall window', 'int', 100, min=10,
                      help='Steps without progress toward the goal before reporting a local minimum (unsolved).'),
            ParamSpec('bilinear_interpolation', 'Bilinear interpolation', 'bool', True,
                      help='Sample the potential continuously during gradient descent.'),
            _gpu_param(),
        ],
    ),
    AlgorithmSpec(
        id='potential_field_nd',
        label='Potential Field ND (baseline)',
        dims=None,   # any dimensionality
        module_file=os.path.join('Algorithms', 'Baseline Algorithms', 'potentialfieldnd.py'),
        class_name='PotentialFieldAlgorithmND',
        params=[
            ParamSpec('q_star', 'Influence radius (q*)', 'float', 30.0, min=1.0, step=1.0,
                      help='Obstacles repel within this distance (in cells). Scale with map size.'),
            ParamSpec('k_att', 'Attractive gain', 'float', 1.0, min=0.0, step=0.1,
                      help='Weight of the goal attraction term.'),
            ParamSpec('k_rep', 'Repulsive gain', 'float', 10000.0, min=0.0, step=100.0,
                      help='Weight of the obstacle repulsion term.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('max_iters', 'Max iterations', 'int', 10000, min=100,
                      help='Descent steps before giving up.'),
            ParamSpec('stall_window', 'Stall window', 'int', 100, min=10,
                      help='Steps without progress toward the goal before reporting a local minimum (unsolved).'),
            ParamSpec('bilinear_interpolation', 'N-linear interpolation', 'bool', True,
                      help='Sample the potential continuously during gradient descent.'),
            _gpu_param(),
        ],
    ),
    AlgorithmSpec(
        id='rrt',
        label='RRT (baseline)',
        dims=None,   # any dimensionality
        module_file=os.path.join('Algorithms', 'Baseline Algorithms', 'rrt.py'),
        class_name='RRTAlgorithm',
        params=[
            ParamSpec('max_iterations', 'Max samples', 'int', 5000, min=10,
                      help='Random samples before giving up.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.1, step=0.5,
                      help='Maximum tree-extension length per sample (in cells).'),
            ParamSpec('goal_sample_rate', 'Goal sample rate', 'float', 0.05, min=0.0, max=1.0, step=0.01,
                      help='Probability of sampling the goal itself (goal bias).'),
            ParamSpec('goal_tolerance', 'Goal tolerance', 'float', 1.5, min=0.5, step=0.5,
                      help='Distance at which a node connects to the goal.'),
            ParamSpec('seed', 'Seed (-1 = random)', 'int', -1, min=-1,
                      help='Fix for reproducible trees; -1 uses fresh randomness.'),
            _gpu_param(),
        ],
    ),
    AlgorithmSpec(
        id='fvb_laplace',
        label='Frontier Voltage Boost Laplace',
        dims=[2],
        module_file=os.path.join('Algorithms', 'Frontier Voltage Boost', 'frontiervoltageboostlaplace.py'),
        class_name='FrontierVoltageBoostLaplace',
        params=[
            ParamSpec('n_l', 'Laplace iterations (n_l)', 'int', 100, min=1,
                      help='Laplace relaxation iterations per wavefront step.'),
            ParamSpec('epsilon', 'Epsilon', 'float', 1e-8, min=0.0,
                      help='Cell is solved when phi <= v_max - epsilon.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('bilinear_interpolation', 'Bilinear interpolation', 'bool', True,
                      help='Sample phi continuously during gradient descent.'),
            _gpu_param(),
        ],
    ),
    AlgorithmSpec(
        id='fvb_laplace_3d',
        label='Frontier Voltage Boost Laplace 3D',
        dims=[3],
        module_file=os.path.join('Algorithms', 'Frontier Voltage Boost', '3dfrontiervoltageboostlaplace.py'),
        class_name='FrontierVoltageBoostLaplace3D',
        params=[
            ParamSpec('n_l', 'Laplace iterations (n_l)', 'int', 100, min=1,
                      help='Laplace relaxation iterations per wavefront step.'),
            ParamSpec('epsilon', 'Epsilon', 'float', 1e-8, min=0.0,
                      help='Cell is solved when phi <= v_max - epsilon.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('bilinear_interpolation', 'Trilinear interpolation', 'bool', True,
                      help='Sample phi continuously during gradient descent.'),
            _gpu_param(),
        ],
    ),
    AlgorithmSpec(
        id='fvb_laplace_nd',
        label='Frontier Voltage Boost Laplace ND',
        dims=None,   # any dimensionality
        module_file=os.path.join('Algorithms', 'Frontier Voltage Boost', 'ndfrontiervoltageboostlaplace.py'),
        class_name='FrontierVoltageBoostLaplaceND',
        params=[
            ParamSpec('n_l', 'Laplace iterations (n_l)', 'int', 100, min=1,
                      help='Laplace relaxation iterations per wavefront step.'),
            ParamSpec('epsilon', 'Epsilon', 'float', 1e-8, min=0.0,
                      help='Cell is solved when phi <= v_max - epsilon.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('bilinear_interpolation', 'N-linear interpolation', 'bool', True,
                      help='Sample phi continuously during gradient descent.'),
            _gpu_param(),
        ],
    ),
    AlgorithmSpec(
        id='rrt_laplace_random',
        label='RRT–Laplace (random sampling)',
        dims=[2],
        module_file=os.path.join('Algorithms', 'Incremental Algorithms', 'rrtlaplacerandomsampling.py'),
        class_name='RRTLaplaceRandomSampling',
        params=[
            ParamSpec('n_l', 'Laplace iterations (n_l)', 'int', 100, min=1,
                      help='Laplace relaxation iterations per outer step.'),
            ParamSpec('n_w', 'Warm-up iterations (n_w)', 'int', 200, min=0,
                      help='Laplace iterations run once before the main loop.'),
            ParamSpec('n_r', 'Samples per step (n_r)', 'int', 5, min=1,
                      help='Random seed points tried per outer step.'),
            ParamSpec('prob_sp', 'Start-sample probability', 'float', 0.05, min=0.0, max=1.0, step=0.01,
                      help='Probability a seed is the start point (goal-directed bias).'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('bilinear_interpolation', 'Bilinear interpolation', 'bool', True,
                      help='Sample phi continuously during gradient descent.'),
            ParamSpec('max_outer_iters', 'Max outer iterations', 'int', 10000, min=1,
                      help='Outer loops before giving up (reports unsolved).'),
            ParamSpec('seed', 'Seed (-1 = random)', 'int', -1, min=-1,
                      help='Fix for reproducible sampling; -1 uses fresh randomness.'),
            _gpu_param(),
        ],
    ),
    AlgorithmSpec(
        id='rrt_laplace_frontier',
        label='RRT–Laplace (frontier sampling)',
        dims=[2],
        module_file=os.path.join('Algorithms', 'Incremental Algorithms', 'rrtlaplacefrontier.py'),
        class_name='RRTLaplaceFrontier',
        params=[
            ParamSpec('n_l', 'Laplace iterations (n_l)', 'int', 100, min=1,
                      help='Laplace relaxation iterations per outer step.'),
            ParamSpec('n_r', 'Samples per step (n_r)', 'int', 5, min=1,
                      help='Frontier seed points tried per outer step.'),
            ParamSpec('epsilon', 'Epsilon', 'float', 1e-8, min=0.0, max=1.0,
                      help='A cell is solved (part of the frontier interior) when phi <= 1 - epsilon.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('bilinear_interpolation', 'Bilinear interpolation', 'bool', True,
                      help='Sample phi continuously during gradient descent.'),
            ParamSpec('max_outer_iters', 'Max outer iterations', 'int', 10000, min=1,
                      help='Outer loops before giving up (reports unsolved).'),
            ParamSpec('seed', 'Seed (-1 = random)', 'int', -1, min=-1,
                      help='Fix for reproducible sampling; -1 uses fresh randomness.'),
            _gpu_param(),
        ],
    ),
]


# ---------------------------------------------------------------------------
# External comparison planners — upstream implementations pinned and built by
# `python third_party/setup_external.py` (see third_party/README.md).
# ---------------------------------------------------------------------------

def _descent_params() -> list:
    """Path-extraction knobs shared by the field-descent wrappers."""
    return [
        ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                  help='Gradient descent step size (cells).'),
        ParamSpec('bilinear_interpolation', 'N-linear interpolation', 'bool', True,
                  help='Sample the field continuously during gradient descent.'),
        ParamSpec('max_iters', 'Max iterations', 'int', 20000, min=100,
                  help='Descent steps before giving up.'),
        ParamSpec('stall_window', 'Stall window', 'int', 200, min=10,
                  help='Steps without the field value decreasing before reporting '
                       'a local minimum (unsolved).'),
    ]


REGISTRY += [
    AlgorithmSpec(
        id='fast_marching',
        label='Fast Marching Method (thinks/fast-marching-method)',
        dims=[2, 3, 4, 5, 6],
        module_file=os.path.join('Algorithms', 'External Algorithms', 'fastmarching.py'),
        class_name='FastMarchingAlgorithm',
        external='fmm',
        params=[
            ParamSpec('clearance_saturation', 'Clearance saturation (FM²)', 'float', 0.0,
                      min=0.0, step=1.0,
                      help='0 = plain FMM (unit speed: geodesic distance, shortest paths). '
                           '> 0 = Fast Marching Square: speed = min(1, d / value) with d '
                           'the distance to the nearest obstacle (cells).'),
            ParamSpec('obstacle_speed', 'Obstacle speed', 'float', 1e-6, min=1e-12,
                      help='Front speed inside obstacles (the solver needs speed > 0).'),
            ParamSpec('high_accuracy', 'High-accuracy solver', 'bool', True,
                      help='Second-order eikonal solver (upstream default).'),
            *_descent_params(),
        ],
    ),
    AlgorithmSpec(
        id='navigation_function',
        label='Koditschek–Rimon Navigation Function (nfsim port)',
        dims=None,   # any dimensionality
        module_file=os.path.join('Algorithms', 'External Algorithms', 'navigationfunction.py'),
        class_name='KoditschekRimonNF',
        params=[
            ParamSpec('k', 'k (0 = auto)', 'float', 0.0, min=0.0, step=1.0,
                      help='KR tuning exponent; 0 uses #obstacles + 1. Larger k weakens '
                           'obstacle repulsion (fewer spurious minima, paths nearer walls).'),
            *_descent_params(),
        ],
    ),
    AlgorithmSpec(
        id='harmonic_map',
        label='Harmonic Map Navigation (hntf2d)',
        dims=[2],
        module_file=os.path.join('Algorithms', 'External Algorithms', 'harmonicmap2d.py'),
        class_name='HarmonicMapNavigation2D',
        external='hntf2d',
        params=[
            ParamSpec('panel_length', 'Panel length', 'float', 1.0, min=0.1, step=0.1,
                      help='Max boundary panel length (cells) for the panel-method map. '
                           'Smaller = more accurate map, cubic cost in the panel count.'),
            ParamSpec('step_size', 'Max step (cells)', 'float', 1.0, min=0.05, step=0.05,
                      help='The paper integrates with 10 ms Euler steps; a step that '
                           'would travel further than this is taken as a shorter Euler step.'),
            ParamSpec('k_d', 'Goal strength k_d', 'float', 20.0, min=0.1, step=1.0,
                      help='Initial goal source strength (paper §6.1: 20).'),
            ParamSpec('k_i', 'Obstacle strength k_i', 'float', 1.0, min=0.0, step=0.1,
                      help='Initial obstacle source strength (paper: 1).'),
            ParamSpec('k_bar', 'Obstacle strength bound k̄_i', 'float', 20.0, min=0.0, step=1.0,
                      help='Upper bound of the adaptive obstacle strengths (paper: 20).'),
            ParamSpec('w_phi', 'Potential scale w_φ', 'float', 20.0, min=0.1, step=1.0,
                      help='tanh scaling of the harmonic potential (paper: 20).'),
            ParamSpec('K_k', 'Adaptation gain K_k', 'float', 100.0, min=0.0, step=10.0,
                      help='Decay gain of the obstacle strengths near the outer boundary.'),
            ParamSpec('max_iters', 'Max iterations', 'int', 20000, min=100,
                      help='Integration steps before giving up.'),
        ],
    ),
    AlgorithmSpec(
        id='custom_fields',
        label='Custom Harmonic Fields (cust_fields)',
        dims=[2],
        module_file=os.path.join('Algorithms', 'External Algorithms', 'customfields.py'),
        class_name='CustomHarmonicFields2D',
        external='cust_fields',
        params=[
            ParamSpec('world_size', 'World size', 'float', 5.0, min=0.5, step=0.5,
                      help='Map is scaled so its longer side spans this many world units '
                           '(upstream constants are tuned for ~5).'),
            ParamSpec('nf_lambda', 'Star→sphere λ', 'float', 1e5, min=1.0,
                      help='Switch parameter of the star-to-sphere map (upstream demo: 1e5). '
                           'Map-dependent — worth sweeping.'),
            ParamSpec('squircle_s', 'Squircle s', 'float', 0.99, min=0.5, max=0.9999, step=0.01,
                      help='Squareness of the rectangle obstacles (upstream demo: 0.99).'),
            ParamSpec('max_iters', 'Max iterations', 'int', 3000, min=100,
                      help='Upstream integrator step limit (its default: 3000).'),
        ],
    ),
]


def get_spec(algo_id: str) -> AlgorithmSpec:
    for spec in REGISTRY:
        if spec.id == algo_id:
            return spec
    raise KeyError(f"Unknown algorithm id: {algo_id!r}")


def registry_json() -> list:
    return [spec.to_json() for spec in REGISTRY]
