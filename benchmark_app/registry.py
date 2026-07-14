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
            ParamSpec('timeout', 'Search timeout (s)', 'float', 30.0, min=1.0,
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
            ParamSpec('step_size', 'Step size', 'float', 3.0, min=0.1, step=0.5,
                      help='Maximum tree-extension length per sample (in cells).'),
            ParamSpec('goal_sample_rate', 'Goal sample rate', 'float', 0.05, min=0.0, max=1.0, step=0.01,
                      help='Probability of sampling the goal itself (goal bias).'),
            ParamSpec('goal_tolerance', 'Goal tolerance', 'float', 1.5, min=0.5, step=0.5,
                      help='Distance at which a node connects to the goal.'),
            ParamSpec('seed', 'Seed (-1 = random)', 'int', -1, min=-1,
                      help='Fix for reproducible trees; -1 uses fresh randomness.'),
        ],
    ),
    AlgorithmSpec(
        id='fvb_laplace',
        label='Frontier Voltage Boost Laplace',
        dims=[2],
        module_file=os.path.join('Algorithms', 'Frontier Voltage Boost', 'frontiervoltageboostlaplace.py'),
        class_name='FrontierVoltageBoostLaplace',
        params=[
            ParamSpec('n_l', 'Laplace iterations (n_l)', 'int', 10, min=1,
                      help='Laplace relaxation iterations per wavefront step.'),
            ParamSpec('epsilon', 'Epsilon', 'float', 0.5, min=0.0, step=0.05,
                      help='Cell is solved when phi <= v_max - epsilon.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('bilinear_interpolation', 'Bilinear interpolation', 'bool', True,
                      help='Sample phi continuously during gradient descent.'),
        ],
    ),
    AlgorithmSpec(
        id='fvb_laplace_3d',
        label='Frontier Voltage Boost Laplace 3D',
        dims=[3],
        module_file=os.path.join('Algorithms', 'Frontier Voltage Boost', '3dfrontiervoltageboostlaplace.py'),
        class_name='FrontierVoltageBoostLaplace3D',
        params=[
            ParamSpec('n_l', 'Laplace iterations (n_l)', 'int', 10, min=1,
                      help='Laplace relaxation iterations per wavefront step.'),
            ParamSpec('epsilon', 'Epsilon', 'float', 0.5, min=0.0, step=0.05,
                      help='Cell is solved when phi <= v_max - epsilon.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('bilinear_interpolation', 'Trilinear interpolation', 'bool', True,
                      help='Sample phi continuously during gradient descent.'),
        ],
    ),
    AlgorithmSpec(
        id='fvb_laplace_nd',
        label='Frontier Voltage Boost Laplace ND',
        dims=None,   # any dimensionality
        module_file=os.path.join('Algorithms', 'Frontier Voltage Boost', 'ndfrontiervoltageboostlaplace.py'),
        class_name='FrontierVoltageBoostLaplaceND',
        params=[
            ParamSpec('n_l', 'Laplace iterations (n_l)', 'int', 10, min=1,
                      help='Laplace relaxation iterations per wavefront step.'),
            ParamSpec('epsilon', 'Epsilon', 'float', 0.5, min=0.0, step=0.05,
                      help='Cell is solved when phi <= v_max - epsilon.'),
            ParamSpec('step_size', 'Step size', 'float', 1.0, min=0.05, step=0.05,
                      help='Gradient descent step size.'),
            ParamSpec('bilinear_interpolation', 'N-linear interpolation', 'bool', True,
                      help='Sample phi continuously during gradient descent.'),
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
