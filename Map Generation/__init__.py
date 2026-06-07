import sys
import os
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)


def _load(name, filename):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(_here, filename))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


Map = _load('map', 'map.py').Map
Map2D = _load('map2d', '2dmap.py').Map2D
Map3D = _load('map3d', '3dmap.py').Map3D
