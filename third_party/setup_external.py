"""
Fetch and build the external comparison planners.

    python third_party/setup_external.py              # fetch (pinned) + build all
    python third_party/setup_external.py fmm hntf2d   # only these components
    python third_party/setup_external.py --force      # re-fetch and rebuild
    python third_party/setup_external.py --check      # report what is available
    python third_party/setup_external.py --selftest   # solve a small map with each planner

Upstream sources are checked out at pinned commits into ``third_party/src/``
and the compiled extension modules land in ``third_party/build/``. Both
directories are git-ignored: the native modules are platform-specific (a macOS
arm64 build will not load on a Linux VM), so every machine builds its own from
the same pinned sources. Only the pybind11 bindings in ``third_party/bindings/``
are ours; upstream code is used as checked out, except for the bug fixes in
``third_party/patches/<name>.patch``, applied right after checkout (see
third_party/README.md).

Requirements: git, a C++17 compiler, and ``pip install pybind11 pyyaml``.
"""

import argparse
import hashlib
import importlib.machinery
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, 'src')
BUILD_DIR = os.path.join(HERE, 'build')
BINDINGS_DIR = os.path.join(HERE, 'bindings')
PATCHES_DIR = os.path.join(HERE, 'patches')

# name -> (git url, ref to fetch, expected commit, license)
SOURCES = {
    'fast-marching-method': (
        'https://github.com/thinks/fast-marching-method.git',
        'b20f0b224b97935e07bd0ba888c2611ba2786f76',
        'b20f0b224b97935e07bd0ba888c2611ba2786f76', 'MIT'),
    'hntf2d': (
        'https://github.com/maxchaos/hntf2d.git',
        '583527e04a29ccd9482b0785b4cc132e1f654445',
        '583527e04a29ccd9482b0785b4cc132e1f654445', 'GPL-3.0'),
    'cust_fields': (
        'https://github.com/Shuaikang-Wang/cust_fields.git',
        '5ca178e588b4b766cbc537af4a2c4e8ebb53895a',
        '5ca178e588b4b766cbc537af4a2c4e8ebb53895a', 'none stated (all rights reserved)'),
    # hntf2d's C++ dependencies (header-only Eigen; tinyxml2 is compiled in).
    'eigen': (
        'https://gitlab.com/libeigen/eigen.git', '3.4.0',
        '3147391d946bb4b6c68edd901f2add6ac1f31f8c', 'MPL-2.0'),
    'tinyxml2': (
        'https://github.com/leethomason/tinyxml2.git', '10.0.0',
        '321ea883b7190d4e85cae5512a12e5eaa8f8731f', 'Zlib'),
}

# component -> sources it needs
COMPONENTS = {
    'fmm': ['fast-marching-method'],
    'hntf2d': ['hntf2d', 'eigen', 'tinyxml2'],
    'cust_fields': ['cust_fields'],
}


def _git(*args, cwd=None):
    subprocess.run(['git', *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def fetch(name: str, force: bool = False) -> str:
    url, ref, commit, _ = SOURCES[name]
    dest = os.path.join(SRC_DIR, name)
    if os.path.isdir(dest) and not force:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=dest,
                              capture_output=True, text=True).stdout.strip()
        if head == commit and is_patched(name):
            return dest
        if head == commit and not os.path.isfile(_patch_marker(name)):
            _apply_patch(name, dest)   # pinned and untouched: just patch it
            return dest
        print(f'  {name}: checkout is at {head[:10] or "?"} (pinned {commit[:10]}) or carries '
              'an outdated patch — re-fetching')
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    print(f'  fetching {name} @ {ref}')
    _git('init', '-q', cwd=dest)
    _git('remote', 'add', 'origin', url, cwd=dest)
    _git('fetch', '-q', '--depth', '1', 'origin', ref, cwd=dest)
    _git('checkout', '-q', 'FETCH_HEAD', cwd=dest)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=dest,
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise RuntimeError(f'{name}: fetched {head}, expected pinned commit {commit}')
    # Upstream cust_fields ships stale __pycache__ dirs; never import those.
    for root, dirs, _ in os.walk(dest):
        for d in list(dirs):
            if d == '__pycache__':
                shutil.rmtree(os.path.join(root, d))
                dirs.remove(d)
    _apply_patch(name, dest)
    return dest


def _patch_file(name: str):
    path = os.path.join(PATCHES_DIR, f'{name}.patch')
    return path if os.path.isfile(path) else None


def _patch_marker(name: str) -> str:
    return os.path.join(SRC_DIR, name, '.applied-patch')


def _patch_digest(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def is_patched(name: str) -> bool:
    """The checkout carries the current patch (or none is needed)."""
    patch = _patch_file(name)
    if patch is None:
        return True
    try:
        with open(_patch_marker(name)) as f:
            return f.read().strip() == _patch_digest(patch)
    except OSError:
        return False


def _apply_patch(name: str, dest: str):
    patch = _patch_file(name)
    if patch is None:
        return
    print(f'  applying patches/{name}.patch')
    _git('apply', '--whitespace=nowarn', patch, cwd=dest)
    with open(_patch_marker(name), 'w') as f:
        f.write(_patch_digest(patch) + '\n')


def _build_extension(ext):
    from setuptools import Distribution
    from pybind11.setup_helpers import build_ext

    os.makedirs(BUILD_DIR, exist_ok=True)
    dist = Distribution({'name': ext.name, 'ext_modules': [ext]})
    cmd = build_ext(dist)
    cmd.build_lib = BUILD_DIR
    with tempfile.TemporaryDirectory() as tmp:
        cmd.build_temp = tmp
        cmd.ensure_finalized()
        cmd.run()


def build_fmm():
    from pybind11.setup_helpers import Pybind11Extension
    src = os.path.join(SRC_DIR, 'fast-marching-method')
    _build_extension(Pybind11Extension(
        'external_fmm',
        [os.path.join(BINDINGS_DIR, 'fmm_module.cpp')],
        include_dirs=[os.path.join(src, 'include')],
        cxx_std=17,
        extra_compile_args=['-O3', '-DNDEBUG'],
    ))


def build_hntf2d():
    from pybind11.setup_helpers import Pybind11Extension
    src = os.path.join(SRC_DIR, 'hntf2d')
    tinyxml2 = os.path.join(SRC_DIR, 'tinyxml2')
    # Silence warnings that only concern the unmodified upstream map.cpp.
    extra = ['-O3', '-DNDEBUG', '-Wno-sign-compare', '-Wno-unused-variable',
             '-Wno-unused-but-set-variable', '-Wno-deprecated-declarations']
    if sys.platform == 'darwin' or 'clang' in (os.environ.get('CXX') or ''):
        extra.append('-Wno-vla-extension')   # upstream map.cpp uses VLAs
    _build_extension(Pybind11Extension(
        'external_hntf2d',
        [os.path.join(BINDINGS_DIR, 'hntf2d_module.cpp'),
         os.path.join(src, 'cpp', 'src', 'map.cpp'),
         os.path.join(tinyxml2, 'tinyxml2.cpp')],
        include_dirs=[os.path.join(src, 'cpp', 'include'),
                      os.path.join(SRC_DIR, 'eigen'),
                      tinyxml2],
        cxx_std=14,
        extra_compile_args=extra,
    ))


def build_cust_fields():
    try:
        import yaml  # noqa: F401  (cust_fields' World loader needs PyYAML)
    except ImportError:
        raise RuntimeError('cust_fields needs PyYAML: pip install pyyaml')


BUILDERS = {'fmm': build_fmm, 'hntf2d': build_hntf2d, 'cust_fields': build_cust_fields}


# component -> (compiled module or None, files that must exist in third_party/src)
_RUNTIME = {
    'fmm': ('external_fmm', []),
    'hntf2d': ('external_hntf2d', [('hntf2d', 'cy', 'hntf2d', 'utils.py')]),
    'cust_fields': (None, [('cust_fields', 'NF', 'transformation.py'),
                           ('cust_fields', 'TOPO', '__init__.py')]),
}


def is_ready(component: str) -> bool:
    """True when ``component`` can run under *this* Python interpreter.

    Checks files only (no imports): a compiled module must exist with this
    interpreter's extension suffix, so a build made with another Python
    version counts as missing.
    """
    module, files = _RUNTIME[component]
    if module and not any(os.path.isfile(os.path.join(BUILD_DIR, module + suffix))
                          for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        return False
    return (all(os.path.isfile(os.path.join(SRC_DIR, *f)) for f in files)
            and all(is_patched(name) for name in COMPONENTS[component]))


def check() -> dict:
    """Return {component: (ok, detail)} without fetching or building."""
    py = f'Python {sys.version_info.major}.{sys.version_info.minor}'
    return {comp: (True, f'ready for {py}') if is_ready(comp)
            else (False, f'not set up for {py} ({sys.executable})')
            for comp in COMPONENTS}


def selftest() -> bool:
    """Solve one small map with every external planner (through the app's
    registry and solver subprocess, exactly as a benchmark run would)."""
    import numpy as np
    root = os.path.dirname(HERE)
    sys.path.insert(0, root)
    from benchmark_app.registry import get_spec
    from benchmark_app.solver_worker import solve_with_timeout

    # Two free-standing blocks off the start-goal diagonal (inside every
    # method's world class, so a failure here means a broken setup).
    grid2 = np.zeros((40, 40), dtype=np.uint8)
    grid2[4:12, 4:12] = 1
    grid2[26:34, 26:34] = 1
    grid3 = np.zeros((16, 16, 16), dtype=np.uint8)
    grid3[4:12, 6:10, 4:12] = 1
    s2, e2, s3, e3 = (4, 35), (35, 4), (2, 2, 2), (13, 13, 13)
    cases = [('fast_marching', grid2, s2, e2), ('fast_marching', grid3, s3, e3),
             ('navigation_function', grid2, s2, e2), ('navigation_function', grid3, s3, e3),
             ('harmonic_map', grid2, s2, e2), ('custom_fields', grid2, s2, e2)]
    ok = True
    for algo_id, grid, start, end in cases:
        out = solve_with_timeout(algo_id, {}, grid, start, end, timeout_s=120)
        detail = (f'{len(out["path"])} points, {out["time_s"]:.3f}s' if out['solved']
                  else (out['error'] or '').strip().splitlines()[-1])
        print(f'{get_spec(algo_id).label:55s} {grid.ndim}D  {"OK " if out["solved"] else "-- "} {detail}')
        ok &= out['solved']
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('components', nargs='*', choices=[[]] + list(COMPONENTS),
                        help='components to set up (default: all)')
    parser.add_argument('--force', action='store_true', help='re-fetch and rebuild')
    parser.add_argument('--check', action='store_true', help='only report status')
    parser.add_argument('--selftest', action='store_true',
                        help='solve a small map with every external planner')
    args = parser.parse_args()

    if args.check:
        for comp, (ok, detail) in check().items():
            print(f'{comp:12s} {"OK " if ok else "-- "} {detail}')
        return
    if args.selftest:
        sys.exit(0 if selftest() else 'selftest failed')

    wanted = args.components or list(COMPONENTS)
    failures = []
    for comp in wanted:
        print(f'[{comp}]')
        try:
            for name in COMPONENTS[comp]:
                fetch(name, force=args.force)
            BUILDERS[comp]()
            print(f'  {comp}: ready')
        except Exception as exc:   # keep going so one failure doesn't block the rest
            detail = getattr(exc, 'stderr', None)
            if isinstance(detail, bytes):
                detail = detail.decode(errors='replace').strip()
            failures.append(comp)
            print(f'  {comp}: FAILED — {exc}' + (f'\n{detail}' if detail else ''))
    if failures:
        sys.exit(f'failed: {", ".join(failures)}')


if __name__ == '__main__':
    main()
