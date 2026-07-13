# Test Suite

**The CLI scripts that used to live here are retired.** Map generation, algorithm
runs, metrics, and result management all moved into the benchmark web app:

```bash
python run_app.py
```

See the root [README](../README.md) for what the app can do.

## What remains here

- `output/` — map folders produced by the old CLI scripts. You can pull these
  into the app via **Map Sets → Import from folder…** (point it at e.g.
  `Test Suite/output/2d_maps`). The app copies the files; nothing here is modified.

Results are written to `<maps_dir>/fvb_results/`, one file per map per output type:

```
fvb_results/
├── map_0_map.png
├── map_0_path.png
├── map_0_phi.png
├── map_1_map.png
└── ...
```

---

## run_test_suite_frontier_voltage_boost_laplace_3d.py

Runs 3D Frontier Voltage Boost Laplace against a folder of 3D maps produced by `init_test_suite.py -dims 3`.

```bash
# Basic run — all output types
python "Test Suite/run_test_suite_frontier_voltage_boost_laplace_3d.py" `
    --maps_dir "Test Suite/output/3d_maps" `
    --laplace_iters 200 `
    --epsilon 0.0001

# Custom step size, selective output
python "Test Suite/run_test_suite_frontier_voltage_boost_laplace_3d.py" `
    --maps_dir "Test Suite/output/3d_maps" `
    --laplace_iters 200 `
    --epsilon 0.0001 `
    --step_size 0.5 `
    --output paths phi
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--maps_dir` | yes | — | Folder with `.npy` files and `start_end_points.csv` |
| `--laplace_iters` | yes | — | Laplace iterations per wavefront step (`n_l`) |
| `--epsilon` | yes | — | Solved threshold: cell solved when `φ ≤ v_max − ε` |
| `--step_size` | no | `1.0` | Gradient descent step size |
| `--no_bilinear` | no | off | Disable trilinear interpolation for gradient sampling (on by default) |
| `--output` | no | all | Space-separated: `maps`, `paths`, `phi` |

**`--output` values:**

| Value | Saves |
|-------|-------|
| `maps` | Per-z-slice PNGs of the raw map |
| `paths` | Per-z-slice PNGs with path overlay — blue path, green start, red end |
| `phi` | Per-z-slice PNGs of the potential field φ as a grayscale heatmap |

Results are written to `<maps_dir>/fvb_results/<map_stem>/`, one subfolder per map:

```
fvb_results/
├── map_0/
│   ├── map_slice_0000.png
│   ├── path_slice_0000.png
│   ├── phi_slice_0000.png
│   └── ...
├── map_1/
└── ...
```
Once you've imported anything you care about, this folder can be deleted.
