# Test Suite

**The CLI scripts that used to live here are retired.** Map generation, algorithm
runs (all algorithms, all dimensionalities), metrics, visualization, and result
management all live in the benchmark web app:

```bash
python run_app.py
```

See the root [README](../README.md) for what the app can do.

## What remains here

- `output/` — map folders produced by the old CLI scripts. Pull these into the
  app via **Map Sets → Import from folder…** (point it at e.g.
  `Test Suite/output/2d_maps`). The app copies the files; nothing here is modified.

Once you've imported anything you care about, this folder can be deleted.
