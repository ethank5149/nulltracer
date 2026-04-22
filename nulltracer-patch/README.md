# Nulltracer P1–P4 Patch

Applies fixes **P1** (API-contract + defaults), **P2** (hero cell rewrite),
**P3** (notebook code fixes), and **P4** (section renumbering) from the
publication-readiness review.

P5–P10 (Kerr-Newman shadow analytic, richer shadow metrics, package hygiene,
tests & CI, docs, reproducibility capstone) are not included — apply these
first, run the notebook top-to-bottom, commit the executed output, *then*
tackle the next layer.

## What's in the tarball

```
nulltracer-patch/
├── nulltracer/
│   ├── renderer.py        # REPLACE — adds _resolve_inclination/_resolve_steps
│   └── compare.py         # REPLACE — uses canonical 'inclination' key
└── notebooks/
    └── nulltracer.ipynb   # REPLACE — 33 cells, reordered, renumbered,
                           # every code cell syntax-clean on Python 3.9+
```

## How to apply

From the repository root:

```bash
tar xzvf nulltracer-patch.tar.gz
cp -v nulltracer-patch/nulltracer/renderer.py nulltracer/renderer.py
cp -v nulltracer-patch/nulltracer/compare.py  nulltracer/compare.py
cp -v nulltracer-patch/notebooks/nulltracer.ipynb notebooks/nulltracer.ipynb
```

Then, on a box with a CUDA GPU:

```bash
pip install -e ".[analysis,notebook]"
jupyter nbconvert --to notebook --execute --inplace notebooks/nulltracer.ipynb
git add -u && git commit -m "P1–P4: fix incl/steps API, rebuild hero notebook"
```

## Fix inventory

### P1 — `nulltracer/renderer.py`

| Bug | Fix |
|---|---|
| `params.get("inclination", 80.0)` silently ignores the `'incl'` key used by every caller in the repo; every dict-based render ran at θ=80° regardless of input | New helper `_resolve_inclination(params)` accepts both keys; used by `render_frame`, `render_frame_timed`, `trace_single_ray` |
| `params.get("steps", 200)` defaults to 200 — at `obs_dist=500, step_size=0.15` that's 30 M of affine length, rays never reach the BH | New helper `_resolve_steps(...)` calls `auto_steps()` when no explicit budget is given (matches the free `render_frame` function) |
| `show_grid` defaulted to `True` in the dict path — overlaid debug grid on every render | Default changed to `False` |

### P1 — `nulltracer/compare.py`

- `compare_integrators()` line 207: `'incl': inclination_deg` changed to `'inclination': inclination_deg` so the canonical-key path is exercised. (With the P1 alias this is cosmetic, but it clarifies the contract.)

### P2–P4 — `notebooks/nulltracer.ipynb`

**P2 — Cell 1 order.** Imports → GPU probe → skymap load → `renderer = CudaRenderer(); renderer.initialize()` → `compile_all()` → hero renders. The `NameError` on `renderer` is gone. Dead imports (`Ellipse`, `scipy.ndimage`, `time`) removed.

**P3 — Code fixes.**
- Cell 7 (shadow-boundary sanity check): prints the numeric comparison and asserts `< 1e-4` relative error.
- Cell 12 (analytic overlay): pixel mapping corrected to match kernel's `(ix+0.5)/W` centre convention — `px = (α/(FOV·asp)+1)·W/2 − 0.5`.
- Cells 22 & 33 (spin sweeps): undefined `source` variable removed — M87 sweep uses `M87['incl']`, Sgr A\* sweep uses `SgrA['incl']`.
- Cell 37 (in-fall grid): inner f-string now uses double quotes so the cell parses on Python 3.9/3.10/3.11 (previous version needed 3.12+).

**P4 — Section numbering.** Single monotonic ordering with no gaps, no duplicates: §1 Intro → §2 Physics → §3 Implementation → §4 Analytic shadow validation → §5 Spin×θ gallery → §6 Shadow extraction → §7 M87\* → §8 Sgr A\* → §9 Advanced integration → §10 Characteristic radii → §11 Integrator comparison & in-fall → §12 Discussion. Empty markdown cells removed; the duplicated "Discussion & Conclusions" header consolidated into §12 with the full reference list.

## Verification

After applying, before running the notebook:

```python
from nulltracer.renderer import _resolve_inclination, _resolve_steps
assert _resolve_inclination({'inclination': 30}) == 30.0
assert _resolve_inclination({'incl': 30})        == 30.0   # alias accepted
assert _resolve_inclination({})                  == 80.0   # fallback
```

To turn this into a proper regression, add to `tests/test_renderer_params.py`:

```python
def test_renderer_accepts_incl_alias():
    from nulltracer.renderer import _resolve_inclination
    assert _resolve_inclination({'incl': 17}) == 17.0
    assert _resolve_inclination({'inclination': 50}) == 50.0
```

## What's still broken after this patch

These stay on the P5–P10 list:

- `shadow_boundary()` has no Kerr-Newman branch (§9.2 narrative is unsupported by the analytic utility).
- `extract_shadow_metrics` reports only a circle-fit diameter; ellipse-fit major axis not surfaced.
- `tests/test_cache.py` imports `from server.cache import ImageCache` — no `server/` package exists; fails at collection.
- README "Project Structure" section describes a layout (`server/`, top-level `index.html`, `docs/images/hero.jpg`, …) that does not match the actual tree.
- Root-level `full_kernel.cu` (1447 LOC) is referenced only by `nulltracer_portfolio.ipynb` — dead weight from a parallel architecture.
- `__init__.py::__all__` lists `render_classify` which is not in the lazy loader.
- `fov` docstring says "degrees" but the kernel treats it as screen half-width in $M$.
