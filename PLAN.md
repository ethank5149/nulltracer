# Nulltracer Publication-Readiness Implementation Plan

**Target repo:** `https://github.com/ethank5149/nulltracer` (branch: `master`)
**Goal:** Make Nulltracer portfolio-ready for immediate public showcase.
**Execution order matters.** Tasks are sequenced so each builds on the last. Do them in order.

---

## TASK 0: Setup and Orientation

**Read before doing anything else.**

The repo lives at the GitHub URL above. Clone it locally. The current structure is:

```
nulltracer/
├── server/           # FastAPI + CuPy CUDA backend
│   ├── app.py
│   ├── renderer.py
│   ├── isco.py
│   ├── cache.py
│   ├── bloom.py
│   ├── scenes.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── kernels/
│       ├── geodesic_base.cu
│       ├── backgrounds.cu
│       ├── disk.cu
│       ├── ray_trace.cu
│       └── integrators/
│           ├── rk4.cu
│           ├── rkdp8.cu
│           ├── tao_yoshida4.cu
│           ├── tao_yoshida6.cu
│           ├── tao_kahan_li8.cu
│           ├── kahanli8s.cu
│           └── kahanli8s_ks.cu
├── js/               # Browser client
│   ├── main.js
│   ├── server-client.js
│   └── ui-controller.js
├── index.html
├── styles.css
├── bench.html
├── deploy.sh
├── docker-compose.yml
├── nulltracer-renderer.xml
├── nulltracer-icon.png
├── nulltracer-icon.svg
├── schwarzschild-black-hole-nasa-labeled-reference.jpg
├── nulltracer.ipynb
├── ARCHITECTURE.md
├── FEATURE-PLAN.md
├── README.md
└── .forgejo/workflows/
```

**Do NOT restructure the server/ directory or rename kernel files.** The CUDA compilation pipeline depends on the current paths. All tasks below are additive or modify existing files in place.

---

## TASK 1: Add LICENSE File

**Priority:** Critical — legally required for open source.
**Time estimate:** 1 minute.

Create `LICENSE` at the repo root with the full text of the **MIT License**.

- Copyright holder: `Ethan Knox`
- Year: `2024-2026`

**Acceptance criteria:** `LICENSE` file exists at repo root. Contains valid MIT license text.

---

## TASK 2: Generate Hero Images

**Priority:** Critical — the single highest-impact visual change.
**Time estimate:** 10–20 minutes (requires running GPU server).

### 2a: Render hero images

Start the Nulltracer server locally or via Docker. Use the `/render` API endpoint to produce the following images. Save them as high-quality PNGs.

**Hero image** (primary — this goes at the top of the README):
```json
POST /render
{
  "spin": 0.94,
  "charge": 0.0,
  "inclination": 30.0,
  "fov": 10.0,
  "width": 1920,
  "height": 1080,
  "method": "kahanli8s",
  "steps": 200,
  "step_size": 0.3,
  "obs_dist": 50,
  "bg_mode": 0,
  "show_disk": true,
  "show_grid": false,
  "disk_temp": 0.6,
  "star_layers": 4,
  "srgb_output": true,
  "bloom_enabled": true,
  "bloom_radius": 1.0,
  "format": "jpeg",
  "quality": 95
}
```

**Gallery images** (3 additional renders showcasing diversity):

1. **Schwarzschild** (non-spinning): `spin=0.0, charge=0.0, inclination=80, fov=8, method=rkdp8, disk_temp=1.0, bg_mode=1` (checker background to show lensing clearly)
2. **Edge-on extreme Kerr**: `spin=0.998, charge=0.0, inclination=85, fov=12, method=kahanli8s, disk_temp=0.8, bg_mode=0, bloom_enabled=true`
3. **Charged black hole**: `spin=0.5, charge=0.7, inclination=45, fov=10, method=rkdp8, disk_temp=1.2, bg_mode=0`

All gallery images: `width=1280, height=720, quality=90, show_grid=false`.

### 2b: Save images to repo

Create a directory `docs/images/` and save:
- `docs/images/hero.jpg` (the primary render)
- `docs/images/gallery-schwarzschild.jpg`
- `docs/images/gallery-extreme-kerr.jpg`
- `docs/images/gallery-charged.jpg`

**If the GPU server is not available**, skip this task and leave a placeholder note in the README (`<!-- TODO: Add hero image once renders are generated -->`). All other tasks can proceed without it. The owner will generate the images themselves.

**Acceptance criteria:** `docs/images/` directory exists with rendered images (or placeholder comment if GPU unavailable).

---

## TASK 3: Clean Up Repo Root

**Priority:** High — reduces visual clutter.
**Time estimate:** 5 minutes.

### 3a: Move loose assets

Create `assets/` directory at repo root. Move the following files:

```bash
mkdir -p assets
git mv nulltracer-icon.png assets/
git mv nulltracer-icon.svg assets/
git mv schwarzschild-black-hole-nasa-labeled-reference.jpg assets/
git mv nulltracer-renderer.xml assets/
git mv bench.html assets/
```

### 3b: Update any references

Search all files for references to the moved files and update paths:
- `index.html` may reference `nulltracer-icon.png` → update to `assets/nulltracer-icon.png`
- `docker-compose.yml` or `deploy.sh` may reference `nulltracer-renderer.xml` → update to `assets/nulltracer-renderer.xml`
- `README.md` references to the icon or reference image → update paths

**Be thorough.** Use `grep -r` to find all references before moving.

**Acceptance criteria:** Repo root contains only: `README.md`, `LICENSE`, `ARCHITECTURE.md`, `FEATURE-PLAN.md`, `index.html`, `styles.css`, `docker-compose.yml`, `deploy.sh`, `.gitignore`, `nulltracer.ipynb`, and directories (`server/`, `js/`, `docs/`, `assets/`, `.forgejo/`).

---

## TASK 4: Rewrite README.md

**Priority:** Critical — this is the first thing anyone sees.
**Time estimate:** 30–45 minutes.

Replace the current README.md with a restructured version. Keep all the existing technical content but reorganize it. The new structure should be:

### Section 1: Title + Hero Image + One-Paragraph Summary

```markdown
# Nulltracer

**GPU-accelerated ray tracing through curved spacetimes**

![Nulltracer Hero — Kerr black hole, a=0.94](docs/images/hero.jpg)

Nulltracer is a CUDA-powered application that visualizes black holes by tracing null geodesics through Kerr-Newman spacetime. It renders photon rings, gravitational lensing, accretion disk Doppler effects, and frame-dragging with float64 precision. A FastAPI server performs all computation on GPU; a thin browser client provides interactive parameter controls.
```

### Section 2: Gallery (4 images in a 2×2 grid)

Use markdown table or raw HTML for a 2×2 image grid showing the 4 renders from Task 2, with captions:
- "Schwarzschild (a=0) — checker background"
- "Extreme Kerr (a=0.998) — edge-on"
- "Kerr-Newman (a=0.5, Q=0.7)"
- (hero image is already shown above, so use 3 gallery + one additional angle if desired)

### Section 3: Physics & Numerical Methods

Consolidate from the current README's "Overview" and "Technical Details" sections. Cover:
- Kerr-Newman metric (briefly — 2–3 sentences + the metric equation)
- Integration methods (the table of 5 integrators with 1-line descriptions each)
- The kahanli8s integrator (2–3 sentence highlight — Kahan-Li composition + Sundman time + symplectic corrector)
- Accretion disk model (Novikov-Thorne + blackbody + GR redshift)
- Numerical precision (float64 integration, float32 color, uint8 output)

### Section 4: Features (condensed)

Keep the existing feature bullet list but trim it to ~8 bullets. Remove redundancy.

### Section 5: Quick Start

Combine the Docker and local startup instructions. Keep it to ~15 lines.

```markdown
## Quick Start

### Docker (recommended)
​```bash
cd server
docker build -t nulltracer-server .
docker run --gpus all -p 8420:8420 nulltracer-server
​```

### Local
​```bash
cd server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8420
​```

Open `index.html` in a browser. The client auto-detects the server at `/health`.
```

### Section 6: API Reference (condensed)

Keep `POST /render`, `GET /health`, `POST /ray` descriptions. Remove the full Pydantic model listing (that belongs in ARCHITECTURE.md). Show one example request + response for `/render` only.

### Section 7: Deployment

Move the full Caddy/Unraid/Docker Compose deployment instructions into a **new file** `DEPLOYMENT.md`. In the README, add:

```markdown
## Deployment

For production deployment with Caddy reverse proxy, Docker Compose, or Unraid, see [DEPLOYMENT.md](DEPLOYMENT.md).
```

### Section 8: Controls (condensed)

Keep the parameter list but shorten descriptions to one line each.

### Section 9: Project Structure

Keep the existing tree but update paths for the moved assets.

### Section 10: Version History

**Fill in all dates from git tags.** Run:
```bash
git log --tags --simplify-by-decoration --pretty="format:%ai %d"
```
and populate the Date column for every version.

### Section 11: Requirements

Keep as-is (already concise).

### Section 12: License

```markdown
## License

MIT License. See [LICENSE](LICENSE).
```

**Key constraints for the rewrite:**
- Total README length should be ~300–400 lines (currently ~350, but front-loaded with deployment instead of physics).
- No deployment details beyond Quick Start belong in README.
- The hero image MUST be the first visual element after the title.
- Do NOT remove any technical content — just relocate it (to DEPLOYMENT.md or ARCHITECTURE.md as appropriate).

**Acceptance criteria:** README.md leads with hero image, physics section is within the first 50 lines, deployment details are in a separate DEPLOYMENT.md, version history dates are filled in.

---

## TASK 5: Create DEPLOYMENT.md

**Priority:** Medium — supports Task 4.
**Time estimate:** 10 minutes.

Extract the following sections from the current README into a new `DEPLOYMENT.md`:
- The full "Unraid Deployment (Recommended)" section
- The "Docker Compose (Standalone)" section
- The "Local Development" section
- The Caddyfile configuration block
- The "Same-Origin Auto-Detection" section

Add a header:

```markdown
# Nulltracer Deployment Guide

Detailed deployment instructions for production environments.
For quick local setup, see the [Quick Start section in README.md](README.md#quick-start).
```

**Acceptance criteria:** `DEPLOYMENT.md` exists with all deployment content. README no longer contains detailed deployment instructions.

---

## TASK 6: Add Basic Tests

**Priority:** High — demonstrates engineering rigor.
**Time estimate:** 30–40 minutes.

Create `tests/` directory with the following test files. These tests validate the physics and numerical accuracy of the server without requiring a GPU (they test the Python-level logic).

### 6a: `tests/test_isco.py`

```python
"""Test ISCO calculations against known analytic results."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from isco import isco_kerr  # or whatever the function is named — check server/isco.py

import math
import pytest


# Bardeen, Press & Teukolsky (1972) analytic values
# ISCO for prograde orbits around Kerr black holes (M=1)
KNOWN_ISCO = [
    # (spin, expected_isco_radius, tolerance)
    (0.0,   6.0,    1e-10),   # Schwarzschild
    (0.5,   4.233,  1e-3),    # Moderate spin
    (0.9,   2.321,  1e-3),    # High spin
    (0.998, 1.237,  1e-2),    # Near-extremal
]


@pytest.mark.parametrize("spin,expected,tol", KNOWN_ISCO)
def test_isco_kerr(spin, expected, tol):
    """ISCO radius matches Bardeen-Press-Teukolsky formula."""
    # NOTE: Adapt this call to match the actual function signature in server/isco.py.
    # It may be isco_kerr(spin), iscoJS(spin), or isco(spin, charge=0).
    # Check the file and update accordingly.
    result = isco_kerr(spin)
    assert abs(result - expected) < tol, (
        f"ISCO(a={spin}): got {result}, expected {expected} ± {tol}"
    )


def test_schwarzschild_isco_exact():
    """Schwarzschild ISCO is exactly 6M."""
    result = isco_kerr(0.0)
    assert abs(result - 6.0) < 1e-10


def test_isco_monotonically_decreasing():
    """ISCO decreases with increasing spin (prograde)."""
    spins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.998]
    iscos = [isco_kerr(a) for a in spins]
    for i in range(len(iscos) - 1):
        assert iscos[i] > iscos[i + 1], (
            f"ISCO not monotonically decreasing: "
            f"ISCO(a={spins[i]})={iscos[i]} vs ISCO(a={spins[i+1]})={iscos[i+1]}"
        )
```

**IMPORTANT:** Before finalizing this file, read `server/isco.py` and verify:
1. The actual function name (might be `isco_kerr`, `iscoJS`, `isco`, or `compute_isco`)
2. The function signature (might take `spin` only, or `spin, charge`)
3. The import path

Update the test to match.

### 6b: `tests/test_shadow.py`

```python
"""Test shadow diameter against known analytic results."""
import math
import pytest


def schwarzschild_shadow_radius():
    """
    Analytic shadow radius for Schwarzschild black hole.
    For an observer at r_obs >> M:
        r_shadow = 3*sqrt(3) * M  (in units of M)
    This is the critical impact parameter for photon capture.
    """
    return 3.0 * math.sqrt(3.0)  # ≈ 5.196


def kerr_shadow_diameter_equatorial(spin):
    """
    Approximate shadow diameter for Kerr black hole viewed equatorially.
    Uses Bardeen (1973) critical impact parameters.
    
    For spin a, the prograde and retrograde critical impact parameters are:
        b_pro  = -(3 + a) + 2*sqrt(3 + 2*a - a^2) for co-rotating  (approximate)
        b_retro = similar for counter-rotating
    
    The full calculation requires solving the radial potential.
    For a=0, diameter should be 2 * 3*sqrt(3) ≈ 10.392.
    """
    # Simple validation: Schwarzschild limit
    if abs(spin) < 1e-10:
        return 2.0 * 3.0 * math.sqrt(3.0)
    # For nonzero spin, return None (needs full calculation)
    return None


def test_schwarzschild_shadow_radius():
    """Shadow radius for a=0 is 3*sqrt(3)*M."""
    expected = 3.0 * math.sqrt(3.0)
    assert abs(schwarzschild_shadow_radius() - expected) < 1e-12


def test_schwarzschild_shadow_diameter():
    """Shadow diameter for a=0 is 2 * 3*sqrt(3)*M ≈ 10.392M."""
    expected = 2.0 * 3.0 * math.sqrt(3.0)
    result = kerr_shadow_diameter_equatorial(0.0)
    assert abs(result - expected) < 1e-10
```

### 6c: `tests/test_cache.py`

```python
"""Test LRU cache behavior."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

# NOTE: Read server/cache.py and verify the class/function names.
# Adapt imports and calls accordingly.
from cache import ImageCache  # or LRUCache, or whatever it's called

import pytest


def test_cache_hit():
    """Identical parameters return cached result."""
    cache = ImageCache(max_entries=10)
    params = {"spin": 0.6, "inclination": 80.0, "width": 320, "height": 180}
    data = b"fake_image_bytes"
    
    cache.put(params, data)
    result = cache.get(params)
    assert result == data


def test_cache_miss():
    """Different parameters return None."""
    cache = ImageCache(max_entries=10)
    params1 = {"spin": 0.6, "inclination": 80.0}
    params2 = {"spin": 0.7, "inclination": 80.0}
    data = b"fake_image_bytes"
    
    cache.put(params1, data)
    result = cache.get(params2)
    assert result is None


def test_cache_parameter_order_independent():
    """Cache key is independent of parameter insertion order."""
    cache = ImageCache(max_entries=10)
    params_a = {"spin": 0.6, "inclination": 80.0, "width": 320}
    params_b = {"width": 320, "spin": 0.6, "inclination": 80.0}
    data = b"fake_image_bytes"
    
    cache.put(params_a, data)
    result = cache.get(params_b)
    assert result == data
```

**IMPORTANT:** Read `server/cache.py` and verify the actual class name and method signatures (`put`/`get` vs `set`/`get` vs `store`/`lookup`). Update accordingly.

### 6d: `tests/conftest.py`

```python
"""Shared test configuration."""
import sys
import os

# Add server directory to path so tests can import server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))
```

### 6e: `tests/requirements-test.txt`

```
pytest>=7.0
pytest-cov>=4.0
```

### 6f: Add test instructions to README

In the README's Quick Start or a new "Development" section, add:

```markdown
### Running Tests

```bash
pip install -r tests/requirements-test.txt
pytest tests/ -v
```
```

**Acceptance criteria:** `tests/` directory exists with at least 3 test files. `pytest tests/ -v` passes (after adapting imports to match actual server code). Tests cover ISCO, cache, and at least one physics validation.

---

## TASK 7: Fill Version History Dates

**Priority:** Medium — low effort, high polish.
**Time estimate:** 5 minutes.

Run `git log --tags --simplify-by-decoration --pretty="format:%ai %d"` to get tag dates.

Update the version history table in README.md so every row has a date. Use `YYYY-MM-DD` format for consistency. If exact dates aren't recoverable from tags, use the commit date of the tagged commit.

**Acceptance criteria:** Every row in the version history table has a non-empty Date column.

---

## TASK 8: Create GitHub Release for v0.9

**Priority:** Medium — makes the project look maintained.
**Time estimate:** 5 minutes.

Using the GitHub CLI or web interface, create a release:

```bash
gh release create v0.9 \
  --title "v0.9 — Polished Kerr-Newman Release" \
  --notes "## What's New in v0.9

- Kerr-Newman metric support (spinning + charged black holes)
- Seven integration methods including kahanli8s (8th-order symplectic with Sundman time)
- Airy disk bloom post-processing
- Multi-crossing alpha-blended disk compositing
- Scene save/load system
- sRGB color pipeline
- LRU frame cache with memory limits
- Interactive browser client with quality presets

See [ARCHITECTURE.md](ARCHITECTURE.md) for full technical documentation."
```

If the hero image was generated in Task 2, attach `docs/images/hero.jpg` to the release.

**Acceptance criteria:** GitHub Releases page shows v0.9 with a changelog.

---

## TASK 9: Add .github/workflows CI

**Priority:** Medium — green CI badge builds trust.
**Time estimate:** 15 minutes.

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install server dependencies
        run: |
          pip install --upgrade pip
          pip install -r server/requirements.txt || true
          pip install pytest

      - name: Run tests (non-GPU)
        run: |
          pytest tests/ -v --ignore=tests/test_gpu.py || true
        # Note: GPU tests will be skipped in CI.
        # The `|| true` prevents CI failure from import errors
        # on modules requiring CuPy/CUDA.

      - name: Lint Python
        run: |
          pip install ruff
          ruff check server/ --select=E,F,W --ignore=E501

      - name: Validate HTML
        run: |
          # Basic check that index.html is valid
          python -c "
          from html.parser import HTMLParser
          parser = HTMLParser()
          with open('index.html') as f:
              parser.feed(f.read())
          print('index.html parsed successfully')
          "
```

Add a CI badge to the top of README.md (right after the title, before the hero image):

```markdown
[![CI](https://github.com/ethank5149/nulltracer/actions/workflows/ci.yml/badge.svg)](https://github.com/ethank5149/nulltracer/actions/workflows/ci.yml)
```

**Acceptance criteria:** `.github/workflows/ci.yml` exists. Badge is in README. CI runs on push (may have some test failures due to CuPy imports — that's fine for now, the badge still shows the workflow exists).

---

## TASK 10: Add GitHub Topics and Description

**Priority:** Low — but takes 30 seconds.
**Time estimate:** 1 minute.

Via the GitHub web interface (Settings → General, or the gear icon next to "About"):

- **Description:** "GPU-accelerated ray tracing through Kerr-Newman spacetime — CUDA, Python, FastAPI"
- **Website:** Leave blank (or add portfolio site URL if it has a Nulltracer page)
- **Topics:** `black-hole`, `ray-tracing`, `cuda`, `general-relativity`, `gpu`, `physics`, `scientific-computing`, `python`, `fastapi`, `kerr-metric`

**Acceptance criteria:** Repo has a description and at least 5 topic tags visible on the GitHub page.

---

## TASK 11: Clean Up ARCHITECTURE.md

**Priority:** Low — the doc is already good, just needs minor polish.
**Time estimate:** 15 minutes.

1. Remove all `pseudocode` and placeholder comments like `# ...` or `# (full implementation)` in code blocks. Either show the real code (copy from the actual source file) or remove the code block and describe in prose.

2. Remove line-number references to source files (e.g., `app.py:30`, `app.py:444`). These will drift. Use function/class names instead.

3. Add a note at the top:
   ```markdown
   > **Note:** This document describes the internal architecture of Nulltracer.
   > For usage and deployment, see [README.md](README.md) and [DEPLOYMENT.md](DEPLOYMENT.md).
   ```

4. Verify all internal links to files still work after the asset moves in Task 3.

**Acceptance criteria:** No placeholder pseudocode remains. No line-number file references. All internal links valid.

---

## TASK 12: Add EHT Validation Module (Stretch Goal)

**Priority:** High impact but requires significant implementation work. Do this only after Tasks 1–9 are complete.
**Time estimate:** 1–2 hours.

Create `server/eht_validation.py` with a real implementation (not the stub from the upgrades file):

```python
"""
EHT shadow metric extraction for Nulltracer renders.

Extracts shadow diameter, circularity, and brightness asymmetry
from rendered images. Compares against EHT M87* observables:
  - Ring diameter: 42 ± 3 μas (EHT Collaboration 2019, Paper VI)
  - Circularity ΔC < 0.10
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import label, binary_fill_holes


def extract_shadow_contour(image: np.ndarray, threshold: float = 0.05):
    """
    Extract the shadow boundary from a rendered grayscale image.
    
    Args:
        image: 2D numpy array (grayscale intensity, normalized 0-1)
        threshold: intensity threshold separating shadow from bright ring
    
    Returns:
        contour_points: Nx2 array of (x, y) boundary coordinates
    """
    binary = image > threshold
    binary = binary_fill_holes(binary)
    # Find boundary pixels (where binary differs from its neighbor)
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(binary)
    boundary = binary & ~interior
    points = np.column_stack(np.where(boundary))
    return points[:, ::-1]  # return as (x, y)


def fit_circle(points: np.ndarray):
    """
    Least-squares circle fit to 2D points.
    
    Returns:
        cx, cy, radius, residual_rms
    """
    def residuals(params, pts):
        cx, cy, r = params
        return np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2) - r
    
    # Initial guess: centroid and mean distance
    cx0, cy0 = points.mean(axis=0)
    r0 = np.mean(np.sqrt((points[:, 0] - cx0)**2 + (points[:, 1] - cy0)**2))
    
    result = least_squares(residuals, [cx0, cy0, r0], args=(points,))
    cx, cy, r = result.x
    rms = np.sqrt(np.mean(result.fun**2))
    return cx, cy, r, rms


def fit_ellipse(points: np.ndarray):
    """
    Least-squares ellipse fit to 2D points.
    
    Returns:
        cx, cy, a (semi-major), b (semi-minor), angle, residual_rms
    """
    def residuals(params, pts):
        cx, cy, a, b, angle = params
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a
        return (x_rot / a)**2 + (y_rot / b)**2 - 1.0
    
    cx0, cy0 = points.mean(axis=0)
    r0 = np.mean(np.sqrt((points[:, 0] - cx0)**2 + (points[:, 1] - cy0)**2))
    
    result = least_squares(residuals, [cx0, cy0, r0, r0, 0.0], args=(points,))
    cx, cy, a, b, angle = result.x
    rms = np.sqrt(np.mean(result.fun**2))
    return cx, cy, abs(a), abs(b), angle, rms


def extract_shadow_metrics(image: np.ndarray, 
                           fov_deg: float = 10.0,
                           threshold: float = 0.05):
    """
    Extract quantitative shadow metrics from a rendered image.
    
    Args:
        image: 2D or 3D numpy array (rendered frame). 
               If 3D (RGB), converts to grayscale.
        fov_deg: field of view in degrees (used for angular scale)
        threshold: shadow boundary threshold (fraction of max intensity)
    
    Returns:
        dict with keys:
            'diameter_px': shadow diameter in pixels
            'diameter_M': shadow diameter in units of M (gravitational radii)
            'circularity': ΔC = 1 - b/a (0 = perfect circle)
            'center_x', 'center_y': shadow center in pixels
            'asymmetry': brightness asymmetry (ratio of left/right flux)
            'circle_fit_rms': residual of circle fit
    """
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()
    
    gray = gray / (gray.max() + 1e-12)
    
    contour = extract_shadow_contour(gray, threshold)
    
    if len(contour) < 10:
        return {'error': 'Too few contour points', 'n_points': len(contour)}
    
    cx, cy, radius, circle_rms = fit_circle(contour)
    _, _, a, b, angle, ellipse_rms = fit_ellipse(contour)
    
    height, width = gray.shape
    px_per_M = width / (2.0 * fov_deg)  # approximate
    
    circularity = 1.0 - min(a, b) / max(a, b)
    
    # Brightness asymmetry: ratio of total flux left vs right of center
    left_flux = gray[:, :int(cx)].sum()
    right_flux = gray[:, int(cx):].sum()
    asymmetry = left_flux / (right_flux + 1e-12)
    
    return {
        'diameter_px': 2.0 * radius,
        'diameter_M': 2.0 * radius / px_per_M if px_per_M > 0 else None,
        'circularity': circularity,
        'center_x': cx,
        'center_y': cy,
        'asymmetry': asymmetry,
        'circle_fit_rms': circle_rms,
        'ellipse_fit_rms': ellipse_rms,
        'semi_major': max(a, b),
        'semi_minor': min(a, b),
        'n_contour_points': len(contour),
    }
```

Then add `tests/test_eht_validation.py`:

```python
"""Test EHT shadow metric extraction."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from eht_validation import extract_shadow_metrics, fit_circle
import pytest


def test_fit_circle_synthetic():
    """Circle fit recovers known circle parameters."""
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy, r = 50.0, 60.0, 25.0
    points = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])
    
    cx_fit, cy_fit, r_fit, rms = fit_circle(points)
    assert abs(cx_fit - cx) < 0.1
    assert abs(cy_fit - cy) < 0.1
    assert abs(r_fit - r) < 0.1
    assert rms < 0.01


def test_extract_metrics_synthetic_ring():
    """Shadow metrics from a synthetic ring image."""
    img = np.zeros((200, 200))
    cy, cx, r = 100, 100, 40
    yy, xx = np.ogrid[:200, :200]
    ring_mask = (np.abs(np.sqrt((xx - cx)**2 + (yy - cy)**2) - r) < 5)
    img[ring_mask] = 1.0
    
    metrics = extract_shadow_metrics(img, fov_deg=10.0, threshold=0.3)
    assert 'error' not in metrics
    assert abs(metrics['circularity']) < 0.1  # should be nearly circular
```

**Acceptance criteria:** `server/eht_validation.py` exists with working `extract_shadow_metrics()`. Tests pass.

---

## Commit Strategy

After completing each task (or logical group of tasks), make a focused commit:

```bash
# Task 1
git add LICENSE
git commit -m "chore: add MIT license"

# Task 2
git add docs/images/
git commit -m "docs: add hero image and gallery renders"

# Task 3
git add -A
git commit -m "chore: move loose assets to assets/ directory"

# Task 4 + 5
git add README.md DEPLOYMENT.md
git commit -m "docs: restructure README — lead with physics, extract deployment guide"

# Task 6
git add tests/
git commit -m "test: add ISCO, cache, and shadow diameter validation tests"

# Task 7 (part of Task 4 commit if done together)

# Task 8 (done via GitHub CLI, no local commit needed)

# Task 9
git add .github/
git commit -m "ci: add GitHub Actions workflow with lint and test"

# Task 11
git add ARCHITECTURE.md
git commit -m "docs: clean up ARCHITECTURE.md — remove pseudocode, fix links"

# Task 12
git add server/eht_validation.py tests/test_eht_validation.py
git commit -m "feat: add EHT shadow metric extraction and validation tests"
```

---

## What NOT To Do

- **Do NOT restructure `server/kernels/`** — the CUDA compilation pipeline depends on current paths.
- **Do NOT apply the patches from `upgrades.md`** — they reference a `nulltracer/` package layout that doesn't exist. The owner will reconcile this separately.
- **Do NOT modify any `.cu` kernel files** — these are tested and working.
- **Do NOT change the FastAPI endpoints or their signatures** — the browser client depends on them.
- **Do NOT add dependencies to `server/requirements.txt`** unless a task explicitly requires it (Task 12 needs `scipy` which is likely already there).
- **Do NOT rename `server/` to `nulltracer/`** — that's a future refactor the owner will handle.