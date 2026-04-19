# Nulltracer Migration Plan: Server → Python Package + Notebook

**Target repo:** `https://github.com/ethank5149/nulltracer` (or local Gitea equivalent)
**Goal:** Restructure Nulltracer from a server-only FastAPI application into an importable Python package (`nulltracer`) with the FastAPI server as an optional thin wrapper. The notebook becomes the primary showcase; `pip install -e .` becomes the primary setup path.

**Prerequisites:** The publication-readiness tasks (LICENSE, README rewrite, tests, CI) should ideally be done first or in parallel. This plan assumes the current repo structure is still rooted in `server/`.

---

## Architecture Overview: Before and After

### BEFORE (current)

```
nulltracer/                     ← repo root
├── server/                     ← ALL Python + CUDA lives here
│   ├── app.py                  ← FastAPI (the only entry point)
│   ├── renderer.py             ← CuPy kernel compilation + launch
│   ├── isco.py                 ← ISCO calculation
│   ├── cache.py                ← LRU image cache (server-only concern)
│   ├── bloom.py                ← Airy disk bloom
│   ├── scenes.py               ← Scene management (server-only concern)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── __init__.py             ← empty
│   └── kernels/                ← CUDA .cu files
│       ├── geodesic_base.cu
│       ├── backgrounds.cu
│       ├── disk.cu
│       ├── ray_trace.cu
│       └── integrators/*.cu
├── js/                         ← Browser client
├── index.html
├── styles.css
├── nulltracer.ipynb            ← Notebook (can't import anything)
└── ...
```

Usage: `docker run --gpus all ... ` → open browser → POST /render

### AFTER (target)

```
nulltracer/                     ← repo root (CLEAN — no loose assets)
├── nulltracer/                 ← importable Python package
│   ├── __init__.py             ← public API exports
│   ├── renderer.py             ← CuPy kernel compilation + launch (no FastAPI)
│   ├── isco.py                 ← ISCO calculation (unchanged)
│   ├── bloom.py                ← Airy disk bloom (unchanged)
│   ├── eht_validation.py       ← EHT shadow metrics (new or from recent work)
│   ├── _params.py              ← RenderParams ctypes struct (extracted from renderer.py)
│   └── kernels/                ← CUDA .cu files (moved here)
│       ├── geodesic_base.cu
│       ├── backgrounds.cu
│       ├── disk.cu
│       ├── ray_trace.cu
│       └── integrators/*.cu
├── server/                     ← OPTIONAL thin FastAPI wrapper
│   ├── app.py                  ← imports from nulltracer package
│   ├── cache.py                ← server-only LRU cache
│   ├── scenes.py               ← server-only scene management
│   ├── scenes/                 ← built-in scene JSONs
│   ├── Dockerfile
│   ├── docker-compose.yml      ← moved from root
│   ├── deploy.sh               ← moved from root
│   └── requirements.txt
├── web/                        ← browser client (companion to server)
│   ├── index.html              ← moved from root
│   ├── styles.css              ← moved from root
│   ├── bench.html              ← moved from root
│   └── js/                     ← moved from root
│       ├── main.js
│       ├── server-client.js
│       └── ui-controller.js
├── notebooks/
│   └── nulltracer.ipynb        ← PRIMARY showcase notebook
├── tests/                      ← package tests
├── docs/
│   ├── images/                 ← hero image, gallery renders
│   ├── assets/                 ← icons, logos
│   └── internal/               ← FEATURE-PLAN.md (archived)
├── pyproject.toml              ← package metadata + dependencies
├── README.md                   ← leads with package usage, not server
├── ARCHITECTURE.md             ← developer documentation
├── DEPLOYMENT.md               ← server deployment guide
├── LICENSE                     ← MIT
└── .gitignore
```

Usage: `pip install -e .` → `from nulltracer import render_frame` → numpy array

---

## Execution Plan

### TASK 0: Create a Migration Branch

**Do this first. Do not work on master/main.**

```bash
git checkout -b refactor/package-migration
```

All work happens on this branch. Merge to master only after everything passes.

---

### TASK 1: Create the Package Directory and Move Core Modules

**Time estimate:** 15 minutes.
**Risk:** Medium — path references in renderer.py will break. That's expected and fixed in Task 3.

#### 1a: Create `nulltracer/` package directory

```bash
mkdir -p nulltracer/kernels/integrators
```

#### 1b: Copy (not move) core Python modules from server/ to nulltracer/

**COPY, don't move yet.** The server needs to keep working during the migration.

```bash
cp server/renderer.py    nulltracer/renderer.py
cp server/isco.py        nulltracer/isco.py
cp server/bloom.py       nulltracer/bloom.py
cp server/__init__.py    nulltracer/__init__.py
```

If `server/eht_validation.py` exists (from the publication plan), copy that too:
```bash
cp server/eht_validation.py nulltracer/eht_validation.py 2>/dev/null || true
```

#### 1c: Copy ALL kernel files

```bash
cp server/kernels/geodesic_base.cu    nulltracer/kernels/
cp server/kernels/backgrounds.cu      nulltracer/kernels/
cp server/kernels/disk.cu             nulltracer/kernels/
cp server/kernels/ray_trace.cu        nulltracer/kernels/
cp server/kernels/integrators/*.cu    nulltracer/kernels/integrators/
```

#### 1d: Create `nulltracer/kernels/__init__.py`

```python
# Empty — marks kernels/ as a Python subpackage for resource discovery.
```

**Acceptance criteria:** `nulltracer/` directory exists with all .py and .cu files. `server/` is UNCHANGED at this point. Both directories have identical copies of the shared files.

---

### TASK 2: Create `nulltracer/__init__.py` with Public API

**Time estimate:** 10 minutes.

Replace the empty `nulltracer/__init__.py` with:

```python
"""
Nulltracer — GPU-accelerated ray tracing through Kerr-Newman spacetime.

Usage:
    from nulltracer import render_frame, isco_radius

    image = render_frame(spin=0.94, inclination=30, fov=10)
    # image is a numpy array of shape (height, width, 3), dtype uint8
"""

__version__ = "0.9.0"

from .renderer import render_frame, render_classify, trace_single_ray
from .isco import isco_kerr, isco_kn
from .bloom import airy_disk_bloom

# Optional imports — don't fail if scipy not installed
try:
    from .eht_validation import extract_shadow_metrics
except ImportError:
    pass

__all__ = [
    "render_frame",
    "render_classify",
    "trace_single_ray",
    "isco_kerr",
    "isco_kn",
    "airy_disk_bloom",
    "extract_shadow_metrics",
]
```

**IMPORTANT:** The function names above (`render_frame`, `isco_kerr`, `isco_kn`, etc.) are GUESSES based on the architecture doc. Before finalizing this file:

1. Read `nulltracer/renderer.py` and find the actual function/method that performs a render. It's probably a method on the `CudaRenderer` class. You'll need to either expose a module-level wrapper function or export the class.
2. Read `nulltracer/isco.py` and find the actual function names. They might be `iscoJS`, `iscoKN`, `isco_kerr`, `compute_isco`, etc.
3. Read `nulltracer/bloom.py` and find the actual function name.

Update ALL imports to match the real names. If the renderer exposes a class rather than a function, the wrapper approach is described in Task 3.

**Acceptance criteria:** `nulltracer/__init__.py` exports a working public API. `from nulltracer import render_frame` doesn't error (assuming CuPy is available).

---

### TASK 3: Refactor `nulltracer/renderer.py` — Decouple from FastAPI

**Time estimate:** 30–45 minutes. This is the most important and most delicate task.

The current `server/renderer.py` contains the `CudaRenderer` class. This class handles:
1. Kernel source loading and `#include` resolution
2. CuPy RawKernel compilation and caching
3. GPU memory allocation and kernel launch
4. Output buffer transfer to CPU
5. Return as numpy array

Items 1–5 are ALL reusable in the package. What needs to change:

#### 3a: Fix kernel file path resolution

The current renderer finds `.cu` files relative to `server/kernels/`. This must change to find them relative to the `nulltracer` package.

Find the code in `renderer.py` that constructs kernel file paths. It likely looks something like:

```python
kernel_dir = os.path.join(os.path.dirname(__file__), "kernels")
```

or

```python
KERNEL_DIR = Path(__file__).parent / "kernels"
```

**This should already work correctly** after the copy, because `__file__` will resolve to `nulltracer/renderer.py` and `kernels/` is a sibling directory. Verify this by inspection. If the path is hardcoded to `"server/kernels"` anywhere, change it to be relative to `__file__`.

#### 3b: Add a module-level convenience function

The `CudaRenderer` class likely requires initialization (`renderer.initialize()`) before use. For notebook users, we want a simpler API. Add to the bottom of `nulltracer/renderer.py`:

```python
# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default_renderer = None


def _get_renderer():
    """Lazily initialize and return the default CudaRenderer singleton."""
    global _default_renderer
    if _default_renderer is None:
        _default_renderer = CudaRenderer()
        _default_renderer.initialize()
    return _default_renderer


def render_frame(
    spin: float = 0.6,
    charge: float = 0.0,
    inclination: float = 80.0,
    fov: float = 8.0,
    width: int = 1280,
    height: int = 720,
    method: str = "rkdp8",
    steps: int = 200,
    step_size: float = 0.3,
    obs_dist: int = 40,
    bg_mode: int = 0,
    show_disk: bool = True,
    show_grid: bool = False,
    disk_temp: float = 1.0,
    star_layers: int = 3,
    srgb_output: bool = True,
    disk_alpha: float = 1.0,
    disk_max_crossings: int = 1,
    bloom_enabled: bool = False,
    bloom_radius: float = 1.0,
    **kwargs,
) -> "numpy.ndarray":
    """
    Render a black hole frame and return as a numpy array.

    Returns:
        numpy.ndarray: RGB image of shape (height, width, 3), dtype uint8.
    """
    renderer = _get_renderer()
    params = {
        "spin": spin,
        "charge": charge,
        "inclination": inclination,
        "fov": fov,
        "width": width,
        "height": height,
        "method": method,
        "steps": steps,
        "step_size": step_size,
        "obs_dist": obs_dist,
        "bg_mode": bg_mode,
        "show_disk": show_disk,
        "show_grid": show_grid,
        "disk_temp": disk_temp,
        "star_layers": star_layers,
        "srgb_output": srgb_output,
        "disk_alpha": disk_alpha,
        "disk_max_crossings": disk_max_crossings,
        "bloom_enabled": bloom_enabled,
        "bloom_radius": bloom_radius,
        **kwargs,
    }
    raw_bytes = renderer.render_frame(params)

    # Convert raw bytes to numpy array if not already
    import numpy as np
    if isinstance(raw_bytes, np.ndarray):
        return raw_bytes
    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width, 3)
```

**CRITICAL ADAPTATION REQUIRED:** The above is a template. You MUST:

1. Read the actual `CudaRenderer.render_frame()` method signature and return type.
2. Check what parameters it expects (dict? individual args? a ctypes struct?).
3. Check what it returns (raw bytes? numpy array? PIL Image?).
4. Adapt the wrapper function to match. The goal is: user passes keyword arguments, gets back a numpy array.

If `CudaRenderer.render_frame()` already returns a numpy array, the conversion at the bottom is unnecessary.

#### 3c: Add `render_classify` and `trace_single_ray` wrappers if applicable

If `renderer.py` or `ray_trace.cu` support classification mode (returning ray fate — escape/horizon/disk — instead of color), add a similar wrapper called `render_classify`.

If the `/ray` endpoint logic is in `renderer.py` (single-ray tracing with trajectory output), add a `trace_single_ray` wrapper.

If these don't exist yet in the renderer class, skip them and remove from `__init__.py`.

#### 3d: Remove any FastAPI-specific imports

Search `nulltracer/renderer.py` for:
- `from fastapi import ...`
- `import asyncio`
- `asyncio.Lock`
- `Pydantic` models
- `PIL` / `Pillow` image encoding (JPEG/WebP)

Remove or guard these. The package version should not depend on FastAPI or Pillow. The render function returns a numpy array; encoding to JPEG is a server concern.

If asyncio.Lock is used to serialize GPU access, replace it with `threading.Lock` (which works in both sync and async contexts):

```python
import threading
_gpu_lock = threading.Lock()

def render_frame(self, params):
    with _gpu_lock:
        # ... existing render logic ...
```

**Acceptance criteria:** `nulltracer/renderer.py` can be imported without FastAPI installed. `render_frame()` takes keyword arguments and returns a numpy array. GPU initialization happens lazily on first call.

---

### TASK 4: Create `nulltracer/_params.py` — Extract RenderParams Struct

**Time estimate:** 10 minutes.

If the `RenderParams` ctypes struct definition is embedded in `renderer.py`, extract it into its own file `nulltracer/_params.py`. This keeps the struct definition clean and importable.

```python
"""RenderParams ctypes structure for CUDA kernel interface."""
import ctypes


class RenderParams(ctypes.Structure):
    """
    C-compatible parameter struct passed to CUDA kernels.
    
    All fields are float64 for alignment consistency between
    Python ctypes and CUDA. Integer values are stored as doubles
    and cast within the kernel.
    """
    _fields_ = [
        # Copy the EXACT field list from the current renderer.py.
        # Do not modify field names, types, or order — the CUDA
        # kernels depend on this struct layout being identical.
        ("width", ctypes.c_double),
        ("height", ctypes.c_double),
        ("spin", ctypes.c_double),
        ("charge", ctypes.c_double),
        ("incl", ctypes.c_double),
        ("fov", ctypes.c_double),
        ("phi0", ctypes.c_double),
        ("isco", ctypes.c_double),
        ("steps", ctypes.c_double),
        ("obs_dist", ctypes.c_double),
        ("esc_radius", ctypes.c_double),
        ("disk_outer", ctypes.c_double),
        ("step_size", ctypes.c_double),
        ("bg_mode", ctypes.c_double),
        ("star_layers", ctypes.c_double),
        ("show_disk", ctypes.c_double),
        ("show_grid", ctypes.c_double),
        ("disk_temp", ctypes.c_double),
        ("srgb_output", ctypes.c_double),
        ("disk_alpha", ctypes.c_double),
        ("disk_max_crossings", ctypes.c_double),
        ("bloom_enabled", ctypes.c_double),
        # ... ADD ALL FIELDS from the original struct.
        # The field order MUST match the CUDA struct in geodesic_base.cu.
    ]
```

**CRITICAL:** Copy the fields EXACTLY from the existing code. Do not add, remove, reorder, or rename fields. The CUDA kernels read this struct by byte offset. Any mismatch = garbage rendering or segfaults.

Then update `nulltracer/renderer.py` to import from `_params`:

```python
from ._params import RenderParams
```

If the struct is already clean and self-contained within renderer.py and you'd rather not split it, skip this task. It's a nice-to-have for code organization, not a requirement.

**Acceptance criteria:** If done, `RenderParams` is in `_params.py` and imported by `renderer.py`. Struct field order is byte-identical to the original.

---

### TASK 5: Create `pyproject.toml`

**Time estimate:** 10 minutes.

Create `pyproject.toml` at the repo root:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "nulltracer"
version = "0.9.0"
description = "GPU-accelerated ray tracing through Kerr-Newman spacetime"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [
    {name = "Ethan Knox", email = "ethank5149@gmail.com"},
]
keywords = [
    "black-hole",
    "ray-tracing",
    "cuda",
    "general-relativity",
    "gpu",
    "scientific-computing",
    "kerr-metric",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Topic :: Scientific/Engineering :: Visualization",
]

dependencies = [
    "numpy>=1.24",
    "cupy-cuda12x>=13.0",
]

[project.optional-dependencies]
analysis = [
    "scipy>=1.10",
    "matplotlib>=3.7",
]
server = [
    "fastapi>=0.104",
    "uvicorn[standard]>=0.24",
    "Pillow>=10.0",
    "pydantic>=2.0",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
]

[project.urls]
Homepage = "https://github.com/ethank5149/nulltracer"
Repository = "https://github.com/ethank5149/nulltracer"

[tool.setuptools.packages.find]
include = ["nulltracer*"]

[tool.setuptools.package-data]
nulltracer = ["kernels/*.cu", "kernels/integrators/*.cu"]

[tool.ruff]
line-length = 120
select = ["E", "F", "W"]
ignore = ["E501"]
```

**Key detail:** The `[tool.setuptools.package-data]` section ensures the `.cu` kernel files are included when the package is installed. Without this, `pip install -e .` won't copy the kernels and the renderer will fail to find them.

**Acceptance criteria:** `pip install -e .` from the repo root installs the `nulltracer` package. `python -c "import nulltracer; print(nulltracer.__version__)"` prints `0.9.0`.

---

### TASK 6: Gut `server/` — Thin Wrapper Only

**Time estimate:** 30–40 minutes. This is where the old architecture gets torn down.

The principle: after this task, `server/` contains ONLY the things that are about being an HTTP server. Zero physics, zero rendering, zero CUDA. Those all live in the `nulltracer` package now.

#### 6a: Delete ALL duplicated / now-obsolete modules from server/

```bash
# Physics and rendering — now in nulltracer/
rm server/renderer.py
rm server/isco.py
rm server/bloom.py
rm server/eht_validation.py  2>/dev/null || true

# Kernels — now in nulltracer/kernels/
rm -rf server/kernels/

# Empty init — server is not a package anymore, it's a script directory
rm server/__init__.py
```

**What remains in server/ after this step:**
- `app.py` — FastAPI entry point (will be rewritten below)
- `cache.py` — server-only LRU image cache
- `scenes.py` — server-only scene management
- `scenes/` — built-in scene JSON files
- `Dockerfile` — container definition (will be rewritten below)
- `requirements.txt` — server-only deps (will be rewritten below)

**Nothing else.** If there are other `.py` files in `server/` not listed above (e.g. `models.py`, helper scripts), inspect each one:
- If it contains physics/math/rendering logic → move it to `nulltracer/`
- If it contains HTTP/caching/encoding logic → keep in `server/`
- If it's dead code → delete it

#### 6b: Rewrite `server/app.py` to import from the package

The core change is replacing:
```python
from renderer import CudaRenderer
from isco import isco_kerr
```

with:
```python
from nulltracer.renderer import CudaRenderer
from nulltracer.isco import isco_kerr, isco_kn
from nulltracer.bloom import airy_disk_bloom
```

Search the ENTIRE file for local imports (`from renderer`, `from isco`, `from bloom`, `import renderer`, etc.) and update every one. Miss even one and the server crashes on startup.

The FastAPI endpoints (`/render`, `/ray`, `/health`, `/scenes`) stay in `server/app.py`. They continue to:
1. Accept HTTP requests
2. Validate parameters (Pydantic)
3. Call `nulltracer` package functions
4. Encode results as JPEG/WebP (Pillow)
5. Manage the LRU cache
6. Return HTTP responses

**Do NOT move FastAPI logic into the package.** The package returns numpy arrays. The server encodes them into images. This separation is the whole point.

#### 6c: Update `server/requirements.txt`

Replace the current requirements with:

```
# Core rendering — installed via the nulltracer package
# (pip install -e .. or pip install .. from this directory)

# Server-only dependencies
fastapi>=0.104
uvicorn[standard]>=0.24
Pillow>=10.0
pydantic>=2.0
```

The nulltracer package itself is installed via `pyproject.toml` at the repo root. The server requirements only list the server-specific dependencies that are NOT in the core package.

#### 6d: Update `server/Dockerfile`

The Dockerfile currently copies only `server/` into the container. It now needs access to the `nulltracer/` package too. Update the Dockerfile:

```dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the package first (for layer caching)
COPY pyproject.toml .
COPY nulltracer/ nulltracer/

# Install the package
RUN pip3 install --no-cache-dir .

# Copy server-specific files
COPY server/ server/
RUN pip3 install --no-cache-dir -r server/requirements.txt

EXPOSE 8420
WORKDIR /app/server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8420", "--workers", "1"]
```

**Acceptance criteria:** `server/` contains exactly: `app.py`, `cache.py`, `scenes.py`, `scenes/`, `Dockerfile`, `requirements.txt`. No `.cu` files, no physics modules, no `__init__.py`. `docker build` succeeds. `POST /render` returns an image. All physics logic is imported from the `nulltracer` package.

---

### TASK 6B: Clean Up Repo Root — Remove Dead Weight

**Time estimate:** 15–20 minutes. This is the housekeeping that makes the repo look professional.

After the migration, the repo root is still cluttered with files that were part of the server-centric architecture. These need to be relocated or removed.

#### 6B-a: Relocate browser client into `web/`

The browser client (`index.html`, `styles.css`, `js/`) is a companion to the server, not to the Python package. Move it out of the repo root:

```bash
mkdir -p web
git mv index.html web/
git mv styles.css web/
git mv js/ web/js/
```

Then update `server/Dockerfile` and `docker-compose.yml` if they reference these files. Also update `DEPLOYMENT.md` (if it exists) to reflect the new paths.

**NOTE:** If Caddy or the Docker setup serves static files from the repo root, the Caddyfile `root` directive must change from `/srv/nulltracer` to `/srv/nulltracer/web`. Check `deploy.sh` and any Caddyfile references.

#### 6B-b: Relocate deployment files into `server/`

These are server deployment concerns, not package concerns:

```bash
git mv docker-compose.yml server/
git mv deploy.sh server/
git mv nulltracer-renderer.xml server/  2>/dev/null || true
```

If a `Caddyfile` or `Caddyfile.current` exists at root, move it too:
```bash
git mv Caddyfile* server/  2>/dev/null || true
```

#### 6B-c: Relocate or remove loose assets

```bash
# Icons — move to docs/ or assets/ (keep for README/web use)
mkdir -p docs/assets
git mv nulltracer-icon.png docs/assets/  2>/dev/null || true
git mv nulltracer-icon.svg docs/assets/  2>/dev/null || true

# Reference image — this is a development reference, not a deliverable
# Delete it unless it's used in the README or notebook
git rm schwarzschild-black-hole-nasa-labeled-reference.jpg  2>/dev/null || true

# Benchmark page — move to web/ (it's a browser thing) or delete
git mv bench.html web/  2>/dev/null || true
```

**Before deleting anything**, grep the entire repo to verify nothing references it:
```bash
grep -r "schwarzschild-black-hole" --include="*.md" --include="*.html" --include="*.py" .
grep -r "bench.html" --include="*.md" --include="*.html" --include="*.py" .
```

If something does reference it, update the reference before deleting.

#### 6B-d: Remove or archive planning docs

```bash
# FEATURE-PLAN.md — the features are implemented. This is internal planning, not user docs.
# Either delete it or move it to docs/internal/
mkdir -p docs/internal
git mv FEATURE-PLAN.md docs/internal/  

# ARCHITECTURE.md — keep at root (it's developer-facing documentation)
# But it needs updating in Task 10 to reflect the new structure.
```

#### 6B-e: Remove `.forgejo/workflows/` if `.github/workflows/` exists

If the publication plan added `.github/workflows/ci.yml`, the Forgejo workflow is redundant on the GitHub mirror. However, if you're still using Gitea as primary, keep both. Make a decision:

- **Gitea is primary, GitHub is mirror:** Keep `.forgejo/`, keep `.github/`
- **GitHub is primary:** `git rm -rf .forgejo/`
- **Both are equal:** Keep both

#### 6B-f: Verify clean repo root

After all moves/deletes, the repo root should contain ONLY:

```
nulltracer/                 ← repo root
├── nulltracer/             ← the Python package (the main thing)
│   ├── __init__.py
│   ├── renderer.py
│   ├── isco.py
│   ├── bloom.py
│   ├── _params.py
│   ├── eht_validation.py
│   └── kernels/
│       └── integrators/
├── server/                 ← optional FastAPI wrapper (thin)
│   ├── app.py
│   ├── cache.py
│   ├── scenes.py
│   ├── scenes/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── deploy.sh
│   └── requirements.txt
├── web/                    ← browser client (companion to server)
│   ├── index.html
│   ├── styles.css
│   ├── bench.html
│   └── js/
├── notebooks/              ← primary showcase
│   └── nulltracer.ipynb
├── tests/                  ← package tests
├── docs/                   ← documentation assets
│   ├── images/
│   ├── assets/
│   └── internal/
├── .github/workflows/      ← CI (if applicable)
├── .forgejo/workflows/     ← Gitea CI (if applicable)
├── pyproject.toml          ← package metadata
├── README.md               ← leads with package usage
├── ARCHITECTURE.md         ← developer docs
├── DEPLOYMENT.md           ← server deployment guide
├── LICENSE                 ← MIT
└── .gitignore
```

**That's it.** No loose images, no XML templates, no deployment scripts, no planning docs at root. Every file is in a directory that explains what it is.

**Acceptance criteria:** `ls` at repo root shows only the items above (directories + 5–6 files). No orphaned assets, no server-specific files at root level.

---

### TASK 7: Move and Update the Notebook

**Time estimate:** 15 minutes.

#### 7a: Move the notebook

```bash
mkdir -p notebooks
git mv nulltracer.ipynb notebooks/nulltracer.ipynb
```

#### 7b: Rewrite notebook to use the package API

Open `notebooks/nulltracer.ipynb` and restructure it. The notebook should demonstrate the package, not the server. Here is the target cell structure:

**Cell 1 — Setup:**
```python
import nulltracer as nt
import numpy as np
import matplotlib.pyplot as plt

print(f"Nulltracer v{nt.__version__}")
```

**Cell 2 — Basic Render:**
```python
image = nt.render_frame(spin=0.94, inclination=30, fov=10, disk_temp=0.6)
plt.figure(figsize=(16, 9))
plt.imshow(image)
plt.axis("off")
plt.title("Kerr Black Hole — a = 0.94, θ = 30°")
plt.tight_layout()
plt.show()
```

**Cell 3 — Parameter Exploration:**
```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
spins = [0.0, 0.3, 0.6, 0.9, 0.95, 0.998]

for ax, a in zip(axes.flat, spins):
    img = nt.render_frame(spin=a, inclination=80, fov=8, width=640, height=360)
    ax.imshow(img)
    ax.set_title(f"a = {a}")
    ax.axis("off")

plt.suptitle("Effect of Spin Parameter on Black Hole Appearance", fontsize=14)
plt.tight_layout()
plt.show()
```

**Cell 4 — ISCO Calculation:**
```python
spins = np.linspace(0, 0.998, 100)
iscos = [nt.isco_kerr(a) for a in spins]

plt.figure(figsize=(8, 5))
plt.plot(spins, iscos, "k-", linewidth=2)
plt.xlabel("Spin parameter a/M")
plt.ylabel("ISCO radius r/M")
plt.title("Innermost Stable Circular Orbit vs. Spin")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Cell 5 — Integrator Comparison (optional):**
```python
methods = ["rk4", "rkdp8", "tao_yoshida4", "kahanli8s"]
fig, axes = plt.subplots(1, len(methods), figsize=(20, 5))

for ax, method in zip(axes, methods):
    img = nt.render_frame(spin=0.998, inclination=85, method=method, 
                          width=640, height=360, steps=200)
    ax.imshow(img)
    ax.set_title(method)
    ax.axis("off")

plt.suptitle("Integrator Comparison at Extreme Parameters (a=0.998, θ=85°)")
plt.tight_layout()
plt.show()
```

**Cell 6 — EHT Validation (if eht_validation.py exists):**
```python
try:
    from nulltracer import extract_shadow_metrics
    
    img = nt.render_frame(spin=0.94, inclination=17, fov=10,
                          width=1024, height=1024, show_disk=False)
    metrics = extract_shadow_metrics(img, fov_deg=10.0)
    
    print("Shadow Metrics:")
    print(f"  Diameter:    {metrics['diameter_M']:.2f} M")
    print(f"  Circularity: ΔC = {metrics['circularity']:.4f}")
    print(f"  Asymmetry:   {metrics['asymmetry']:.3f}")
    print(f"  Schwarzschild prediction: {2 * 3 * np.sqrt(3):.2f} M")
except ImportError:
    print("EHT validation module not available (install scipy)")
```

**IMPORTANT:** Adapt all function names to match what actually exists in the package after Task 3. The above is a template assuming the API described in Task 2. If `render_frame` has different parameter names, update accordingly.

**Acceptance criteria:** `notebooks/nulltracer.ipynb` runs top-to-bottom with `pip install -e .` as the only setup. No server required. Produces publication-quality figures.

---

### TASK 8: Update Tests to Import from Package

**Time estimate:** 10 minutes.

If `tests/` exists from the publication plan, update all imports from:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))
from isco import ...
```

to:
```python
from nulltracer.isco import ...
from nulltracer.renderer import render_frame
```

Remove all `sys.path` hacks. The package should be importable via normal Python imports once `pip install -e .` is done.

Update `tests/conftest.py` accordingly — it should not need any path manipulation.

**Acceptance criteria:** `pip install -e . && pytest tests/ -v` passes with no sys.path hacks.

---

### TASK 9: Update CI Workflow

**Time estimate:** 5 minutes.

If `.github/workflows/ci.yml` exists, update the test step:

```yaml
      - name: Install package
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest tests/ -v
```

This replaces any previous approach of installing `server/requirements.txt` separately.

**Acceptance criteria:** CI installs the package via `pip install -e ".[dev]"` and runs tests against the installed package.

---

### TASK 10: Update README, ARCHITECTURE.md, and DEPLOYMENT.md

**Time estimate:** 20 minutes.

#### 10a: README Quick Start

Replace the server-first Quick Start with a package-first approach:

```markdown
## Quick Start

### As a Python Package (recommended for analysis)

```bash
pip install -e .
```

```python
import nulltracer as nt

image = nt.render_frame(spin=0.94, inclination=30, fov=10)
# image is a numpy array (height, width, 3), dtype uint8
```

See `notebooks/nulltracer.ipynb` for a full walkthrough.

### As a Web Server (for interactive exploration)

```bash
pip install -e ".[server]"
cd server
uvicorn app:app --host 0.0.0.0 --port 8420
```

Open `web/index.html` in a browser. For production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md).
```

#### 10b: Update project structure tree

Update the project structure in README.md to match the AFTER diagram from this plan. Remove all references to files at the repo root that have been moved (index.html, styles.css, docker-compose.yml, deploy.sh, etc.).

#### 10c: Update ARCHITECTURE.md

Add a section at the top explaining the package vs. server distinction:

```markdown
## Package vs. Server

Nulltracer is structured as a Python package (`nulltracer/`) with an optional
FastAPI server (`server/`) and browser client (`web/`).

- **`nulltracer/`** — The core library. Contains all CUDA kernels, the renderer,
  ISCO calculations, and analysis tools. Returns numpy arrays. No web dependencies.
  Install with `pip install -e .`.

- **`server/`** — A thin FastAPI wrapper around the package. Adds HTTP endpoints,
  image encoding (JPEG/WebP), LRU caching, and scene management. Install with
  `pip install -e ".[server]"`.

- **`web/`** — Static browser client. Communicates with the server via HTTP.
  No build step required — open `web/index.html` in a browser.

The notebook (`notebooks/nulltracer.ipynb`) uses the package directly. The browser
client communicates with the server.
```

Also update all file path references in ARCHITECTURE.md (e.g., `index.html` → `web/index.html`, `js/main.js` → `web/js/main.js`, references to files that moved into `server/`).

#### 10d: Update DEPLOYMENT.md

If DEPLOYMENT.md exists, update all path references:
- `index.html` → `web/index.html`
- `docker-compose.yml` at root → `server/docker-compose.yml`
- `deploy.sh` at root → `server/deploy.sh`
- Caddy `root * /srv/nulltracer` → `root * /srv/nulltracer/web`
- Any `nulltracer-renderer.xml` reference → `server/nulltracer-renderer.xml`

#### 10e: Update .gitignore

Add standard Python package ignores if not already present:

```
# Python package
*.egg-info/
dist/
build/
__pycache__/
*.pyc

# CUDA compilation cache
*.cubin
*.fatbin

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv/
```

**Acceptance criteria:** README shows package-first workflow. All file path references across all docs match the new directory structure. No doc references orphaned files.

---

### TASK 11: Final Verification

**Time estimate:** 15 minutes.

Run through the full verification checklist:

```bash
# 1. Clean install from scratch
pip install -e ".[dev,analysis]"

# 2. Package imports work
python -c "
import nulltracer as nt
print(f'Version: {nt.__version__}')
print(f'Modules: {dir(nt)}')
"

# 3. Tests pass
pytest tests/ -v

# 4. Notebook runs (requires GPU)
# jupyter nbconvert --execute notebooks/nulltracer.ipynb --to notebook

# 5. Server still works
pip install -e ".[server]"
cd server
python -c "from app import app; print('Server app imports OK')"

# 6. Docker still builds (requires Docker + NVIDIA)
# docker build -t nulltracer-server -f server/Dockerfile .

# 7. Verify cleanup — no physics code remains in server/
echo "--- Files in server/ (should be ~6 files + scenes dir): ---"
find server/ -type f | sort
# Expected: app.py, cache.py, scenes.py, Dockerfile, requirements.txt,
#           scenes/*.json. NOTHING ELSE.

# 8. Verify cleanup — no loose files at repo root
echo "--- Repo root contents: ---"
ls -1
# Expected: nulltracer/ server/ web/ notebooks/ tests/ docs/
#           pyproject.toml README.md ARCHITECTURE.md DEPLOYMENT.md
#           LICENSE .gitignore
# NOT expected: index.html, styles.css, bench.html, deploy.sh,
#               docker-compose.yml, *.png, *.svg, *.jpg, *.xml,
#               FEATURE-PLAN.md

# 9. Verify no orphaned imports
grep -r "from renderer import" server/ nulltracer/ || echo "OK: no old imports"
grep -r "from isco import" server/ nulltracer/ || echo "OK: no old imports"
grep -r "from bloom import" server/ nulltracer/ || echo "OK: no old imports"
# Any hits here are bugs — all imports should be `from nulltracer.X import Y`
```

**Acceptance criteria:** Steps 1–3 pass without GPU. Steps 4–6 pass with GPU/Docker. Steps 7–9 confirm no dead code remains.

---

### TASK 12: Merge and Tag

```bash
# Squash or merge the migration branch
git checkout master
git merge refactor/package-migration

# Tag the new version
git tag -a v1.0.0 -m "v1.0.0: Package-based architecture

- nulltracer is now an installable Python package (pip install -e .)
- render_frame() returns numpy arrays directly
- FastAPI server is an optional wrapper
- Notebook is the primary showcase
- Server, kernels, and analysis tools all work from the package"

git push origin master --tags
```

---

## What NOT To Do

- **Do NOT modify any `.cu` kernel file contents.** Only move/copy them. The kernels are tested and working.
- **Do NOT change the RenderParams struct field order.** The CUDA kernels read it by byte offset.
- **Do NOT delete `server/app.py`, `server/cache.py`, or `server/scenes.py`.** These are the thin wrapper. Everything else in server/ gets removed or relocated.
- **Do NOT add FastAPI as a core dependency of the package.** It goes in `[project.optional-dependencies.server]`.
- **Do NOT break the Docker build.** The Dockerfile must be updated (Task 6d) to copy the package into the container. Test it.
- **Do NOT rename kernel files.** The renderer's kernel loading logic matches files by name.
- **Do NOT change the `server/app.py` endpoint signatures or response formats.** The browser client depends on them.
- **Do NOT delete files without grepping for references first.** Use `grep -r "filename" .` before every `git rm`.
- **Do NOT leave orphaned files at the repo root.** After Task 6B, the root should have only: `nulltracer/`, `server/`, `web/`, `notebooks/`, `tests/`, `docs/`, config files (pyproject.toml, .gitignore), and docs (README, ARCHITECTURE, DEPLOYMENT, LICENSE). Nothing else.

---

## Dependency Graph

```
Task 0 (branch)
  └─→ Task 1 (create package dir, copy files)
       └─→ Task 2 (__init__.py)
       └─→ Task 3 (refactor renderer.py)  ← hardest task
       └─→ Task 4 (_params.py) — optional
            └─→ Task 5 (pyproject.toml)
                 └─→ Task 6  (gut server/, rewire imports, update Dockerfile)
                 └─→ Task 6B (clean repo root, relocate web/, remove dead files)
                 └─→ Task 7  (notebook)
                 └─→ Task 8  (tests)
                 └─→ Task 9  (CI)
                 └─→ Task 10 (docs)
                      └─→ Task 11 (verify)
                           └─→ Task 12 (merge + tag)
```

Tasks 6, 6B, 7, 8, 9, 10 can be done in parallel once Task 5 is complete.
Task 3 is the critical path — everything else is mechanical once the renderer works as a standalone module.
Task 6B depends on Task 6 (don't clean up root until server is rewired).