# Nulltracer — GPU-Accelerated Curved Spacetime Rendering Architecture

## 1. Overview

This document describes the CUDA-only rendering architecture for Nulltracer. The system consists of a lightweight browser client that displays server-rendered frames from a GPU-accelerated backend. All rendering is performed server-side using CUDA compute kernels; there is no local client-side rendering.

### Design Principles

- **Server-only rendering** — All black hole ray tracing and image synthesis occurs on the server via CUDA kernels
- **Thin client** — The browser provides parameter controls (sliders/buttons) and displays the server image in a fixed viewport
- **Stateless API** — Each render request is self-contained; the server caches results by parameter hash
- **High precision** — float64 for geodesic integration and metric evaluation; float32 for color; uint8 for final output

### Rendering Pipeline Overview

```mermaid
sequenceDiagram
    participant User
    participant Client as Client - Browser
    participant Server as Server - FastAPI
    participant GPU as GPU - CUDA Kernels

    User->>Client: Adjusts parameter (spin, charge, etc.)
    Client->>Client: Schedule 200ms debounce timer
    Note over Client: Timer resets on each new change
    Client->>Server: POST /render with all parameters
    Server->>Server: Check LRU image cache by param hash
    alt Cache hit
        Server-->>Client: Return cached JPEG/WebP
    else Cache miss
        Server->>GPU: Launch CUDA kernel (one thread per pixel)
        GPU->>GPU: Initialize ray, integrate geodesics (float64)
        GPU->>GPU: Determine termination (escaped/horizon/disk)
        GPU->>GPU: Compute ray color, write uint8 RGBA output
        GPU-->>Server: Transfer output buffer to CPU
        Server->>Server: Encode as JPEG/WebP via Pillow
        Server->>Server: Store in LRU cache
        Server-->>Client: Return encoded image
    end
    Client->>Client: Display image in <img id="server-frame">
```

---

## 2. Server Architecture

### 2.1 Directory Structure

```
nulltracer/server/
├── app.py                 # FastAPI application entry point (/render, /ray, /health endpoints)
├── renderer.py            # CudaRenderer: kernel compilation, execution, GPU management
├── cache.py               # LRU image cache by parameter hash
├── isco.py                # ISCO calculation (port of iscoJS/iscoKN)
├── Dockerfile             # nvidia/cuda:12.2.0-devel-ubuntu22.04 base
├── requirements.txt       # fastapi, uvicorn, cupy-cuda12x, Pillow, numpy
├── __init__.py
└── kernels/
    ├── geodesic_base.cu    # Shared metric functions, geodesic RHS, constants (float64)
    ├── backgrounds.cu      # Background rendering (stars, checker, colormap)
    ├── disk.cu             # Accretion disk emission and color computation
    ├── ray_trace.cu        # Single-ray tracing kernel for /ray endpoint
    └── integrators/
        ├── rk4.cu          # RK4 4th-order Runge-Kutta integrator kernel
        ├── yoshida4.cu       # Yoshida 4th-order symplectic integrator kernel
        ├── rkdp8.cu          # Dormand-Prince adaptive 8th-order RK kernel
        ├── tao_yoshida4.cu   # Tao + Yoshida 4th-order symplectic (extended phase space)
        ├── tao_yoshida6.cu   # Tao + Yoshida 6th-order symplectic (extended phase space)
        └── tao_kahan_li8.cu  # Tao + Kahan-Li 8th-order symplectic (extended phase space)
```

### 2.2 FastAPI Application ([`app.py`](server/app.py))

The main entry point providing a REST API for render requests:

```python
# Pseudocode structure
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Nulltracer Render Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize CUDA renderer on server startup."""
    renderer.initialize()

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": renderer.gpu_info, "backends": ["cuda"]}

@app.post("/render")
async def render(params: RenderRequest) -> Response:
    # 1. Compute cache key from params
    # 2. Check LRU cache
    # 3. On miss: acquire GPU lock, render via CUDA, encode, cache
    # 4. Return image bytes with appropriate Content-Type header
    ...
```

**Startup**: On application startup (`@app.on_event("startup")`), initialize the CUDA context and compile the default kernel (Yoshida 4th order). This context persists for the lifetime of the process.

**Shutdown**: On shutdown, release CUDA device memory and destroy contexts.

### 2.3 CUDA Renderer ([`renderer.py`](server/renderer.py))

The `CudaRenderer` class manages CUDA kernel compilation, caching, and execution:

```python
class CudaRenderer:
    """CUDA-based renderer using CuPy RawKernel."""

    def __init__(self):
        self._kernel_cache: dict[str, cp.RawKernel] = {}
        self._gpu_info: str = "unknown"
        self._initialized = False

    def initialize(self) -> None:
        """Initialize CUDA context and query GPU properties."""
        # Force CUDA context creation on device 0
        cp.cuda.Device(0).use()
        
        # Query GPU info: name, memory, compute capability
        props = cp.cuda.runtime.getDeviceProperties(0)
        self._gpu_info = f"{name} ({mem_gb:.1f} GB, compute {cc})"
        
        # Pre-compile the default kernel
        self._get_kernel("yoshida4")

    def render_frame(self, params: dict) -> bytes:
        """Render a frame via CUDA.
        
        Returns raw RGB bytes (uint8, no padding).
        """
        # 1. Resolve kernel source (inline #include directives)
        # 2. Get or compile CuPy RawKernel for the integration method
        # 3. Prepare GPU memory: input RenderParams, output RGBA buffer
        # 4. Launch kernel: (width, height) blocks × 1 thread per pixel
        # 5. Transfer output buffer back to CPU
        # 6. Return as numpy array
        ...
```

#### Kernel Compilation

Each integration method (`yoshida4`, `rk4`, `rkdp8`, `kahanli8s`, `kahanli8s_ks`, `tao_yoshida4`, `tao_yoshida6`, `tao_kahan_li8`) has a corresponding CUDA kernel in `server/kernels/integrators/*.cu`. The `CudaRenderer` maintains a kernel cache keyed by method name, avoiding recompilation:

```python
_kernel_cache: dict[str, cp.RawKernel] = {}

def _get_kernel(self, method: str) -> cp.RawKernel:
    if method not in self._kernel_cache:
        source, entry_point = self._load_kernel_source(method)
        kernel = cp.RawKernel(source, entry_point)
        self._kernel_cache[method] = kernel
    return self._kernel_cache[method]
```

#### GPU Memory Management

For each render request:

```
1. Create a ctypes-compatible RenderParams structure with all parameters
   (width, height, spin, charge, inclination, FOV, etc.)
2. Allocate GPU output buffer: (width * height * 3) uint8 bytes for RGB
3. Copy RenderParams to GPU constant memory or global memory
4. Launch kernel with grid = (width, height), block = (1, 1, 1)
   - Each thread processes one pixel, writes to output buffer
5. Synchronize with GPU (cudaDeviceSynchronize via CuPy)
6. Copy output buffer back to CPU
7. Return numpy array of shape (height, width, 3) with dtype uint8
```

### 2.4 ISCO Calculation ([`isco.py`](server/isco.py))

A direct port of the JavaScript [`iscoJS()`](index.html:942) and [`iscoKN()`](index.html:861) functions, computing the Innermost Stable Circular Orbit radius for a given black hole spin and charge.

**Logic**:
1. If `Q == 0`: use the analytic Kerr formula
2. If `Q != 0`: use numerical bisection on the effective potential

The ISCO value is computed server-side and passed as a parameter to the CUDA kernel. This ensures the client and server agree on the ISCO radius.

### 2.5 LRU Image Cache ([`cache.py`](server/cache.py))

Cache rendered images to avoid redundant GPU work for identical parameter sets.

**Cache key**: SHA-256 hash of the canonical JSON representation of all render parameters (sorted keys):

```python
def cache_key(params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

**Cache implementation**: Custom LRU dict with:
- **Max entries**: 512 (configurable via `CACHE_MAX_ENTRIES` env var)
- **Max memory**: 256 MB (configurable via `CACHE_MAX_BYTES` env var)
- **Eviction**: LRU by access time; also evict if total memory exceeds limit

**Cache bypass**: Possible future enhancement via `Cache-Control: no-cache` header.

### 2.6 Concurrency Model

The server runs a **single CUDA context** shared across all requests. CUDA operations are not thread-safe, so:

1. **Single-worker uvicorn** with `--workers 1` (GPU context is per-process)
2. **asyncio.Lock** guards all CUDA calls — only one render executes at a time
3. Requests queue behind the lock; FastAPI's async handler yields while waiting
4. CPU-bound work (image encoding) runs in a thread pool via `asyncio.to_thread()`

```python
gpu_lock = asyncio.Lock()

async def render_frame(params: dict) -> bytes:
    async with gpu_lock:
        return await asyncio.to_thread(_render_sync, params)
```

**Scaling**: For higher throughput, run multiple single-worker containers behind a load balancer, each with its own GPU or sharing via NVIDIA MPS.

### 2.7 Docker Containerization

```dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**GPU passthrough**: Run with `--gpus all` (Docker) or `--device nvidia.com/gpu=all` (containerd):

```bash
docker run --gpus all -p 8000:8000 nulltracer-server
```

**Dependencies** ([`requirements.txt`](server/requirements.txt)):

```
fastapi>=0.104
uvicorn[standard]>=0.24
cupy-cuda12x>=13.0
Pillow>=10.0
numpy>=1.24
pydantic>=2.0
```

### 2.8 Server Module Dependency Graph

```
app.py (FastAPI entry point)
├── renderer.py          → Manages CUDA kernel compilation and execution
│   ├── isco.py         → ISCO calculation (called during render)
│   └── kernels/
│       ├── geodesic_base.cu     → Shared metric functions, RHS
│       ├── backgrounds.cu       → Background rendering
│       ├── disk.cu              → Disk color computation
│       └── integrators/*.cu     → Integrator-specific kernels
├── cache.py             → LRU image cache
└── models.py            → Pydantic request validation (RenderRequest)
```

**Module responsibilities:**
- **`app.py`** — FastAPI entry point, request routing, response formatting
- **`renderer.py`** — CUDA kernel compilation, GPU memory management, kernel execution
- **`isco.py`** — ISCO radius calculation from spin and charge
- **`cache.py`** — LRU image caching by parameter hash
- **`kernels/`** — CUDA compute kernels for ray tracing (all computation happens here)

---

## 3. CUDA Kernel Architecture

### 3.1 Kernel Organization

The CUDA rendering pipeline is split into modular kernel files:

| File | Purpose |
|------|---------|
| [`geodesic_base.cu`](server/kernels/geodesic_base.cu) | Metric functions, geodesic RHS, constants, shared utilities (float64) |
| [`backgrounds.cu`](server/kernels/backgrounds.cu) | Procedural background: stars, checker pattern, colormap |
| [`disk.cu`](server/kernels/disk.cu) | Accretion disk emission, blackbody color, redshift |
| [`integrators/rk4.cu`](server/kernels/integrators/rk4.cu) | RK4 4th-order Runge-Kutta integrator kernel |
| [`integrators/yoshida4.cu`](server/kernels/integrators/yoshida4.cu) | Yoshida 4th-order symplectic integrator kernel |
| [`integrators/rkdp8.cu`](server/kernels/integrators/rkdp8.cu) | Dormand-Prince adaptive 8th-order integrator kernel |
| [`integrators/tao_yoshida4.cu`](server/kernels/integrators/tao_yoshida4.cu) | Tao + Yoshida 4th-order symplectic (extended phase space) |
| [`integrators/tao_yoshida6.cu`](server/kernels/integrators/tao_yoshida6.cu) | Tao + Yoshida 6th-order symplectic (extended phase space) |
| [`integrators/tao_kahan_li8.cu`](server/kernels/integrators/tao_kahan_li8.cu) | Tao + Kahan-Li 8th-order symplectic (extended phase space) |
| [`integrators/kahanli8s.cu`](server/kernels/integrators/kahanli8s.cu) | Kahan-Li 8th-order symplectic integrator kernel with Sundman time transformation |

### 3.2 RenderParams Structure

All rendering parameters are passed to the kernel via a single C-compatible struct:

```c
struct RenderParams {
    double width, height;           // Output resolution
    double spin, charge;            // Black hole spin and charge
    double incl;                    // Observer inclination (radians)
    double fov;                     // Field of view
    double phi0;                    // Disk rotation (typically 0)
    double isco;                    // Innermost stable circular orbit
    double steps;                   // Integration step count
    double obs_dist;                // Observer distance in M
    double esc_radius;              // Escape radius (obs_dist + 12)
    double disk_outer;              // Outer disk radius (14.0)
    double step_size;               // Base step size (H_BASE)
    double bg_mode;                 // Background: 0=stars, 1=checker, 2=colormap
    double star_layers;             // Number of background star layers
    double show_disk;               // Show disk: 1.0 or 0.0
    double show_grid;               // Show grid: 1.0 or 0.0
    double disk_temp;               // Disk temperature multiplier
};
```

**Note**: All fields are `double` (float64) for alignment consistency between Python ctypes and CUDA. Integer values are stored as doubles and cast within the kernel.

### 3.3 Kernel Execution Model

Each CUDA kernel is launched as:

```cuda
// Launch: one thread per pixel
kernel<<<dim3(width, height), dim3(1, 1, 1)>>>(params, output_buffer);

__global__ void trace_yoshida4(RenderParams params, uint8_t *output_buffer) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pixel_idx = (y * (int)params.width + x) * 3;  // RGB triple
    
    // 1. Initialize ray from observer position
    // 2. Run Yoshida 4th-order integration loop
    // 3. Determine termination condition (escaped/horizon/disk)
    // 4. Compute ray color
    // 5. Write uint8 RGBA to output_buffer[pixel_idx:pixel_idx+3]
}
```

### 3.4 Numerical Precision

- **Integration**: float64 (double) for all geodesic computations
  - Metric evaluation: `g_μν`
  - Geodesic RHS: `dp/dλ`, `dx/dλ`
  - All intermediate calculations maintain full precision
- **Color computation**: float32 allowed for efficiency (RGB color is inherently low-precision)
- **Output**: uint8 (0-255) per channel for final RGBA

### 3.5 Key Technical Decisions

#### Why CUDA Instead of OpenGL

- **Compute kernels** allow per-pixel parallel execution without graphics pipeline overhead
- **Direct memory access** for ray state and output buffers
- **float64 support** is native and efficient; OpenGL requires extensions or workarounds
- **Kernel caching** avoids recompilation when parameters don't change
- **No graphics context** complexity; CUDA context is simpler to initialize in Docker

#### Why One Thread Per Pixel

- **Simplicity** — Each thread manages one ray independently; minimal inter-thread communication
- **Scalability** — Fully independent workload, embarrassingly parallel
- **Memory efficiency** — Each thread's local state (ray position, velocity, etc.) fits in registers
- **Register pressure** — Modern GPUs have enough registers per thread for full ray state (typically ~30 registers per thread)

### 3.6 kahanli8s: Kahan-Li 8th-Order Symplectic Integrator

The **kahanli8s** integrator (`trace_kahanli8s` in [`kahanli8s.cu`](server/kernels/integrators/kahanli8s.cu)) is a high-accuracy geometric integrator designed for near-horizon black hole ray tracing. It combines multiple numerical techniques to achieve ~10th-order effective accuracy while maintaining symplecticity.

#### Architecture Overview

**kahanli8s** comprises five integrated components:

1. **Kahan-Li s15odr8 composition** — A 15-stage symmetric composition of the leapfrog integrator using optimized coefficients from Kahan & Li (1997). The composition is palindromic (coefficients are symmetric), with maximum weight magnitude |W_i| = 0.797, ensuring all substep displacements remain bounded by the step size.

2. **Sundman/Mino time transformation** — Instead of adaptive step-size selection (which breaks symplecticity; Ge & Marsden 1988), a coordinate time transformation dτ = dλ/Σ is used, where Σ = r² + a²cos²θ. This provides automatic step adaptation (small steps near the horizon where spacetime curvature is extreme, large steps at large radii) while preserving the symplectic structure. The Mino-time budget is computed from the exact Bardeen photon sphere radius, allowing accurate allocation of integration steps.

3. **Compensated (Kahan) summation** — All accumulations of state variables use Kahan summation with a correction term, tracking floating-point round-off error and effectively doubling working precision. This preserves the modified Hamiltonian (the discrete Hamiltonian function of the integrator) to machine precision, crucial for long integration paths.

4. **Symplectic corrector** — A near-identity canonical transformation (h²/24 correction term) applied after each step, raising the effective accuracy from 8th to approximately 10th order at the cost of one additional geodesic RHS evaluation per step (31 RHS calls total per full step: 15 composition stages × 2 half-steps + 1 corrector).

5. **Hamiltonian projection** — After each step, the momenta are algebraically solved onto the H=0 null geodesic constraint by adjusting p_r, ensuring rays remain on the light cone. This is performed exactly (not iteratively), preventing energy drift on long integration paths.

#### Computation Cost

```
Per full integration step:
  • 15 composition stages
  • 2 RHS evaluations per stage (half-step kicks)
  = 30 RHS calls from composition
  + 1 RHS call for symplectic corrector
  ────────────────────────────
  = 31 RHS evaluations per step
```

With default settings (200 steps, 320×180 pixels):
- **RTX 3090 performance**: ~0.6–1.7 seconds per frame
- **Compared to yoshida4**: ~3–8× slower, but with dramatically superior accuracy at extreme inclinations (θ > 85°) and near the event horizon

#### Diagnostic Capability

The integrator computes the **Carter constant** Q₀ at initialization based on impact parameters. This value can be used for post-render validation to confirm geodesic conserved quantities are maintained.

#### Mathematical References

- **Kahan & Li (1997)**: "Composition constants for raising the orders of unconventional schemes for ordinary differential equations." *Math. Comp.* 66(219):1089–1099.
- **Wisdom (2006)**: "Symplectic correctors for canonical heliocentric N-body maps." *Astron. J.* 131:2294–2298.
- **Kahan (1965)**: "Pracniques: further remarks on reducing truncation errors." *Comm. ACM* 8(1):40.
- **Ge & Marsden (1988)**: "Lie-Poisson Hamilton-Jacobi theory and Lie-Poisson integrators." *Phys. Lett. A* 133(3):134–139.
- **Mino (2003)**: "Perturbative approach to an orbital evolution around a supermassive black hole." *Phys. Rev. D* 67:084027.

#### When to Use kahanli8s

- **High-inclination scenarios** (θ > 80°): Visible accuracy improvement over lower-order integrators
- **Near-horizon detail**: Rendering structures very close to r_+ requires the step adaptation of Sundman time
- **Publication-quality output**: The ~10th-order accuracy and symplecticity are suitable for scientific visualization
- **Disk inner edge dynamics**: Precise ray-disk intersection detection near the ISCO

Not recommended for:
- **Real-time interaction** at high resolution; use **yoshida4** for responsive preview
- **Low-inclination views** where geodesics are nearly radial; lower-order integrators converge adequately

---

## 4. API Contract

### 4.1 `POST /render`

**Request body** (`application/json`):

```json
{
  "spin": 0.6,
  "charge": 0.0,
  "inclination": 80.0,
  "fov": 8.0,
  "width": 1920,
  "height": 1080,
  "method": "yoshida4",
  "steps": 200,
  "step_size": 0.3,
  "obs_dist": 40,
  "bg_mode": 1,
  "show_disk": true,
  "show_grid": true,
  "disk_temp": 1.0,
  "star_layers": 3,
  "phi0": 0.0,
  "format": "jpeg",
  "quality": 85
}
```

**Pydantic model** ([`app.py`](server/app.py:30)):

```python
class RenderRequest(BaseModel):
    spin: float = Field(0.6, ge=0, le=0.998)
    charge: float = Field(0.0, ge=0, le=0.998)
    inclination: float = Field(80.0, ge=3, le=89)
    fov: float = Field(8.0, ge=2, le=25)
    width: int = Field(1280, ge=160, le=3840)
    height: int = Field(720, ge=90, le=2160)
    method: str = Field("yoshida4", pattern=r"^(yoshida4|rk4|rkdp8|kahanli8s|kahanli8s_ks|tao_yoshida4|tao_yoshida6|tao_kahan_li8)$")
    steps: int = Field(200, ge=60, le=500)
    step_size: float = Field(0.3, ge=0.1, le=0.8)
    obs_dist: int = Field(40, ge=20, le=100)
    bg_mode: int = Field(1, ge=0, le=2)
    show_disk: bool = Field(True)
    show_grid: bool = Field(True)
    disk_temp: float = Field(1.0, ge=0.2, le=2.5)
    star_layers: int = Field(3, ge=1, le=4)
    phi0: float = Field(0.0)
    format: str = Field("jpeg", pattern=r"^(jpeg|webp)$")
    quality: int = Field(85, ge=10, le=100)

    @model_validator(mode="after")
    def check_spin_charge_constraint(self):
        if self.spin ** 2 + self.charge ** 2 > 1.0:
            raise ValueError("spin² + charge² must be ≤ 1")
        return self
```

**Validation rules**:
- `0 ≤ spin ≤ 0.998`
- `0 ≤ charge ≤ 0.998`
- `spin² + charge² ≤ 1.0` (naked singularity constraint, enforced via validator)
- `160 ≤ width ≤ 3840`
- `90 ≤ height ≤ 2160`
- Total pixel count `width * height ≤ 8,294,400` (4K cap)

**Response** (success — `200 OK`):

```
Content-Type: image/jpeg | image/webp
Content-Length: <bytes>
X-Render-Time-Ms: 142
X-Cache: HIT | MISS
X-Backend: cuda

<binary image data>
```

**Response** (validation error — `422 Unprocessable Entity`):

```json
{
  "detail": [
    {
      "loc": ["body", "spin"],
      "msg": "ensure this value is less than or equal to 0.998",
      "type": "value_error.number.not_le"
    }
  ]
}
```

**Response** (CUDA error — `500 Internal Server Error`):

```json
{
  "error": "cuda_render_failed",
  "detail": "CUDA kernel execution failed: ..."
}
```

**Response** (GPU unavailable — `503 Service Unavailable`):

```json
{
  "error": "gpu_unavailable",
  "detail": "CUDA renderer not initialized"
}
```

### 4.2 `GET /health`

**Response** (`200 OK`):

```json
{
  "status": "ok",
  "gpu": "NVIDIA GeForce RTX 3090 (24.0 GB, compute 8.6)",
  "backends": ["cuda"]
}
```

### 4.3 `POST /ray`

General-purpose single-ray tracing endpoint. Traces one photon geodesic through Kerr-Newman spacetime and returns the full trajectory, equatorial plane crossings with disk physics, and termination conditions.

**Request body** (`application/json`):

```json
{
  "mode": "pixel",
  "ix": 160,
  "iy": 90,
  "spin": 0.6,
  "charge": 0.0,
  "inclination": 80.0,
  "method": "yoshida4",
  "fov": 8.0,
  "width": 320,
  "height": 180,
  "steps": 200,
  "step_size": 0.3,
  "obs_dist": 40,
  "phi0": 0.0,
  "doppler_boost": 2,
  "disk_temp": 1.0,
  "include_trajectory": true,
  "include_disk_physics": true,
  "max_trajectory_points": 200
}
```

**Input modes**:
- `"pixel"` — specify screen coordinates `(ix, iy)` to trace (defaults to image center)
- `"impact_parameter"` — specify `(alpha, beta)` angles directly in radians

**Impact parameter mode example**:

```json
{
  "mode": "impact_parameter",
  "alpha": -0.05,
  "beta": 0.02,
  "spin": 0.998,
  "inclination": 85.0,
  "method": "rk4"
}
```

**Pydantic model** ([`app.py`](server/app.py:444)):

```python
class RayRequest(BaseModel):
    mode: str = Field("pixel", pattern=r"^(pixel|impact_parameter)$")
    ix: Optional[int] = Field(None, ge=0)
    iy: Optional[int] = Field(None, ge=0)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    spin: float = Field(0.6, ge=0, le=0.998)
    charge: float = Field(0.0, ge=0, le=0.998)
    inclination: float = Field(80.0, ge=3, le=89)
    method: str = Field("yoshida4", pattern=r"^(yoshida4|rk4|...)$")
    fov: float = Field(8.0, ge=2, le=25)
    width: int = Field(320, ge=16, le=3840)
    height: int = Field(180, ge=16, le=2160)
    steps: int = Field(200, ge=60, le=500)
    step_size: float = Field(0.3, ge=0.1, le=0.8)
    obs_dist: int = Field(40, ge=20, le=100)
    phi0: float = Field(0.0)
    doppler_boost: int = Field(2, ge=0, le=2)
    disk_temp: float = Field(1.0, ge=0.2, le=2.5)
    include_trajectory: bool = Field(True)
    include_disk_physics: bool = Field(True)
    max_trajectory_points: int = Field(200, ge=10, le=500)
```

**Response** (success — `200 OK`, `application/json`):

```json
{
  "ray": {
    "mode": "pixel",
    "ix": 160,
    "iy": 90,
    "b": -1.234
  },
  "spacetime": {
    "spin": 0.6,
    "charge": 0.0,
    "r_plus": 1.8,
    "r_isco": 3.829,
    "method": "yoshida4",
    "effective_method": "yoshida4"
  },
  "initial_state": {
    "r": 40.0, "theta": 1.396, "phi": 0.0,
    "pr": -0.998, "pth": -0.456
  },
  "final_state": {
    "r": 52.0, "theta": 0.87, "phi": 3.14,
    "pr": 0.12, "pth": -0.03
  },
  "termination": {
    "reason": "escape",
    "steps_used": 147,
    "steps_max": 200
  },
  "trajectory": {
    "points": 147,
    "r": [40.0, 39.7, "..."],
    "theta": [1.396, 1.395, "..."],
    "phi": [0.0, 0.001, "..."],
    "step_sizes": [0.3, 0.29, "..."]
  },
  "disk_crossings": [
    {
      "crossing_index": 42,
      "r": 6.12,
      "phi": 1.23,
      "direction": "north_to_south",
      "g_factor": 1.15,
      "novikov_thorne_flux": 0.73,
      "T_emit": 6400.0,
      "T_observed": 7360.0
    }
  ],
  "timing": {
    "kernel_ms": 0.12,
    "total_ms": 1.45
  }
}
```

**Termination reasons**:
- `"escape"` — ray reached escape radius (background)
- `"horizon"` — ray captured by event horizon
- `"steps_exhausted"` — maximum integration steps reached
- `"nan"` — numerical instability detected
- `"underflow"` — radius dropped below safety threshold

**Integrator support**: The ray trace kernel natively supports all available methods (`yoshida4`, `rk4`, `rkdp8`, `kahanli8s`, `kahanli8s_ks`). The `effective_method` field in the response indicates which integrator was actually used.

**Disk crossing physics**: Each equatorial plane crossing includes:
- `g_factor` — gravitational redshift factor ν_obs/ν_emit
- `novikov_thorne_flux` — normalized Page-Thorne radial flux F/F_max
- `T_emit` — emitted blackbody temperature (K)
- `T_observed` — observed temperature after redshift (K)

### 4.4 CORS Configuration

The server enables CORS for all origins by default (configurable):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Render-Time-Ms", "X-Cache", "X-Backend"],
)
```

---

## 5. Client Architecture

### 5.1 Frontend File Structure

```
nulltracer/
├── index.html              # HTML markup + title/legend + parameter controls
├── styles.css              # All CSS for layout, panels, controls
└── js/
    ├── main.js             # ES6 module entry point, shared state, initialization
    ├── server-client.js    # Server render requests, debouncing, health checks
    └── ui-controller.js    # DOM event handlers, slider logic, presets
```

### 5.2 Client Architecture (ES6 Modular)

The client is a thin server-only viewer with modular JavaScript:

| File | Purpose |
|------|---------|
| [`index.html`](index.html) | HTML markup, `<img id="server-frame">` viewport, parameter controls, panels |
| [`styles.css`](styles.css) | CSS layout, responsive design, panel styling |
| [`js/main.js`](js/main.js) | ES6 entry point, shared state, module initialization |
| [`js/server-client.js`](js/server-client.js) | Server render fetch, debouncing, auto-detection, health checks |
| [`js/ui-controller.js`](js/ui-controller.js) | DOM event handlers, slider/button logic, presets, labels |

### 5.3 Rendering Flow

```
1. User adjusts parameter (slider/button)
   ↓
2. DOM event handler increments state, calls scheduleServerRender()
   ↓
3. Debounce timer starts (200ms)
   ↓
4. If user changes another parameter within 200ms, timer resets
   ↓
5. When timer fires (200ms idle), construct JSON params
   ↓
6. POST /render to server URL
   ↓
7. Server checks cache, on miss renders via CUDA
   ↓
8. Server returns JPEG/WebP image
   ↓
9. Client displays in <img id="server-frame">
```

### 5.4 Server Auto-Detection

On page load, the client attempts to detect a same-origin server:

```javascript
export async function autoDetectServer() {
    // Check if server is at default origin (same origin, port 8000)
    const url = window.location.origin + ':8000';
    const ok = await checkServerHealth();
    if (ok) {
        setServerUrl(url);
        scheduleServerRender();  // Begin rendering
    } else {
        // Show error message, wait for manual server URL entry
    }
}
```

### 5.5 Shared Application State

All modules reference a shared state object for parameter values:

```javascript
const state = {
    spin: 0,
    charge: 0,
    incl: 89,
    diskTemp: 1,
    showDisk: 1,
    showGrid: 1,
    autoRotate: false,
    rotAngle: 0,
    fov: 8,
    bgMode: 1,
    qMethod: 'yoshida4',
    qSteps: 200,
    qResScale: 1.0,
    qStepSize: 0.3,
    qObsDist: 40,
    qStarLayers: 3,
    renderMode: 'server',
};
```

This is passed to all modules so they can read/write shared parameters and coordinate.

### 5.6 Spin/Charge Constraint Enforcement

The UI enforces the Kerr-Newman naked singularity constraint: `spin² + charge² ≤ 1.0`

When one parameter changes, the other is clamped if necessary:

```javascript
document.getElementById('spin').addEventListener('input', function(){
    stateRef.spin = +this.value;
    // Enforce: spin² + charge² <= 1
    const maxQ = Math.sqrt(Math.max(1 - stateRef.spin*stateRef.spin, 0));
    if (stateRef.charge > maxQ) {
        stateRef.charge = Math.floor(maxQ * 500) / 500;
        document.getElementById('charge').value = stateRef.charge;
    }
    scheduleServerRender();
});
```

### 5.7 Quality Presets

Presets adjust integration method, steps, and observer distance for performance/quality trade-off:

```javascript
const presets = {
    low:    {method:'yoshida4', steps:80,  stepSize:0.5,  obsDist:30, starLayers:1},
    medium: {method:'yoshida4', steps:200, stepSize:0.3,  obsDist:40, starLayers:3},
    high:   {method:'rkdp8',    steps:180, stepSize:0.40, obsDist:50, starLayers:4},
    ultra:  {method:'rkdp8',    steps:200, stepSize:0.45, obsDist:60, starLayers:4},
    extreme:{method:'kahanli8s', steps:180, stepSize:0.35, obsDist:70, starLayers:4},
};
```

Presets are applied via buttons in the settings panel, updating sliders and state:
- **low** — Fast preview (yoshida4, 80 steps) for rapid exploration
- **medium** — Default balance (yoshida4, 200 steps)
- **high** — Enhanced accuracy (rkdp8, 180 steps)
- **ultra** — Maximum accuracy (rkdp8, 200 steps) with excellent convergence
- **extreme** — Maximum accuracy (kahanli8s, 180 steps) for publication-quality output at extreme inclinations

---

## 6. Rendering Pipeline Detail

### 6.1 Server-Side CUDA Render Steps

```mermaid
flowchart TD
    Start["Receive RenderRequest"]
    
    subgraph Cache["CACHE LOOKUP (Fast Path ⚡)"]
        A{"Image cached?"}
        A -->|"Yes ✓"| CacheHit["Return cached image"]
    end
    
    Start --> A
    A -->|"No"| CalcISCO["Compute ISCO from spin + charge"]
    
    subgraph KernelLoad["CUDA KERNEL COMPILATION & CACHING"]
        B{"Kernel cached?"}
        B -->|"Yes ✓"| KernelReuse["Reuse compiled kernel"]
        B -->|"No"| LoadSource["Load kernel source + #include resolution"]
        LoadSource --> LoadErr{"Load succeeded?"}
        LoadErr -->|"No ✗"| ErrLoad["Return 500 error"]
        LoadErr -->|"Yes"| Compile["Compile kernel via CuPy"]
        Compile --> CompErr{"Compilation succeeded?"}
        CompErr -->|"No ✗"| ErrComp["Log CUDA compilation error"]
        ErrComp --> CleanComp["Return 500 error"]
        CompErr -->|"Yes"| Store["Store in kernel cache"]
        Store --> KernelReuse
    end
    
    CalcISCO --> B
    ErrLoad --> End1["End - 500 Error"]
    CleanComp --> End2["End - 500 Error"]
    
    subgraph GPURender["GPU RENDERING & READBACK"]
        C["Allocate GPU output buffer (width × height × 3 uint8)"]
        C --> D["Copy RenderParams to GPU memory"]
        D --> E["Launch kernel: grid=(width,height), block=(1,1,1)"]
        E --> F["Each thread: initialize ray, integrate, write color"]
        F --> G["Synchronize GPU (cudaDeviceSynchronize)"]
        G --> H["Copy output buffer to CPU"]
        H --> ReadErr{"Readback succeeded?"}
        ReadErr -->|"No ✗"| ErrRead["Log readback error"]
        ErrRead --> CleanRead["Free GPU memory"]
        CleanRead --> EndRead["Return 500 error"]
        ReadErr -->|"Yes"| ToPillow["Pillow Image.frombytes"]
    end
    
    KernelReuse --> C
    End1 --> EndAll["End"]
    End2 --> EndAll
    EndRead --> EndAll
    
    subgraph ImageEncode["IMAGE ENCODING & CACHING"]
        I["Encode JPEG/WebP at quality setting"]
        EncErr{"Encoding succeeded?"}
        EncErr -->|"No ✗"| ErrEnc["Log encoding error"]
        ErrEnc --> CleanEnc["Return 500 error"]
        EncErr -->|"Yes"| Cache2["Store in image cache"]
    end
    
    ToPillow --> I
    I --> EncErr
    CleanEnc --> EndEnc["End - 500 Error"]
    
    subgraph ResourceCleanup["RESOURCE CLEANUP"]
        S["Free GPU output buffer"]
    end
    
    Cache2 --> S
    S --> CacheHit
    CacheHit --> Success["✓ Return image to client"]
    
    style Cache fill:#90EE90
    style CacheHit fill:#32CD32,color:#fff
    style KernelReuse fill:#87CEEB
    style GPURender fill:#FFB6C1
    style ImageEncode fill:#DDA0DD
    style ResourceCleanup fill:#F0E68C
    style Start fill:#fff,stroke:#333,stroke-width:2px
    style Success fill:#32CD32,color:#fff,stroke:#333,stroke-width:2px
```

### 6.2 Per-Pixel Ray Tracing

Each CUDA thread traces a single ray from the observer:

```
1. Compute ray direction from pixel coordinates (x, y)
   - Convert to normalized device coords in [-1, 1]
   - Apply FOV and inclination transformation
   - Get ray (α, β) impact parameters in observer frame
2. Initialize ray position at observer: r = obs_dist, θ = π/2, φ = 0
3. Initialize ray direction: (pr, pθ, pφ) from impact parameters
4. Photon loop:
   - Integrate geodesic equations (float64) for N steps
   - Check termination: escaped (r > esc_radius)? horizon? disk hit?
   - If escaped: color = background (stars/checker/colormap)
   - If horizon: color = black
   - If disk: compute disk blackbody color + redshift
   - Break
5. Convert linear RGB to sRGB gamma
6. Convert to uint8 (0-255) per channel
7. Write RGBA to output buffer
```

### 6.3 Integrator Kernels

Each integrator kernel implements the same interface:

```cuda
__global__ void trace_yoshida4(RenderParams params, uint8_t *output_buffer) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    
    // 1. Compute ray direction
    // 2. Initialize ray state (position, velocity)
    // 3. Integration loop (Yoshida 4th-order symplectic)
    //    while (not_terminated && step < params.steps) {
    //        x += h * dx/dlambda
    //        p += h * dp/dlambda
    //    }
    // 4. Terminate & compute color
    // 5. Write to output
}
```

All integrators use the same `geoRHS()` function from [`geodesic_base.cu`](server/kernels/geodesic_base.cu) to compute the geodesic equations in Kerr-Newman spacetime.

### 6.4 Kerr-Newman Metric and Geodesics

The metric is computed in Boyer-Lindquist coordinates with mass M = 1, spin parameter `a`, and charge `Q`. The [`geodesic_base.cu`](server/kernels/geodesic_base.cu) kernel provides:

- **Metric components**: `g_tt`, `g_tφ`, `g_rr`, `g_θθ`, `g_φφ`
- **Christoffel symbols**: Partial derivatives of metric for geodesic equations
- **Geodesic RHS**: `dx^μ/dλ` and `dp_μ/dλ` in first-order Hamiltonian form

All metric computations use float64 for maximum accuracy.

### 6.5 Background Rendering

The [`backgrounds.cu`](server/kernels/backgrounds.cu) kernel provides three background modes:

1. **Stars** (`bg_mode=0`): Procedural Perlin-like noise with multiple layers
2. **Checker** (`bg_mode=1`): Repeating checkerboard pattern in equatorial coordinates
3. **Colormap** (`bg_mode=2`): Gradient-based color map (celestial sphere visualization)

Each is selected at runtime based on `params.bg_mode`.

### 6.6 Accretion Disk Color

The [`disk.cu`](server/kernels/disk.cu) kernel computes disk color when a ray hits the equatorial plane within the disk bounds:

```
1. Check if radius is within disk (ISCO * 0.85 to RDISK)
2. Compute temperature profile: T(r) = T_base * u_temp * (r / r_isco)^(-0.75)
3. Compute bolometric intensity: I ~ T^4 / r
4. Apply GR redshift factor:
   - Compute metric at emission point
   - Compute circular orbit angular velocity
   - Compute redshift: g = 1 / (u^t * (1 - b*Ω))
5. Apply redshift: T_obs = g * T_emit, I_obs = g^4 * I_emit
6. Convert temperature to blackbody color (linear sRGB)
7. Apply turbulence texture (Perlin-like noise)
8. Return final RGB color
```

---

## 7. Performance Characteristics

### 7.1 Bottleneck Analysis

**CUDA kernel time** (per frame, typical):
- Resolution: 1920×1080 (2.07 MP)
- Integration steps: 200
- Typical GPU (RTX 3090): ~100-300 ms per frame

**Breakdown**:
- Ray initialization: ~5 ms (small)
- Integration loop (200 steps × ~60 ops per step): ~200-250 ms
- Background/disk color: ~20 ms
- Disk I/O (GPU → CPU → encode): ~10-30 ms

**Encoding**:
- JPEG encoding (Pillow): ~10-20 ms
- WebP encoding (Pillow): ~20-40 ms

**Memory bandwidth**:
- Output buffer: 1920 × 1080 × 3 bytes = 6.2 MB
- PCIe 3.0 (12 GB/s): <1 ms to transfer

**Total wall time**: ~130-350 ms per render (highly dependent on GPU and integration method)

### 7.2 Optimization Opportunities

- **Adaptive step size** (RKDP8): Uses fewer steps in smooth regions
- **Kernel compilation caching**: Avoids recompilation when parameters don't change
- **Image caching**: Identical parameters reuse previous renders
- **Multi-GPU**: Run multiple containers, each with its own GPU

---

## 8. Error Handling

| Scenario | Server Response | Client Behavior |
|----------|----------------|-----------------|
| Invalid parameters | 422 with validation details | Show error message; retry with valid params |
| CUDA compilation failure | 500 with error detail | Retry; may indicate GPU driver issue |
| GPU out of memory | 500 with OOM detail | Retry with lower resolution |
| Request timeout (>30s) | Client-side timeout | Show error; retry |
| Server unreachable | Network error | Increment failure counter; disable after 3 failures |
| `spin² + charge² > 1` | 422 validation error | Client pre-validates; never send invalid combos |

---

## 9. Future Considerations

These are **out of scope** for the initial CUDA-only implementation but worth noting for roadmap planning:

- **WebSocket streaming**: Replace REST polling with a WebSocket that streams progressive output as the render completes
- **Tile-based rendering**: Render and stream tiles progressively for faster perceived response
- **Advanced queue system**: Redis-backed job queue for high-traffic scenarios
- **Multi-GPU rendering**: Distribute work across multiple GPUs for higher throughput
- **Adaptive LOD**: Adjust integration steps dynamically based on local ray complexity
- **Pre-computed lookup tables**: Cache commonly-accessed values (metric components at fixed points, etc.)

---

## 10. Key Technical Decisions

### 10.1 Why CUDA (Not OpenGL/OpenCL)

- **Mature ecosystem**: CUDA is battle-tested, well-documented
- **Performance**: Native float64 support; OpenGL requires extensions
- **Simplicity**: CUDA context setup is cleaner than EGL for Docker
- **Future extensibility**: CUDA Graphs for optimization, cuDNN integration possible

### 10.2 Why float64 for Integration

- **Stability**: Geodesic integration is sensitive to numerical error
- **Accuracy**: long-term stability (200+ integration steps) requires double precision
- **Drift reduction**: float32 diverges visibly; float64 keeps rays stable

### 10.3 Why One Thread Per Pixel

- **Simplicity** — No inter-thread communication complexity
- **Scalability** — Fully independent workloads scale linearly
- **Debugging** — Easy to verify per-pixel behavior
- **Memory** — Ray state fits in registers (~30 regs/thread); minimal shared memory needed

### 10.4 Cache Key Design

The cache key is a truncated SHA-256 hash of the canonical JSON of all parameters:

```python
canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
key = hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

This ensures:
- Identical parameters always produce the same key
- Different parameter orderings produce the same key
- 16-character hex = 64 bits collision resistance (sufficient for 512 entries)
- Floating-point precision is controlled by Pydantic serialization

---

## 11. File Structure Summary

### Frontend

| File | Purpose |
|------|---------|
| [`index.html`](index.html) | HTML markup + `<img id="server-frame">` viewport + controls |
| [`styles.css`](styles.css) | All CSS for layout, responsive design, panels |
| [`js/main.js`](js/main.js) | ES6 module entry point, shared state, initialization |
| [`js/server-client.js`](js/server-client.js) | Server render fetch, debounce, health checks, auto-detect |
| [`js/ui-controller.js`](js/ui-controller.js) | DOM handlers, sliders, buttons, presets, labels |

### Backend

| File | Purpose |
|------|---------|
| [`server/app.py`](server/app.py) | FastAPI entry point, `/render`, `/ray`, `/bench`, and `/health` endpoints |
| [`server/renderer.py`](server/renderer.py) | CudaRenderer: kernel compilation, GPU memory, execution, single-ray tracing |
| [`server/isco.py`](server/isco.py) | ISCO radius calculation (Kerr-Newman) |
| [`server/cache.py`](server/cache.py) | LRU image cache with memory limits |
| [`server/kernels/geodesic_base.cu`](server/kernels/geodesic_base.cu) | Metric functions, geodesic RHS, shared utilities |
| [`server/kernels/backgrounds.cu`](server/kernels/backgrounds.cu) | Background rendering (stars, checker, colormap) |
| [`server/kernels/disk.cu`](server/kernels/disk.cu) | Disk emission, blackbody color, GR redshift |
| [`server/kernels/ray_trace.cu`](server/kernels/ray_trace.cu) | Single-ray tracing kernel with trajectory recording and disk physics |
| [`server/kernels/integrators/rk4.cu`](server/kernels/integrators/rk4.cu) | RK4 integrator kernel |
| [`server/kernels/integrators/yoshida4.cu`](server/kernels/integrators/yoshida4.cu) | Yoshida 4th-order integrator kernel |
| [`server/kernels/integrators/rkdp8.cu`](server/kernels/integrators/rkdp8.cu) | Dormand-Prince adaptive integrator kernel |
| [`server/kernels/integrators/tao_yoshida4.cu`](server/kernels/integrators/tao_yoshida4.cu) | Tao + Yoshida 4th-order symplectic kernel |
| [`server/kernels/integrators/tao_yoshida6.cu`](server/kernels/integrators/tao_yoshida6.cu) | Tao + Yoshida 6th-order symplectic kernel |
| [`server/kernels/integrators/tao_kahan_li8.cu`](server/kernels/integrators/tao_kahan_li8.cu) | Tao + Kahan-Li 8th-order symplectic kernel |
| [`server/kernels/integrators/kahanli8s.cu`](server/kernels/integrators/kahanli8s.cu) | Kahan-Li 8th-order symplectic integrator kernel |
| [`server/Dockerfile`](server/Dockerfile) | NVIDIA CUDA 12.2 container definition |
| [`server/requirements.txt`](server/requirements.txt) | Python dependencies (fastapi, cupy, etc.) |

### Deployment

| File | Purpose |
|------|---------|
| [`docker-compose.yml`](docker-compose.yml) | Multi-service Docker compose (client + server) |
| [`Caddyfile.current`](Caddyfile.current) | Reverse proxy configuration |
| [`deploy.sh`](deploy.sh) | Deployment helper script |

---

## 12. Development Workflow

### Local Development (CPU Only)

For development without a GPU, you can run the client standalone:

```bash
# Serve static files
python3 -m http.server 8080 --directory .
# Open http://localhost:8080 in browser
```

The client will show an error waiting for server connection, but you can prototype UI changes.

### With Docker

Build and run the complete stack:

```bash
docker-compose up --build
# Frontend: http://localhost
# Server: http://localhost/api/render
```

### Integration Testing

The server includes validation tests for parameter ranges and CUDA kernel compilation. Run:

```bash
cd server
python3 -m pytest tests/ -v
```

---

## 13. Monitoring and Debugging

### Server Logs

Enable debug logging:

```bash
RUST_LOG=debug python3 -m uvicorn app:app --log-level debug
```

Key log lines:
- `CUDA GPU: ...` — GPU detection on startup
- `Kernel compiled: ...` — Kernel compilation/cache hit
- `CUDA render failed: ...` — Render errors
- `X-Render-Time-Ms: ...` — Per-request timing

### Client Debugging

Open browser DevTools (F12) → Console tab:

```javascript
// Check server connection
fetch('/health').then(r => r.json()).then(console.log);

// Check render request
fetch('/render', {
    method: 'POST',
    body: JSON.stringify({/* params */})
}).then(r => r.blob()).then(b => console.log(`Received ${b.size} bytes`));
```

### GPU Memory

Monitor with NVIDIA tools:

```bash
nvidia-smi                   # One-shot GPU stats
nvidia-smi dmon              # Per-process GPU monitor
nvtop                        # Interactive GPU monitor
```

---

## 14. Physics and Mathematics Reference

### Kerr-Newman Spacetime

The metric in Boyer-Lindquist coordinates with mass M = 1:

```
ds² = -(Δ - a²sin²θ)/Σ dt² + 2a(2 - Q²sin²θ)/Σ dt dφ
      + Σ/Δ dr² + Σ dθ² + (r² + a² + a²sin²θ(2 - Q²sin²θ)/Σ) sin²θ dφ²
```

where:
- `Σ = r² + a²cos²θ`
- `Δ = r² - 2r + a² + Q²`
- `a` = spin parameter (0 ≤ a < 1)
- `Q` = charge parameter (0 ≤ Q < 1)
- Constraint: `a² + Q² ≤ 1` (no naked singularity)

### Geodesic Equations

The geodesic equations are integrated in the form:

```
dx^μ/dλ = p^μ
dp_μ/dλ = -Γ^λ_αβ p^α p^β
```

with affine parameter λ (not necessarily proper time). All integrators use this first-order formulation.

### Horizon Location

The event horizon is at:

```
r_+ = 1 + √(1 - a² - Q²)
```

### ISCO (Innermost Stable Circular Orbit)

- **Kerr** (Q=0): Analytic formula (Bardeen, Press & Teukolsky 1972)
- **Kerr-Newman** (Q≠0): Computed numerically via bisection on effective potential

---

## References and Further Reading

- **Kerr-Newman Metric**: Carter (1968), de Felice (1975)
- **Ray Tracing**: Luminet (1979), Cunningham (1975)
- **Geodesic Integration**: Stoer & Bulirsch (1980), Numerical Recipes
- **CUDA Programming**: NVIDIA CUDA C Programming Guide
- **CuPy Documentation**: https://docs.cupy.dev/
