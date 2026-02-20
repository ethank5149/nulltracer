# Nulltracer

**GPU-accelerated ray tracing through curved spacetimes** — a CUDA-powered server application that visualizes the appearance of black holes by tracing light paths (null geodesics) through curved spacetime. A thin browser client displays rendered frames with interactive parameter controls.

## Overview

Nulltracer simulates the visual appearance of black holes as they would appear to an external observer. By tracing null geodesics (light paths) through curved spacetime using GPU-accelerated CUDA compute kernels, the application renders realistic depictions of black hole phenomena including:

- The **black hole shadow** — the dark silhouette cast by the event horizon
- **Photon rings** — unstable light orbits around the black hole
- **Accretion disk emissions** — with Doppler boosting effects that make the approaching side appear hotter and brighter
- **Gravitational lensing** — the bending of light from background stars and structures
- **Frame-dragging effects** — the warping of spacetime by the black hole's rotation

The simulator supports both **Kerr black holes** (spinning) and **Kerr-Newman black holes** (spinning with electric charge), allowing exploration of how these parameters affect the visual appearance. All rendering is performed on a CUDA-enabled GPU server, which the browser client communicates with via HTTP.

## Features

- **Server-based CUDA rendering** — all ray tracing performed on GPU-accelerated CUDA compute kernels
- **Kerr-Newman metric support** — model rotating, electrically charged black holes
- **Interactive browser controls** — modify spin parameter (a), electric charge (Q), and observer inclination (θ) and receive updated frames in real time
- **Multiple background modes** — Stars (cube-mapped), Checker pattern, or Color-mapped sphere
- **Accretion disk rendering** — with Doppler temperature boosting
- **Quality presets** — Low, Medium, High, and Ultra quality settings with resolution and integration tuning
- **Advanced controls** — configure integration steps, ray step size, and observer distance
- **Integrator options** — RK4, Yoshida 4th/6th/8th order, or RKDP8 adaptive methods
- **Full precision computation** — float64 precision for accurate geodesic integration
- **Server-side LRU caching** — intelligent frame caching to reduce redundant computations
- **One thread per pixel** — maximum CUDA parallelism for performance
- **Single GPU server model** — simple deployment and configuration

## Usage

Nulltracer requires a CUDA-enabled GPU server to operate. Start the server first, then open the client in your browser.

```
Open nulltracer/index.html in your browser (after starting the server)
```

On page load, the client automatically probes `/health` on the same origin. If a server responds, rendering is enabled. Otherwise, the UI shows a connection error.

## Server Rendering

The Nulltracer server is a FastAPI application that performs all ray tracing using CUDA compute kernels. It is required for operation.

### Quick Start (Docker)

```bash
cd nulltracer/server
docker build -t nulltracer-server .
docker run --gpus all -p 8420:8420 nulltracer-server
```

### Quick Start (Local)

```bash
cd nulltracer/server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8420
```

### Connecting the Client

1. Start the server (Docker or local)
2. Open `index.html` in a browser (served from the same origin as the server via Caddy, or locally on a different port)
3. The client automatically probes `/health` and connects if available
4. If served from a different origin, the server URL can be entered manually via the settings panel

### API Endpoint

- `POST /render` — renders a frame with specified parameters (black hole properties, ray-tracing settings), returns JPEG or WebP image
- `GET /health` — server health check and status

## Deployment

The recommended deployment uses your **existing Caddy reverse proxy** (e.g., on Unraid) alongside a standalone renderer container. Caddy serves the static client and proxies API requests to the renderer, unifying everything under a single origin. This eliminates CORS issues and enables automatic same-origin server detection — the client auto-discovers the API without any manual URL configuration.

### Unraid Deployment (Recommended)

If you already have Caddy running as a Docker container on Unraid:

#### Step 1: Build and start the renderer

```bash
# Build the image
docker build -t nulltracer-renderer /mnt/user/scripts/nulltracer/server/

# Run via docker-compose (renderer only)
cd /mnt/user/scripts/nulltracer
docker-compose up -d
```

Or create the container manually via the Unraid Docker tab using the template at [`nulltracer-renderer.xml`](nulltracer-renderer.xml). Key settings:
- **Image:** `nulltracer-renderer` (built above)
- **Port:** `8420:8420`
- **Extra Parameters:** `--runtime=nvidia`
- **Network:** Same network as your Caddy container

#### Step 2: Mount the static files in Caddy

Add a path mapping to your existing Caddy container:

| Container Path | Host Path | Mode |
|---|---|---|
| `/srv/nulltracer` | `/mnt/user/scripts/nulltracer` | Read-Only |

#### Step 3: Add the site block to your Caddyfile

Add the following to your existing Caddy configuration (see [`Caddyfile`](Caddyfile) for all options):

```
nulltracer.yourdomain.com {
    root * /srv/nulltracer
    file_server

    handle /render {
        reverse_proxy nulltracer-renderer:8420
    }
    handle /health {
        reverse_proxy nulltracer-renderer:8420
    }

    try_files {path} /index.html
}
```

Replace `nulltracer.yourdomain.com` with your actual subdomain. Caddy handles HTTPS automatically.

#### Step 4: Reload Caddy

Restart your Caddy container or send a reload signal. Open `https://nulltracer.yourdomain.com` — the client will auto-detect the server via the `/health` endpoint.

### Docker Compose (Standalone)

If you don't have an existing Caddy setup, the `docker-compose.yml` runs the renderer container. You'll need to set up a reverse proxy separately or access the renderer directly:

```bash
cd nulltracer
docker-compose up -d
```

Then open `index.html` locally and enter `http://your-server:8420` as the server URL in settings.

### Local Development

```bash
# Start the renderer
cd nulltracer/server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8420

# In another terminal, serve the static files (any HTTP server works)
cd nulltracer
python3 -m http.server 8080
```

Open `http://localhost:8080` — the client auto-detects the renderer at the same origin if you set up a reverse proxy, or enter `http://localhost:8420` manually in settings.

### Same-Origin Auto-Detection

When served through Caddy (or any reverse proxy), the client automatically probes `/health` at the same origin on page load. If the API responds, the server URL is configured automatically — no manual setup required.

## Controls

- **Spin (a):** Adjust black hole rotation from 0 (non-rotating Schwarzschild) to near-maximal values
- **Charge (Q):** Set electric charge parameter for Kerr-Newman black holes
- **Inclination (θ):** Change observer viewing angle relative to the black hole's rotation axis
- **Disk Temperature:** Adjust the color temperature of the accretion disk
- **Quality Preset:** Choose from Low/Medium/High/Ultra to balance visual fidelity and performance
- **Integration Method:** Select between different integration algorithms (RK4, Yoshida, RKDP8)
- **Integration Steps:** Control ray-tracing precision
- **Resolution Scaling:** Adjust internal rendering resolution for performance
- **Background Mode:** Switch between different background textures and patterns

## Technical Details

### Ray Tracing Approach

Nulltracer uses **CUDA compute kernels** to perform ray tracing on the server. Each pixel is computed by a separate thread, with each thread tracing a light ray backward from the observer's eye through spacetime in float64 precision. The integration follows the equations of motion for null geodesics in the Kerr-Newman metric.

### Kerr-Newman Metric

The application solves the geodesic equations in Boyer-Lindquist coordinates, supporting both:
- **Kerr metric** — spinning (uncharged) black holes
- **Kerr-Newman metric** — spinning black holes with electric charge

### Integration Methods

The server supports multiple high-order integration methods:

1. **RK4** — classical 4th-order Runge-Kutta, good balance of speed and stability
2. **Yoshida 4th-order** — symplectic method, preserves phase space structure
3. **Yoshida 6th-order** — higher-order symplectic, improved accuracy
4. **Yoshida 8th-order** — maximum-order symplectic for highest precision
5. **RKDP8** — adaptive 8th-order Runge-Kutta-Dormand-Prince, automatically adjusts step size

### Optimizations

- **μ = cos(θ) coordinate substitution** for robust pole handling
- **Adaptive stepping refinements** to balance accuracy and performance
- **Smooth regularization** techniques for numerical stability
- **Equal-area sphere tiling** to eliminate polar distortion in background rendering

### Server Architecture

The Nulltracer server is built with **FastAPI** and **CuPy CUDA compute kernels**:

- **FastAPI** — async HTTP server for parameter acceptance and image delivery
- **CuPy RawKernel** — custom-compiled CUDA kernels in C++ with one thread per pixel
- **Float64 precision** — all geodesic calculations use 64-bit floating point for accuracy
- **LRU frame cache** — intelligent caching by parameter hash to avoid redundant computation
- **GPU serialization** — single-worker with `asyncio.Lock` to prevent concurrent GPU access
- **NVIDIA Container Toolkit** — Dockerfile based on `nvidia/cuda:12.2.0-devel-ubuntu22.04` for easy deployment

## Version History

All versions are preserved as git tags. To view a previous release, use `git checkout v0.X`:

| Version | Tag | Date | Key Changes |
|---------|-----|------|-------------|
| v0.0.1 | `v0.0.1` | Initial | Kerr black hole with Hamiltonian RK4 integration |
| v0.1 | `v0.1` | | Refactored to separated first-order equations (~40% faster) |
| v0.2 | `v0.2` | | UX overhaul: legend, settings panel, multiple backgrounds |
| v0.3 | `v0.3` | | Equal-area sphere tiling (fixes polar pinching) |
| v0.4 | `v0.4` | | Numerical stability improvements |
| v0.5 | `v0.5` | | Smooth regularization and cube-map projection |
| v0.6 | `v0.6` | | μ=cos(θ) coordinate substitution for pole handling |
| v0.7 | `v0.7` | | Adaptive stepping refinements |
| v0.8 | `v0.8` | | Kerr-Newman extension (electric charge parameter) |
| v0.9 | `v0.9` | Current | Polished Kerr-Newman release |

**Accessing previous versions:**
```bash
git checkout v0.8    # View the Kerr-Newman release before current
git checkout v0.1    # View the first performance-optimized version
git checkout main    # Return to the latest version
```

## Requirements

### Client Requirements
- **Modern web browser** (Chrome 56+, Firefox 51+, Safari 15+, Edge 79+)
- **Network connection** to the CUDA render server
- No local GPU required — browser is a thin client that displays server-rendered JPEG images

### Server Requirements (Required)
- **Python 3.8+**
- **NVIDIA GPU** with CUDA support (RTX 3090, A100, H100, etc.)
- **CUDA Toolkit 12.x** with cupy-cuda12x installed
- **Docker with NVIDIA Container Toolkit** (recommended for easy deployment)

### Browser Compatibility

- Chrome/Chromium 56+
- Firefox 51+
- Safari 15+ (on macOS/iOS)
- Edge 79+

## Project Structure

```
nulltracer/
├── index.html                # Browser client UI — parameter controls and image display
├── styles.css                # Client stylesheet
├── js/                        # Client JavaScript
│   ├── main.js               # App initialization
│   ├── server-client.js      # HTTP /render communication and /health auto-detection
│   └── ui-controller.js      # Slider/button event handlers
├── Caddyfile.current          # Caddy reverse proxy configuration (optional)
├── docker-compose.yml        # Docker Compose configuration for renderer
├── nulltracer-renderer.xml   # Unraid Docker template
├── README.md                 # This file
├── ARCHITECTURE.md           # Detailed technical documentation
└── server/                   # CUDA-accelerated FastAPI server
    ├── app.py                # FastAPI application and /render, /health endpoints
    ├── renderer.py           # CuPy CUDA renderer (entry point)
    ├── isco.py               # Kerr-Newman ISCO calculations
    ├── cache.py              # LRU frame cache
    ├── requirements.txt      # Python dependencies
    ├── Dockerfile            # Docker image definition
    └── kernels/              # CUDA compute kernels (C++)
        ├── geodesic_base.cu  # Metric functions and geodesic integration
        ├── backgrounds.cu    # Background rendering (stars, checker, colormap)
        ├── disk.cu           # Accretion disk emission and color
        └── integrators/      # Integration method kernels
            ├── rk4.cu        # 4th-order Runge-Kutta
            ├── rkdp8.cu          # Adaptive Runge-Kutta-Dormand-Prince
            ├── tao_yoshida4.cu   # Tao + Yoshida 4th-order symplectic
            ├── tao_yoshida6.cu   # Tao + Yoshida 6th-order symplectic
            └── tao_kahan_li8.cu  # Tao + Kahan-Li 8th-order symplectic
```

---

**License:** Check the repository for licensing information.

**Author:** Nulltracer project contributors
