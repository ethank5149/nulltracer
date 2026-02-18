# Nulltracer

**A GPU-accelerated Kerr-Newman black hole ray tracer** — an interactive WebGL application that visualizes the appearance of rotating and electrically charged black holes by tracing light paths (null geodesics) through curved spacetime.

## Overview

Nulltracer simulates the visual appearance of black holes as they would appear to an external observer. By tracing null geodesics (light paths) through Kerr-Newman spacetime using GPU-accelerated WebGL fragment shaders, the application renders realistic depictions of black hole phenomena including:

- The **black hole shadow** — the dark silhouette cast by the event horizon
- **Photon rings** — unstable light orbits around the black hole
- **Accretion disk emissions** — with Doppler boosting effects that make the approaching side appear hotter and brighter
- **Gravitational lensing** — the bending of light from background stars and structures
- **Frame-dragging effects** — the warping of spacetime by the black hole's rotation

The simulator supports both **Kerr black holes** (spinning) and **Kerr-Newman black holes** (spinning with electric charge), allowing exploration of how these parameters affect the visual appearance. For mobile devices and low-power systems, Nulltracer can offload rendering to a GPU-accelerated server while maintaining a responsive local preview.

## Features

- **Real-time interactive rendering** — adjust black hole parameters and see results instantly
- **Kerr-Newman metric support** — model rotating, electrically charged black holes
- **Interactive controls** — modify spin parameter (a), electric charge (Q), and observer inclination (θ)
- **Multiple background modes** — Stars (cube-mapped), Checker pattern, or Color-mapped sphere
- **Accretion disk rendering** — with Doppler temperature boosting
- **Quality presets** — Low, Medium, High, and Ultra quality settings with performance tuning
- **Advanced controls** — configure integration steps, resolution scaling, step size, and observer distance
- **Integrator options** — switch between separated first-order equations or Hamiltonian integration
- **Full-screen capable** — for immersive visualization
- **Hybrid server rendering** — offload GPU work to a server for mobile and low-power device support
- **Three rendering modes** — Local Only (default), Hybrid (local preview + server quality), Server Only
- **Mobile auto-detection** — automatically optimizes settings for mobile devices
- **Server-side caching** — LRU cache for rendered frames to reduce redundant computations

## Usage

Simply open `index.html` in a modern web browser with WebGL 2.0 support. For server rendering, see the [Server Rendering](#server-rendering) section below.

```
Open nulltracer/index.html in your browser
```

## Server Rendering

The Nulltracer server provides GPU-accelerated frame rendering for mobile devices and low-power systems. The client can work in three modes: **Local Only** (default), **Hybrid**, or **Server Only**.

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

1. Open `index.html` in a browser
2. Click the ⚙ (settings) button
3. Enter the server URL (e.g., `http://your-server:8420`)
4. Select **Hybrid** or **Server Only** mode

### Rendering Modes

- **Local Only** (default) — all rendering on client GPU, no server needed. Best for desktop browsers.
- **Hybrid** — instant low-res local preview in the browser, plus high-quality server frames with crossfade. Optimal for mobile and low-power devices.
- **Server Only** — dims local canvas, relies entirely on server rendering. Use when client GPU is unavailable.

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

Restart your Caddy container or send a reload signal. Open `https://nulltracer.yourdomain.com` — the client will auto-detect the server and enable hybrid mode on mobile devices.

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

When served through Caddy (or any reverse proxy), the client automatically probes `/health` at the same origin on page load. If the API responds, the server URL is configured automatically and hybrid mode is enabled on mobile devices — no manual setup required.

## Controls

- **Spin (a):** Adjust black hole rotation from 0 (non-rotating Schwarzschild) to near-maximal values
- **Charge (Q):** Set electric charge parameter for Kerr-Newman black holes
- **Inclination (θ):** Change observer viewing angle relative to the black hole's rotation axis
- **Disk Temperature:** Adjust the color temperature of the accretion disk
- **Quality Preset:** Choose from Low/Medium/High/Ultra to balance visual fidelity and performance
- **Integration Method:** Select between Separated equations (faster) or Hamiltonian (more stable)
- **Integration Steps:** Control ray-tracing precision
- **Resolution Scaling:** Adjust internal rendering resolution for performance
- **Background Mode:** Switch between different background textures and patterns

## Technical Details

### Ray Tracing Approach

Nulltracer uses **WebGL 2.0 fragment shaders** to perform real-time ray tracing. Each pixel on screen corresponds to a light ray traced backward from the observer's eye through spacetime. The integration follows the equations of motion for null geodesics in the Kerr-Newman metric.

### Kerr-Newman Metric

The application solves the geodesic equations in Boyer-Lindquist coordinates, supporting both:
- **Kerr metric** — spinning (uncharged) black holes
- **Kerr-Newman metric** — spinning black holes with electric charge

### Integration Methods

1. **Separated First-Order Equations** (~40% faster) — optimized for performance
2. **Hamiltonian Integration** — uses conserved quantities for improved numerical stability

### Optimizations

- **μ = cos(θ) coordinate substitution** for robust pole handling
- **Adaptive stepping refinements** to balance accuracy and performance
- **Smooth regularization** techniques for numerical stability
- **Equal-area sphere tiling** to eliminate polar distortion in background rendering

### Server Architecture

The Nulltracer server is built with **FastAPI** and a headless **EGL/OpenGL renderer** that mirrors the client's WebGL implementation:

- **FastAPI + headless rendering** — uses EGL for GPU acceleration without a display server
- **GLSL shader porting** — same core shaders as the client, generated dynamically in Python
- **LRU cache** — 512 entries with 256MB capacity, keyed on parameter hash for frame reuse
- **GPU serialization** — single-worker with `asyncio.Lock` to prevent concurrent GPU access
- **Container image** — Dockerfile based on `nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04`

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
- **Modern web browser** with WebGL 2.0 support
- **GPU acceleration** strongly recommended for real-time performance
- No external dependencies or server required — runs entirely in the browser

### Server Requirements (Optional)
- **Python 3.8+**
- **NVIDIA GPU** with EGL support (for high-quality server rendering)
- **Docker with NVIDIA Container Toolkit** (recommended for easy deployment)

### Browser Compatibility

- Chrome/Chromium 56+
- Firefox 51+
- Safari 15+ (on macOS/iOS)
- Edge 79+

## Project Structure

```
nulltracer/
├── index.html                # Main client application
├── Caddyfile                 # Caddy site block snippet (add to existing config)
├── docker-compose.yml        # Renderer-only Docker Compose
├── nulltracer-renderer.xml   # Unraid Docker template
├── README.md                 # This file
├── ARCHITECTURE.md           # Detailed technical documentation
└── server/                   # GPU-accelerated FastAPI server
    ├── app.py                # FastAPI application and /render endpoint
    ├── renderer.py           # Headless EGL/OpenGL renderer
    ├── shader.py             # GLSL shader generation
    ├── isco.py               # Kerr-Newman ISCO calculations
    ├── cache.py              # LRU frame cache
    ├── requirements.txt      # Python dependencies
    └── Dockerfile            # Docker image definition
```

---

**License:** Check the repository for licensing information.

**Author:** Nulltracer project contributors
