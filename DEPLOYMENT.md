# Nulltracer Deployment Guide

Detailed deployment instructions for production environments.
For quick local setup, see the [Quick Start section in README.md](README.md#quick-start).

The recommended deployment uses an **existing Caddy reverse proxy** (e.g., on Unraid) alongside a standalone renderer container. Caddy serves the static client in `web/` and proxies API requests to the CUDA renderer, unifying everything under a single origin. This eliminates CORS issues and enables automatic same-origin server detection — the client auto-discovers the API without any manual URL configuration.

> **Deployment state:** the FastAPI server module is being consolidated. The sections below assume the package exposes an ASGI app at `nulltracer.server:app` (`uvicorn nulltracer.server:app`). If a dedicated `server/` tree is reintroduced, update the `docker build` context accordingly.

## Unraid Deployment (Recommended)

If you already have Caddy running as a Docker container on Unraid:

### Step 1: Build and start the renderer

A minimal Dockerfile at the repository root (building from the full source tree, since the kernel `.cu` files live in `nulltracer/kernels/`) is the cleanest option:

```bash
# Build the image from the repository root
docker build -t nulltracer-renderer .
```

Launch with NVIDIA runtime, publishing port 8420:

```bash
docker run -d --name nulltracer-renderer \
    --runtime=nvidia --gpus all \
    -p 8420:8420 \
    nulltracer-renderer
```

An Unraid Community Application template is provided at [`assets/nulltracer-renderer.xml`](assets/nulltracer-renderer.xml) (if present). Key settings:

- **Image:** `nulltracer-renderer` (built above)
- **Port:** `8420:8420`
- **Extra Parameters:** `--runtime=nvidia`
- **Network:** Same network as your Caddy container

### Step 2: Mount the static files in Caddy

Add a path mapping to your existing Caddy container so it can serve `web/`:

| Container Path | Host Path | Mode |
|----------------|-----------|------|
| `/srv/nulltracer` | `/path/to/nulltracer/web` | Read-Only |

### Step 3: Add the site block to your Caddyfile

```caddyfile
nulltracer.yourdomain.com {
    root * /srv/nulltracer
    file_server

    handle /render {
        reverse_proxy nulltracer-renderer:8420
    }
    handle /health {
        reverse_proxy nulltracer-renderer:8420
    }
    handle /ray {
        reverse_proxy nulltracer-renderer:8420
    }

    try_files {path} /index.html
}
```

Replace `nulltracer.yourdomain.com` with your actual subdomain. Caddy handles HTTPS automatically.

### Step 4: Reload Caddy

Restart your Caddy container or send a reload signal. Open `https://nulltracer.yourdomain.com` — the client will auto-detect the server via the `/health` endpoint.

## Docker Compose (Standalone)

If you don't maintain an existing Caddy setup, a `docker-compose.yml` at the repository root can run the renderer container on its own:

```bash
docker-compose up -d
```

Then open `web/index.html` locally and enter `http://your-server:8420` as the server URL in the client's settings panel.

## Local Development

No container, no reverse proxy — just run the renderer directly against the static client:

```bash
# Start the renderer
pip install -e ".[server]"
uvicorn nulltracer.server:app --host 0.0.0.0 --port 8420

# In another terminal, serve the static files (any HTTP server works)
cd web && python3 -m http.server 8080
```

Open `http://localhost:8080` in a browser. Either set up a local reverse proxy to unify origins, or enter `http://localhost:8420` as the server URL in the client's settings panel.

## Same-Origin Auto-Detection

When the static client and renderer are served from the same origin (via Caddy or any reverse proxy), the client automatically probes `/health` at page load. If the API responds, the server URL is configured without any manual setup. Set `server_url` explicitly in the client settings only when running on different origins.
