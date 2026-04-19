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
