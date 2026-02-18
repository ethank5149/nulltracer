"""
FastAPI application for the Nulltracer server-side renderer.

Provides a POST /render endpoint that accepts render parameters,
renders a black hole frame via headless OpenGL, and returns a
JPEG or WebP image.
"""

import asyncio
import io
import logging
import time

# Configure root logger so all module loggers (renderer, shader, etc.) are visible
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .cache import ImageCache
from .renderer import Renderer
from .renderer_cuda import CudaRenderer

logger = logging.getLogger(__name__)

# ── Pydantic request model ──────────────────────────────────

class RenderRequest(BaseModel):
    spin: float = Field(0.6, ge=0, le=0.998)
    charge: float = Field(0.0, ge=0, le=0.998)
    inclination: float = Field(80.0, ge=3, le=89)
    fov: float = Field(8.0, ge=2, le=25)
    width: int = Field(1280, ge=160, le=3840)
    height: int = Field(720, ge=90, le=2160)
    method: str = Field("yoshida4", pattern=r"^(yoshida4|rk4|yoshida6|yoshida8|rkdp8)$")
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
            raise ValueError("spin² + charge² must be ≤ 1 (naked singularity constraint)")
        return self


# ── Global state ─────────────────────────────────────────────

renderer = Renderer()
cuda_renderer = CudaRenderer()
cache = ImageCache()
gpu_lock = asyncio.Lock()
cuda_lock = asyncio.Lock()

# ── FastAPI app ──────────────────────────────────────────────

app = FastAPI(title="Nulltracer Render Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize both renderers on server startup."""
    logger.info("Initializing OpenGL renderer...")
    renderer.initialize()
    logger.info("OpenGL renderer ready: %s", renderer.gpu_info)

    logger.info("Initializing CUDA renderer...")
    try:
        cuda_renderer.initialize()
        logger.info("CUDA renderer ready: %s", cuda_renderer.gpu_info)
    except Exception as e:
        logger.warning("CUDA renderer failed to initialize: %s (OpenGL-only mode)", e)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu": renderer.gpu_info,
        "cuda_gpu": cuda_renderer.gpu_info if cuda_renderer._initialized else "not initialized",
        "backends": ["opengl"] + (["cuda"] if cuda_renderer._initialized else []),
    }


@app.post("/render")
async def render(req: RenderRequest):
    """Render a black hole frame and return the image.

    Checks the LRU cache first. On cache miss, renders via headless
    OpenGL, encodes to JPEG/WebP, caches, and returns the image.
    """
    # Build the full parameter dict for cache keying and rendering
    params = req.model_dump()

    # Separate format/quality from render params for cache key
    # (format and quality affect encoding, not the raw render)
    render_params = {k: v for k, v in params.items() if k not in ("format", "quality")}
    cache_key_params = params  # include format+quality in cache key

    # Check cache
    cached = cache.get(cache_key_params)
    if cached is not None:
        media_type = "image/jpeg" if req.format == "jpeg" else "image/webp"
        return Response(
            content=cached,
            media_type=media_type,
            headers={
                "X-Cache": "HIT",
                "X-Render-Time-Ms": "0",
            },
        )

    # Cache miss — render
    t0 = time.monotonic()

    try:
        # Serialize GPU access. We run synchronously on the main thread
        # because the EGL/OpenGL context is thread-local and was created
        # on this thread. With --workers 1 and the gpu_lock, this is safe.
        async with gpu_lock:
            raw_rgb = renderer.render_frame(render_params)
    except Exception as e:
        logger.error("Render failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Render failed: {e}")

    # Encode to image format
    width = req.width
    height = req.height
    img = Image.frombytes("RGB", (width, height), raw_rgb)

    buf = io.BytesIO()
    if req.format == "jpeg":
        img.save(buf, format="JPEG", quality=req.quality)
        media_type = "image/jpeg"
    else:
        img.save(buf, format="WEBP", quality=req.quality)
        media_type = "image/webp"

    image_bytes = buf.getvalue()
    render_ms = (time.monotonic() - t0) * 1000.0

    # Cache the result
    cache.put(cache_key_params, image_bytes)

    logger.info(
        "Rendered %dx%d %s in %.1fms (%d bytes)",
        width, height, req.format, render_ms, len(image_bytes),
    )

    return Response(
        content=image_bytes,
        media_type=media_type,
        headers={
            "X-Cache": "MISS",
            "X-Render-Time-Ms": f"{render_ms:.1f}",
        },
    )


@app.post("/render_cuda")
async def render_cuda(req: RenderRequest):
    """Render via CUDA backend (for A/B validation against OpenGL /render).

    Same parameters and response format as /render, but uses the CUDA
    compute kernel with float64 precision instead of OpenGL fragment shaders.
    """
    if not cuda_renderer._initialized:
        raise HTTPException(status_code=503, detail="CUDA renderer not available")

    params = req.model_dump()
    render_params = {k: v for k, v in params.items() if k not in ("format", "quality")}

    # Use separate cache namespace for CUDA renders
    cache_key_params = {**params, "_backend": "cuda"}
    cached = cache.get(cache_key_params)
    if cached is not None:
        media_type = "image/jpeg" if req.format == "jpeg" else "image/webp"
        return Response(
            content=cached,
            media_type=media_type,
            headers={
                "X-Cache": "HIT",
                "X-Render-Time-Ms": "0",
                "X-Backend": "cuda",
            },
        )

    t0 = time.monotonic()

    try:
        async with cuda_lock:
            raw_rgb = cuda_renderer.render_frame(render_params)
    except Exception as e:
        logger.error("CUDA render failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"CUDA render failed: {e}")

    width = req.width
    height = req.height
    img = Image.frombytes("RGB", (width, height), raw_rgb)

    buf = io.BytesIO()
    if req.format == "jpeg":
        img.save(buf, format="JPEG", quality=req.quality)
        media_type = "image/jpeg"
    else:
        img.save(buf, format="WEBP", quality=req.quality)
        media_type = "image/webp"

    image_bytes = buf.getvalue()
    render_ms = (time.monotonic() - t0) * 1000.0

    cache.put(cache_key_params, image_bytes)

    logger.info(
        "CUDA rendered %dx%d %s in %.1fms (%d bytes)",
        width, height, req.format, render_ms, len(image_bytes),
    )

    return Response(
        content=image_bytes,
        media_type=media_type,
        headers={
            "X-Cache": "MISS",
            "X-Render-Time-Ms": f"{render_ms:.1f}",
            "X-Backend": "cuda",
        },
    )
