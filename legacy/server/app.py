"""
FastAPI application for the Nulltracer server-side renderer.
GPU-accelerated ray tracing through curved spacetimes.

Provides a POST /render endpoint that accepts render parameters,
renders a black hole frame via CUDA compute kernels, and returns a
JPEG or WebP image.

Also provides a POST /bench endpoint that renders a frame using ALL
available integrators and returns runtime statistics + images for
side-by-side comparison.

Also provides a POST /ray endpoint for single-ray tracing that returns
the full geodesic trajectory, equatorial plane crossings with disk
physics (g-factor, Novikov-Thorne flux, temperature), and termination
conditions.
"""

import asyncio
import base64
import io
import json
import logging
import struct
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

# Configure root logger so all module loggers (renderer, etc.) are visible
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .cache import ImageCache
from .renderer import CudaRenderer
from .scenes import SceneManager

logger = logging.getLogger(__name__)

# ── Pydantic request model ──────────────────────────────────

class RenderRequest(BaseModel):
    spin: float = Field(0.6, ge=0, le=0.998)
    charge: float = Field(0.0, ge=0, le=0.998)
    inclination: float = Field(80.0, ge=3, le=89)
    fov: float = Field(8.0, ge=2, le=25)
    width: int = Field(1280, ge=160, le=3840)
    height: int = Field(720, ge=90, le=2160)
    method: str = Field("rkdp8", pattern=r"^(rk4|rkdp8|kahanli8s|kahanli8s_ks|tao_yoshida4|tao_yoshida6|tao_kahan_li8)$")
    steps: int = Field(200, ge=60, le=500)
    step_size: float = Field(0.3, ge=0.1, le=0.8)
    obs_dist: int = Field(40, ge=20, le=100)
    bg_mode: int = Field(1, ge=0, le=2)
    show_disk: bool = Field(True)
    show_grid: bool = Field(True)
    disk_temp: float = Field(1.0, ge=0.2, le=2.5)
    star_layers: int = Field(3, ge=1, le=4)
    phi0: float = Field(0.0)
    doppler_boost: int = Field(default=2, ge=0, le=2, description="Doppler boost mode: 0=off, 1=g^3 optically thin, 2=g^4 optically thick")
    srgb_output: bool = Field(default=True, description="Apply IEC 61966-2-1 sRGB transfer function (proper gamma for standard displays)")
    disk_alpha: float = Field(default=0.95, ge=0.0, le=1.0, description="Base opacity per disk crossing")
    disk_max_crossings: int = Field(default=5, ge=1, le=20, description="Maximum disk crossings to accumulate")
    bloom_enabled: bool = Field(default=False, description="Enable Airy disk bloom post-processing")
    bloom_radius: float = Field(default=1.0, ge=0.1, le=5.0, description="Bloom radius multiplier (1.0 = physical default)")
    format: str = Field("jpeg", pattern=r"^(jpeg|webp)$")
    quality: int = Field(85, ge=10, le=100)

    @model_validator(mode="after")
    def check_spin_charge_constraint(self):
        if self.spin ** 2 + self.charge ** 2 > 1.0:
            raise ValueError("spin² + charge² must be ≤ 1 (naked singularity constraint)")
        return self


class BenchRequest(BaseModel):
    """Request model for the /bench endpoint.

    Accepts the same physics parameters as RenderRequest but renders
    at fixed 1920×1080 using all (or a subset of) integrators for
    side-by-side comparison with detailed timing statistics.
    """
    spin: float = Field(0.6, ge=0, le=0.998)
    charge: float = Field(0.0, ge=0, le=0.998)
    inclination: float = Field(80.0, ge=3, le=89)
    fov: float = Field(8.0, ge=2, le=25)
    steps: int = Field(200, ge=60, le=500)
    step_size: float = Field(0.3, ge=0.1, le=0.8)
    obs_dist: int = Field(40, ge=20, le=100)
    bg_mode: int = Field(1, ge=0, le=2)
    show_disk: bool = Field(True)
    show_grid: bool = Field(True)
    disk_temp: float = Field(1.0, ge=0.2, le=2.5)
    star_layers: int = Field(3, ge=1, le=4)
    phi0: float = Field(0.0)
    doppler_boost: int = Field(default=2, ge=0, le=2, description="Doppler boost mode: 0=off, 1=g^3 optically thin, 2=g^4 optically thick")
    srgb_output: bool = Field(default=True, description="Apply IEC 61966-2-1 sRGB transfer function")
    disk_alpha: float = Field(default=0.95, ge=0.0, le=1.0, description="Base opacity per disk crossing")
    disk_max_crossings: int = Field(default=5, ge=1, le=20, description="Maximum disk crossings to accumulate")
    bloom_enabled: bool = Field(default=False, description="Enable Airy disk bloom post-processing")
    bloom_radius: float = Field(default=1.0, ge=0.1, le=5.0, description="Bloom radius multiplier (1.0 = physical default)")
    methods: Optional[list[str]] = Field(
        None,
        description="Subset of integrator methods to benchmark. Defaults to all available methods."
    )
    format: str = Field("webp", pattern=r"^(jpeg|webp|png)$")
    quality: int = Field(90, ge=10, le=100)
    include_images: bool = Field(True, description="Include base64-encoded images in response. Set false for stats-only.")

    @model_validator(mode="after")
    def check_spin_charge_constraint(self):
        if self.spin ** 2 + self.charge ** 2 > 1.0:
            raise ValueError("spin² + charge² must be ≤ 1 (naked singularity constraint)")
        return self


# ── Global state ─────────────────────────────────────────────

renderer = CudaRenderer()
cache = ImageCache()
gpu_lock = asyncio.Lock()
scene_manager = SceneManager()

# ── FastAPI app ──────────────────────────────────────────────

app = FastAPI(title="Nulltracer — GPU-Accelerated Curved Spacetime Renderer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the CUDA renderer on server startup."""
    logger.info("Initializing CUDA renderer...")
    try:
        renderer.initialize()
        logger.info("CUDA renderer ready: %s", renderer.gpu_info)
        # Pre-compile all integrator kernels for bench endpoint
        compile_results = renderer.precompile_all()
        compiled = sum(1 for v in compile_results.values() if v)
        logger.info(
            "Pre-compiled %d/%d integrator kernels",
            compiled, len(compile_results),
        )
    except Exception as e:
        raise RuntimeError(f"CUDA renderer failed to initialize: {e}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu": renderer.gpu_info,
        "backends": ["cuda"] if renderer._initialized else [],
    }


@app.post("/purge-cache")
async def purge_cache():
    """Purge all CUDA kernel caches (in-memory and CuPy disk cache).

    Use after updating kernel source files to force recompilation.
    """
    count = renderer.purge_cache()
    # Also clear the image cache since renders may differ after kernel changes
    cache.clear()
    logger.info("Cache purged: %d kernels cleared, image cache cleared", count)
    return {"status": "purged", "kernels_cleared": count}


# ── Scene management endpoints ──────────────────────────────

@app.get("/scenes")
async def list_scenes():
    """List all available scenes."""
    return {"scenes": scene_manager.list_scenes()}


@app.get("/scenes/{name}")
async def get_scene(name: str):
    """Load a scene by name."""
    scene = scene_manager.get_scene(name)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Scene '{name}' not found")
    return scene


@app.post("/scenes/{name}")
async def save_scene(name: str, params: dict):
    """Save a scene with the given name and parameters."""
    if not scene_manager.validate_name(name):
        raise HTTPException(
            status_code=400,
            detail="Invalid scene name. Use alphanumeric characters, hyphens, underscores, and spaces (1-63 chars).",
        )
    success = scene_manager.save_scene(name, params)
    if not success:
        raise HTTPException(status_code=409, detail=f"Cannot overwrite built-in scene '{name}'")
    return {"status": "saved", "name": name}


@app.delete("/scenes/{name}")
async def delete_scene(name: str):
    """Delete a scene by name."""
    success = scene_manager.delete_scene(name)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Scene '{name}' not found or is a built-in scene",
        )
    return {"status": "deleted", "name": name}


@app.post("/render")
async def render(req: RenderRequest):
    """Render a black hole frame via CUDA and return the image.

    Checks the LRU cache first. On cache miss, renders via CUDA
    compute kernels with float64 precision, encodes to JPEG/WebP,
    caches, and returns the image.
    """
    if not renderer._initialized:
        raise HTTPException(status_code=503, detail="CUDA renderer not available")

    params = req.model_dump()
    render_params = {k: v for k, v in params.items() if k not in ("format", "quality")}

    # Check cache
    cached = cache.get(params)
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

    # Cache miss — render
    t0 = time.monotonic()

    try:
        async with gpu_lock:
            raw_rgb = renderer.render_frame(render_params)
    except Exception as e:
        logger.error("CUDA render failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"CUDA render failed: {e}")

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
    cache.put(params, image_bytes)

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


# ── Bench endpoint (SSE streaming) ───────────────────────────

BENCH_WIDTH = 1920
BENCH_HEIGHT = 1080


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/bench")
async def bench(req: BenchRequest):
    """Benchmark all integrators with SSE streaming.

    Renders the scene at 1920×1080 using every available integrator
    (or a user-specified subset), streaming results as Server-Sent Events
    so the client can display each integrator's result as it completes.

    SSE event types:
        started  — benchmark metadata (bench_id, methods, gpu, resolution)
        progress — emitted before each integrator starts rendering
        result   — emitted after each integrator completes (includes image)
        complete — final summary with timing statistics

    This endpoint bypasses the image cache entirely to ensure
    accurate timing measurements.
    """
    if not renderer._initialized:
        raise HTTPException(status_code=503, detail="CUDA renderer not available")

    bench_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Determine which methods to benchmark
    all_methods = renderer.available_methods
    if req.methods is not None:
        invalid = [m for m in req.methods if m not in all_methods]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown integrator method(s): {invalid}. Available: {all_methods}",
            )
        methods = req.methods
    else:
        methods = all_methods

    # Build base render params (shared across all integrators)
    base_params = {
        "spin": req.spin,
        "charge": req.charge,
        "inclination": req.inclination,
        "fov": req.fov,
        "width": BENCH_WIDTH,
        "height": BENCH_HEIGHT,
        "steps": req.steps,
        "step_size": req.step_size,
        "obs_dist": req.obs_dist,
        "bg_mode": req.bg_mode,
        "show_disk": req.show_disk,
        "show_grid": req.show_grid,
        "disk_temp": req.disk_temp,
        "star_layers": req.star_layers,
        "phi0": req.phi0,
        "doppler_boost": req.doppler_boost,
        "srgb_output": req.srgb_output,
        "disk_alpha": req.disk_alpha,
        "disk_max_crossings": req.disk_max_crossings,
        "bloom_enabled": req.bloom_enabled,
        "bloom_radius": req.bloom_radius,
    }

    # Echo input parameters for the started event
    input_params = req.model_dump()
    input_params.pop("methods", None)
    input_params.pop("format", None)
    input_params.pop("quality", None)
    input_params.pop("include_images", None)

    async def event_generator():
        # Emit started event
        yield _sse_event("started", {
            "bench_id": bench_id,
            "timestamp": timestamp,
            "gpu": renderer.gpu_info,
            "resolution": {"width": BENCH_WIDTH, "height": BENCH_HEIGHT},
            "parameters": input_params,
            "methods": methods,
            "total": len(methods),
        })

        results = []
        t_bench_start = time.monotonic()

        # Hold GPU lock for entire benchmark to ensure consistent timing
        async with gpu_lock:
            for idx, method in enumerate(methods):
                # Emit progress event before rendering
                yield _sse_event("progress", {
                    "method": method,
                    "index": idx + 1,
                    "total": len(methods),
                    "status": "rendering",
                    "elapsed_ms": round((time.monotonic() - t_bench_start) * 1000.0, 2),
                })

                render_params = {**base_params, "method": method}
                result_entry = {"method": method}

                try:
                    # Run timed render (blocking GPU work)
                    timed = await asyncio.get_running_loop().run_in_executor(
                        None, renderer.render_frame_timed, render_params
                    )

                    raw_rgb = timed["raw_rgb"]
                    kernel_ms = timed["kernel_ms"]
                    total_render_ms = timed["total_ms"]
                    gpu_mem_alloc = timed["gpu_mem_alloc_bytes"]

                    # Encode image
                    t_enc_start = time.monotonic()
                    img = Image.frombytes("RGB", (BENCH_WIDTH, BENCH_HEIGHT), raw_rgb)
                    buf = io.BytesIO()

                    if req.format == "png":
                        img.save(buf, format="PNG")
                        mime = "image/png"
                    elif req.format == "webp":
                        img.save(buf, format="WEBP", quality=req.quality)
                        mime = "image/webp"
                    else:
                        img.save(buf, format="JPEG", quality=req.quality)
                        mime = "image/jpeg"

                    image_bytes = buf.getvalue()
                    encode_ms = (time.monotonic() - t_enc_start) * 1000.0

                    result_entry.update({
                        "status": "ok",
                        "kernel_ms": round(kernel_ms, 2),
                        "render_ms": round(total_render_ms, 2),
                        "encode_ms": round(encode_ms, 2),
                        "total_ms": round(total_render_ms + encode_ms, 2),
                        "gpu_mem_alloc_bytes": gpu_mem_alloc,
                        "image_size_bytes": len(image_bytes),
                    })

                    if req.include_images:
                        b64 = base64.b64encode(image_bytes).decode("ascii")
                        result_entry["image_base64"] = f"data:{mime};base64,{b64}"
                    else:
                        result_entry["image_base64"] = None

                except Exception as e:
                    logger.error("Bench render failed for method '%s': %s", method, e, exc_info=True)
                    result_entry.update({
                        "status": "error",
                        "error": str(e),
                        "kernel_ms": None,
                        "render_ms": None,
                        "encode_ms": None,
                        "total_ms": None,
                        "gpu_mem_alloc_bytes": None,
                        "image_size_bytes": None,
                        "image_base64": None,
                    })

                results.append(result_entry)

                # Emit result event immediately after this integrator completes
                yield _sse_event("result", result_entry)

        total_bench_ms = (time.monotonic() - t_bench_start) * 1000.0

        # Compute summary statistics
        ok_results = [r for r in results if r["status"] == "ok"]
        summary = {}
        if ok_results:
            fastest = min(ok_results, key=lambda r: r["kernel_ms"])
            slowest = max(ok_results, key=lambda r: r["kernel_ms"])
            summary = {
                "fastest_method": fastest["method"],
                "fastest_kernel_ms": fastest["kernel_ms"],
                "slowest_method": slowest["method"],
                "slowest_kernel_ms": slowest["kernel_ms"],
                "speedup_ratio": round(slowest["kernel_ms"] / fastest["kernel_ms"], 2) if fastest["kernel_ms"] > 0 else None,
                "methods_ok": len(ok_results),
                "methods_failed": len(results) - len(ok_results),
            }

        logger.info(
            "Bench completed: %d methods in %.1fms (fastest: %s %.1fms)",
            len(results),
            total_bench_ms,
            summary.get("fastest_method", "N/A"),
            summary.get("fastest_kernel_ms", 0),
        )

        # Emit complete event with summary
        yield _sse_event("complete", {
            "bench_id": bench_id,
            "total_bench_time_ms": round(total_bench_ms, 2),
            "summary": summary,
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx/proxy buffering
        },
    )


# ── Ray request model ────────────────────────────────────────

class RayRequest(BaseModel):
    """Request model for the /ray endpoint.

    Traces a single photon ray through Kerr-Newman spacetime and
    returns the full trajectory, equatorial plane crossings, and
    disk physics (g-factor, Novikov-Thorne flux, temperature).

    Supports two input modes:
      - pixel: specify (ix, iy) pixel coordinates
      - impact_parameter: specify (alpha, beta) directly in radians
    """
    # ── Ray specification ──────────────────────────────────
    mode: str = Field("pixel", pattern=r"^(pixel|impact_parameter)$",
                      description="Input mode: 'pixel' for screen coordinates, 'impact_parameter' for direct angles")

    # Pixel mode inputs
    ix: Optional[int] = Field(None, ge=0, description="Pixel x-coordinate (required if mode='pixel')")
    iy: Optional[int] = Field(None, ge=0, description="Pixel y-coordinate (required if mode='pixel')")

    # Impact parameter mode inputs
    alpha: Optional[float] = Field(None, description="Horizontal impact parameter in radians (required if mode='impact_parameter')")
    beta: Optional[float] = Field(None, description="Vertical impact parameter in radians (required if mode='impact_parameter')")

    # ── Black hole parameters ──────────────────────────────
    spin: float = Field(0.6, ge=0, le=0.998)
    charge: float = Field(0.0, ge=0, le=0.998)
    inclination: float = Field(80.0, ge=3, le=89)

    # ── Integration parameters ─────────────────────────────
    method: str = Field("rkdp8", pattern=r"^(rk4|rkdp8|kahanli8s|kahanli8s_ks|tao_yoshida4|tao_yoshida6|tao_kahan_li8)$")
    fov: float = Field(8.0, ge=2, le=25)
    width: int = Field(320, ge=16, le=3840)
    height: int = Field(180, ge=16, le=2160)
    steps: int = Field(200, ge=60, le=500)
    step_size: float = Field(0.3, ge=0.1, le=0.8)
    obs_dist: int = Field(40, ge=20, le=100)
    phi0: float = Field(0.0)
    doppler_boost: int = Field(default=2, ge=0, le=2,
                               description="Doppler boost mode: 0=off, 1=g^3 optically thin, 2=g^4 optically thick")
    srgb_output: bool = Field(default=True, description="Apply IEC 61966-2-1 sRGB transfer function")
    disk_alpha: float = Field(default=0.95, ge=0.0, le=1.0, description="Base opacity per disk crossing")
    disk_max_crossings: int = Field(default=5, ge=1, le=20, description="Maximum disk crossings to accumulate")

    # ── Disk parameters ────────────────────────────────────
    disk_temp: float = Field(1.0, ge=0.2, le=2.5)

    # ── Output control ─────────────────────────────────────
    include_trajectory: bool = Field(True, description="Include full trajectory arrays in response")
    include_disk_physics: bool = Field(True, description="Include disk crossing physics in response")
    max_trajectory_points: int = Field(200, ge=10, le=500,
                                       description="Maximum number of trajectory points to record")

    @model_validator(mode="after")
    def check_spin_charge_constraint(self):
        if self.spin ** 2 + self.charge ** 2 > 1.0:
            raise ValueError("spin² + charge² must be ≤ 1 (naked singularity constraint)")
        return self

    @model_validator(mode="after")
    def check_mode_inputs(self):
        if self.mode == "pixel":
            if self.ix is None:
                self.ix = self.width // 2
            if self.iy is None:
                self.iy = self.height // 2
        elif self.mode == "impact_parameter":
            if self.alpha is None or self.beta is None:
                raise ValueError("alpha and beta are required when mode='impact_parameter'")
        return self


@app.post("/ray")
async def ray(req: RayRequest):
    """Trace a single photon ray through Kerr-Newman spacetime.

    General-purpose single-ray tracing endpoint that returns the full
    geodesic trajectory, equatorial plane crossings with disk physics
    (g-factor, Novikov-Thorne flux, temperature), and termination
    conditions.

    Supports two input modes:
      - pixel: specify screen coordinates (ix, iy) to trace
      - impact_parameter: specify (α, β) angles directly

    The response includes:
      - Initial and final ray state (r, θ, φ, pr, pθ)
      - Full trajectory arrays (r, θ, φ, step sizes)
      - Disk crossing details with GR redshift and thermal physics
      - Termination reason and step count
      - GPU kernel timing
    """
    if not renderer._initialized:
        raise HTTPException(status_code=503, detail="CUDA renderer not available")

    # Build params dict for the renderer
    params = {
        "mode": req.mode,
        "spin": req.spin,
        "charge": req.charge,
        "inclination": req.inclination,
        "method": req.method,
        "fov": req.fov,
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "step_size": req.step_size,
        "obs_dist": req.obs_dist,
        "phi0": req.phi0,
        "doppler_boost": req.doppler_boost,
        "srgb_output": req.srgb_output,
        "disk_alpha": req.disk_alpha,
        "disk_max_crossings": req.disk_max_crossings,
        "disk_temp": req.disk_temp,
        "max_trajectory_points": req.max_trajectory_points,
    }

    if req.mode == "pixel":
        params["ix"] = req.ix
        params["iy"] = req.iy
    else:
        params["alpha"] = req.alpha
        params["beta"] = req.beta

    try:
        async with gpu_lock:
            result = await asyncio.get_running_loop().run_in_executor(
                None, renderer.trace_single_ray, params
            )
    except Exception as e:
        logger.error("Ray trace failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ray trace failed: {e}")

    # Optionally strip trajectory data
    if not req.include_trajectory:
        result["trajectory"] = {"points": result["trajectory"]["points"]}

    # Optionally strip disk physics from crossings
    if not req.include_disk_physics:
        for crossing in result.get("disk_crossings", []):
            for key in ("g_factor", "novikov_thorne_flux", "T_emit", "T_observed"):
                crossing.pop(key, None)

    # Build warnings list for API consumers
    warnings = []
    if result["spacetime"]["effective_method"] != result["spacetime"]["method"]:
        warnings.append(
            f"Requested method '{result['spacetime']['method']}' is not natively supported "
            f"for single-ray tracing; fell back to '{result['spacetime']['effective_method']}'. "
            f"Results may differ from /bench endpoint."
        )
        logger.warning(
            "RAY METHOD MISMATCH: requested '%s' but executed '%s' — "
            "results will differ from /bench endpoint for this method",
            result["spacetime"]["method"],
            result["spacetime"]["effective_method"],
        )
    result["warnings"] = warnings

    logger.info(
        "Ray traced: mode=%s method=%s (effective=%s) spin=%.3f term=%s steps=%d/%d crossings=%d in %.1fms",
        req.mode, req.method, result["spacetime"]["effective_method"], req.spin,
        result["termination"]["reason"],
        result["termination"]["steps_used"],
        result["termination"]["steps_max"],
        len(result["disk_crossings"]),
        result["timing"]["total_ms"],
    )

    return result


# ── WebSocket streaming endpoint ────────────────────────────


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    client_info = websocket.client
    logger.info("WebSocket client connected: %s", client_info)

    # Shared state between receiver and render loop
    latest_request = None        # Most recent validated (render_params, fmt, quality, width, height)
    request_event = asyncio.Event()  # Signaled when a new request arrives
    closed = False

    async def receiver():
        """Read messages from the client, validate, and update latest_request."""
        nonlocal latest_request, closed
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    raw = json.loads(data)
                    req = RenderRequest(**raw)
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": str(e)}))
                    continue

                render_params = req.model_dump()
                fmt = render_params.pop("format", "jpeg")
                quality = render_params.pop("quality", 80)
                width = render_params["width"]
                height = render_params["height"]

                latest_request = (render_params, fmt, quality, width, height)
                request_event.set()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug("WS receiver ended: %s", e)
        finally:
            closed = True
            request_event.set()  # Unblock render loop so it can exit

    async def render_loop():
        """Wait for requests and render the latest one."""
        nonlocal latest_request
        try:
            while not closed:
                # Wait for a new request
                await request_event.wait()
                request_event.clear()

                if closed:
                    break

                # Grab the latest request (may have been updated multiple times)
                current = latest_request
                if current is None:
                    continue

                render_params, fmt, quality, width, height = current

                # Check cache first
                cached = cache.get(render_params)
                if cached:
                    fmt_byte = 1 if fmt == "webp" else 0
                    header = struct.pack("<HHB3x", width, height, fmt_byte)
                    try:
                        await websocket.send_bytes(header + cached)
                    except Exception:
                        break
                    continue

                # Check if a newer request arrived while we were checking cache
                if latest_request is not current:
                    continue

                # Acquire GPU and render
                t0 = time.time()
                async with gpu_lock:
                    # Check again after acquiring lock — a newer request may have arrived
                    if latest_request is not current:
                        continue

                    try:
                        raw_rgb = await asyncio.get_running_loop().run_in_executor(
                            None, renderer.render_frame, render_params
                        )
                    except Exception as e:
                        try:
                            await websocket.send_text(json.dumps({"error": f"Render failed: {e}"}))
                        except Exception:
                            break
                        continue

                render_ms = (time.time() - t0) * 1000

                # Discard if a newer request arrived during rendering
                if latest_request is not current:
                    logger.debug("WS render discarded (stale) after %.1fms", render_ms)
                    continue

                # Encode image
                # Note: renderer.render_frame() already applies np.flipud()
                # to convert from GPU bottom-to-top to standard top-to-bottom
                # row order, so no additional flip is needed here.
                img = Image.frombytes("RGB", (width, height), raw_rgb)

                buf = io.BytesIO()
                if fmt == "webp":
                    img.save(buf, "WEBP", quality=quality)
                    fmt_byte = 1
                else:
                    img.save(buf, "JPEG", quality=quality)
                    fmt_byte = 0
                image_bytes = buf.getvalue()

                # Cache the result
                cache.put(render_params, image_bytes)

                # Send frame with header
                header = struct.pack("<HHB3x", width, height, fmt_byte)
                try:
                    await websocket.send_bytes(header + image_bytes)
                    logger.debug("WS frame sent: %dx%d %s %.1fms", width, height, fmt, render_ms)
                except Exception:
                    break
        except Exception as e:
            logger.error("WS render loop error: %s", e)

    # Run receiver and render loop concurrently
    receiver_task = asyncio.create_task(receiver())
    render_task = asyncio.create_task(render_loop())

    try:
        # Wait for either task to finish (receiver finishes on disconnect)
        done, pending = await asyncio.wait(
            [receiver_task, render_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        logger.error("WS stream error: %s", e)
    finally:
        logger.info("WebSocket client disconnected: %s", client_info)
