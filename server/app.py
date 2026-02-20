"""
FastAPI application for the Nulltracer server-side renderer.
GPU-accelerated ray tracing through curved spacetimes.

Provides a POST /render endpoint that accepts render parameters,
renders a black hole frame via CUDA compute kernels, and returns a
JPEG or WebP image.

Also provides a POST /bench endpoint that renders a frame using ALL
available integrators and returns runtime statistics + images for
side-by-side comparison.
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

logger = logging.getLogger(__name__)

# ── Pydantic request model ──────────────────────────────────

class RenderRequest(BaseModel):
    spin: float = Field(0.6, ge=0, le=0.998)
    charge: float = Field(0.0, ge=0, le=0.998)
    inclination: float = Field(80.0, ge=3, le=89)
    fov: float = Field(8.0, ge=2, le=25)
    width: int = Field(1280, ge=160, le=3840)
    height: int = Field(720, ge=90, le=2160)
    method: str = Field("yoshida4", pattern=r"^(yoshida4|rk4|yoshida6|yoshida8|rkdp8|kahanli8s|kahanli8s_ks)$")
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

# ── FastAPI app ──────────────────────────────────────────────

app = FastAPI(title="Nulltracer — GPU-Accelerated Curved Spacetime Renderer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
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


@app.get("/debug_ray")
async def debug_ray(
    ix: int = 160,
    iy: int = 90,
    spin: float = 0.6,
    charge: float = 0.0,
    inclination: float = 80.0,
    fov: float = 8.0,
    width: int = 320,
    height: int = 180,
    steps: int = 200,
    step_size: float = 0.3,
    obs_dist: int = 40,
    bg_mode: int = 1,
    star_layers: int = 3,
    disk_temp: float = 1.0,
):
    """Debug endpoint: trace a single ray and report its trajectory."""
    import math as _math
    import numpy as _np
    import cupy as _cp

    if not renderer._initialized:
        raise HTTPException(status_code=503, detail="CUDA renderer not available")

    from .isco import isco as _isco
    isco_r = _isco(spin, charge)

    debug_src = r'''
struct RenderParams {
    double width, height, spin, charge, incl, fov, phi0, isco;
    double steps, obs_dist, esc_radius, disk_outer, step_size;
    double bg_mode, star_layers, show_disk, show_grid, disk_temp;
    double doppler_boost;
};
#define PI  3.14159265358979323846
#define S2_EPS 0.0004

__device__ void geoRHS_dbg(
    double r, double th, double pr, double pth,
    double a, double b, double Q2,
    double *dr, double *dth, double *dphi, double *dpr, double *dpth
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth*sth + S2_EPS, c2 = cth*cth;
    double a2 = a*a, r2 = r*r;
    double sig = r2 + a2*c2;
    double del = r2 - 2.0*r + a2 + Q2;
    double sdel = fmax(del, 1e-14);
    double rpa2 = r2 + a2;
    double w = 2.0*r - Q2;
    double A_ = rpa2*rpa2 - sdel*a2*s2;
    double isig = 1.0/sig;
    double SD = sig*sdel;
    double iSD = 1.0/SD;
    double is2 = 1.0/s2;
    double grr = sdel*isig, gthth = isig;
    double gff = (sig - w)*iSD*is2;
    double gtf = -a*w*iSD;
    *dr = grr*pr; *dth = gthth*pth; *dphi = gff*b - gtf;
    double dsig_r=2.0*r, ddel_r=2.0*r-2.0;
    double dA_r=4.0*r*rpa2-ddel_r*a2*s2;
    double dSD_r=dsig_r*sdel+sig*ddel_r;
    double dgtt_r=-(dA_r*SD-A_*dSD_r)/(SD*SD);
    double dgtf_r=-a*(2.0*SD-w*dSD_r)/(SD*SD);
    double dgrr_r=(ddel_r*sig-sdel*dsig_r)/(sig*sig);
    double dgthth_r=-dsig_r*isig*isig;
    double num_ff=sig-w, den_ff=SD*s2;
    double dgff_r=((dsig_r-2.0)*den_ff-num_ff*dSD_r*s2)/(den_ff*den_ff);
    *dpr=-0.5*(dgtt_r-2.0*b*dgtf_r+dgrr_r*pr*pr+dgthth_r*pth*pth+dgff_r*b*b);
    double dsig_th=-2.0*a2*sth*cth, ds2_th=2.0*sth*cth;
    double dA_th=-sdel*a2*ds2_th, dSD_th=dsig_th*sdel;
    double dgtt_th=-(dA_th*SD-A_*dSD_th)/(SD*SD);
    double dgtf_th=a*w*dSD_th/(SD*SD);
    double dgrr_th=-sdel*dsig_th/(sig*sig);
    double dgthth_th=-dsig_th*isig*isig;
    double dgff_th=(dsig_th*den_ff-num_ff*(dsig_th*sdel*s2+SD*ds2_th))/(den_ff*den_ff);
    *dpth=-0.5*(dgtt_th-2.0*b*dgtf_th+dgrr_th*pr*pr+dgthth_th*pth*pth+dgff_th*b*b);
}

extern "C" __global__
void debug_trace(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;
    int ix = (int)output[200], iy = (int)output[201];
    double asp = p.width / p.height;
    double ux = (2.0*(ix+0.5)/p.width - 1.0);
    double uy = (2.0*(iy+0.5)/p.height - 1.0);
    double alpha = ux*p.fov*asp, beta = uy*p.fov;
    double a = p.spin, a2 = a*a, Q2 = p.charge*p.charge;
    double thObs = p.incl, sO = sin(thObs), cO = cos(thObs);
    double b = -alpha*sO;
    double r = p.obs_dist, th = thObs, phi = p.phi0;
    double sth = sin(thObs), cth = cos(thObs);
    double s2 = sth*sth+S2_EPS, c2 = cth*cth;
    double r0 = p.obs_dist, r02 = r0*r0;
    double sig = r02+a2*c2, del = r02-2.0*r0+a2+Q2;
    double sdel = fmax(del,1e-14), rpa2 = r02+a2;
    double A_ = rpa2*rpa2-sdel*a2*s2;
    double iSD = 1.0/(sig*sdel), is2 = 1.0/s2;
    double grr = sdel/sig, gthi = 1.0/sig;
    double w_init = 2.0*r0-Q2;
    double pth = -beta;
    double rest = -A_*iSD + 2.0*a*b*w_init*iSD + gthi*beta*beta + (sig-w_init)*iSD*is2*b*b;
    double pr2 = -rest/grr;
    double pr = (pr2 > 0.0) ? -sqrt(pr2) : 0.0;
    double rp = 1.0 + sqrt(fmax(1.0-a2-Q2, 0.0));

    output[0]=r; output[1]=th; output[2]=pr; output[3]=pth;
    output[4]=b; output[5]=rp; output[6]=alpha; output[7]=beta;
    output[14]=p.esc_radius; output[15]=p.obs_dist;
    output[16]=p.step_size; output[17]=p.steps;
    output[18]=p.width; output[19]=p.height;

    int STEPS = (int)p.steps, term_reason = 0, steps_used = 0;
    double Y4_W1=1.3512071919596576, Y4_W0=-1.7024143839193153;
    double Y4_D1=0.6756035959798288, Y4_D0=-0.1756035959798288;

    for (int i = 0; i < STEPS; i++) {
        steps_used = i+1;
        double he = p.step_size * fmin(fmax((r-rp)*0.4, 0.04), 1.0);
        he = fmin(fmax(he, 0.012), 0.6);
        if (i < 100) output[20+i] = r;
        if (i < 20) output[120+i] = he;
        double oldR = r, oldTh = th;
        double dr_,dth_,dphi_,dpr_,dpth_;
        geoRHS_dbg(r,th,pr,pth,a,b,Q2,&dr_,&dth_,&dphi_,&dpr_,&dpth_);
        r+=he*Y4_D1*dr_; th+=he*Y4_D1*dth_; phi+=he*Y4_D1*dphi_;
        geoRHS_dbg(r,th,pr,pth,a,b,Q2,&dr_,&dth_,&dphi_,&dpr_,&dpth_);
        pr+=he*Y4_W1*dpr_; pth+=he*Y4_W1*dpth_;
        geoRHS_dbg(r,th,pr,pth,a,b,Q2,&dr_,&dth_,&dphi_,&dpr_,&dpth_);
        r+=he*Y4_D0*dr_; th+=he*Y4_D0*dth_; phi+=he*Y4_D0*dphi_;
        geoRHS_dbg(r,th,pr,pth,a,b,Q2,&dr_,&dth_,&dphi_,&dpr_,&dpth_);
        pr+=he*Y4_W0*dpr_; pth+=he*Y4_W0*dpth_;
        geoRHS_dbg(r,th,pr,pth,a,b,Q2,&dr_,&dth_,&dphi_,&dpr_,&dpth_);
        r+=he*Y4_D1*dr_; th+=he*Y4_D1*dth_; phi+=he*Y4_D1*dphi_;
        geoRHS_dbg(r,th,pr,pth,a,b,Q2,&dr_,&dth_,&dphi_,&dpr_,&dpth_);
        pr+=he*Y4_W1*dpr_; pth+=he*Y4_W1*dpth_;
        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI-0.005) { th = PI-0.005; pth = -fabs(pth); }
        if (r <= rp*1.01) { term_reason = 1; break; }
        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }
    output[8]=r; output[9]=th; output[10]=pr; output[11]=pth;
    output[12]=(double)term_reason; output[13]=(double)steps_used;
}
'''

    kernel = _cp.RawKernel(debug_src, 'debug_trace', options=('--std=c++14',))

    from .renderer import RenderParams as RP
    rp_struct = RP(
        width=width, height=height,
        spin=float(spin), charge=float(charge),
        incl=_math.radians(float(inclination)),
        fov=float(fov), phi0=0.0, isco=float(isco_r),
        steps=int(steps), obs_dist=float(obs_dist),
        esc_radius=float(obs_dist) + 12.0,
        disk_outer=14.0, step_size=float(step_size),
        bg_mode=int(bg_mode), star_layers=int(star_layers),
        show_disk=1, show_grid=1, disk_temp=float(disk_temp),
        doppler_boost=2.0,
    )

    params_bytes = bytes(rp_struct)
    h_params = _np.frombuffer(params_bytes, dtype=_np.uint8)
    d_params = _cp.asarray(h_params)
    d_output = _cp.zeros(202, dtype=_cp.float64)
    d_output[200] = float(ix)
    d_output[201] = float(iy)

    async with gpu_lock:
        kernel((1,), (1,), (d_params, d_output))
        _cp.cuda.Stream.null.synchronize()

    result = d_output.get()
    term_names = {0: "steps_exhausted", 1: "horizon", 2: "escape", 3: "nan", 4: "underflow"}
    steps_used = int(result[13])

    return {
        "pixel": {"ix": ix, "iy": iy},
        "initial": {
            "r": result[0], "th": result[1], "pr": result[2], "pth": result[3],
            "b": result[4], "rp": result[5], "alpha": result[6], "beta": result[7],
        },
        "final": {
            "r": result[8], "th": result[9], "pr": result[10], "pth": result[11],
        },
        "termination": term_names.get(int(result[12]), f"unknown({int(result[12])})"),
        "steps_used": steps_used,
        "params_read_by_kernel": {
            "esc_radius": result[14], "obs_dist": result[15],
            "step_size": result[16], "steps": result[17],
            "width": result[18], "height": result[19],
        },
        "r_trajectory_sample": [result[20 + i] for i in range(min(100, steps_used))],
        "he_first_20": [result[120 + i] for i in range(min(20, steps_used))],
    }


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
