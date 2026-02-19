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
    """Initialize renderers on server startup.

    CUDA renderer is the primary backend. OpenGL renderer is optional
    (kept temporarily for A/B validation during the CUDA migration).
    The server starts successfully if at least one backend initializes.
    """
    opengl_ok = False
    cuda_ok = False

    # Try OpenGL renderer (optional — may fail in CUDA-only containers)
    logger.info("Initializing OpenGL renderer...")
    try:
        renderer.initialize()
        logger.info("OpenGL renderer ready: %s", renderer.gpu_info)
        opengl_ok = True
    except Exception as e:
        logger.warning("OpenGL renderer failed to initialize: %s (CUDA-only mode)", e)

    # Initialize CUDA renderer (primary backend)
    logger.info("Initializing CUDA renderer...")
    try:
        cuda_renderer.initialize()
        logger.info("CUDA renderer ready: %s", cuda_renderer.gpu_info)
        cuda_ok = True
    except Exception as e:
        logger.warning("CUDA renderer failed to initialize: %s", e)

    if not opengl_ok and not cuda_ok:
        raise RuntimeError("No rendering backend available. Both OpenGL and CUDA failed to initialize.")

    logger.info("Backends available: %s",
                ", ".join(b for b, ok in [("OpenGL", opengl_ok), ("CUDA", cuda_ok)] if ok))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu": renderer.gpu_info,
        "cuda_gpu": cuda_renderer.gpu_info if cuda_renderer._initialized else "not initialized",
        "backends": (["opengl"] if renderer._initialized else []) + (["cuda"] if cuda_renderer._initialized else []),
    }


@app.post("/render")
async def render(req: RenderRequest):
    """Render a black hole frame and return the image.

    Checks the LRU cache first. On cache miss, renders via headless
    OpenGL (if available) or falls back to CUDA, encodes to JPEG/WebP,
    caches, and returns the image.
    """
    # If OpenGL is not available, redirect to CUDA backend
    if not renderer._initialized:
        if cuda_renderer._initialized:
            return await render_cuda(req)
        raise HTTPException(status_code=503, detail="No rendering backend available")
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

    if not cuda_renderer._initialized:
        raise HTTPException(status_code=503, detail="CUDA renderer not available")

    from .isco import isco as _isco
    isco_r = _isco(spin, charge)

    debug_src = r'''
struct RenderParams {
    double width, height, spin, charge, incl, fov, phi0, isco;
    double steps, obs_dist, esc_radius, disk_outer, step_size;
    double bg_mode, star_layers, show_disk, show_grid, disk_temp;
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
    double pth = beta;
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

    from .renderer_cuda import RenderParams as RP
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
    )

    params_bytes = bytes(rp_struct)
    h_params = _np.frombuffer(params_bytes, dtype=_np.uint8)
    d_params = _cp.asarray(h_params)
    d_output = _cp.zeros(202, dtype=_cp.float64)
    d_output[200] = float(ix)
    d_output[201] = float(iy)

    async with cuda_lock:
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
