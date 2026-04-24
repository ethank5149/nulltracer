import io
import json
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

from .renderer import CudaRenderer

app = FastAPI(title="Nulltracer Render Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

renderer = CudaRenderer()

@app.on_event("startup")
async def startup_event():
    """Initialize CUDA renderer on server startup."""
    renderer.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up CUDA resources."""
    renderer.shutdown()

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": renderer.gpu_info, "backends": ["cuda"]}

class RenderRequest(BaseModel):
    spin: float = 0.6
    charge: float = 0.0
    inclination: float = 80.0
    fov: float = 8.0
    obs_dist: float = 40.0
    width: int = 1280
    height: int = 720
    method: str = "rkn86"
    steps: Optional[int] = 2000
    step_size: float = 0.08
    bg_mode: int = 1
    show_disk: bool = True
    disk_mode: int = 1
    show_grid: bool = False
    disk_temp: float = 1.0
    star_layers: int = 3
    doppler_boost: int = 2
    phi0: float = 0.0
    srgb_output: bool = True
    disk_alpha: float = 0.95
    disk_max_crossings: int = 8
    disk_outer: float = 50.0
    aa_samples: int = 1
    qed_coupling: float = 0.0
    hawking_boost: float = 0.0
    format: str = "jpeg" # "jpeg" or "webp"

class RayRequest(BaseModel):
    spin: float = 0.6
    charge: float = 0.0
    inclination: float = 80.0
    fov: float = 8.0
    obs_dist: float = 40.0
    method: str = "rkn86"
    steps: Optional[int] = 2000
    step_size: float = 0.08
    mode: str = "pixel"
    ix: Optional[int] = None
    iy: Optional[int] = None
    alpha: float = 0.0
    beta: float = 0.0
    width: int = 1280
    height: int = 720
    max_trajectory_points: int = 200
    disk_temp: float = 1.0
    doppler_boost: int = 2
    phi0: float = 0.0

@lru_cache(maxsize=128)
def _cached_render(params_json: str, format: str) -> bytes:
    params = json.loads(params_json)
    res = renderer.render_frame_timed(params)
    img_data = res["raw_rgb"]
    width = params.get("width", 1280)
    height = params.get("height", 720)
    
    img_arr = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
    img = Image.fromarray(img_arr)
    
    buf = io.BytesIO()
    if format.lower() == "webp":
        img.save(buf, format="WEBP", quality=90)
    else:
        img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

@app.post("/render")
async def render_endpoint(req: RenderRequest):
    params_dict = req.dict(exclude={"format"})
    # Convert any None values to null in JSON, though we don't expect many
    params_json = json.dumps(params_dict, sort_keys=True)
    img_bytes = _cached_render(params_json, req.format)
    
    media_type = "image/webp" if req.format.lower() == "webp" else "image/jpeg"
    return Response(content=img_bytes, media_type=media_type)

@app.post("/ray")
async def ray_endpoint(req: RayRequest):
    params_dict = req.dict(exclude_none=True)
    res = renderer.trace_single_ray(params_dict)
    return res
