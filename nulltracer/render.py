"""
Core rendering and shadow-classification functions.

Two rendering paths:

- :func:`render_frame` - Full visual pipeline (disk, stars, Doppler,
  tone mapping).  Returns an ``(H, W, 3)`` uint8 sRGB image.
- :func:`classify_shadow` - Lightweight shadow measurement.  Returns
  a boolean mask plus disk-crossing radii and g-factors.

Both dispatch to the same production CUDA kernels in ``kernels/``.
"""

from __future__ import annotations

import math
import warnings
import time as _time
from dataclasses import dataclass, field
from typing import Optional

import cupy as cp
import numpy as np
import tqdm

from ._kernel_utils import KernelCache
from ._params import RenderParams
from .isco import isco, isco_kerr
from .skymap import get_skymap

__all__ = [
    "render_frame",
    "classify_shadow",
    "auto_steps",
    "RenderInfo",
    "ClassifyInfo",
]

# Module-level kernel cache (shared across calls).
_kc = KernelCache()


def compile_all(*, verbose: bool = True) -> None:
    """Pre-compile every CUDA kernel.

    Useful at the top of a notebook to front-load compilation latency.
    """
    _kc.compile_all(verbose=verbose)


def available_methods() -> list[str]:
    """Return all integration method names with render kernels."""
    return _kc.available_methods


# ---- Step-budget heuristic ----


# Import shared utilities
from ._physics_utils import auto_steps


# ---- Render (full visual pipeline) ----


@dataclass
class RenderInfo:
    """Metadata returned alongside a rendered image."""

    render_ms: float
    method: str
    max_steps: int
    obs_dist: float
    width: int = 0
    height: int = 0


def render_frame(
    spin: float,
    inclination_deg: float,
    *,
    charge: float = 0.0,
    width: int = 512,
    height: int = 512,
    fov: float = 7.0,
    obs_dist: float = 40.0,
    max_steps: int | None = None,
    step_size: float = 0.3,
    method: str = "rkn86",
    show_disk: bool = True,
    bg_mode: int = 0,
    star_layers: int = 3,
    disk_temp: float = 1.0,
    doppler_boost: int = 2,
    phi0: float = 0.0,
    srgb_output: bool = True,
    disk_alpha: float = 0.95,
    disk_max_crossings: int = 5,
    disk_outer: float = 50.0,
    aa_samples: int = 1,
) -> tuple[np.ndarray, RenderInfo]:
    """Render a black-hole image using the production visual pipeline.

    Parameters
    ----------
    spin : float
        Dimensionless spin parameter, 0 <= a < 1.
    inclination_deg : float
        Observer inclination in degrees (0 = pole-on, 90 = equatorial).
    charge : float
        Dimensionless charge (Kerr-Newman), default 0.
    width, height : int
        Output resolution in pixels.
    fov : float
        Screen half-width in units of M (gravitational radii). Matches the
        CUDA kernel, which maps pixel column ux in [-1, +1] to impact
        parameter alpha = ux * fov * aspect. NOT an angle, despite the name.
    obs_dist : float
        Observer distance in gravitational radii (M).
    max_steps : int, optional
        Integration steps.  If *None*, estimated automatically.
    step_size : float
        Base affine-parameter step size.
    method : str
        Integration method (see :func:`available_methods`). Note: 'rkn86' is recommended for best performance and accuracy; 'tkl108' for long-term stability and photon rings.
    show_disk : bool
        Render the accretion disk.
    bg_mode : int
        Background mode: 0 = stars, 1 = checker, 2 = colormap, 3 = skymap.
    star_layers : int
        Number of procedural star layers (bg_mode 0).
    disk_temp : float
        Disk colour-temperature multiplier.
    doppler_boost : int
        0 = off, 1 = g^3 (optically thin), 2 = g^4 (optically thick).
    phi0 : float
        Azimuthal rotation offset (radians).
    srgb_output : bool
        Apply IEC 61966-2-1 sRGB transfer function.
    disk_alpha : float
        Base opacity per disk crossing.
    disk_max_crossings : int
        Maximum disk crossings to accumulate.
    disk_outer : float
        Outer edge of the disk in units of M (default 50.0). For Novikov-Thorne
        the emission falls off as ~1/r, so most of the bright region sits inside
        r less-than  5-10 M; a smaller ``disk_outer`` crops the invisible outer halo and
        can produce cleaner hero images.
    aa_samples : int
        Number of stochastic super-samples per pixel (default 1). Each sample
        jitters within the pixel and averages the result. 4-8 samples remove
        the visible staircase on lensed arcs at the cost of a proportional
        linear slow-down.


    Returns
    -------
    (image, info)
        ``image`` is an ``(H, W, 3)`` uint8 sRGB NumPy array.
        ``info`` is a :class:`RenderInfo` dataclass.
    """
    if method not in ['rkn86', 'verner98', 'tkl108']:
        warnings.warn(
            f"Integration method '{method}' is not recognized. "
            "Use 'rkn86' (default), 'tkl108' (symplectic), or 'verner98' (highest accuracy).",
            UserWarning,
            stacklevel=2
        )

    params = {
        "spin": spin, "charge": charge, "inclination": inclination_deg,
        "width": width, "height": height, "fov": fov, "obs_dist": obs_dist,
        "max_steps": max_steps, "step_size": step_size, "method": method,
        "show_disk": show_disk, "bg_mode": bg_mode, "star_layers": star_layers,
        "disk_temp": disk_temp, "doppler_boost": doppler_boost, "phi0": phi0,
        "srgb_output": srgb_output, "disk_alpha": disk_alpha,
        "disk_max_crossings": disk_max_crossings, "disk_outer": disk_outer,
        "aa_samples": aa_samples,
    }
    
    from .renderer import CudaRenderer
    renderer = CudaRenderer()
    renderer.initialize()

    total_pixels = width * height
    with tqdm.tqdm(total=total_pixels, desc="Rendering frame", unit="px", unit_scale=True) as pbar:
        def progress_update(pixels_done):
            pbar.update(pixels_done)
        
        res = renderer.render_frame_timed(params, progress_callback=progress_update)

    img = np.frombuffer(res["raw_rgb"], dtype=np.uint8).reshape((height, width, 3))
    
    info = RenderInfo(
        render_ms=res["kernel_ms"],
        method=method,
        max_steps=res["max_steps"],
        obs_dist=obs_dist,
        width=width,
        height=height,
    )
    return img, info


# ---- Classify (shadow measurement) ----


@dataclass
class ClassifyInfo:
    """Metadata returned alongside a shadow classification."""

    render_ms: float
    max_steps: int
    obs_dist: float
    r_disk: np.ndarray = field(repr=False)
    g_factor: np.ndarray = field(repr=False)


def classify_shadow(
    spin: float,
    inclination_deg: float,
    *,
    width: int = 512,
    height: int = 512,
    fov: float = 7.0,
    obs_dist: float = 500.0,
    max_steps: int | None = None,
    step_size: float = 0.15,
) -> tuple[np.ndarray, ClassifyInfo]:
    """Classify pixels as shadow / disk / escape for shadow measurement.

    Uses a dedicated lightweight RK4 kernel (no visual pipeline) with
    the production ``geoRHS`` for maximum accuracy.  The large default
    ``obs_dist`` (500 M) approximates the asymptotic observer limit
    needed for comparing against analytic shadow curves.

    Parameters
    ----------
    spin : float
        Dimensionless spin, 0 <= a < 1.
    inclination_deg : float
        Observer inclination in degrees.
    width, height : int
        Classification grid resolution.
    fov : float
        Screen half-width in units of M (gravitational radii). Matches the
        CUDA kernel, which maps pixel column ux in [-1, +1] to impact
        parameter alpha = ux * fov * aspect. NOT an angle, despite the name.
    obs_dist : float
        Observer distance in M.
    max_steps : int, optional
        If *None*, estimated automatically.
    step_size : float
        Base step size (smaller implies more accurate near horizon).

    Returns
    -------
    (shadow_mask, info)
        ``shadow_mask`` is a boolean ``(H, W)`` array (True = shadow).
        ``info`` is a :class:`ClassifyInfo` with ``r_disk`` and
        ``g_factor`` arrays.
    """
    kernel = _kc.get_classify_kernel()

    if max_steps is None:
        max_steps = auto_steps(
            obs_dist, step_size, spin=spin, method="_classify"
        )

    isco_r = isco_kerr(spin)
    n = width * height

    d_class = cp.zeros(n, dtype=cp.float64)
    d_rdisk = cp.zeros(n, dtype=cp.float64)
    d_g = cp.zeros(n, dtype=cp.float64)
    d_b = cp.zeros(n, dtype=cp.float64)

    block = (16, 16)
    grid = ((width + 15) // 16, (height + 15) // 16)

    t0 = _time.perf_counter()
    kernel(
        grid,
        block,
        (
            d_class,
            d_rdisk,
            d_g,
            d_b,
            np.int32(width),
            np.int32(height),
            np.float64(spin),
            np.float64(math.radians(inclination_deg)),
            np.float64(fov),
            np.float64(obs_dist),
            np.float64(float(isco_r)),
            np.int32(max_steps),
            np.float64(step_size),
        ),
    )
    cp.cuda.Device().synchronize()
    ms = (_time.perf_counter() - t0) * 1000.0

    h_class = np.flipud(d_class.get().reshape(height, width))
    h_rdisk = np.flipud(d_rdisk.get().reshape(height, width))
    h_g = np.flipud(d_g.get().reshape(height, width))

    info = ClassifyInfo(
        render_ms=ms,
        max_steps=max_steps,
        obs_dist=obs_dist,
        r_disk=h_rdisk,
        g_factor=h_g,
    )
    return (h_class == 1.0), info
