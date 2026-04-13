"""
Nulltracer — GPU-accelerated null geodesic ray tracer for Kerr–Newman
black holes.

Quick start::

    import nulltracer as nt

    # Pre-compile all CUDA kernels (optional, avoids first-call latency)
    nt.compile_all()

    # Render an image
    img, info = nt.render_frame(spin=0.94, inclination_deg=17.0)

    # Classify shadow for measurement
    mask, info = nt.classify_shadow(spin=0.94, inclination_deg=17.0)

    # Trace a single diagnostic ray
    data = nt.trace_ray(spin=0.94, inclination_deg=17.0)
"""

__version__ = "1.0.0"

# ── Core rendering ─────────────────────────────────────────────
from .render import (
    render_frame,
    classify_shadow,
    compile_all,
    available_methods,
    auto_steps,
    RenderInfo,
    ClassifyInfo,
)

# ── Single-ray diagnostics ─────────────────────────────────────
from .ray import trace_ray

# ── ISCO and horizon ───────────────────────────────────────────
from .isco import isco, isco_kerr, isco_kn, isco_kerr_vec, r_plus

# ── Analytic shadow and comparison tools ───────────────────────
from .compare import (
    shadow_boundary,
    compare_integrators,
    fit_ellipse_to_shadow,
)

# ── Skymap management ──────────────────────────────────────────
from .skymap import load_skymap, get_skymap, clear_skymap

# ── Bloom post-processing ─────────────────────────────────────
from .bloom import apply_bloom

__all__ = [
    # rendering
    "render_frame",
    "classify_shadow",
    "compile_all",
    "available_methods",
    "auto_steps",
    "RenderInfo",
    "ClassifyInfo",
    # ray tracing
    "trace_ray",
    # isco / horizon
    "isco",
    "isco_kerr",
    "isco_kn",
    "isco_kerr_vec",
    "r_plus",
    # analytic / comparison
    "shadow_boundary",
    "compare_integrators",
    "fit_ellipse_to_shadow",
    # skymap
    "load_skymap",
    "get_skymap",
    "clear_skymap",
    # bloom
    "apply_bloom",
]
