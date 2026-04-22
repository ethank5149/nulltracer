"""
Nulltracer — GPU-accelerated ray tracing through Kerr-Newman spacetime.

Usage:
    from nulltracer import render_frame, isco_kerr

    image = render_frame(spin=0.94, inclination=30, fov=10)
    # image is a numpy array of shape (height, width, 3), dtype uint8
"""

__version__ = "0.9.0"

# ISCO calculations - no GPU required
from .isco import isco_kerr, isco_kn, isco


# Lazy imports for GPU-dependent modules
def __getattr__(name):
    """Lazy load GPU-dependent modules."""
    if name in ("apply_bloom", "extract_shadow_metrics", "CudaRenderer", "RenderParams"):
        try:
            if name == "apply_bloom":
                from .bloom import apply_bloom

                return apply_bloom
            elif name == "extract_shadow_metrics":
                from .eht_validation import extract_shadow_metrics

                return extract_shadow_metrics
            elif name in ("CudaRenderer", "RenderParams"):
                from .renderer import CudaRenderer, RenderParams

                if name == "CudaRenderer":
                    return CudaRenderer
                return RenderParams
        except ImportError:
            raise AttributeError(f"Module 'nulltracer' has no attribute '{name}' (CuPy may not be installed)")

    # Old API modules
    old_api_names = (
        "render_frame",
        "classify_shadow",
        "compile_all",
        "available_methods",
        "auto_steps",
        "RenderInfo",
        "ClassifyInfo",
        "trace_ray",
        "shadow_boundary",
        "compare_integrators",
        "fit_ellipse_to_shadow",
        "load_skymap",
        "get_skymap",
        "clear_skymap",
        "KernelCache",
    )
    if name in old_api_names:
        try:
            if name in (
                "render_frame",
                "classify_shadow",
                "compile_all",
                "available_methods",
                "auto_steps",
                "RenderInfo",
                "ClassifyInfo",
            ):
                from .render import (
                    render_frame,
                    classify_shadow,
                    compile_all,
                    available_methods,
                    auto_steps,
                    RenderInfo,
                    ClassifyInfo,
                )

                return locals()[name]
            elif name == "trace_ray":
                from .ray import trace_ray

                return trace_ray
            elif name in ("shadow_boundary", "compare_integrators", "fit_ellipse_to_shadow"):
                from .compare import shadow_boundary, compare_integrators, fit_ellipse_to_shadow

                return locals()[name]
            elif name in ("load_skymap", "get_skymap", "clear_skymap"):
                from .skymap import load_skymap, get_skymap, clear_skymap

                return locals()[name]
            elif name == "KernelCache":
                from ._kernel_utils import KernelCache

                return KernelCache
        except ImportError:
            raise AttributeError(f"Module 'nulltracer' has no attribute '{name}' (CuPy may not be installed)")

    raise AttributeError(f"Module 'nulltracer' has no attribute '{name}'")


__all__ = [
    "isco_kerr",
    "isco_kn",
    "isco",
    "apply_bloom",
    "extract_shadow_metrics",
    "CudaRenderer",
    "RenderParams",
    "render_frame",
    "trace_ray",
    "classify_shadow",
    "compile_all",
    "available_methods",
    "auto_steps",
    "RenderInfo",
    "ClassifyInfo",
    "shadow_boundary",
    "compare_integrators",
    "fit_ellipse_to_shadow",
    "load_skymap",
    "get_skymap",
    "clear_skymap",
    "KernelCache",
]
