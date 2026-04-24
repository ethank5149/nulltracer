"""
Nulltracer ??? GPU-accelerated ray tracing through Kerr-Newman spacetime.

Usage:
    from nulltracer import render_frame, isco_kerr

    image = render_frame(spin=0.94, inclination=30, fov=10)
    # image is a numpy array of shape (height, width, 3), dtype uint8
"""

__version__ = "0.9.0"

# ISCO calculations - no GPU required
from .isco import isco_kerr, isco_kn, isco


_LAZY_IMPORTS = {
    "extract_shadow_metrics": ".eht_validation",
    "CudaRenderer": ".renderer",
    "RenderParams": ".renderer",
    "render_frame": ".render",
    "classify_shadow": ".render",
    "compile_all": ".render",
    "available_methods": ".render",
    "auto_steps": ".render",
    "RenderInfo": ".render",
    "ClassifyInfo": ".render",
    "trace_ray": ".ray",
    "shadow_boundary": ".compare",
    "compare_integrators": ".compare",
    "fit_ellipse_to_shadow": ".compare",
    "load_skymap": ".skymap",
    "get_skymap": ".skymap",
    "clear_skymap": ".skymap",
    "KernelCache": "._kernel_utils",
}

def __getattr__(name):
    """Lazy load GPU-dependent modules."""
    if name in _LAZY_IMPORTS:
        try:
            import importlib
            module = importlib.import_module(_LAZY_IMPORTS[name], package=__name__)
            return getattr(module, name)
        except ImportError:
            raise AttributeError(f"Module 'nulltracer' has no attribute '{name}' (CuPy may not be installed)")
    raise AttributeError(f"Module 'nulltracer' has no attribute '{name}'")


__all__ = [
    "isco_kerr",
    "isco_kn",
    "isco",
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
