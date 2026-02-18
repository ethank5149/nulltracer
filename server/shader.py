"""
Shader orchestrator for the Nulltracer server-side renderer.

Composes the GLSL fragment shader from modular components:
  - shader_base: header defines, utility functions, geoRHS
  - backgrounds: bgStars, bgChecker, bgColorMap, background, sphereDir, disk
  - integrators: per-method main() loop (rk4, yoshida4, yoshida6, yoshida8, rkdp8)

The build_frag_src() function signature and return value are unchanged.
"""

from .shader_base import shader_header, common_glsl_prefix, geo_rhs_function
from .backgrounds import background_functions
from .integrators import INTEGRATORS

VERTEX_SHADER_SRC = "#version 120\n" \
    "attribute vec2 a_pos; varying vec2 v_uv; void main(){ v_uv=a_pos; gl_Position=vec4(a_pos,0,1); }"


def build_frag_src(opts: dict) -> str:
    """Build the fragment shader source from options.

    Args:
        opts: dict with keys method, steps, obsDist, starLayers, stepSize, bgMode

    Returns:
        Complete GLSL fragment shader source string with #version 120 prefix.
    """
    STEPS = opts.get("steps", 200)
    METHOD = opts.get("method", "yoshida4")
    R0 = opts.get("obsDist", 40)
    RESC = R0 + 12
    STAR_LAYERS = opts.get("starLayers", 3)
    H_BASE = opts.get("stepSize", 0.30)
    BG_MODE = opts.get("bgMode", 1)

    # Compose the shader from modular components
    src = shader_header(STEPS, R0, RESC, H_BASE, STAR_LAYERS, BG_MODE)
    src += common_glsl_prefix()
    src += background_functions()
    src += geo_rhs_function()

    # Get the integrator-specific main() loop
    integrator_fn = INTEGRATORS.get(METHOD, INTEGRATORS["yoshida4"])
    src += integrator_fn()

    return "#version 120\n" + src
