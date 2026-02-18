"""
Integrator registry for the Nulltracer shader generator.

Each integrator module exports a function that returns the GLSL source
for its specific main() loop (including constants like Yoshida coefficients).
The geoRHS() function is shared and defined once in shader_base.py.
"""

from .rk4 import rk4_integrator
from .yoshida4 import yoshida4_integrator
from .yoshida6 import yoshida6_integrator
from .yoshida8 import yoshida8_integrator
from .rkdp8 import rkdp8_integrator

INTEGRATORS = {
    "rk4": rk4_integrator,
    "yoshida4": yoshida4_integrator,
    "yoshida6": yoshida6_integrator,
    "yoshida8": yoshida8_integrator,
    "rkdp8": rkdp8_integrator,
}
