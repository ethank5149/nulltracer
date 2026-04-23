"""
Single-ray geodesic tracing for diagnostics and validation.

Traces one photon through Kerr???Newman spacetime and returns the
full trajectory, equatorial-plane crossings, and disk physics.
"""

from __future__ import annotations

import math
import time as _time
from typing import Any

import cupy as cp
import numpy as np

from ._kernel_utils import KernelCache
from ._params import RenderParams
from .isco import isco

__all__ = ["trace_ray"]

_kc = KernelCache()


def trace_ray(**kwargs) -> dict[str, Any]:
    """Trace a single null geodesic and return full diagnostic data.

    Parameters
    ----------
    **kwargs
        Forwarded to :func:`CudaRenderer.trace_single_ray`.

    Returns
    -------
    dict
        Keys: ``ray``, ``spacetime``, ``initial_state``,
        ``final_state``, ``termination``, ``trajectory``,
        ``disk_crossings``, ``timing``.
    """
    from .renderer import CudaRenderer
    renderer = CudaRenderer()
    renderer.initialize()
    return renderer.trace_single_ray(kwargs)
