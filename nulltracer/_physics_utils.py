"""
Physics utilities shared between rendering and ray tracing modules.

This module contains functions that are used by both render.py and renderer.py
to avoid circular import dependencies.
"""

import math


def auto_steps(
    obs_dist: float,
    step_size: float = 0.3,
    *,
    spin: float = 0.0,
    charge: float = 0.0,
    method: str = "rkn86",
    safety: float = 3.0,
) -> int:
    """Estimate the number of integration steps needed.

    Accounts for observer distance, step size, and integrator type.
    Symplectic methods use a fixed-step budget; adaptive methods
    scale with an approximate path-length estimate.

    Parameters
    ----------
    obs_dist : float
        Observer distance in M.
    step_size : float
        Base affine-parameter step size.
    spin, charge : float
        Black-hole parameters (used for horizon radius estimate).
    method : str
        Integrator name.
    safety : float
        Multiplicative safety factor.
    """
    rp = 1.0 + math.sqrt(max(1.0 - spin**2 - charge**2, 0.0))
    is_symp = method.startswith("tao") or method.startswith("kahan")

    if method == "_classify":
        N_near = 20.0 / step_size
        N_far = (2 * rp / step_size) * math.log(max(obs_dist / rp, 2.0))
        return max(int((N_near + N_far) * safety), 400)
    elif is_symp:
        return max(int((obs_dist + 200.0 / step_size) * safety), 400)
    else:
        h_scaled = step_size * (obs_dist / 30.0) * 1.7
        h_max = 3.0  # rkn86 and verner98 use similar step sizing
        if method in ("verner98", "rkn86"):
            # Higher-order adaptive methods need fewer steps
            h_max = 4.0
        N = obs_dist / min(h_scaled, h_max) + 60.0 / step_size
        return max(int(N * safety), 200)