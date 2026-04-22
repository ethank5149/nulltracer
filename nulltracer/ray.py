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


def trace_ray(
    *,
    spin: float = 0.6,
    charge: float = 0.0,
    inclination_deg: float = 80.0,
    fov: float = 8.0,
    obs_dist: float = 40.0,
    method: str = "rkdp8",
    steps: int = 200,
    step_size: float = 0.30,
    mode: str = "pixel",
    ix: int | None = None,
    iy: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    width: int = 320,
    height: int = 180,
    max_trajectory_points: int = 200,
    disk_temp: float = 1.0,
    doppler_boost: float = 2.0,
    phi0: float = 0.0,
) -> dict[str, Any]:
    """Trace a single null geodesic and return full diagnostic data.

    Parameters
    ----------
    spin, charge : float
        Black-hole parameters.
    inclination_deg : float
        Observer inclination in degrees.
    mode : str
        ``"pixel"`` (specify *ix*, *iy*) or ``"impact_parameter"``
        (specify *alpha*, *beta* in degrees).
    ix, iy : int
        Pixel coordinates when ``mode="pixel"``.  Default: image centre.
    alpha, beta : float
        Impact-parameter angles (degrees) when ``mode="impact_parameter"``.
    max_trajectory_points : int
        Maximum trajectory samples to record (capped at 500).

    Returns
    -------
    dict
        Keys: ``ray``, ``spacetime``, ``initial_state``,
        ``final_state``, ``termination``, ``trajectory``,
        ``disk_crossings``, ``timing``.
    """
    kernel = _kc.get_ray_trace_kernel(method)
    max_traj = min(max_trajectory_points, 500)

    isco_r = isco(spin, charge)

    rp = RenderParams(
        width=float(width),
        height=float(height),
        spin=float(spin),
        charge=float(charge),
        incl=math.radians(inclination_deg),
        fov=float(fov),
        phi0=float(phi0),
        isco=float(isco_r),
        steps=float(steps),
        obs_dist=float(obs_dist),
        esc_radius=float(obs_dist) + 12.0,
        disk_outer=50.0,
        step_size=float(step_size),
        bg_mode=0.0,
        star_layers=1.0,
        show_disk=1.0,
        show_grid=0.0,
        disk_temp=float(disk_temp),
        doppler_boost=float(doppler_boost),
        srgb_output=1.0,
        disk_alpha=0.95,
        disk_max_crossings=5.0,
        bloom_enabled=0.0,
        sky_width=0.0,
        sky_height=0.0,
    )

    d_params = rp.to_gpu()

    # Output buffer layout:
    #   Header:     20 doubles
    #   Trajectory: max_traj ?? 4 doubles (r, ??, ??, h_e)
    #   Crossings:  1 + 16 ?? 8 = 129 doubles
    max_crossings = 16
    output_size = 20 + max_traj * 4 + 1 + max_crossings * 8
    d_output = cp.zeros(output_size, dtype=cp.float64)

    # Write input specification
    if mode == "impact_parameter":
        d_output[0] = 1.0
        d_output[1] = float(alpha or 0.0)
        d_output[2] = float(beta or 0.0)
    else:
        d_output[0] = 0.0
        d_output[1] = float(ix if ix is not None else width // 2)
        d_output[2] = float(iy if iy is not None else height // 2)
    d_output[3] = float(max_traj)

    # Launch
    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()

    t_wall = _time.perf_counter()
    start_ev.record()
    kernel((1,), (1,), (d_params, d_output))
    end_ev.record()
    end_ev.synchronize()
    kernel_ms = cp.cuda.get_elapsed_time(start_ev, end_ev)
    total_ms = (_time.perf_counter() - t_wall) * 1000.0

    result = d_output.get()

    # Parse header
    steps_used = int(result[14])
    term_code = int(result[13])
    term_names = {
        0: "steps_exhausted",
        1: "horizon",
        2: "escape",
        3: "nan",
        4: "underflow",
    }

    def _safe(v: float):
        return None if (math.isnan(v) or math.isinf(v)) else v

    # Parse trajectory
    traj_pts = min(steps_used, max_traj)
    traj_base = 20
    traj = {
        "points": traj_pts,
        "r": [float(result[traj_base + i * 4]) for i in range(traj_pts)],
        "theta": [float(result[traj_base + i * 4 + 1]) for i in range(traj_pts)],
        "phi": [float(result[traj_base + i * 4 + 2]) for i in range(traj_pts)],
        "step_sizes": [float(result[traj_base + i * 4 + 3]) for i in range(traj_pts)],
    }

    # Parse disk crossings
    cx_base = traj_base + max_traj * 4
    n_cx = int(result[cx_base])
    crossings = []
    for j in range(n_cx):
        off = cx_base + 1 + j * 8
        crossings.append({
            "crossing_index": int(result[off]),
            "r": float(result[off + 1]),
            "phi": float(result[off + 2]),
            "direction": "north_to_south" if result[off + 3] > 0 else "south_to_north",
            "g_factor": float(result[off + 4]),
            "novikov_thorne_flux": float(result[off + 5]),
            "T_emit": float(result[off + 6]),
            "T_observed": float(result[off + 7]),
        })

    # Build ray info
    ray_info: dict = {"mode": mode}
    if mode == "impact_parameter":
        ray_info["alpha"] = alpha or 0.0
        ray_info["beta"] = beta or 0.0
    else:
        ray_info["ix"] = ix if ix is not None else width // 2
        ray_info["iy"] = iy if iy is not None else height // 2
    ray_info["b"] = float(result[5])

    return {
        "ray": ray_info,
        "spacetime": {
            "spin": float(spin),
            "charge": float(charge),
            "r_plus": float(result[15]),
            "r_isco": float(result[16]),
            "method": method,
        },
        "initial_state": {
            "r": float(result[0]),
            "theta": float(result[1]),
            "phi": float(result[2]),
            "pr": float(result[3]),
            "pth": float(result[4]),
        },
        "final_state": {
            "r": _safe(float(result[8])),
            "theta": _safe(float(result[9])),
            "phi": _safe(float(result[10])),
            "pr": _safe(float(result[11])),
            "pth": _safe(float(result[12])),
        },
        "termination": {
            "reason": term_names.get(term_code, f"unknown({term_code})"),
            "steps_used": steps_used,
            "steps_max": steps,
        },
        "trajectory": traj,
        "disk_crossings": crossings,
        "timing": {
            "kernel_ms": round(float(kernel_ms), 3),
            "total_ms": round(total_ms, 3),
        },
    }
