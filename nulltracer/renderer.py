"""
CUDA renderer using CuPy RawKernel for Kerr-Newman black hole ray tracing.

Replaces the EGL/OpenGL renderer with direct CUDA compute kernels.
All geodesic integration uses float64 for maximum accuracy.
"""

import ctypes
import logging
import math
import os
import time as _time
import warnings
from pathlib import Path
from typing import Optional

import cupy as cp
import numpy as np

from .isco import isco
from ._params import RenderParams
from ._kernel_utils import resolve_includes

logger = logging.getLogger(__name__)


def _safe_float(v: float) -> float:
    """Convert a float to a JSON-safe value, replacing NaN/Inf with None-safe sentinels."""
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _resolve_inclination(params: dict) -> float:
    """Return inclination in **degrees** from a params dict.

    Accepts either ``inclination`` (canonical, documented) or the shorthand
    alias ``incl`` that the notebook/UI uses. Defaults to 80.0 if neither
    is provided. Fixes the historical silent bug where dict-based renders
    always ran at 80?? regardless of the caller's ``incl`` setting.
    """
    if "inclination" in params:
        return float(params["inclination"])
    if "incl" in params:
        return float(params["incl"])
    return 80.0


def _resolve_steps(params: dict, *, spin: float, charge: float, method: str,
                   obs_dist: float, step_size: float) -> int:
    """Return the integration step budget.

    Uses an explicit ``steps`` / ``max_steps`` value if supplied, otherwise
    calls :func:`nulltracer.render.auto_steps` ??? matching the behaviour of
    the free ``nulltracer.render_frame`` entry point. Without this, dict-based
    renders at large ``obs_dist`` would terminate before reaching the black
    hole (the default of 200 steps ?? 0.15 step_size = 30 M of affine length).
    """
    if "steps" in params:
        return int(params["steps"])
    if "max_steps" in params and params["max_steps"] is not None:
        return int(params["max_steps"])
    # Lazy import to avoid circular dependency (render.py imports from renderer.py via __init__).
    from .render import auto_steps
    return auto_steps(obs_dist, step_size, spin=spin, charge=charge, method=method)

# Path to kernel source files
_KERNEL_DIR = Path(__file__).parent / "kernels"
_INTEGRATOR_DIR = _KERNEL_DIR / "integrators"

# Map method names to kernel source files and entry point names
_KERNEL_REGISTRY = {
    "rk4":      ("rk4.cu",      "trace_rk4"),
    "rk4_cks":  ("rk4_cks.cu",  "trace_rk4_cks"),
    "rkdp8":    ("rkdp8.cu",    "trace_rkdp8"),
    "rkdp8_cks":("rkdp8_cks.cu","trace_rkdp8_cks"),
    "symplectic8": ("symplectic8.cu", "trace_symplectic8"),
}

# Map method names to single-ray trace kernel entry points.
_RAY_TRACE_REGISTRY = {
    "rk4":          "ray_trace_rk4",
    "rk4_cks":      "ray_trace_rk4_cks",
    "rkdp8":        "ray_trace_rkdp8",
    "rkdp8_cks":    "ray_trace_rkdp8_cks",
    "symplectic8":  "ray_trace_symplectic8",
}


class CudaRenderer:
    """CUDA-based renderer for black hole ray tracing using CuPy."""

    def __init__(self):
        self._kernel_cache: dict[str, cp.RawKernel] = {}
        self._gpu_info: str = "unknown"
        self._initialized = False

    def initialize(self) -> None:
        """Initialize CUDA context and query GPU info."""
        if self._initialized:
            return

        try:
            # Query current device instead of forcing 0
            dev_id = cp.cuda.runtime.getDevice()

            # Query GPU info
            props = cp.cuda.runtime.getDeviceProperties(dev_id)
            name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            mem_gb = props["totalGlobalMem"] / (1024**3)
            cc = f"{props['major']}.{props['minor']}"
            self._gpu_info = f"{name} ({mem_gb:.1f} GB, compute {cc})"
            logger.info("CUDA GPU: %s", self._gpu_info)

            # Log CUDA version
            cuda_ver = cp.cuda.runtime.runtimeGetVersion()
            logger.info("CUDA runtime version: %d", cuda_ver)

            # Pre-compile the default kernel (rkdp8)
            self._get_kernel("rkdp8")
            logger.info("Default kernel (rkdp8) pre-compiled")

        except Exception as e:
            logger.error("CUDA initialization failed: %s", e, exc_info=True)
            raise RuntimeError(f"CUDA initialization failed: {e}")

        self._initialized = True
        logger.info("CUDA renderer initialized successfully")

    def purge_cache(self) -> int:
        """Clear all cached compiled kernels, forcing recompilation on next use.

        Also clears CuPy's internal kernel cache to ensure source changes
        are picked up even if the on-disk cache wasn't cleared.

        Returns the number of kernels that were cached.
        """
        count = len(self._kernel_cache)
        self._kernel_cache.clear()

        # Also clear CuPy's internal compilation cache
        try:
            cp.cuda.compiler._kernel_cache = {}
        except AttributeError:
            pass  # CuPy internals may vary by version

        logger.info("Purged %d cached kernels", count)
        return count

    @property
    def gpu_info(self) -> str:
        """Return GPU information string."""
        return self._gpu_info

    def _load_kernel_source(self, method: str) -> tuple[str, str]:
        """Load CUDA kernel source for the given integration method.

        Returns (source_code, entry_point_name).
        """
        if method not in _KERNEL_REGISTRY:
            raise ValueError(
                f"Unknown integration method '{method}'. "
                f"Available: {list(_KERNEL_REGISTRY.keys())}"
            )

        filename, entry_point = _KERNEL_REGISTRY[method]
        source_path = _INTEGRATOR_DIR / filename

        if not source_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {source_path}")

        # Read the kernel source
        source = source_path.read_text()

        # CuPy RawKernel doesn't support #include, so we need to
        # inline the included files. Replace #include directives
        # with the actual file contents.
        source = resolve_includes(source, source_path.parent)

        return source, entry_point

    def _get_kernel(self, method: str) -> cp.RawKernel:
        """Get or compile a CUDA kernel for the given method."""
        if method in self._kernel_cache:
            return self._kernel_cache[method]

        if method not in ['rk4', 'rkdp8', 'symplectic8']:
            warnings.warn(
                f"Integration method '{method}' is not recognized. "
                "Use 'rk4', 'rkdp8', or 'symplectic8'.",
                UserWarning,
                stacklevel=2
            )

        logger.info("Compiling CUDA kernel for method: %s", method)
        source, entry_point = self._load_kernel_source(method)

        # Compile with CuPy RawKernel
        # Options: enable double precision, set architecture
        kernel = cp.RawKernel(
            source,
            entry_point,
            options=(
                "--std=c++14",
                "-use_fast_math",  # Fast math for float32 (doesn't affect float64)
            ),
        )

        self._kernel_cache[method] = kernel
        logger.info("Kernel '%s' compiled (entry: %s)", method, entry_point)
        return kernel

    def render_frame(self, params: dict) -> bytes:
        """Render a single frame and return raw RGB pixel data.

        Args:
            params: dict with all render parameters.

        Returns:
            Raw RGB bytes (width * height * 3), top-to-bottom row order.
        """
        return self.render_frame_timed(params)["raw_rgb"]

    def render_frame_timed(self, params: dict, progress_callback=None) -> dict:
        """Render a single frame with precise CUDA event timing.

        Like render_frame() but returns a dict with raw RGB bytes and
        detailed timing statistics using CUDA events for GPU-side
        kernel timing (microsecond accuracy).

        Args:
            params: dict with all render parameters (same as render_frame).

        Returns:
            dict with keys:
                raw_rgb: bytes ??? raw RGB pixel data (top-to-bottom)
                kernel_ms: float ??? GPU kernel execution time in ms
                total_ms: float ??? total wall-clock time (kernel + readback)
                gpu_mem_alloc_bytes: int ??? GPU memory allocated for buffers
        """
        if not self._initialized:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")

        t_wall_start = _time.monotonic()

        width = params.get("width", 1280)
        height = params.get("height", 720)
        method = params.get("method", "rkdp8")

        # Get compiled kernel
        kernel = self._get_kernel(method)

        # Compute ISCO
        spin = params.get("spin", 0.6)
        charge = params.get("charge", 0.0)
        isco_radius = isco(spin, charge)

        # Build RenderParams struct
        obs_dist = float(params.get("obs_dist", 40))
        step_size = float(params.get("step_size", 0.30))
        inclination_deg = _resolve_inclination(params)
        steps = int(params.get("steps", 2000))
        rp = RenderParams(
            width=width,
            height=height,
            spin=float(spin),
            charge=float(charge),
            incl=math.radians(inclination_deg),
            fov=float(params.get("fov", 8.0)),
            phi0=float(params.get("phi0", 0.0)),
            isco=float(isco_radius),
            steps=steps,
            obs_dist=obs_dist,
            esc_radius=obs_dist + 12.0,
            disk_outer=50.0,
            step_size=float(params.get("step_size", 0.08)),
            bg_mode=int(params.get("bg_mode", 1)),
            star_layers=int(params.get("star_layers", 3)),
            show_disk=1 if params.get("show_disk", True) else 0,
            show_grid=1 if params.get("show_grid", False) else 0,
            disk_temp=float(params.get("disk_temp", 1.0)),
            doppler_boost=float(params.get("doppler_boost", 2.0)),
            srgb_output=1.0 if params.get("srgb_output", True) else 0.0,
            disk_alpha=float(params.get("disk_alpha", 0.95)),
            disk_max_crossings=float(params.get("disk_max_crossings", 5)),
            disk_mode=float(params.get("disk_mode", 1)),
            aa_samples=float(params.get("aa_samples", 1)),
            debug_trace=1.0 if params.get("debug_trace", False) else 0.0,
            sky_width=0.0,
            sky_height=0.0,
            qed_coupling=float(params.get("qed_coupling", 0.0)),
            hawking_boost=float(params.get("hawking_boost", 0.0)),
        )

        # Copy params struct to GPU as a byte array
        params_bytes = bytes(rp)
        h_params = np.frombuffer(params_bytes, dtype=np.uint8)
        d_params = cp.asarray(h_params)

        # Allocate output buffer on GPU (RGB, uint8)
        output_size = height * width * 3
        d_output = cp.zeros(output_size, dtype=cp.uint8)

        # Track GPU memory allocated for this render
        gpu_mem_alloc = len(params_bytes) + output_size

        # Launch kernel with CUDA event timing on a non-blocking stream
        block_size = (16, 16)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1],
        )

        d_progress_counter = cp.zeros(1, dtype=cp.uint32)
        stream = cp.cuda.Stream(non_blocking=True)
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()  # on default stream
        with stream:
            d_skymap = cp.zeros(1, dtype=cp.float32)
            kernel(
                grid_size,
                block_size,
                (d_params, d_output, d_skymap, d_progress_counter),
            )
        end_event.record(stream)

        if progress_callback:
            total_pixels = width * height
            last_reported_progress = 0
            while not stream.done:
                current_progress = d_progress_counter.get().item()
                if current_progress > last_reported_progress:
                    progress_callback(current_progress - last_reported_progress)
                    last_reported_progress = current_progress
                _time.sleep(0.01)  # poll every 10ms
            
            # Final update to ensure 100% completion
            stream.synchronize()
            final_progress = d_progress_counter.get().item()
            if final_progress > last_reported_progress:
                progress_callback(final_progress - last_reported_progress)

        end_event.synchronize()
        kernel_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        # Copy to host
        h_output = d_output.get()

        # Reshape and flip (same as render_frame)
        pixel_array = h_output.reshape(height, width, 3)
        pixel_array = np.flipud(pixel_array)



        t_wall_end = _time.monotonic()
        total_ms = (t_wall_end - t_wall_start) * 1000.0

        return {
            "raw_rgb": pixel_array.tobytes(),
            "kernel_ms": float(kernel_ms),
            "total_ms": float(total_ms),
            "gpu_mem_alloc_bytes": gpu_mem_alloc,
            "max_steps": steps,
        }

    @property
    def available_methods(self) -> list[str]:
        """Return list of all available integration method names."""
        return list(_KERNEL_REGISTRY.keys())

    def precompile_all(self) -> dict[str, bool]:
        """Pre-compile all CUDA kernels and return compilation status.

        Returns:
            dict mapping method name to True (compiled) or False (failed).
        """
        results = {}
        for method in _KERNEL_REGISTRY:
            try:
                self._get_kernel(method)
                results[method] = True
            except Exception as e:
                logger.error("Failed to compile kernel '%s': %s", method, e)
                results[method] = False
        return results

    # ?????? Single-ray tracing ?????????????????????????????????????????????????????????????????????????????????????????????????????????

    def _get_ray_trace_kernel(self, method: str) -> cp.RawKernel:
        """Get or compile a single-ray trace CUDA kernel for the given method.

        The ray_trace.cu kernel provides per-integrator entry points.
        Methods not in _RAY_TRACE_REGISTRY fall back to rkdp8.
        """
        cache_key = f"ray_trace_{method}"
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        # Determine entry point (fall back to rkdp8 for unsupported methods)
        effective_method = method if method in _RAY_TRACE_REGISTRY else "rkdp8"
        entry_point = _RAY_TRACE_REGISTRY[effective_method]

        if effective_method != method:
            logger.warning(
                "RAY TRACE FALLBACK: method '%s' has no native ray trace kernel ??? "
                "falling back to '%s'. Results will NOT match /bench output for this method! "
                "Native ray trace methods: %s",
                method, effective_method, list(_RAY_TRACE_REGISTRY.keys()),
            )

        logger.info(
            "Compiling ray trace kernel for method: %s (entry: %s, effective: %s)",
            method, entry_point, effective_method,
        )

        # Load ray_trace.cu source and resolve includes
        source_path = _KERNEL_DIR / "ray_trace.cu"
        if not source_path.exists():
            raise FileNotFoundError(f"Ray trace kernel not found: {source_path}")

        source = source_path.read_text()
        source = resolve_includes(source, source_path.parent)

        kernel = cp.RawKernel(
            source,
            entry_point,
            options=("--std=c++14", "-use_fast_math"),
        )

        self._kernel_cache[cache_key] = kernel
        logger.info("Ray trace kernel '%s' compiled (entry: %s)", method, entry_point)
        return kernel

    @property
    def ray_trace_methods(self) -> list[str]:
        """Return list of methods with native ray trace kernel support."""
        return list(_RAY_TRACE_REGISTRY.keys())

    def trace_single_ray(self, params: dict) -> dict:
        """Trace a single photon ray and return detailed trajectory data.

        This is the core computation for the /ray endpoint. It traces
        one geodesic through Kerr-Newman spacetime and returns the full
        trajectory, equatorial plane crossings, and disk physics.

        Args:
            params: dict with keys:
                mode: str ??? "pixel" or "impact_parameter"
                ix, iy: int ??? pixel coordinates (if mode="pixel")
                alpha, beta: float ??? impact parameters in radians (if mode="impact_parameter")
                spin, charge, inclination (degrees), fov, width, height,
                method, steps, step_size, obs_dist, phi0, disk_temp,
                doppler_boost, max_trajectory_points

        Returns:
            dict with keys:
                ray: dict ??? input ray specification
                spacetime: dict ??? black hole parameters
                initial_state: dict ??? r, theta, phi, pr, pth
                final_state: dict ??? r, theta, phi, pr, pth
                termination: dict ??? reason, steps_used, steps_max
                trajectory: dict ??? r[], theta[], phi[], step_sizes[]
                disk_crossings: list[dict] ??? crossing details with physics
                timing: dict ??? kernel_ms, total_ms
        """
        if not self._initialized:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")

        t_wall_start = _time.monotonic()

        method = params.get("method", "rkdp8")
        max_traj = min(params.get("max_trajectory_points", 200), 500)

        # Diagnostic: log method coverage gap
        effective_method = method if method in _RAY_TRACE_REGISTRY else "rkdp8"
        if effective_method != method:
            logger.warning(
                "DISCREPANCY ALERT: /ray requested method='%s' but will use '%s'. "
                "The /bench endpoint uses the real '%s' kernel. "
                "Missing ray trace kernels: %s",
                method, effective_method, method,
                [m for m in _KERNEL_REGISTRY if m not in _RAY_TRACE_REGISTRY],
            )

        # Get compiled ray trace kernel
        kernel = self._get_ray_trace_kernel(method)

        # Compute ISCO
        spin = params.get("spin", 0.6)
        charge = params.get("charge", 0.0)
        isco_radius = isco(spin, charge)

        # Build RenderParams struct
        obs_dist = float(params.get("obs_dist", 40))
        step_size = float(params.get("step_size", 0.30))
        width = params.get("width", 320)
        height = params.get("height", 180)
        inclination_deg = _resolve_inclination(params)
        steps = int(params.get("steps", 2000))

        rp = RenderParams(
            width=width,
            height=height,
            spin=float(spin),
            charge=float(charge),
            incl=math.radians(inclination_deg),
            fov=float(params.get("fov", 8.0)),
            phi0=float(params.get("phi0", 0.0)),
            isco=float(isco_radius),
            steps=steps,
            obs_dist=obs_dist,
            esc_radius=obs_dist + 12.0,
            disk_outer=50.0,
            step_size=float(params.get("step_size", 0.08)),
            bg_mode=0,  # Not used for ray tracing
            star_layers=1,  # Not used for ray tracing
            show_disk=1,  # Always detect disk crossings
            show_grid=0,  # Not used for ray tracing
            disk_temp=float(params.get("disk_temp", 1.0)),
            doppler_boost=float(params.get("doppler_boost", 2.0)),
            srgb_output=1.0 if params.get("srgb_output", True) else 0.0,
            disk_alpha=float(params.get("disk_alpha", 0.95)),
            disk_max_crossings=float(params.get("disk_max_crossings", 5)),
            disk_mode=float(params.get("disk_mode", 1)),
            debug_trace=1.0 if params.get("debug_trace", False) else 0.0,
            sky_width=0.0,
            sky_height=0.0,
            qed_coupling=float(params.get("qed_coupling", 0.0)),
            hawking_boost=float(params.get("hawking_boost", 0.0)),
        )

        # Copy params struct to GPU
        params_bytes = bytes(rp)
        h_params = np.frombuffer(params_bytes, dtype=np.uint8)
        d_params = cp.asarray(h_params)

        # Allocate output buffer:
        #   Header: 24 doubles
        #   Trajectory: max_traj * 4 doubles (r, th, phi, he per step)
        #   Crossings: 1 + 16 * 8 = 129 doubles
        max_crossings = 16
        output_size = 24 + max_traj * 4 + 1 + max_crossings * 8
        d_output = cp.zeros(output_size, dtype=cp.float64)

        # Write input parameters to output buffer
        mode = params.get("mode", "pixel")
        if mode == "impact_parameter":
            d_output[0] = 1.0  # impact_parameter mode
            d_output[1] = float(params.get("alpha", 0.0))
            d_output[2] = float(params.get("beta", 0.0))
        else:
            d_output[0] = 0.0  # pixel mode
            d_output[1] = float(params.get("ix", width // 2))
            d_output[2] = float(params.get("iy", height // 2))
        d_output[3] = float(max_traj)

        # Launch kernel with CUDA event timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        kernel((1,), (1,), (d_params, d_output))
        end_event.record()
        end_event.synchronize()

        kernel_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        # Read back results
        result = d_output.get()

        t_wall_end = _time.monotonic()
        total_ms = (t_wall_end - t_wall_start) * 1000.0

        # Parse results
        steps_used = int(result[14])
        term_reason_code = int(result[13])
        term_names = {
            0: "steps_exhausted",
            1: "horizon",
            2: "escape",
            3: "nan",
            4: "underflow",
        }

        # Effective method (may differ from requested if no native kernel)
        effective_method = method if method in _RAY_TRACE_REGISTRY else "rkdp8"

        # Build ray specification
        ray_info = {"mode": mode}
        if mode == "impact_parameter":
            ray_info["alpha"] = params.get("alpha", 0.0)
            ray_info["beta"] = params.get("beta", 0.0)
        else:
            ray_info["ix"] = params.get("ix", width // 2)
            ray_info["iy"] = params.get("iy", height // 2)
        ray_info["b"] = float(result[5])

        # Parse trajectory
        traj_points = min(steps_used, max_traj)
        traj_base = 24
        traj_r = []
        traj_th = []
        traj_phi = []
        traj_he = []
        for i in range(traj_points):
            off = traj_base + i * 4
            traj_r.append(float(result[off + 0]))
            traj_th.append(float(result[off + 1]))
            traj_phi.append(float(result[off + 2]))
            traj_he.append(float(result[off + 3]))

        # Parse disk crossings
        crossing_base = traj_base + max_traj * 4
        num_crossings = int(result[crossing_base])
        crossings = []
        for j in range(num_crossings):
            coff = crossing_base + 1 + j * 8
            crossing = {
                "crossing_index": int(result[coff + 0]),
                "r": float(result[coff + 1]),
                "phi": float(result[coff + 2]),
                "direction": "north_to_south" if result[coff + 3] > 0 else "south_to_north",
                "g_factor": float(result[coff + 4]),
                "novikov_thorne_flux": float(result[coff + 5]),
                "T_emit": float(result[coff + 6]),
                "T_observed": float(result[coff + 7]),
            }
            crossings.append(crossing)

        return {
            "ray": ray_info,
            "spacetime": {
                "spin": float(spin),
                "charge": float(charge),
                "r_plus": float(result[15]),
                "r_isco": float(result[16]),
                "method": method,
                "effective_method": effective_method,
            },
            "initial_state": {
                "r": float(result[0]),
                "theta": float(result[1]),
                "phi": float(result[2]),
                "pr": float(result[3]),
                "pth": float(result[4]),
                "H": float(result[17]),
                "Q": float(result[19]),
            },
            "final_state": {
                "r": _safe_float(float(result[8])),
                "theta": _safe_float(float(result[9])),
                "phi": _safe_float(float(result[10])),
                "pr": _safe_float(float(result[11])),
                "pth": _safe_float(float(result[12])),
                "H": _safe_float(float(result[18])),
                "Q": _safe_float(float(result[20])),
            },
            "termination": {
                "reason": term_names.get(term_reason_code, f"unknown({term_reason_code})"),
                "steps_used": steps_used,
                "steps_max": int(params.get("steps", 200)),
            },
            "trajectory": {
                "points": traj_points,
                "r": traj_r,
                "theta": traj_th,
                "phi": traj_phi,
                "step_sizes": traj_he,
            },
            "disk_crossings": crossings,
            "timing": {
                "kernel_ms": round(float(kernel_ms), 3),
                "total_ms": round(total_ms, 3),
            },
        }

    def shutdown(self) -> None:
        """Clean up CUDA resources."""
        if not self._initialized:
            return

        self._kernel_cache.clear()
        self._initialized = False
        logger.info("CUDA renderer shut down")
