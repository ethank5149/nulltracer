"""
CUDA renderer using CuPy RawKernel for Kerr-Newman black hole ray tracing.

Replaces the EGL/OpenGL renderer with direct CUDA compute kernels.
All geodesic integration uses float64 for maximum accuracy.
"""

import ctypes
import logging
import math
import os
from pathlib import Path
from typing import Optional

import cupy as cp
import numpy as np

from .isco import isco

logger = logging.getLogger(__name__)

# Path to kernel source files
_KERNEL_DIR = Path(__file__).parent / "kernels"
_INTEGRATOR_DIR = _KERNEL_DIR / "integrators"

# Map method names to kernel source files and entry point names
_KERNEL_REGISTRY = {
    "yoshida4": ("yoshida4.cu", "trace_yoshida4"),
    "rk4":      ("rk4.cu",      "trace_rk4"),
    "yoshida6": ("yoshida6.cu", "trace_yoshida6"),
    "yoshida8": ("yoshida8.cu", "trace_yoshida8"),
    "rkdp8":    ("rkdp8.cu",    "trace_rkdp8"),
}


class RenderParams(ctypes.Structure):
    """C-compatible struct matching the CUDA RenderParams definition.

    Must be kept in sync with server/kernels/geodesic_base.cu.
    """
    _fields_ = [
        ("width",       ctypes.c_int),
        ("height",      ctypes.c_int),
        ("spin",        ctypes.c_double),
        ("charge",      ctypes.c_double),
        ("incl",        ctypes.c_double),
        ("fov",         ctypes.c_double),
        ("phi0",        ctypes.c_double),
        ("isco",        ctypes.c_double),
        ("steps",       ctypes.c_int),
        ("obs_dist",    ctypes.c_double),
        ("esc_radius",  ctypes.c_double),
        ("disk_outer",  ctypes.c_double),
        ("step_size",   ctypes.c_double),
        ("bg_mode",     ctypes.c_int),
        ("star_layers", ctypes.c_int),
        ("show_disk",   ctypes.c_int),
        ("show_grid",   ctypes.c_int),
        ("disk_temp",   ctypes.c_float),
    ]


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
            # Force CUDA context creation
            cp.cuda.Device(0).use()

            # Query GPU info
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            mem_gb = props["totalGlobalMem"] / (1024**3)
            cc = f"{props['major']}.{props['minor']}"
            self._gpu_info = f"{name} ({mem_gb:.1f} GB, compute {cc})"
            logger.info("CUDA GPU: %s", self._gpu_info)

            # Log CUDA version
            cuda_ver = cp.cuda.runtime.runtimeGetVersion()
            logger.info("CUDA runtime version: %d", cuda_ver)

            # Pre-compile the default kernel (yoshida4)
            self._get_kernel("yoshida4")
            logger.info("Default kernel (yoshida4) pre-compiled")

        except Exception as e:
            logger.error("CUDA initialization failed: %s", e, exc_info=True)
            raise RuntimeError(f"CUDA initialization failed: {e}")

        self._initialized = True
        logger.info("CUDA renderer initialized successfully")

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
        source = self._resolve_includes(source, source_path.parent)

        return source, entry_point

    def _resolve_includes(self, source: str, base_dir: Path) -> str:
        """Resolve #include directives by inlining file contents.

        CuPy's RawKernel doesn't support the #include preprocessor
        directive, so we manually inline all includes.
        """
        import re
        include_pattern = re.compile(r'#include\s+"([^"]+)"')

        resolved = set()  # Track resolved files to avoid duplicates

        def replace_include(match):
            rel_path = match.group(1)
            abs_path = (base_dir / rel_path).resolve()

            # Avoid double-inclusion (header guards handle this in real CUDA,
            # but we need to handle it here since we're inlining)
            if abs_path in resolved:
                return f"/* Already included: {rel_path} */"
            resolved.add(abs_path)

            if not abs_path.exists():
                raise FileNotFoundError(
                    f"Include file not found: {rel_path} "
                    f"(resolved to {abs_path})"
                )

            included_source = abs_path.read_text()
            # Recursively resolve includes in the included file
            included_source = self._resolve_includes(
                included_source, abs_path.parent
            )
            return f"/* Begin include: {rel_path} */\n{included_source}\n/* End include: {rel_path} */"

        return include_pattern.sub(replace_include, source)

    def _get_kernel(self, method: str) -> cp.RawKernel:
        """Get or compile a CUDA kernel for the given method."""
        if method in self._kernel_cache:
            return self._kernel_cache[method]

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
            params: dict with all render parameters including:
                spin, charge, inclination (degrees), fov, width, height,
                method, steps, step_size, obs_dist, bg_mode, show_disk,
                show_grid, disk_temp, star_layers, phi0

        Returns:
            Raw RGB bytes (width * height * 3), top-to-bottom row order.
        """
        if not self._initialized:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")

        width = params.get("width", 1280)
        height = params.get("height", 720)
        method = params.get("method", "yoshida4")

        # Get compiled kernel
        kernel = self._get_kernel(method)

        # Compute ISCO
        spin = params.get("spin", 0.6)
        charge = params.get("charge", 0.0)
        isco_radius = isco(spin, charge)

        # Build RenderParams struct
        obs_dist = float(params.get("obs_dist", 40))
        rp = RenderParams(
            width=width,
            height=height,
            spin=float(spin),
            charge=float(charge),
            incl=math.radians(float(params.get("inclination", 80.0))),
            fov=float(params.get("fov", 8.0)),
            phi0=float(params.get("phi0", 0.0)),
            isco=float(isco_radius),
            steps=int(params.get("steps", 200)),
            obs_dist=obs_dist,
            esc_radius=obs_dist + 12.0,
            disk_outer=14.0,
            step_size=float(params.get("step_size", 0.30)),
            bg_mode=int(params.get("bg_mode", 1)),
            star_layers=int(params.get("star_layers", 3)),
            show_disk=1 if params.get("show_disk", True) else 0,
            show_grid=1 if params.get("show_grid", True) else 0,
            disk_temp=float(params.get("disk_temp", 1.0)),
        )

        # Copy params struct to GPU
        params_bytes = bytes(rp)
        d_params = cp.cuda.alloc(len(params_bytes))
        d_params.copy_from_host(params_bytes, len(params_bytes))

        # Allocate output buffer on GPU (RGB, uint8)
        d_output = cp.zeros(height * width * 3, dtype=cp.uint8)

        # Launch kernel
        block_size = (16, 16)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1],
        )

        kernel(
            grid_size,
            block_size,
            (d_params, d_output),
        )

        # Synchronize and read back
        cp.cuda.Stream.null.synchronize()

        # Copy to host
        h_output = d_output.get()

        # Reshape to (H, W, 3) — kernel writes in row-major order
        # with y=0 at top (matching image convention, no flip needed)
        pixel_array = h_output.reshape(height, width, 3)

        return pixel_array.tobytes()

    def shutdown(self) -> None:
        """Clean up CUDA resources."""
        if not self._initialized:
            return

        self._kernel_cache.clear()
        self._initialized = False
        logger.info("CUDA renderer shut down")
