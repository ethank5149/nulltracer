"""
Equirectangular skymap loader for background star fields.

Supports JPEG/PNG (8-bit sRGB), 16-bit TIFF/PNG, and OpenEXR (HDR).
The loaded texture is stored on the GPU as a float32 linear-light array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp
import numpy as np

__all__ = ["load_skymap", "get_skymap", "clear_skymap"]

# Module-level GPU cache: (device_array, width, height) or None
_cache: Optional[tuple[cp.ndarray, int, int]] = None


def load_skymap(
    path: str | Path | None = None,
    brightness: float = 5.0,
) -> tuple[cp.ndarray, int, int]:
    """Load an equirectangular skymap to GPU memory.

    Parameters
    ----------
    path : str or Path, optional
        Image file to load.  Supported formats: JPEG, PNG (8/16-bit),
        TIFF, OpenEXR.  If *None*, attempts to find the default Gaia
        EDR3 skymap adjacent to the package.
    brightness : float
        Multiplicative scaling applied after linearisation.

    Returns
    -------
    (d_skymap, width, height)
        GPU float32 array (H×W×3 flattened) and its dimensions.
    """
    global _cache

    if path is None:
        # Search common locations relative to package
        pkg = Path(__file__).resolve().parent
        candidates = [
            pkg.parent / "data" / "starmap_gaia_edr3_flux_cartesian_4k.jpg",
            pkg.parent / "starmap_gaia_edr3_flux_cartesian_4k.jpg",
            pkg.parent / "starmap_gaia_edr3_flux_cartesian_4k.png",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
        if path is None:
            raise FileNotFoundError(
                "No skymap found. Pass an explicit path or place a skymap "
                "in the data/ directory."
            )

    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".exr":
        try:
            import OpenEXR
            import Imath

            exr = OpenEXR.InputFile(str(path))
            header = exr.header()
            dw = header["dataWindow"]
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = []
            for ch in ("R", "G", "B"):
                raw = exr.channel(ch, pt)
                channels.append(
                    np.frombuffer(raw, dtype=np.float32).reshape(h, w)
                )
            data = np.stack(channels, axis=-1)
            fmt_info = f"EXR float32"
        except ImportError:
            raise ImportError(
                "OpenEXR is required for .exr files. "
                "Install with: pip install OpenEXR"
            )
    else:
        from PIL import Image

        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        raw = np.array(img)

        if raw.dtype == np.uint16:
            data = raw.astype(np.float32) / 65535.0
            fmt_info = f"16-bit ({ext})"
        elif raw.dtype == np.uint8:
            data = raw.astype(np.float32) / 255.0
            fmt_info = f"8-bit ({ext})"
        else:
            data = raw.astype(np.float32)
            fmt_info = f"{raw.dtype} ({ext})"

        # sRGB → linear (inverse IEC 61966-2-1)
        mask = data <= 0.04045
        data[mask] /= 12.92
        data[~mask] = ((data[~mask] + 0.055) / 1.055) ** 2.4

    data = np.ascontiguousarray(data[:, :, :3].astype(np.float32))
    data *= brightness

    d_skymap = cp.asarray(data.ravel())
    _cache = (d_skymap, w, h)

    print(
        f"Skymap loaded: {path.name} ({w}×{h}, {fmt_info}, "
        f"brightness={brightness}×, {data.nbytes / 1e6:.1f} MB on GPU)"
    )
    return _cache


def get_skymap() -> tuple[cp.ndarray, int, int]:
    """Return the cached GPU skymap, or a null placeholder if none loaded."""
    if _cache is not None:
        return _cache
    return (cp.zeros(1, dtype=cp.float32), 0, 0)


def clear_skymap() -> None:
    """Release GPU skymap memory."""
    global _cache
    _cache = None
