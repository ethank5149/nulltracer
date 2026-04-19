"""
Airy disk bloom post-processing for Nulltracer.

Physically-motivated bloom simulating diffraction from the human eye's pupil.
Based on the Starless reference implementation with improvements.

The Airy disk function is (2 * J1(x) / x)^2 where J1 is the Bessel function
of the first kind. It's spectrally dependent: red light diffracts more than
blue (scale factors [1.0, 0.86, 0.61]). The diffraction radius comes from
1.22 × 650nm / (4mm pupil) scaled by FOV and resolution.
"""

import numpy as np
from scipy.special import j1
from scipy.signal import fftconvolve

import logging
logger = logging.getLogger(__name__)


def airy_disk(x: np.ndarray) -> np.ndarray:
    """Compute the Airy disk intensity pattern.

    The Airy pattern is (2*J1(x)/x)^2, with the singularity at x=0
    handled by L'Hôpital's rule (limit = 1.0).
    """
    result = np.ones_like(x)
    mask = x != 0
    result[mask] = (2.0 * j1(x[mask]) / x[mask]) ** 2
    return result


# Spectral scaling: red diffracts more than blue.
# Approximate positions of R, G, B in the visible spectrum.
SPECTRUM = np.array([1.0, 0.86, 0.61])


def generate_kernel(scale: float, size: int) -> np.ndarray:
    """Generate a (2*size+1, 2*size+1, 3) Airy disk convolution kernel.

    Args:
        scale: Base radius of the Airy disk in pixels (for red channel).
        size: Half-size of the kernel in pixels.

    Returns:
        Normalized 3-channel kernel array.
    """
    x = np.arange(-size, size + 1, dtype=np.float64)
    xs, ys = np.meshgrid(x, x)
    r = np.sqrt(xs**2 + ys**2) + 1e-10  # avoid exact zero

    kernel = np.zeros((2 * size + 1, 2 * size + 1, 3))

    for c in range(3):
        channel_scale = scale * SPECTRUM[c]
        kernel[:, :, c] = airy_disk(r / max(channel_scale, 0.01))

    # Normalize each channel independently
    for c in range(3):
        total = kernel[:, :, c].sum()
        if total > 0:
            kernel[:, :, c] /= total

    return kernel


def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 image to linear float32 RGB."""
    linear = arr.astype(np.float32) / 255.0
    # Inverse sRGB transfer function
    mask = linear > 0.04045
    linear[mask] = ((linear[mask] + 0.055) / 1.055) ** 2.4
    linear[~mask] /= 12.92
    return linear


def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    """Convert linear float32 RGB to sRGB uint8."""
    result = np.copy(arr)
    result = np.clip(result, 0.0, None)  # no negative values
    mask = result > 0.0031308
    result[mask] = 1.055 * np.power(result[mask], 1.0 / 2.4) - 0.055
    result[~mask] *= 12.92
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result


def apply_bloom(image: np.ndarray, fov: float = 8.0,
                bloom_radius: float = 1.0,
                width: int = 0) -> np.ndarray:
    """Apply Airy disk bloom to a rendered image.

    Args:
        image: Input image as (H, W, 3) uint8 sRGB array.
        fov: Field of view in degrees (used to compute diffraction scale).
        bloom_radius: User-adjustable radius multiplier (1.0 = physical default).
        width: Image width (for resolution scaling). If 0, uses image.shape[1].

    Returns:
        Bloomed image as (H, W, 3) uint8 sRGB array.
    """
    h, w = image.shape[:2]
    if width <= 0:
        width = w

    # Convert to linear space for physically correct convolution
    linear = srgb_to_linear(image)

    # Compute diffraction radius in pixels
    # 1.22 * 650nm / 4mm = 0.00019825 radians (human eye diffraction limit for red)
    fov_rad = np.radians(fov) if fov > 0.1 else np.radians(8.0)
    radd = 0.00019825 * width / fov_rad
    radd *= bloom_radius

    # Adaptive kernel size based on maximum brightness
    max_intensity = np.amax(linear)
    if max_intensity < 0.01:
        # Image is too dark for visible bloom
        return image

    # Kernel pixel radius scales with cube root of max intensity and resolution
    kern_radius = int(25 * (max_intensity / 5.0) ** (1.0 / 3.0) * width / 1920.0)
    kern_radius = max(kern_radius, 5)
    kern_radius = min(kern_radius, 100)  # cap for performance

    logger.debug("Bloom: radius=%.2f, kernel_size=%d, max_intensity=%.2f",
                 radd, kern_radius, max_intensity)

    # Generate Airy disk kernel
    kernel = generate_kernel(radd, kern_radius)

    # Convolve each channel with its spectral kernel using FFT
    result = np.zeros_like(linear)
    for c in range(3):
        result[:, :, c] = fftconvolve(
            linear[:, :, c], kernel[:, :, c], mode='same'
        )

    # Convert back to sRGB uint8
    return linear_to_srgb(result)
