"""
Airy disk bloom post-processing for Nulltracer.

Physically-motivated bloom simulating diffraction from the human eye's pupil.
This implementation is GPU-accelerated using CuPy.

The Airy disk function is (2 * J1(x) / x)^2 where J1 is the Bessel function
of the first kind. It's spectrally dependent: red light diffracts more than
blue (scale factors [1.0, 0.86, 0.61]). The diffraction radius comes from
1.22 ?? 650nm / (4mm pupil) scaled by FOV and resolution.
"""

import numpy as np
import logging

try:
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve
    from cupyx.scipy.ndimage import zoom as cp_zoom
    from cupyx.scipy.special import j1
    CUPY_AVAILABLE = True
except ImportError:
    from scipy.signal import fftconvolve
    from scipy.ndimage import zoom as cp_zoom
    from scipy.special import j1
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def airy_disk(x: np.ndarray) -> np.ndarray:
    """Compute the Airy disk intensity pattern.

    The Airy pattern is (2*J1(x)/x)^2, with the singularity at x=0
    handled by L'H??pital's rule (limit = 1.0).
    """
    result = np.ones_like(x)
    mask = x != 0
    result[mask] = (2.0 * j1(x[mask]) / x[mask]) ** 2
    return result


# Spectral scaling: red diffracts more than blue.
# Approximate positions of R, G, B in the visible spectrum.
SPECTRUM = np.array([1.0, 0.86, 0.61])


def generate_kernel(scale: float, size: int, xp) -> 'cp.ndarray | np.ndarray':
    """Generate a (2*size+1, 2*size+1, 3) Airy disk convolution kernel.

    Args:
        scale: Base radius of the Airy disk in pixels (for red channel).
        size: Half-size of the kernel in pixels.
        xp: The array module to use (cupy or numpy).

    Returns:
        Normalized 3-channel kernel array on the specified device.
    """
    x = xp.arange(-size, size + 1, dtype=xp.float64)
    xs, ys = xp.meshgrid(x, x)
    r = xp.sqrt(xs**2 + ys**2) + 1e-10  # avoid exact zero

    kernel = xp.zeros((2 * size + 1, 2 * size + 1, 3), dtype=xp.float32)

    for c in range(3):
        channel_scale = scale * SPECTRUM[c]
        # Move r to CPU for airy_disk (scipy) then move back
        r_cpu = r.get() if hasattr(r, 'get') else r
        airy_values = airy_disk(r_cpu / max(channel_scale, 0.01))
        kernel[:, :, c] = xp.asarray(airy_values)

    # Normalize each channel independently
    for c in range(3):
        total = kernel[:, :, c].sum()
        if total > 0:
            kernel[:, :, c] /= total

    return kernel


def srgb_to_linear(arr: np.ndarray, xp) -> 'cp.ndarray | np.ndarray':
    """Convert sRGB uint8 image to linear float32 RGB."""
    linear = xp.asarray(arr, dtype=xp.float32) / 255.0
    # Inverse sRGB transfer function
    mask = linear > 0.04045
    linear[mask] = ((linear[mask] + 0.055) / 1.055) ** 2.4
    linear[~mask] /= 12.92
    return linear


def linear_to_srgb(arr: 'cp.ndarray | np.ndarray', xp) -> np.ndarray:
    """Convert linear float32/64 RGB to sRGB uint8."""
    result = xp.clip(arr, 0.0, None)  # no negative values
    mask = result > 0.0031308
    result[mask] = 1.055 * xp.power(result[mask], 1.0 / 2.4) - 0.055
    result[~mask] *= 12.92
    result = xp.clip(result * 255.0, 0, 255)
    
    # If on GPU, move back to CPU before converting to uint8
    if hasattr(result, 'get'):
        result = result.get()
        
    return result.astype(np.uint8)


def apply_bloom(image: np.ndarray, fov: float = 8.0,
                bloom_radius: float = 1.0,
                width: int = 0,
                obs_dist: float = 40.0) -> np.ndarray:
    """Apply Airy disk bloom to a rendered image, using GPU if available.

    Args:
        image: Input image as (H, W, 3) uint8 sRGB array.
        fov: Field of view (half-width) in units of M (gravitational radii).
        bloom_radius: User-adjustable radius multiplier (1.0 = physical default).
        width: Image width (for resolution scaling). If 0, uses image.shape[1].
        obs_dist: Observer distance in units of M.

    Returns:
        Bloomed image as (H, W, 3) uint8 sRGB array.
    """
    xp = cp if CUPY_AVAILABLE else np
    h, w = image.shape[:2]
    if width <= 0:
        width = w

    # Convert to linear space for physically correct convolution
    linear = srgb_to_linear(image, xp)

    # Compute diffraction radius in pixels
    # fov is in units of M. The angular half-width is roughly fov / obs_dist.
    angular_fov = fov / max(obs_dist, 1.0)
    radd = 0.00019825 * width / angular_fov
    radd *= bloom_radius

    # Isolate bright areas that cause bloom
    threshold = 0.95
    bright_pixels = xp.clip(linear - threshold, 0, None)
    max_intensity = xp.amax(bright_pixels)
    if max_intensity < 0.01:
        return image

    # Kernel pixel radius scales with cube root of max intensity and resolution
    kern_radius = int(30 * (max_intensity / 5.0) ** (1.0 / 3.0) * width / 1920.0)
    kern_radius = max(kern_radius, 5)
    kern_radius = min(kern_radius, 120)  # cap for performance

    logger.debug("Bloom: radius=%.2f, kernel_size=%d, max_intensity=%.2f, GPU=%s",
                 radd, kern_radius, max_intensity, CUPY_AVAILABLE)

    kernel = generate_kernel(radd, kern_radius, xp)

    # Adaptive downsampling for performance
    downsample_factor = max(1, int(kern_radius / 8.0))
    down_w = w // downsample_factor
    down_h = h // downsample_factor

    if CUPY_AVAILABLE:
        # GPU implementation
        down_bright = cp_zoom(bright_pixels, (down_h/h, down_w/w, 1), order=1)
        down_convolved = xp.zeros_like(down_bright)
        for c in range(3):
            down_convolved[:, :, c] = fftconvolve(
                down_bright[:, :, c], kernel[:, :, c], mode='same'
            )
        bloom_img = cp_zoom(down_convolved, (h/down_h, w/down_w, 1), order=1)
    else:
        # CPU fallback
        bright_pixels_cpu = bright_pixels if isinstance(bright_pixels, np.ndarray) else bright_pixels.get()
        down_bright = cp_zoom(bright_pixels_cpu, (down_h/h, down_w/w, 1), order=1)
        down_convolved = np.zeros_like(down_bright)
        for c in range(3):
            down_convolved[:, :, c] = fftconvolve(
                down_bright[:, :, c], kernel[:, :, c], mode='same'
            )
        bloom_img = xp.asarray(cp_zoom(down_convolved, (h/down_h, w/down_w, 1), order=1))

    bloom_weight = 0.4 * np.log1p(max_intensity)
    result = linear + bloom_img * bloom_weight

    return linear_to_srgb(result, xp)
