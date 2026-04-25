"""
EHT shadow and ring metric extraction for Nulltracer renders.

Extracts shadow diameter, circularity, and brightness asymmetry
from rendered images.  Provides two measurement modes:

  **Shadow mode** (show_disk=False): measures the dark-region boundary,
  corresponding to the critical impact parameter.  For Schwarzschild
  this is 2·3√3 ≈ 10.39 M.  Compare against analytic predictions.

  **Ring mode** (show_disk=True): measures the bright emission-ring
  peak.  This is the **EHT observable**: the M87* and Sgr A* papers
  quote the diameter of the bright ring, not the dark shadow interior.

Key distinction (EHT Paper VI, §5):
  The emission ring is ~5–10 % larger than the shadow because the
  synchrotron-emitting plasma extends *outside* the photon ring.
  The calibration factor ring/shadow ≈ 1.10 ± 0.05.

EHT observational references:
  - M87*  ring: 42 ± 3 μas  (EHT 2019 Paper I)
  - M87*  ring: 43.9 ± 0.6 μas  (EHT 2017–2021 variability paper)
  - M87*  ΔC < 0.10  (EHT 2019 Paper VI)
  - Sgr A* ring: 51.8 ± 2.3 μas  (EHT 2022 Paper I)
  - Sgr A* shadow: 48.7 ± 7.0 μas  (EHT 2022)
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import binary_fill_holes, binary_erosion, label

# ── EHT reference constants ──────────────────────────────────

_G = 6.67430e-11        # m³ kg⁻¹ s⁻²
_c = 2.99792458e8       # m s⁻¹
_M_sun = 1.98892e30     # kg
_pc = 3.08567758e16     # m
_Mpc = _pc * 1e6

# Canonical values used by EHT Collaboration
EHT_M87 = {
    "mass_kg":    6.5e9 * _M_sun,
    "mass_Msun":  6.5e9,
    "dist_m":     16.8 * _Mpc,
    "dist_Mpc":   16.8,
    "incl_deg":   17.0,             # from jet position angle
    "ring_uas":   42.0,             # Paper I (2019)
    "ring_err":   3.0,
    "ring_2021_uas": 43.9,          # 2017–2021 variability
    "ring_2021_err": 0.6,
    "delta_C":    0.10,             # upper bound, Paper VI
    "calib_ring_shadow": 1.10,      # ring_D / shadow_D, Paper VI §5
    "calib_err":  0.05,
}

EHT_SGRA = {
    "mass_kg":    4.0e6 * _M_sun,
    "mass_Msun":  4.0e6,
    "dist_m":     8.28e3 * _pc,
    "dist_kpc":   8.28,
    "incl_deg":   30.0,             # favoured by EHT; high-i disfavoured
    "ring_uas":   51.8,             # Paper I (2022)
    "ring_err":   2.3,
    "shadow_uas": 48.7,             # Paper I (2022)
    "shadow_err": 7.0,
}

GARGANTUA = {
    "mass_Msun":  1.0e8,            # Thorne (2014)
    "spin":       0.99999,          # near-extremal
    "incl_deg":   80.0,             # movie framing (edge-on)
    "doppler":    False,            # suppressed in film
}


def M_to_uas(d_M: float, mass_kg: float, dist_m: float) -> float:
    """Convert a diameter in units of M to microarcseconds.

    Parameters
    ----------
    d_M : float
        Diameter in gravitational radii (M = GM/c²).
    mass_kg : float
        Black hole mass in kg.
    dist_m : float
        Distance to the source in metres.

    Returns
    -------
    float
        Angular diameter in μas.
    """
    r_g = _G * mass_kg / (_c ** 2)
    theta_rad = d_M * r_g / dist_m
    return theta_rad * (180 / np.pi) * 3600 * 1e6


def uas_per_M(mass_kg: float, dist_m: float) -> float:
    """Angular scale: μas per gravitational radius M."""
    r_g = _G * mass_kg / (_c ** 2)
    return r_g / dist_m * (180 / np.pi) * 3600 * 1e6


# ── Contour extraction ────────────────────────────────────────


def extract_shadow_contour(image: np.ndarray, threshold: float = 0.05):
    """
    Extract the shadow boundary from a rendered grayscale image.

    Args:
        image: 2D numpy array (grayscale intensity, normalized 0-1)
        threshold: intensity threshold separating shadow from bright ring

    Returns:
        contour_points: Nx2 array of (x, y) boundary coordinates
    """
    # The shadow is the dark region (intensity <= threshold)
    binary = image <= threshold
    
    # Use connected components to find the largest central dark blob.
    # This avoids picking up the image border or other dark regions.
    labeled, num_features = label(binary)
    if num_features > 0:
        # Find the center pixel
        h, w = binary.shape
        cy, cx = h // 2, w // 2
        
        # If the center pixel is dark, use its component
        if labeled[cy, cx] > 0:
            binary = labeled == labeled[cy, cx]
        else:
            # Otherwise, use the largest component
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest_label = np.argmax(sizes) + 1
            binary = labeled == largest_label
            
    # Now find the border of this specific component
    interior = binary_erosion(binary)
    boundary = binary & ~interior
    
    # Still exclude the outermost pixel border just in case the shadow touches the edge
    boundary[0, :] = False
    boundary[-1, :] = False
    boundary[:, 0] = False
    boundary[:, -1] = False
    
    points = np.column_stack(np.where(boundary))
    return points[:, ::-1]  # return as (x, y)


# ── Geometric fitting ─────────────────────────────────────────


def fit_circle(points: np.ndarray):
    """
    Least-squares circle fit to 2D points.

    Returns:
        cx, cy, radius, residual_rms
    """

    def residuals(params, pts):
        cx, cy, r = params
        return np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) - r

    # Initial guess: centroid and mean distance
    cx0, cy0 = points.mean(axis=0)
    r0 = np.mean(np.sqrt((points[:, 0] - cx0) ** 2 + (points[:, 1] - cy0) ** 2))

    result = least_squares(residuals, [cx0, cy0, r0], args=(points,))
    cx, cy, r = result.x
    rms = np.sqrt(np.mean(result.fun**2))
    return cx, cy, r, rms


def fit_ellipse(points: np.ndarray):
    """
    Least-squares ellipse fit to 2D points.

    Returns:
        cx, cy, a (semi-major), b (semi-minor), angle, residual_rms
    """

    def residuals(params, pts):
        cx, cy, a, b, angle = params
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a
        return (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1.0

    cx0, cy0 = points.mean(axis=0)
    r0 = np.mean(np.sqrt((points[:, 0] - cx0) ** 2 + (points[:, 1] - cy0) ** 2))

    result = least_squares(residuals, [cx0, cy0, r0, r0, 0.0], args=(points,))
    cx, cy, a, b, angle = result.x
    rms = np.sqrt(np.mean(result.fun**2))
    return cx, cy, abs(a), abs(b), angle, rms


# ── Shadow metrics ────────────────────────────────────────────


def extract_shadow_metrics(
    image: np.ndarray, fov_deg: float = 10.0, threshold: float = 0.05
):
    """
    Extract quantitative shadow metrics from a rendered image.

    Args:
        image: 2D or 3D numpy array (rendered frame).
               If 3D (RGB), converts to grayscale.
        fov_deg: screen half-width in units of M (note: historically
                 named ``fov_deg``, but the kernel treats this as an
                 impact-parameter half-extent, not an angle).
        threshold: shadow boundary threshold (fraction of max intensity)

    Returns:
        dict with keys:

        Shadow-boundary metrics (from the dark region):

            'diameter_px':     shadow diameter in pixels (2 × circle-fit radius)
            'diameter_M':      shadow diameter in units of M (circle fit).
            'diameter_beta_M': shadow extent in the β (vertical) direction in M.
                               **Use this for EHT comparisons at low inclination**
                               where the shadow is D-shaped and the circle fit
                               overestimates the diameter.
            'ring_diameter_M': ellipse-fit major-axis diameter in M.
            'circularity':     ΔC = 1 − b/a (0 = perfect circle)

        Photometry:

            'asymmetry':       brightness asymmetry (left/right flux ratio)
            'center_x', 'center_y': shadow center in pixels

        Fit quality:

            'circle_fit_rms', 'ellipse_fit_rms': residuals in pixels
            'semi_major', 'semi_minor': ellipse axes in pixels
            'n_contour_points': number of extracted boundary pixels
    """
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()

    gray = gray / (gray.max() + 1e-12)

    contour = extract_shadow_contour(gray, threshold)

    if len(contour) < 10:
        return {"error": "Too few contour points", "n_points": len(contour)}

    cx, cy, radius, circle_rms = fit_circle(contour)
    _, _, a_ell, b_ell, angle, ellipse_rms = fit_ellipse(contour)

    height, width = gray.shape
    px_per_M = width / (2.0 * fov_deg)  # kernel maps ux∈[-1,1] to α∈[-fov,+fov]×M

    semi_major_px = max(a_ell, b_ell)
    semi_minor_px = min(a_ell, b_ell)
    circularity = 1.0 - semi_minor_px / semi_major_px

    # β-direction extent: the vertical span of the contour.
    # This is the best diameter metric for low-inclination spinning BHs
    # where the shadow is D-shaped (α-extent >> β-extent).
    y_extent_px = contour[:, 1].max() - contour[:, 1].min()

    # Brightness asymmetry: ratio of total flux left vs right of center
    left_flux = gray[:, : int(cx)].sum()
    right_flux = gray[:, int(cx):].sum()
    asymmetry = left_flux / (right_flux + 1e-12)

    return {
        "diameter_px": 2.0 * radius,
        "diameter_M": 2.0 * radius / px_per_M if px_per_M > 0 else None,
        "diameter_beta_M": y_extent_px / px_per_M if px_per_M > 0 else None,
        "ring_diameter_M": 2.0 * semi_major_px / px_per_M if px_per_M > 0 else None,
        "circularity": circularity,
        "center_x": cx,
        "center_y": cy,
        "asymmetry": asymmetry,
        "circle_fit_rms": circle_rms,
        "ellipse_fit_rms": ellipse_rms,
        "semi_major": semi_major_px,
        "semi_minor": semi_minor_px,
        "n_contour_points": len(contour),
    }


# ── Ring-peak extraction (for disk-on renders) ────────────────


def extract_ring_diameter(
    image: np.ndarray,
    fov: float = 7.0,
    n_radial: int = 200,
    theta_samples: int = 360,
) -> dict:
    """Extract the bright emission-ring diameter from a disk-on render.

    Unlike :func:`extract_shadow_metrics` which measures the *dark*
    interior boundary, this function finds the *peak* of the bright
    ring - the quantity the EHT actually measures.

    The algorithm:
      1. Compute the azimuthally averaged radial intensity profile I(r)
         centred on the image midpoint.
      2. Find the radius r_peak where I(r) is maximised.
      3. The ring diameter is 2 × r_peak.

    Parameters
    ----------
    image : ndarray
        2D or 3D rendered frame (with disk and Doppler enabled).
    fov : float
        Screen half-width in M (same as the kernel's ``fov`` parameter).
    n_radial : int
        Number of radial bins for the profile.
    theta_samples : int
        Number of azimuthal samples per radial bin.

    Returns
    -------
    dict
        'ring_peak_M':    ring diameter in M (2 × peak radius)
        'ring_peak_px':   ring diameter in pixels
        'peak_intensity': intensity at the ring peak (0–1)
        'profile_r_M':    radial bin centres in M
        'profile_I':      azimuthally averaged intensity profile
    """
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    gray = gray / (gray.max() + 1e-12)

    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    px_per_M = w / (2.0 * fov)

    # Maximum radius to probe (half the image)
    r_max_px = min(cx, cy) * 0.95
    r_bins = np.linspace(0, r_max_px, n_radial + 1)
    r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Azimuthal averaging
    theta = np.linspace(0, 2 * np.pi, theta_samples, endpoint=False)
    profile = np.zeros(n_radial)

    for i in range(n_radial):
        r_mid = r_centres[i]
        xs = cx + r_mid * np.cos(theta)
        ys = cy + r_mid * np.sin(theta)
        # Bilinear sample (clamp to image bounds)
        xi = np.clip(xs.astype(int), 0, w - 1)
        yi = np.clip(ys.astype(int), 0, h - 1)
        profile[i] = gray[yi, xi].mean()

    # Find peak (skip innermost bins which may be noisy)
    search_start = max(3, n_radial // 10)
    peak_idx = search_start + np.argmax(profile[search_start:])
    peak_r_px = r_centres[peak_idx]
    peak_r_M = peak_r_px / px_per_M

    return {
        "ring_peak_M": 2.0 * peak_r_M,
        "ring_peak_px": 2.0 * peak_r_px,
        "peak_intensity": float(profile[peak_idx]),
        "profile_r_M": r_centres / px_per_M,
        "profile_I": profile,
    }
