"""
EHT shadow metric extraction for Nulltracer renders.

Extracts shadow diameter, circularity, and brightness asymmetry
from rendered images. Compares against EHT M87* observables:
  - Ring diameter: 42 ?? 3 ??as (EHT Collaboration 2019, Paper VI)
  - Circularity ??C < 0.10
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import binary_fill_holes, binary_erosion, label


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


def extract_shadow_metrics(
    image: np.ndarray, fov_deg: float = 10.0, threshold: float = 0.05
):
    """
    Extract quantitative shadow metrics from a rendered image.

    Args:
        image: 2D or 3D numpy array (rendered frame).
               If 3D (RGB), converts to grayscale.
        fov_deg: screen half-width in units of M (note: historically
                 named `fov_deg`, but the kernel treats this as an
                 impact-parameter half-extent, not an angle).
        threshold: shadow boundary threshold (fraction of max intensity)

    Returns:
        dict with keys:
            'diameter_px': shadow diameter in pixels (2 * circle-fit radius)
            'diameter_M':  shadow diameter in units of M (circle fit).
                           Best for comparing against analytic critical
                           impact parameters like 2*3*sqrt(3) M.
            'ring_diameter_M': ellipse-fit major-axis diameter in M.
                           This is the **EHT observable** -- the M87* and
                           Sgr A* papers quote the bright ring's major
                           axis, not a circle-averaged diameter.
            'circularity': Delta C = 1 - b/a (0 = perfect circle)
            'center_x', 'center_y': shadow center in pixels
            'asymmetry': brightness asymmetry (ratio of left/right flux)
            'circle_fit_rms', 'ellipse_fit_rms': residuals
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

    # Brightness asymmetry: ratio of total flux left vs right of center
    left_flux = gray[:, : int(cx)].sum()
    right_flux = gray[:, int(cx):].sum()
    asymmetry = left_flux / (right_flux + 1e-12)

    return {
        "diameter_px": 2.0 * radius,
        "diameter_M": 2.0 * radius / px_per_M if px_per_M > 0 else None,
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
