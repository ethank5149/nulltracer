"""
Analytic shadow boundary and integrator comparison utilities.

:func:`shadow_boundary` computes the Bardeen (1973) / Chandrasekhar (1983)
analytic shadow contour for a Kerr black hole, useful for validating
the numerical ray tracer.

:func:`compare_integrators` renders with every available method for
side-by-side visual and performance comparison.

:func:`fit_ellipse_to_shadow` extracts geometric shadow observables
(diameter, asymmetry, centroid offset) from a classified shadow mask.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .isco import isco_kerr

__all__ = [
    "shadow_boundary",
    "shadow_observables",
    "compare_integrators",
    "fit_ellipse_to_shadow",
]


# ── Analytic shadow observables ──────────────────────────────

def shadow_observables(
    a: float,
    theta_obs: float,
    Q: float = 0.0,
    N: int = 2000,
) -> dict:
    """Compute analytic shadow observables for EHT comparison.

    Returns a dict with every diameter metric needed for comparing
    against EHT measurements, along with the angular sizes for
    M87* and Sgr A*.

    Parameters
    ----------
    a : float
        Dimensionless spin.
    theta_obs : float
        Observer inclination in **radians**.
    Q : float
        Dimensionless charge.
    N : int
        Number of contour points.

    Returns
    -------
    dict
        Geometric quantities (in M = GM/c²):
            diameter_circle_M   – mean-radius circle fit × 2
            diameter_alpha_M    – α-direction extent
            diameter_beta_M     – β-direction extent (**best for EHT**)
            centroid_shift_M    – α-centroid offset from spin
            circularity_delta_C – 1 − min/max extent
        Angular sizes (in μas):
            m87_shadow_uas      – shadow D_β at M87* scale
            m87_ring_est_uas    – estimated ring (shadow × 1.10 calib)
            sgra_shadow_uas     – shadow D_β at Sgr A* scale
            sgra_ring_est_uas   – estimated ring (shadow × 1.10 calib)
    """
    from .eht_validation import EHT_M87, EHT_SGRA, uas_per_M

    alpha, bp, bm = shadow_boundary(a, theta_obs, Q=Q, N=N)

    # Full contour
    alpha_full = np.concatenate([alpha, alpha[::-1]])
    beta_full = np.concatenate([bp, bm[::-1]])

    # Circle fit
    cx_a = (alpha.max() + alpha.min()) / 2.0
    r_vals = np.sqrt((alpha_full - cx_a) ** 2 + beta_full ** 2)
    d_circle = 2.0 * r_vals.mean()

    d_alpha = alpha.max() - alpha.min()
    d_beta_top = bp.max()
    d_beta_bot = -bm.min()
    d_beta = d_beta_top + d_beta_bot

    d_max = max(d_alpha, d_beta)
    d_min = min(d_alpha, d_beta)
    delta_C = 1.0 - d_min / d_max if d_max > 0 else 0.0

    scale_m87 = uas_per_M(EHT_M87["mass_kg"], EHT_M87["dist_m"])
    scale_sgra = uas_per_M(EHT_SGRA["mass_kg"], EHT_SGRA["dist_m"])
    calib = EHT_M87["calib_ring_shadow"]

    return {
        "diameter_circle_M": d_circle,
        "diameter_alpha_M": d_alpha,
        "diameter_beta_M": d_beta,
        "centroid_shift_M": cx_a,
        "circularity_delta_C": delta_C,
        "m87_shadow_uas": d_beta * scale_m87,
        "m87_ring_est_uas": d_beta * scale_m87 * calib,
        "sgra_shadow_uas": d_beta * scale_sgra,
        "sgra_ring_est_uas": d_beta * scale_sgra * calib,
    }


# ?????? Analytic shadow boundary ??????????????????????????????????????????????????????????????????????????????????????????????????????


def shadow_boundary(
    a: float,
    theta_obs: float,
    Q: float = 0.0,
    N: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic shadow boundary for a Kerr or Kerr-Newman black hole.

    Derived from the spherical photon-orbit conditions :math:`R(r) = 0`,
    :math:`R'(r) = 0` for null geodesics in the Kerr-Newman spacetime
    (Boyer-Lindquist coordinates, geometric units :math:`M = 1`). The
    critical impact parameters are

    .. math::

        \\xi_c(r) &= -\\frac{r^3 - 3r^2 + a^2 r + a^2 + 2 r Q^2}{a(r - 1)}, \\\\
        \\eta_c(r) &= \\frac{r^2\\bigl[4 a^2 \\Delta - (r^2 - 3 r + 2 a^2 + 2 Q^2)^2\\bigr]}{a^2 (r - 1)^2},

    where :math:`\\Delta = r^2 - 2 r + a^2 + Q^2`. The photon-orbit radii
    (endpoints of the spherical orbit range) are the real roots of

    .. math::

        r^4 - 6 r^3 + (9 + 4 Q^2) r^2 - 4(a^2 + 3 Q^2) r + 4 Q^2 (a^2 + Q^2) = 0

    lying outside the outer horizon :math:`r_+ = 1 + \\sqrt{1 - a^2 - Q^2}`.
    For :math:`Q = 0` this reduces to Bardeen's (1973) Kerr result. For
    :math:`a = 0` (Reissner-Nordstr??m) the shadow is a circle of radius
    :math:`r_{\\rm ph}^2/\\sqrt{r_{\\rm ph}^2 - 2 r_{\\rm ph} + Q^2}` where
    :math:`r_{\\rm ph} = (3 + \\sqrt{9 - 8 Q^2})/2`.

    Parameters
    ----------
    a : float
        Dimensionless spin, :math:`0 \\le a^2 + Q^2 < 1`.
    theta_obs : float
        Observer inclination in **radians**.
    Q : float
        Dimensionless electric charge (Kerr-Newman). Default 0 (Kerr).
    N : int
        Number of points along the contour.

    Returns
    -------
    (alpha, beta_plus, beta_minus)
        Impact-parameter coordinates of the shadow edge (upper and lower halves),
        in units of :math:`M`.

    Notes
    -----
    Requires :math:`a^2 + Q^2 \\le 1` (cosmic-censorship bound). For
    values close to the extremal boundary the quartic roots may approach
    the horizon; the caller should sanity-check the returned contour.
    """
    if a**2 + Q**2 > 1.0:
        raise ValueError(
            f"Hyperextremal configuration: a?? + Q?? = {a**2 + Q**2:.4f} > 1 "
            f"(a={a}, Q={Q}). No horizon exists."
        )

    sO = np.sin(theta_obs)
    cO = np.cos(theta_obs)

    if abs(sO) < 1e-7:
        # Pole-on observer: limit where xi -> 0, shadow is a perfect circle.
        coeffs = [1.0, -3.0, a**2 + 2.0 * Q**2, a**2]
        r_plus = 1.0 + np.sqrt(max(0.0, 1.0 - a**2 - Q**2))
        real_roots = [r.real for r in np.roots(coeffs) if abs(r.imag) < 1e-8 and r.real > r_plus + 1e-10]
        if not real_roots:
            raise RuntimeError(f"Could not locate polar photon orbit for a={a}, Q={Q}.")
        r_c = max(real_roots)
        
        Delta_c = r_c**2 - 2.0 * r_c + a**2 + Q**2
        eta_c = (r_c**2 * (4.0 * a**2 * Delta_c - (r_c**2 - 3.0 * r_c + 2.0 * a**2 + 2.0 * Q**2)**2)) / (a**2 * (r_c - 1.0)**2)
        
        R = np.sqrt(max(0.0, eta_c + a**2))
        phi = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        return R * np.cos(phi), R * np.sin(phi), -R * np.sin(phi)

    # ?????? Schwarzschild / Reissner-Nordstr??m (spherically symmetric) ??????
    if a < 1e-5:
        if Q < 1e-12:
            r_shadow = 3.0 * np.sqrt(3.0)              # Schwarzschild
        else:
            # RN photon sphere r_ph = (3 + ???(9-8Q??))/2 requires Q?? ??? 9/8;
            # for physical 0 ??? Q?? < 1 this is always satisfied.
            r_ph = 0.5 * (3.0 + np.sqrt(9.0 - 8.0 * Q**2))
            r_shadow = r_ph**2 / np.sqrt(r_ph**2 - 2.0 * r_ph + Q**2)
        phi = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        alpha = r_shadow * np.cos(phi)
        beta = r_shadow * np.sin(phi)
        return alpha, beta, -beta

    # ?????? Kerr / Kerr-Newman: parameterise spherical photon orbits ??????
    # Photon-orbit radii are real roots of the quartic outside r_+.
    coeffs = [
        1.0,
        -6.0,
        9.0 + 4.0 * Q**2,
        -4.0 * (a**2 + 3.0 * Q**2),
        4.0 * Q**2 * (a**2 + Q**2),
    ]
    r_plus = 1.0 + np.sqrt(max(0.0, 1.0 - a**2 - Q**2))
    roots = np.roots(coeffs)
    real_roots = np.sort([
        r.real for r in roots
        if abs(r.imag) < 1e-8 and r.real > r_plus + 1e-10
    ])
    if len(real_roots) < 2:
        raise RuntimeError(
            f"shadow_boundary: could not locate two photon-orbit radii "
            f"outside r_+ = {r_plus:.4f} for a={a}, Q={Q}. "
            f"Real roots > r_+: {real_roots}"
        )
    r_min, r_max = real_roots[0], real_roots[-1]

    r = np.linspace(r_min + 1e-6, r_max - 1e-6, N)

    denom = a * (r - 1.0)
    xi = -(r**3 - 3.0 * r**2 + a**2 * r + a**2 + 2.0 * r * Q**2) / denom
    Delta = r**2 - 2.0 * r + a**2 + Q**2
    eta = (
        r**2 * (4.0 * a**2 * Delta - (r**2 - 3.0 * r + 2.0 * a**2 + 2.0 * Q**2) ** 2)
    ) / (a**2 * (r - 1.0) ** 2)

    alpha = -xi / sO
    beta_sq = eta + a**2 * cO**2 - (xi**2) * (cO**2 / (sO**2))
    beta = np.sqrt(np.maximum(beta_sq, 0.0))

    return alpha, beta, -beta


# ?????? Shadow ellipse fitting ????????????????????????????????????????????????????????????????????????????????????????????????????????????


def fit_ellipse_to_shadow(
    shadow_mask: np.ndarray,
    fov: float,
    img_size: int,
) -> dict:
    """Extract geometric observables from a boolean shadow mask.

    Fits an ellipse to the shadow boundary and returns the major/minor
    diameters, centroid offset, and fractional asymmetry.

    Parameters
    ----------
    shadow_mask : ndarray
        Boolean ``(H, W)`` array (True = shadow).
    fov : float
        Field of view in degrees used during classification.
    img_size : int
        Image width (assumed square).

    Returns
    -------
    dict
        Keys: ``diameter_M``, ``diameter_x_M``, ``diameter_y_M``,
        ``asymmetry``, ``centroid_offset_M``.
    """
    scale = 2.0 * fov / img_size  # degrees per pixel

    # Shadow pixel coordinates
    ys, xs = np.where(shadow_mask)
    if len(xs) == 0:
        return {
            "diameter_M": 0.0,
            "diameter_x_M": 0.0,
            "diameter_y_M": 0.0,
            "asymmetry": 0.0,
            "centroid_offset_M": 0.0,
        }

    # The shadow mask is boolean: True for shadow, False otherwise.
    # The kernel Maps ux in [-1, 1] to impact parameter alpha = ux * fov * aspect
    # Here the img_size is assumed square, so aspect = 1.
    # Thus the field of view mapping is from -fov to +fov in M.
    # Since we want impact parameter, cx = (xs - img_size / 2.0) / (img_size / 2.0) * fov
    cx = (xs - img_size / 2.0) / (img_size / 2.0) * fov
    cy = (ys - img_size / 2.0) / (img_size / 2.0) * fov

    x_extent = cx.max() - cx.min()
    y_extent = cy.max() - cy.min()

    centroid_x = cx.mean()
    centroid_y = cy.mean()
    centroid_offset = np.sqrt(centroid_x**2 + centroid_y**2)

    diameter = max(x_extent, y_extent)
    asymmetry = abs(x_extent - y_extent) / max(diameter, 1e-12)

    circularity = (
        min(x_extent, y_extent) / max(x_extent, y_extent)
        if max(x_extent, y_extent) > 0
        else 1.0
    )

    return {
        "diameter_M": float(diameter),
        "diameter_x_M": float(x_extent),
        "diameter_y_M": float(y_extent),
        "asymmetry": float(asymmetry),
        "centroid_offset_M": float(centroid_offset),
        "circularity": float(circularity),
        "delta_C": float(1.0 - circularity),
    }


# ?????? Integrator comparison ???????????????????????????????????????????????????????????????????????????????????????????????????????????????


def compare_integrators(
    spin: float = 0.6,
    inclination_deg: float = 80.0,
    *,
    obs_dist: float = 40.0,
    step_size: float = 0.3,
    width: int = 512,
    height: int = 512,
    fov: float = 7.0,
    methods: list[str] | None = None,
    **render_kwargs,
):
    """Render with multiple integrators for visual + timing comparison.

    Parameters
    ----------
    spin : float
        Dimensionless spin.
    inclination_deg : float
        Observer inclination in degrees.
    methods : list of str, optional
        Integrator names.  Default: all available render methods.
    **render_kwargs
        Forwarded to :func:`render_frame`.

    Returns
    -------
    (results, fig)
        ``results`` is a list of dicts with method, label, render_ms, max_steps.
        ``fig`` is a matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from .renderer import CudaRenderer
    

    renderer = CudaRenderer()
    renderer.initialize()
    if methods is None:
        methods = renderer.available_methods
    
    # ensure all kernels are compiled
    renderer.precompile_all()

    results = []
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for ax, m in zip(axes, methods):
        params = {'spin': spin, 'inclination': inclination_deg, 'width': width,
                  'height': height, 'fov': fov, 'obs_dist': obs_dist,
                  'step_size': step_size, 'method': m, **render_kwargs}
        timed = renderer.render_frame_timed(params)
        import numpy as np
        img = np.frombuffer(timed['raw_rgb'], dtype=np.uint8).reshape((height, width, 3))
        ax.imshow(img)
        ax.axis("off")
        label = m
        ax.set_title(
            f"{label}\n{timed['kernel_ms']:.0f} ms, {timed['max_steps']} steps",
            fontsize=10,
        )
        results.append(
            {
                "method": m,
                "label": label,
                "render_ms": timed['kernel_ms'],
                "max_steps": timed['max_steps'],
            }
        )

    fig.suptitle(
        f"Integrator Comparison ??? $a={spin}$, "
        rf"$\theta={inclination_deg}??$, "
        rf"$r{{\rm obs}}={obs_dist}\,M$",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    print(f"\n{'Method':<40} {'Time':>8} {'Steps':>7}")
    print("???" * 60)
    for r in results:
        print(f"{r['label']:<40} {r['render_ms']:>7.0f}ms {r['max_steps']:>7d}")

    return results, fig
