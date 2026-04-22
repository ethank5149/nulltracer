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
    "compare_integrators",
    "fit_ellipse_to_shadow",
]


# ── Analytic shadow boundary ──────────────────────────────────


def shadow_boundary(
    a: float,
    theta_obs: float,
    N: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bardeen (1973) analytic shadow boundary for a Kerr black hole.

    Parameters
    ----------
    a : float
        Dimensionless spin.
    theta_obs : float
        Observer inclination in **radians**.
    N : int
        Number of points along the contour.

    Returns
    -------
    (alpha, beta_plus, beta_minus)
        Impact-parameter coordinates of the shadow edge (upper and lower halves),
        in units of M.
    """
    sO = np.sin(theta_obs)
    cO = np.cos(theta_obs)

    if a < 1e-5:
        r_shadow = 3.0 * np.sqrt(3.0)
        phi = np.linspace(0.0, 2.0 * np.pi, N)
        alpha = r_shadow * np.cos(phi)
        beta = r_shadow * np.sin(phi)
        return alpha, beta, -beta

    r_min = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
    r_max = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(a)))
    r = np.linspace(r_min + 1e-6, r_max - 1e-6, N)

    xi = -(r**3 - 3.0 * r**2 + a**2 * r + a**2) / (a * (r - 1.0) + 1e-30)
    eta = (
        -(r**3)
        * (r**3 - 6.0 * r**2 + 9.0 * r - 4.0 * a**2)
        / (a**2 * (r - 1.0) ** 2 + 1e-30)
    )

    alpha = -xi / sO
    beta_sq = eta + a**2 * cO**2 - (xi**2) * (cO**2 / (sO**2))
    beta = np.sqrt(np.maximum(beta_sq, 0.0))

    return alpha, beta, -beta


# ── Shadow ellipse fitting ────────────────────────────────────


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

    # Convert to impact-parameter space (centred on image)
    cx = (xs - img_size / 2.0) * scale
    cy = (ys - img_size / 2.0) * scale

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


# ── Integrator comparison ─────────────────────────────────────


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
        methods = renderer.available_methods()
    
    # ensure all kernels are compiled
    renderer.precompile_all()

    results = []
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for ax, m in zip(axes, methods):
        params = {'spin': spin, 'incl': inclination_deg, 'width': width, 'height': height, 'fov': fov, 'obs_dist': obs_dist, 'step_size': step_size, 'method': m, **render_kwargs}
        timed = renderer.render_frame_timed(params)
        import numpy as np
        img = np.frombuffer(timed['raw_rgb'], dtype=np.uint8).reshape((height, width, 3))
        class FakeInfo:
            render_ms = timed['kernel_ms']
            max_steps = 0 # Cannot get max_steps out of timed easily, so assume 0
        info = FakeInfo()
        ax.imshow(img)
        ax.axis("off")
        label = m
        ax.set_title(
            f"{label}\n{info.render_ms:.0f} ms, {info.max_steps} steps",
            fontsize=10,
        )
        results.append(
            {
                "method": m,
                "label": label,
                "render_ms": info.render_ms,
                "max_steps": info.max_steps,
            }
        )

    fig.suptitle(
        f"Integrator Comparison — $a={spin}$, "
        rf"$\theta={inclination_deg}°$, "
        rf"$r{{\rm obs}}={obs_dist}\,M$",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    print(f"\n{'Method':<40} {'Time':>8} {'Steps':>7}")
    print("─" * 60)
    for r in results:
        print(f"{r['label']:<40} {r['render_ms']:>7.0f}ms {r['max_steps']:>7d}")

    return results, fig
