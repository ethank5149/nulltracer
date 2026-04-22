"""Tests for the analytic shadow-boundary utility.

Verifies:
- Kerr limit (Q=0) reproduces the Bardeen (1973) shadow.
- Schwarzschild limit (a=0, Q=0) gives a circle of radius 3√3 M.
- Reissner-Nordström limit (a=0, Q>0) gives the correct analytic radius
  r_ph²/√(r_ph² - 2r_ph + Q²) with r_ph = (3+√(9-8Q²))/2.
- Hyperextremal configurations (a² + Q² > 1) raise ValueError.
"""

import numpy as np
import pytest


def test_schwarzschild_circular_shadow():
    from nulltracer.compare import shadow_boundary
    alpha, beta_p, beta_m = shadow_boundary(0.0, np.pi / 2, N=400)
    r_top = np.sqrt(alpha ** 2 + beta_p ** 2).max()
    r_bot = np.sqrt(alpha ** 2 + beta_m ** 2).max()
    r = max(r_top, r_bot)
    expected = 3.0 * np.sqrt(3.0)
    assert abs(r - expected) / expected < 1e-4, f"got {r}, expected {expected}"


@pytest.mark.parametrize("Q, r_ph_expected", [
    (0.0, 3.000000),
    (0.3, 2.938749),
    (0.6, 2.736932),
    (0.9, 2.293725),
])
def test_reissner_nordstrom_radius(Q, r_ph_expected):
    """For a=0 the shadow is a circle of radius r_ph²/√(r_ph²-2r_ph+Q²)."""
    from nulltracer.compare import shadow_boundary
    b_expected = r_ph_expected ** 2 / np.sqrt(
        r_ph_expected ** 2 - 2 * r_ph_expected + Q ** 2
    )
    alpha, beta_p, _ = shadow_boundary(0.0, np.pi / 2, Q=Q, N=200)
    r = np.sqrt(alpha ** 2 + beta_p ** 2).max()
    assert abs(r - b_expected) / b_expected < 1e-4


def test_rn_shadow_decreases_with_charge():
    """RN shadow shrinks as charge grows (at fixed a=0)."""
    from nulltracer.compare import shadow_boundary
    radii = []
    for Q in [0.0, 0.3, 0.6, 0.9]:
        alpha, beta_p, _ = shadow_boundary(0.0, np.pi / 2, Q=Q, N=100)
        radii.append(np.sqrt(alpha ** 2 + beta_p ** 2).max())
    for i in range(len(radii) - 1):
        assert radii[i] > radii[i + 1], (
            f"shadow radius should decrease with Q: {radii}"
        )


def test_kerr_reduces_to_bardeen():
    """The Q=0 path of the new code must agree with the old Bardeen formula."""
    from nulltracer.compare import shadow_boundary
    for a in [0.3, 0.6, 0.9, 0.998]:
        alpha, beta_p, _ = shadow_boundary(a, np.radians(60.0), Q=0.0, N=500)
        # Bardeen photon orbit endpoints (analytic)
        r_min_bardeen = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
        r_max_bardeen = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(+a)))
        # At the endpoints, η → 0, so β → 0. The α extents of our numerical
        # contour should bracket the Bardeen analytic endpoints.
        xi_min = -(r_min_bardeen ** 3 - 3 * r_min_bardeen ** 2
                   + a ** 2 * r_min_bardeen + a ** 2) / (a * (r_min_bardeen - 1))
        xi_max = -(r_max_bardeen ** 3 - 3 * r_max_bardeen ** 2
                   + a ** 2 * r_max_bardeen + a ** 2) / (a * (r_max_bardeen - 1))
        alpha_min_bardeen = -xi_min / np.sin(np.radians(60.0))
        alpha_max_bardeen = -xi_max / np.sin(np.radians(60.0))
        a_lo = min(alpha_min_bardeen, alpha_max_bardeen)
        a_hi = max(alpha_min_bardeen, alpha_max_bardeen)
        assert abs(alpha.min() - a_lo) / max(abs(a_lo), 1) < 1e-3
        assert abs(alpha.max() - a_hi) / max(abs(a_hi), 1) < 1e-3


def test_kerr_newman_intermediate_case():
    """K-N (a, Q both nonzero) returns a closed, asymmetric contour."""
    from nulltracer.compare import shadow_boundary
    alpha, beta_p, beta_m = shadow_boundary(0.5, np.radians(60.0), Q=0.3, N=500)
    # Closed contour, nonzero extent
    assert alpha.max() > alpha.min()
    assert beta_p.max() > 0
    # Kerr (Q=0) shadow should be larger than K-N (Q>0) at the same a,θ
    alpha_k, _, _ = shadow_boundary(0.5, np.radians(60.0), Q=0.0, N=500)
    width_kn = alpha.max() - alpha.min()
    width_k = alpha_k.max() - alpha_k.min()
    assert width_kn < width_k


def test_hyperextremal_raises():
    """a² + Q² > 1 has no horizon — must reject, not silently return garbage."""
    from nulltracer.compare import shadow_boundary
    with pytest.raises(ValueError, match="Hyperextremal"):
        shadow_boundary(0.9, np.pi / 2, Q=0.9)


def test_kerr_contour_is_closed():
    """beta_plus and beta_minus meet at α endpoints (top and bottom of shadow)."""
    from nulltracer.compare import shadow_boundary
    _, beta_p, beta_m = shadow_boundary(0.7, np.radians(45.0), N=1000)
    # Endpoints are equatorial photon orbits where η=0 ⇒ β=0
    assert abs(beta_p[0]) < 1e-2
    assert abs(beta_p[-1]) < 1e-2
    assert abs(beta_m[0]) < 1e-2
    assert abs(beta_m[-1]) < 1e-2
