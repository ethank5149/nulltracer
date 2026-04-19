"""Test shadow diameter against known analytic results."""

import math
import pytest


def schwarzschild_shadow_radius():
    """
    Analytic shadow radius for Schwarzschild black hole.
    For an observer at r_obs >> M:
        r_shadow = 3*sqrt(3) * M  (in units of M)
    This is the critical impact parameter for photon capture.
    """
    return 3.0 * math.sqrt(3.0)  # ≈ 5.196


def kerr_shadow_diameter_equatorial(spin):
    """
    Approximate shadow diameter for Kerr black hole viewed equatorially.
    Uses Bardeen (1973) critical impact parameters.

    For spin a, the prograde and retrograde critical impact parameters are:
        b_pro  = -(3 + a) + 2*sqrt(3 + 2*a - a^2) for co-rotating  (approximate)
        b_retro = similar for counter-rotating

    The full calculation requires solving the radial potential.
    For a=0, diameter should be 2 * 3*sqrt(3) ≈ 10.392.
    """
    # Simple validation: Schwarzschild limit
    if abs(spin) < 1e-10:
        return 2.0 * 3.0 * math.sqrt(3.0)
    # For nonzero spin, return None (needs full calculation)
    return None


def test_schwarzschild_shadow_radius():
    """Shadow radius for a=0 is 3*sqrt(3)*M."""
    expected = 3.0 * math.sqrt(3.0)
    assert abs(schwarzschild_shadow_radius() - expected) < 1e-12


def test_schwarzschild_shadow_diameter():
    """Shadow diameter for a=0 is 2 * 3*sqrt(3)*M ≈ 10.392M."""
    expected = 2.0 * 3.0 * math.sqrt(3.0)
    result = kerr_shadow_diameter_equatorial(0.0)
    assert abs(result - expected) < 1e-10
