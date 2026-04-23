"""Test ISCO calculations against known analytic results."""

from nulltracer.isco import isco_kerr

import math
import pytest


# Bardeen, Press & Teukolsky (1972) analytic values
# ISCO for prograde orbits around Kerr black holes (M=1)
KNOWN_ISCO = [
    # (spin, expected_isco_radius, tolerance)
    (0.0, 6.0, 1e-10),  # Schwarzschild
    (0.5, 4.233, 1e-3),  # Moderate spin
    (0.9, 2.321, 1e-3),  # High spin
    (0.998, 1.237, 1e-2),  # Near-extremal
]


@pytest.mark.parametrize("spin,expected,tol", KNOWN_ISCO)
def test_isco_kerr(spin, expected, tol):
    """ISCO radius matches Bardeen-Press-Teukolsky formula."""
    result = isco_kerr(spin)
    assert abs(result - expected) < tol, f"ISCO(a={spin}): got {result}, expected {expected} ± {tol}"


def test_schwarzschild_isco_exact():
    """Schwarzschild ISCO is exactly 6M."""
    result = isco_kerr(0.0)
    assert abs(result - 6.0) < 1e-10


def test_isco_monotonically_decreasing():
    """ISCO decreases with increasing spin (prograde)."""
    spins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.998]
    iscos = [isco_kerr(a) for a in spins]
    for i in range(len(iscos) - 1):
        assert iscos[i] > iscos[i + 1], (
            f"ISCO not monotonically decreasing: ISCO(a={spins[i]})={iscos[i]} vs ISCO(a={spins[i + 1]})={iscos[i + 1]}"
        )


def test_isco_kn_reduces_to_kerr():
    """Numerical Kerr-Newman ISCO reduces to analytic Kerr ISCO when Q=0."""
    from nulltracer.isco import isco_kn, isco_kerr
    spins = [0.0, 0.5, 0.9, 0.998]
    for a in spins:
        assert abs(isco_kn(a, 0.0) - isco_kerr(a)) < 1e-4
