"""
Port of ISCO calculations from nulltracer/index.html (lines 861-949).

Computes the Innermost Stable Circular Orbit (ISCO) radius for
Kerr and Kerr-Newman black holes.
"""

import math


def r_plus(a: float, Q: float) -> float:
    """Event horizon radius for a Kerr-Newman black hole.

    Args:
        a: Dimensionless spin parameter (0 <= a < 1).
        Q: Dimensionless electric charge (0 <= Q < 1).

    Returns:
        Outer event horizon radius r+.
    """
    return 1.0 + math.sqrt(max(1.0 - a * a - Q * Q, 0.0))


def isco_kerr(a: float) -> float:
    """Analytic Kerr ISCO (Bardeen, Press & Teukolsky 1972).

    Prograde ISCO for a Kerr black hole with spin parameter a.

    Args:
        a: Dimensionless spin parameter (0 <= a < 1).

    Returns:
        ISCO radius.
    """
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + max(1.0 - a, 0.0) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def isco_kn(a: float, Q: float) -> float:
    """Numerical Kerr-Newman ISCO via bisection on dE/dr = 0.

    Args:
        a: Dimensionless spin parameter.
        Q: Dimensionless electric charge.

    Returns:
        ISCO radius found by bisection.
    """
    a2 = a * a
    Q2 = Q * Q
    rh = 1.0 + math.sqrt(max(1.0 - a2 - Q2, 1e-12))

    def energy(r: float) -> float:
        """Energy of a circular orbit at radius r (equatorial plane)."""
        r2 = r * r
        delta = r2 - 2.0 * r + a2 + Q2
        if delta <= 0:
            return float("nan")

        # Metric components at equator (Sigma = r^2)
        gtt = -(1.0 - (2.0 * r - Q2) / r2)
        gtph = -a * (2.0 * r - Q2) / r2
        gphph = (r2 * r2 + a2 * r2 + a2 * (2.0 * r - Q2)) / r2

        # Derivatives of metric components
        dgtt = 2.0 * (Q2 - r) / (r2 * r)
        dgtph_clean = 2.0 * a * (r - Q2) / (r * r2)
        dgphph = 2.0 * r + a2 * (-2.0 / r2 + 2.0 * Q2 / (r2 * r))

        # Circular orbit condition: dgtt + 2*Om*dgtph + Om^2*dgphph = 0
        disc = dgtph_clean * dgtph_clean - dgtt * dgphph
        if disc < 0:
            return float("nan")
        Om = (-dgtph_clean + math.sqrt(disc)) / dgphph  # prograde

        # E = -(g_tt + g_tphi * Om) * u^t
        # u^t = 1/sqrt(-(g_tt + 2*g_tphi*Om + g_phiphi*Om^2))
        denom = -(gtt + 2.0 * gtph * Om + gphph * Om * Om)
        if denom <= 0:
            return float("nan")
        ut = 1.0 / math.sqrt(denom)
        return -(gtt + gtph * Om) * ut

    # Bisect on dE/dr = 0
    dr = 1e-5

    def dEdr(r: float) -> float:
        Ep = energy(r + dr)
        Em = energy(r - dr)
        if math.isnan(Ep) or math.isnan(Em):
            return float("nan")
        return (Ep - Em) / (2.0 * dr)

    lo = rh + 0.01
    hi = 9.0
    # ISCO is where dE/dr changes from negative to positive
    for _ in range(80):
        mid = (lo + hi) / 2.0
        d = dEdr(mid)
        if math.isnan(d) or d < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def isco(a: float, Q: float) -> float:
    """Compute the ISCO radius for a Kerr-Newman black hole.

    Uses the analytic Kerr formula when Q=0, and numerical
    bisection for the general Kerr-Newman case.

    Args:
        a: Dimensionless spin parameter (0 <= a < 1).
        Q: Dimensionless electric charge (0 <= Q < 1).

    Returns:
        ISCO radius.
    """
    if not Q or Q == 0:
        return isco_kerr(a)
    return isco_kn(a, Q)
