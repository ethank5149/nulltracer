"""
Innermost Stable Circular Orbit (ISCO) for Kerr and Kerr–Newman black holes.

Analytic formula for Kerr (Bardeen, Press & Teukolsky 1972);
numerical bisection for Kerr–Newman.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["isco", "isco_kerr", "isco_kn", "r_plus"]


def r_plus(a: float, Q: float = 0.0) -> float:
    """Outer event-horizon radius.

    Parameters
    ----------
    a : float
        Dimensionless spin, 0 ≤ a < 1.
    Q : float
        Dimensionless charge, 0 ≤ Q.  a² + Q² < 1 required.
    """
    return 1.0 + math.sqrt(max(1.0 - a * a - Q * Q, 0.0))


def isco_kerr(a: float) -> float:
    """Prograde ISCO for a Kerr black hole (Bardeen, Press & Teukolsky 1972).

    Parameters
    ----------
    a : float
        Dimensionless spin parameter, 0 ≤ a < 1.

    Returns
    -------
    float
        ISCO radius in units of M.
    """
    a2 = a * a
    z1 = 1.0 + (1.0 - a2) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + max(1.0 - a, 0.0) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * a2 + z1 * z1)
    return 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def isco_kerr_vec(a: np.ndarray) -> np.ndarray:
    """Vectorised ISCO for arrays of spin values."""
    a2 = a * a
    z1 = 1.0 + (1.0 - a2) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + np.maximum(1.0 - a, 0.0) ** (1.0 / 3.0)
    )
    z2 = np.sqrt(3.0 * a2 + z1 * z1)
    return 3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def isco_kn(a: float, Q: float) -> float:
    """Prograde ISCO for a Kerr–Newman black hole (numerical bisection).

    Parameters
    ----------
    a : float
        Dimensionless spin.
    Q : float
        Dimensionless charge.

    Returns
    -------
    float
        ISCO radius found by bisection on dE/dr = 0.
    """
    a2 = a * a
    Q2 = Q * Q
    rh = 1.0 + math.sqrt(max(1.0 - a2 - Q2, 1e-12))

    def _energy(r: float) -> float:
        r2 = r * r
        delta = r2 - 2.0 * r + a2 + Q2
        if delta <= 0:
            return float("nan")
        gtt = -(1.0 - (2.0 * r - Q2) / r2)
        gtph = -a * (2.0 * r - Q2) / r2
        gphph = (r2 * r2 + a2 * r2 + a2 * (2.0 * r - Q2)) / r2
        dgtt = 2.0 * (Q2 - r) / (r2 * r)
        dgtph = 2.0 * a * (r - Q2) / (r * r2)
        dgphph = 2.0 * r + a2 * (-2.0 / r2 + 2.0 * Q2 / (r2 * r))
        disc = dgtph * dgtph - dgtt * dgphph
        if disc < 0:
            return float("nan")
        Om = (-dgtph + math.sqrt(disc)) / dgphph
        denom = -(gtt + 2.0 * gtph * Om + gphph * Om * Om)
        if denom <= 0:
            return float("nan")
        ut = 1.0 / math.sqrt(denom)
        return -(gtt + gtph * Om) * ut

    dr = 1e-5

    def _dEdr(r: float) -> float:
        Ep = _energy(r + dr)
        Em = _energy(r - dr)
        if math.isnan(Ep) or math.isnan(Em):
            return float("nan")
        return (Ep - Em) / (2.0 * dr)

    lo, hi = rh + 0.01, 9.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        d = _dEdr(mid)
        if math.isnan(d) or d < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def isco(a: float, Q: float = 0.0) -> float:
    """ISCO radius for a Kerr–Newman black hole.

    Dispatches to the analytic Kerr formula when *Q* = 0.

    Parameters
    ----------
    a : float
        Dimensionless spin, 0 ≤ a < 1.
    Q : float
        Dimensionless charge, default 0.

    Returns
    -------
    float
        ISCO radius in units of M.
    """
    if Q == 0 or not Q:
        return isco_kerr(a)
    return isco_kn(a, Q)
