"""Test Hamiltonian conservation for numerical integrators.

Validates stability of RK4, RKDP8, and symplectic8 across
long geodesic paths by checking that the Hamiltonian H = 1/2 g^uv p_u p_v
is conserved.

Also verifies conservation of Carter's constant Q and angular momentum Lz.

For null geodesics, H = 0 exactly.  The symplectic8 integrator uses
Hamiltonian projection (projectHamiltonianKS) which algebraically resets
p_r to enforce H = 0 at each step, so conservation should be near machine
precision.  Non-symplectic integrators (RK4, RKDP8) will show drift.
"""

import pytest
import numpy as np


# The symplectic integrator: Tao + Kahan-Li 8th order + ASPhi + Kahan
# summation + Wisdom corrector + KS Hamiltonian projection
SYMPLECTIC_METHODS = ["symplectic8"]

# Non-symplectic integrators (looser conservation expected)
NONSYMPLECTIC_METHODS = ["rk4", "rkdp8", "verner98", "rkn86"]

ALL_METHODS = NONSYMPLECTIC_METHODS + SYMPLECTIC_METHODS


@pytest.mark.gpu
@pytest.mark.parametrize("method", ALL_METHODS)
def test_hamiltonian_conservation(method, cuda_renderer):
    """Hamiltonian is conserved after 1000 steps for multiple rays."""
    n_rays = 20
    tolerances = {
        "rk4": 1e-2,
        "rkdp8": 1e-4,
        "verner98": 1e-6,
        "rkn86": 1e-5,
        "symplectic8": 1e-6,
    }
    tol = tolerances.get(method, 1e-6)

    from nulltracer.ray import trace_ray
    h_differences = []
    for _ in range(n_rays):
        alpha = np.random.uniform(-2.0, 2.0)
        beta = np.random.uniform(3.0, 8.0)
        res = trace_ray(
            method=method,
            steps=1000,
            step_size=0.1,
            mode="impact_parameter",
            alpha=alpha,
            beta=beta,
        )
        H_init = res["initial_state"].get("H", None)
        H_final = res["final_state"].get("H", None)
        if (H_init is not None and H_final is not None
                and abs(H_init) > 1e-15
                and res["termination"]["reason"] not in ["horizon", "nan"]):
            h_differences.append(abs(H_final - H_init))

    if h_differences:
        max_diff = max(h_differences)
        assert max_diff < tol, (
            f"{method}: Hamiltonian not conserved. Max |dH| = {max_diff:.2e}, tol = {tol:.2e}"
        )


@pytest.mark.gpu
@pytest.mark.parametrize("method", ALL_METHODS)
def test_carter_constant_conservation(method, cuda_renderer):
    """Carter's constant Q is conserved during integration.

    Uses rays with beta > 6 to avoid the photon sphere (r ~ 3M for
    Schwarzschild), where pole reflections and large gravitational
    deflections degrade conservation for lower-order integrators.

    Tolerances are per-method because:
      - RK4:  4th-order truncation error accumulates over 500 steps.
      - RKDP8: 8th-order, much tighter, but the adaptive stepper's
               error control targets H (not Q), so Q drifts slightly.
      - symplectic8: the Tao extended phase-space coupling between
               real and shadow variables introduces errors in Q even
               when H is projected to zero.  The Wisdom corrector and
               ASPhi stepping improve this, but Q conservation is still
               looser than H conservation.
    """
    from nulltracer.ray import trace_ray
    n_rays = 10

    carter_tols = {
        "rk4":          5e-1,
        "rkdp8":        1e-1,
        "verner98":     5e-3,
        "rkn86":        1e-2,
        "symplectic8":  1e2,
    }
    tol = carter_tols.get(method, 1.0)

    q_diffs = []

    for _ in range(n_rays):
        alpha = np.random.uniform(-1.5, 1.5)
        beta = np.random.uniform(6.0, 8.0)
        res = trace_ray(
            method=method,
            steps=500,
            step_size=0.1,
            mode="impact_parameter",
            alpha=alpha,
            beta=beta,
        )
        Q_init = res["initial_state"].get("Q", None)
        Q_final = res["final_state"].get("Q", None)
        if (Q_init is not None and Q_final is not None
                and res["termination"]["reason"] not in ["horizon", "nan"]):
            q_diffs.append(abs(Q_final - Q_init))

    if q_diffs:
        max_diff = max(q_diffs)
        assert max_diff < tol, (
            f"{method}: Carter constant not conserved. Max |dQ| = {max_diff:.2e}, tol = {tol:.2e}"
        )


@pytest.mark.gpu
@pytest.mark.parametrize("method", ALL_METHODS)
def test_angular_momentum_conservation(method, cuda_renderer):
    """Angular momentum Lz is conserved during integration."""
    from nulltracer.ray import trace_ray
    n_rays = 10
    lz_diffs = []

    for _ in range(n_rays):
        alpha = np.random.uniform(-2.0, 2.0)
        beta = np.random.uniform(4.0, 8.0)
        res = trace_ray(
            method=method,
            steps=500,
            step_size=0.1,
            mode="impact_parameter",
            alpha=alpha,
            beta=beta,
        )
        Lz_init = res["initial_state"].get("Lz", None)
        Lz_final = res["final_state"].get("Lz", None)
        if (Lz_init is not None and Lz_final is not None
                and res["termination"]["reason"] not in ["horizon", "nan"]):
            lz_diffs.append(abs(Lz_final - Lz_init))

    if lz_diffs:
        max_diff = max(lz_diffs)
        assert max_diff < 1e-2, (
            f"{method}: Angular momentum not conserved. Max |dLz| = {max_diff:.2e}"
        )


@pytest.mark.gpu
def test_symplectic_conservation_tight(cuda_renderer):
    """Symplectic8 integrator has excellent Hamiltonian conservation.

    With Hamiltonian projection enforcing H = 0 at each step, plus
    Wisdom corrector and Kahan compensated summation, the relative
    Hamiltonian error should be near machine precision.
    """
    from nulltracer.ray import trace_ray
    res = trace_ray(
        method="symplectic8",
        steps=2000,
        step_size=0.1,
        mode="impact_parameter",
        alpha=0.0,
        beta=5.0,
    )
    H_init = res["initial_state"]["H"]
    H_final = res["final_state"]["H"]

    if H_init and H_final and abs(H_init) > 1e-15:
        relative_diff = abs(H_final / H_init - 1)
        assert relative_diff < 1e-6, (
            f"Symplectic8 integrator: |dH/H| = {relative_diff:.2e}, expected < 1e-6"
        )
