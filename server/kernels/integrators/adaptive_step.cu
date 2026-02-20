/* ============================================================
 *  ADAPTIVE_STEP — Shared adaptive step size functions
 *
 *  This file provides reusable adaptive step size computation
 *  for each integrator family.  Both the full-frame render
 *  kernels and the single-ray trace kernel include this file
 *  to guarantee identical step sizing across all code paths.
 *
 *  Requires geodesic_base.cu to be included first.
 * ============================================================ */

#ifndef ADAPTIVE_STEP_CU
#define ADAPTIVE_STEP_CU


/* ── Tao extended phase space integrators ──────────────────── */

/* Geometric heuristic: step size scales with distance from horizon.
 * The extended phase space method handles the non-separability;
 * the step size just needs to scale with distance from horizon. */
__device__ double adaptive_step_tao(double r, double rp,
                                    double step_size, double obs_dist) {
    double h_scaled = step_size * (obs_dist / 30.0);
    double he = h_scaled * fmin(fmax((r - rp) * 0.4, 0.04), 1.0);
    return fmin(fmax(he, 0.012), 1.0);
}


/* ── Non-symplectic RK4 ────────────────────────────────────── */

/* RK4 needs ~1.7× more affine parameter than symplectic integrators
 * to avoid energy-drift-induced ray capture, especially for
 * near-edge-on inclinations in strong gravity. */
__device__ double adaptive_step_rk4(double r, double rp,
                                    double step_size, double obs_dist) {
    double h_scaled = step_size * (obs_dist / 30.0) * 1.7;
    double he = h_scaled * fmin(fmax((r - rp) * 0.5, 0.04), 1.0);
    return fmin(fmax(he, 0.02), 1.4);
}


/* ── RKDP8 initial step estimate ───────────────────────────── */

/* Initial step size for Dormand-Prince 8(7).  Subsequent steps
 * are controlled by the embedded error estimator. */
__device__ double adaptive_step_rkdp8_initial(double r, double rp,
                                              double step_size, double obs_dist,
                                              double h_min, double h_max) {
    double h_scaled = step_size * (obs_dist / 30.0);
    double he = h_scaled * fmin(fmax((r - rp) * 0.4, 0.04), 1.0);
    return fmin(fmax(he, h_min), h_max);
}


/* ── Sundman / Mino time step (kahanli8s, kahanli8s_ks) ───── */

/* Compute the Mino-time step Δτ from the geodesic budget.
 *
 * Sundman (Mino time) transformation: dτ = dλ/Σ.
 * Fixed steps in Mino time τ give physical steps Δλ = Σ·Δτ
 * that automatically shrink near the horizon (small Σ) and
 * grow far away (large Σ).
 *
 * The total Mino time for a round-trip from the photon sphere
 * r_ph to the escape radius r_esc is:
 *   τ_needed = 2·(1/r_ph − 1/r_esc)
 *
 * The Mino-time step is:
 *   Δτ = (1 + step_size) · τ_needed / N
 *
 * Returns dtau (the fixed Mino-time step). */
__device__ double sundman_dtau(double a, double Q2, double rp,
                               double step_size, double esc_radius,
                               int STEPS) {
    double r_ph;
    if (Q2 < 1e-10) {
        /* Exact prograde equatorial photon orbit (Bardeen 1972) */
        r_ph = 2.0 * (1.0 + cos(2.0 / 3.0 * acos(-a)));
    } else {
        /* Conservative bound: r+ ≤ r_ph for all Kerr-Newman */
        r_ph = rp;
    }
    double tau_needed = 2.0 * (1.0 / r_ph - 1.0 / esc_radius);
    return (1.0 + step_size) * tau_needed / (double)STEPS;
}

/* Convert Mino-time step to physical affine parameter step.
 * Physical step Δλ = Δτ·Σ, clamped to prevent vanishing/overshooting. */
__device__ double sundman_physical_step(double dtau, double r, double th,
                                        double a, double obs_dist) {
    double cth = cos(th);
    double sig = r * r + a * a * cth * cth;
    double he = dtau * sig;
    return fmin(fmax(he, 0.005), 0.2 * obs_dist);
}


#endif /* ADAPTIVE_STEP_CU */
