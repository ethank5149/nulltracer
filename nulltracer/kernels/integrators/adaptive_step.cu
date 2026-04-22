/* ============================================================
 *  ADAPTIVE_STEP ??? Shared adaptive step size functions
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


/* -- Tao extended phase space integrators -------------------- */

/* Geometric heuristic: step size scales with distance from horizon.
 * The extended phase space method handles the non-separability;
 * the step size just needs to scale with distance from horizon. */
__device__ double adaptive_step_tao(double r, double rp,
                                    double step_size, double obs_dist) {
    double h_scaled = step_size * (obs_dist / 30.0);
    double he = h_scaled * fmin(fmax((r - rp) * 0.4, 0.04), 1.0);
    return fmin(fmax(he, 0.012), 1.0);
}


/* -- Non-symplectic RK4 -------------------------------------- */

/* RK4 needs ~1.7?? more affine parameter than symplectic integrators
 * to avoid energy-drift-induced ray capture, especially for
 * near-edge-on inclinations in strong gravity. */
__device__ double adaptive_step_rk4(double r, double rp,
                                    double step_size, double obs_dist) {
    double h_scaled = step_size * (obs_dist / 30.0) * 1.7;
    double he = h_scaled * fmin(fmax((r - rp) * 0.5, 0.04), 1.0);
    return fmin(fmax(he, 0.02), 1.4);
}


/* -- RKDP8 initial step estimate ----------------------------- */

/* Initial step size for Dormand-Prince 8(7).  Subsequent steps
 * are controlled by the embedded error estimator. */
__device__ double adaptive_step_rkdp8_initial(double r, double rp,
                                              double step_size, double obs_dist,
                                              double h_min, double h_max) {
    double h_scaled = step_size * (obs_dist / 30.0);
    double he = h_scaled * fmin(fmax((r - rp) * 0.4, 0.04), 1.0);
    return fmin(fmax(he, h_min), h_max);
}


/* -- Sundman / Mino time step (kahanli8s, kahanli8s_ks) ----- */

/* Compute the Mino-time step ???? from the geodesic budget.
 *
 * Sundman (Mino time) transformation: d?? = d??/??.
 * Fixed steps in Mino time ?? give physical steps ???? = ????????
 * that automatically shrink near the horizon (small ??) and
 * grow far away (large ??).
 *
 * The total Mino time for a round-trip from the photon sphere
 * r_ph to the escape radius r_esc is:
 *   ??_needed = 2??(1/r_ph ??? 1/r_esc)
 *
 * The Mino-time step is:
 *   ???? = (1 + step_size) ?? ??_needed / N
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
        /* Conservative bound: r+ ??? r_ph for all Kerr-Newman */
        r_ph = rp;
    }
    double tau_needed = 2.0 * (1.0 / r_ph - 1.0 / esc_radius);
    return (1.0 + step_size) * tau_needed / (double)STEPS;
}

/* Convert Mino-time step to physical affine parameter step.
 * Physical step ???? = ????????, with angular rate limiting.
 *
 * The Sundman transformation gives ???? = ????????, which automatically
 * shrinks near the horizon (small ??) and grows far away (large ??).
 * However, at large r the step can become so large that rays aimed
 * near the coordinate pole overshoot ?? = 0 or ?? = ?? in a single
 * step, triggering the pole reflection boundary and corrupting the
 * ray trajectory.
 *
 * Following ipole (Mo??cibrodzka et al.), we add an angular rate
 * limiter: the step is clamped so that |????| = |p_??/??| ?? ???? ??? ??_max.
 * This ensures no single step can move the ray more than ~17?? in
 * the ?? direction, preventing pole overshooting while preserving
 * the Sundman scaling's benefits for radial motion.
 *
 * Additionally, near the pole (?? < 0.1 or ?? > ?????0.1), the step
 * is further reduced proportionally to the distance from the pole,
 * matching ipole's d2fac pole-proximity factor.
 *
 * Reference: ipole, model_geodesics.c:267 (Mo??cibrodzka et al.)
 */


/* -- ??-variable adaptive stepping (Wu et al. 2024 / Preto & Saha 2009)
 *
 *  The AS??? algorithm wraps the existing S??? integrator with two
 *  additional leapfrog half-steps per outer iteration that evolve
 *  an auxiliary variable ??.  This provides adaptive time stepping
 *  that preserves symplecticity.
 *
 *  The Sundman function here is g = ??/r?? (NOT the implicit g = ??
 *  used in sundman_physical_step).  Combined with ?? ??? j/r, the
 *  physical step h/?? grows with r, providing larger steps far
 *  from the black hole while maintaining small steps near it.
 *
 *  References:
 *    [1] X. Wu et al., "Explicit symplectic methods in black hole
 *        spacetimes," Astrophys. J. 2024.
 *    [2] M. Preto & S. Saha, "On post-Newtonian orbits and the
 *        Galactic-center stars," Astrophys. J. 703:1743, 2009.
 * --------------------------------------------------------------- */

/* Compute the Sundman function g = ??/r?? for the ??-variable method.
 * This differs from the implicit g = ?? used in sundman_physical_step().
 * For large r, g ??? 1 since ?? ??? r??. */
__device__ double phi_var_sundman_g(double r, double th, double a) {
    double cth = cos(th);
    double sig = r * r + a * a * cth * cth;  /* ?? */
    return sig / (r * r);                      /* g = ??/r?? */
}

/* Compute the ?? half-step increment for Boyer-Lindquist coordinates.
 * d?? = ???g ?? h ?? (g^rr ?? p_r) / (2r)
 * where g^rr = ??/?? in BL. */
__device__ double phi_var_dphi_BL(double r, double th, double pr,
                                   double a, double Q2,
                                   double g, double h) {
    double cth = cos(th);
    double sig = r * r + a * a * cth * cth;  /* ?? */
    double del = r * r - 2.0 * r + a * a + Q2;  /* ?? */
    if (del < 1e-14) del = 1e-14;  /* clamp like geoRHS does */
    double grr = del / sig;  /* g^rr in BL */
    double v_r = grr * pr;   /* radial velocity component from p_r */
    return -g * h * v_r / (2.0 * r);
}

/* Compute the ?? half-step increment for Kerr-Schild coordinates.
 * d?? = ???g ?? h ?? (g^rr_KS ?? p_r_KS) / (2r)
 * In KS, g^rr_KS = ??/?? + (2r ??? Q??)/?? = (?? + 2r ??? Q??)/??. */
__device__ double phi_var_dphi_KS(double r, double th, double pr,
                                   double a, double Q2,
                                   double g, double h) {
    double cth = cos(th);
    double sig = r * r + a * a * cth * cth;  /* ?? */
    double del = r * r - 2.0 * r + a * a + Q2;  /* ?? */
    double w = 2.0 * r - Q2;
    double grr_ks = (del + w) / sig;  /* g^rr in KS = (?? + w)/?? */
    double v_r = grr_ks * pr;
    return -g * h * v_r / (2.0 * r);
}

/* Compute the physical step for the ??-variable method.
 * Returns h/?? with angular rate limiting and safety clamping,
 * matching the same limiters as sundman_physical_step(). */
__device__ double phi_var_physical_step(double h, double Phi,
                                         double r, double th,
                                         double pth, double a,
                                         double obs_dist) {
    double he = h / Phi;

    /* Angular rate limiter: |????| ??? 0.3 rad (~17??) per step.
     * d??/d?? = p_?? / ??  (from the geodesic equations). */
    double sth = sin(th);
    if (fabs(sth) > 1e-8) {
        double sig = r * r + a * a * cos(th) * cos(th);
        double dth_rate = fabs(pth / sig);
        double max_dth = 0.3;
        if (fabs(he) * dth_rate > max_dth) {
            he = copysign(max_dth / dth_rate, he);
        }
    }

    /* Pole proximity factor (ipole-inspired): reduce step further
     * when within ~0.1 rad of either pole. */
    double pole_dist = fmin(th, PI - th);
    if (pole_dist < 0.1) {
        double pole_factor = pole_dist / 0.1;
        pole_factor = fmax(pole_factor, 0.05);
        he *= pole_factor;
    }

    /* Final clamp (same bounds as sundman_physical_step) */
    double abs_he = fabs(he);
    abs_he = fmax(abs_he, 0.005);
    abs_he = fmin(abs_he, 0.2 * obs_dist);
    he = copysign(abs_he, he);

    return he;
}


#endif /* ADAPTIVE_STEP_CU */
