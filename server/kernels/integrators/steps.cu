/* ============================================================
 *  STEPS — Shared integrator step functions
 *
 *  This file provides reusable single-step functions for each
 *  integration method.  Both the full-frame render kernels and
 *  the single-ray trace kernel include this file to guarantee
 *  identical integration logic across all code paths.
 *
 *  Requires geodesic_base.cu to be included first.
 * ============================================================ */

#ifndef INTEGRATOR_STEPS_CU
#define INTEGRATOR_STEPS_CU


/* ============================================================
 *  TAO EXTENDED PHASE SPACE INTEGRATORS
 *
 *  Molei Tao (2016), "Explicit symplectic approximation of
 *  nonseparable Hamiltonians: Algorithm and long time performance,"
 *  Phys. Rev. E 94, 043303.
 *
 *  The Kerr-Newman Hamiltonian is non-separable:
 *    H = ½[g^rr(q)·p_r² + g^θθ(q)·p_θ²] + V(q)
 *  where the kinetic energy depends on both q and p.
 *
 *  Tao's method embeds the system in a doubled phase space
 *  (q, p, x, y) with the extended Hamiltonian:
 *    H_ext = H_A(q,y) + H_B(x,p) + ω·H_C
 *  where H_A = H(q,y), H_B = H(x,p), and
 *  H_C = ½(|q-x|² + |p-y|²) is the harmonic coupling.
 *
 *  The three exact flows (Tao 2016, Eq. 1) are:
 *
 *    φ_A^τ: Evaluate H(q,y).
 *            p  → p  − τ·∂_q H(q,y)    [update real momenta]
 *            x  → x  + τ·∂_y H(q,y)    [update shadow positions]
 *            (q and y unchanged)
 *
 *    φ_B^τ: Evaluate H(x,p).
 *            q  → q  + τ·∂_p H(x,p)    [update real positions]
 *            y  → y  − τ·∂_x H(x,p)    [update shadow momenta]
 *            (x and p unchanged)
 *
 *    φ_C^τ: Harmonic rotation coupling q↔x and p↔y.
 *
 *  The 2nd-order Strang base step (Tao 2016, Eq. 2) is:
 *    φ_2^δ = φ_A^{δ/2} ∘ φ_B^{δ/2} ∘ φ_C^δ ∘ φ_B^{δ/2} ∘ φ_A^{δ/2}
 *
 *  Higher-order methods compose the base step via triple-jump
 *  (Yoshida) or optimal palindromic (Kahan-Li) coefficients.
 *
 *  Coupling parameter: ω = TAO_OMEGA_C / h, following Tao's
 *  recommendation that ω·h = O(1).
 * ============================================================ */

/* Tao coupling constant: ω·h = TAO_OMEGA_C.
 * Tao (2016) recommends O(1); values 1-5 work well.
 * Larger values bind shadow variables more tightly but
 * increase stiffness.  2.0 is a good balance. */
#define TAO_OMEGA_C 2.0


/* ── Tao harmonic rotation (φ_C flow) ─────────────────────── */

/* Exact solution of the harmonic coupling ωH_C for time δ.
 *
 * H_C = ½(|q-x|² + |p-y|²).  The time-δ flow of ωH_C
 * rotates each conjugate pair (q_i-x_i, p_i-y_i) by angle 2ωδ
 * (Tao 2016, Eq. 1):
 *
 *   R(δ) = [cos(2ωδ)·I   sin(2ωδ)·I]
 *          [-sin(2ωδ)·I   cos(2ωδ)·I]
 *
 * acting on the difference vector (q-x, p-y).  The center of
 * mass (q+x)/2 and (p+y)/2 are preserved.
 *
 * For each conjugate DOF (q_i, p_i, x_i, y_i):
 *   u = q_i - x_i,  v = p_i - y_i
 *   u' = cos(2ωδ)·u + sin(2ωδ)·v
 *   v' = -sin(2ωδ)·u + cos(2ωδ)·v
 *   q_i' = (q_i+x_i)/2 + u'/2
 *   x_i' = (q_i+x_i)/2 - u'/2
 *   p_i' = (p_i+y_i)/2 + v'/2
 *   y_i' = (p_i+y_i)/2 - v'/2
 *
 * For cyclic coordinates (φ) with shared conserved momentum b,
 * the momentum difference is always 0, so the rotation reduces
 * to an independent rotation of (φ-φs) by cos(2ωδ).
 */
__device__ void tao_rotate_coupled(
    double *q, double *p,    /* real position and momentum */
    double *qs, double *ps,  /* shadow position and momentum */
    double c, double s       /* cos(2ωδ) and sin(2ωδ) */
) {
    double sum_q = *q + *qs, diff_q = *q - *qs;
    double sum_p = *p + *ps, diff_p = *p - *ps;
    double new_diff_q = c * diff_q + s * diff_p;
    double new_diff_p = -s * diff_q + c * diff_p;
    *q  = 0.5 * (sum_q + new_diff_q);
    *qs = 0.5 * (sum_q - new_diff_q);
    *p  = 0.5 * (sum_p + new_diff_p);
    *ps = 0.5 * (sum_p - new_diff_p);
}

/* Rotation for cyclic coordinate φ (no conjugate momentum in
 * the extended phase space — b is shared and constant).
 * The momentum difference is 0, so the rotation simplifies to:
 *   φ'  = (φ+φs)/2 + cos(2ωδ)·(φ-φs)/2
 *   φs' = (φ+φs)/2 - cos(2ωδ)·(φ-φs)/2
 */
__device__ void tao_rotate_cyclic(
    double *q, double *qs,   /* real and shadow cyclic coordinate */
    double c                 /* cos(2ωδ) */
) {
    double sum = *q + *qs, diff = *q - *qs;
    *q  = 0.5 * (sum + c * diff);
    *qs = 0.5 * (sum - c * diff);
}


/* ── Tao φ_A flow ─────────────────────────────────────────── */

/* φ_A^τ: Evaluate geoRHS at (q, y) = (r, th, prs, pths).
 *   geoRHS returns: dr,dth,dphi = ∂_y H(q,y)  [velocity from shadow momenta]
 *                   dpr,dpth    = −∂_q H(q,y)  [force from real positions]
 *
 * Per Tao (2016) Eq. 1:
 *   p  → p  − τ·∂_q H(q,y)  =  p  + τ·dpr   (update real momenta)
 *   x  → x  + τ·∂_y H(q,y)  =  x  + τ·dr    (update shadow positions)
 *   q and y are unchanged.
 */
#define _TAO_PHI_A(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, tau) \
    { \
        double dr_, dth_, dphi_, dpr_, dpth_; \
        geoRHS(r, th, prs, pths, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
        /* Update real momenta (p) using −∂_q H(q,y) */ \
        pr   += (tau) * dpr_; \
        pth  += (tau) * dpth_; \
        /* Update shadow positions (x) using ∂_y H(q,y) */ \
        rs   += (tau) * dr_; \
        ths  += (tau) * dth_; \
        phis += (tau) * dphi_; \
    }


/* ── Tao φ_B flow ─────────────────────────────────────────── */

/* φ_B^τ: Evaluate geoRHS at (x, p) = (rs, ths, pr, pth).
 *   geoRHS returns: dr,dth,dphi = ∂_p H(x,p)  [velocity from real momenta]
 *                   dpr,dpth    = −∂_x H(x,p)  [force from shadow positions]
 *
 * Per Tao (2016) Eq. 1:
 *   q  → q  + τ·∂_p H(x,p)  =  q  + τ·dr    (update real positions)
 *   y  → y  − τ·∂_x H(x,p)  =  y  + τ·dpr   (update shadow momenta)
 *   x and p are unchanged.
 */
#define _TAO_PHI_B(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, tau) \
    { \
        double dr_, dth_, dphi_, dpr_, dpth_; \
        geoRHS(rs, ths, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
        /* Update real positions (q) using ∂_p H(x,p) */ \
        r    += (tau) * dr_; \
        th   += (tau) * dth_; \
        phi  += (tau) * dphi_; \
        /* Update shadow momenta (y) using −∂_x H(x,p) */ \
        prs  += (tau) * dpr_; \
        pths += (tau) * dpth_; \
    }


/* ── Tao φ_C flow (harmonic rotation) ────────────────────── */

/* φ_C^δ: Exact time-δ flow of ωH_C.
 *
 * The rotation angle is 2ωδ (Tao 2016, Eq. 1).
 * Since ω = TAO_OMEGA_C / h, the angle is 2·TAO_OMEGA_C·(δ/h).
 *
 * For each conjugate DOF (r↔pr, θ↔pθ), the coupled rotation
 * mixes position differences with momentum differences.
 * For cyclic φ (shared conserved b), only the position rotates.
 */
#define _TAO_PHI_C(r, th, phi, pr, pth, rs, ths, phis, prs, pths, angle2) \
    { \
        double c_ = cos(angle2), s_ = sin(angle2); \
        tao_rotate_coupled(&r,  &pr,  &rs,  &prs,  c_, s_); \
        tao_rotate_coupled(&th, &pth, &ths, &pths, c_, s_); \
        tao_rotate_cyclic(&phi, &phis, c_); \
    }


/* ── Tao 2nd-order Strang base step (Eq. 2) ──────────────── */

/* φ_2^δ = φ_A^{δ/2} ∘ φ_B^{δ/2} ∘ φ_C^δ ∘ φ_B^{δ/2} ∘ φ_A^{δ/2}
 *
 * Cost: 4 geoRHS evaluations + 1 harmonic rotation per base step.
 * The rotation angle for φ_C^δ is 2ωδ (Tao 2016, Eq. 1).
 */
#define _TAO_STRANG(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                    a, b, Q2, delta, angle2) \
    { \
        double half_delta_ = 0.5 * (delta); \
        _TAO_PHI_A(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, half_delta_) \
        _TAO_PHI_B(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, half_delta_) \
        _TAO_PHI_C(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   angle2) \
        _TAO_PHI_B(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, half_delta_) \
        _TAO_PHI_A(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, half_delta_) \
    }


/* ── Tao + Yoshida 4th-order step ─────────────────────────── */

/* Triple-jump composition (Tao 2016, Eq. 3):
 *   φ_4^δ = φ_2^{γδ} ∘ φ_2^{(1-2γ)δ} ∘ φ_2^{γδ}
 * where γ = 1/(2 − 2^{1/3}) ≈ 1.3512.
 *
 * Cost: 3 × (4 geoRHS + 1 rotation) = 12 geoRHS + 3 rotations.
 * Adjacent φ_A terms at composition boundaries can be merged
 * for efficiency, but we keep them separate for clarity.
 */

#define TAO_Y4_GAMMA  1.3512071919596576340
#define TAO_Y4_GAMMA2 (-1.7024143839193152681)  /* 1 - 2γ */

__device__ void tao_yoshida4_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double *rs, double *ths, double *phis,
    double *prs, double *pths,
    double a, double b, double Q2, double he
) {
    /* ω = TAO_OMEGA_C / h.  For each Strang step with duration δ_i,
     * the rotation angle is 2ωδ_i = 2·TAO_OMEGA_C · (δ_i / h).
     * Since δ_i = γ·h or (1-2γ)·h, the angle is 2·TAO_OMEGA_C · γ
     * or 2·TAO_OMEGA_C · (1-2γ). */
    double d1 = TAO_Y4_GAMMA * he;
    double d0 = TAO_Y4_GAMMA2 * he;
    double angle1 = 2.0 * TAO_OMEGA_C * TAO_Y4_GAMMA;
    double angle0 = 2.0 * TAO_OMEGA_C * TAO_Y4_GAMMA2;

    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1, angle1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d0, angle0)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1, angle1)
}


/* ── Yoshida 6th-order coefficients (Solution A) ──────────── */

/* 6th-order via triple-jump of 4th-order:
 *   φ_6^δ = φ_4^{γ₆δ} ∘ φ_4^{(1-2γ₆)δ} ∘ φ_4^{γ₆δ}
 * where γ₆ = 1/(2 − 2^{1/5}).
 *
 * Each φ_4 is itself a triple-jump of φ_2, giving
 * 3 × 3 = 9 Strang base steps total.
 * Cost: 9 × (4 geoRHS + 1 rotation) = 36 geoRHS + 9 rotations.
 */

#define TAO_Y6_GAMMA  1.1746391730891982610   /* 1/(2 − 2^{1/5}) */
#define TAO_Y6_GAMMA2 (-1.3492783461783965220)  /* 1 − 2γ₆ */

__device__ void tao_yoshida6_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double *rs, double *ths, double *phis,
    double *prs, double *pths,
    double a, double b, double Q2, double he
) {
    /* Outer triple-jump: scale h for each φ_4 call */
    double h1 = TAO_Y6_GAMMA * he;
    double h0 = TAO_Y6_GAMMA2 * he;

    /* Inner triple-jump coefficients for φ_4 */
    double d1_1 = TAO_Y4_GAMMA * h1;
    double d0_1 = TAO_Y4_GAMMA2 * h1;
    double d1_0 = TAO_Y4_GAMMA * h0;
    double d0_0 = TAO_Y4_GAMMA2 * h0;

    /* Rotation angles: 2ωδ = 2·TAO_OMEGA_C · (δ/h).
     * For inner step δ = γ₄·h_outer where h_outer = γ₆·h or (1-2γ₆)·h,
     * angle = 2·TAO_OMEGA_C · γ₄ · γ₆  etc. */
    double a1_1 = 2.0 * TAO_OMEGA_C * TAO_Y4_GAMMA * TAO_Y6_GAMMA;
    double a0_1 = 2.0 * TAO_OMEGA_C * TAO_Y4_GAMMA2 * TAO_Y6_GAMMA;
    double a1_0 = 2.0 * TAO_OMEGA_C * TAO_Y4_GAMMA * TAO_Y6_GAMMA2;
    double a0_0 = 2.0 * TAO_OMEGA_C * TAO_Y4_GAMMA2 * TAO_Y6_GAMMA2;

    /* φ_4^{γ₆·h}: 3 Strang steps */
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1_1, a1_1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d0_1, a0_1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1_1, a1_1)

    /* φ_4^{(1-2γ₆)·h}: 3 Strang steps */
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1_0, a1_0)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d0_0, a0_0)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1_0, a1_0)

    /* φ_4^{γ₆·h}: 3 Strang steps */
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1_1, a1_1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d0_1, a0_1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1_1, a1_1)
}


/* ── Kahan-Li s15odr8 optimal 8th-order coefficients ──────── */

/* From: W. Kahan & R.-C. Li, Math. Comp. 66:1089–1099, 1997.
 * max |W_i| = 0.797 (vs 2.447 for Yoshida Solution D).
 *
 * 15-stage palindromic composition of the Strang base step:
 *   φ_8^δ = φ_2^{w₇δ} ∘ φ_2^{w₆δ} ∘ ... ∘ φ_2^{w₀δ} ∘ ... ∘ φ_2^{w₇δ}
 *
 * Cost: 15 × (4 geoRHS + 1 rotation) = 60 geoRHS + 15 rotations.
 */

#ifndef TAO_KL8_W0
#define TAO_KL8_W0  0.74167036435061295345   /* w₇ (outermost) */
#define TAO_KL8_W1 -0.40910082580003159400   /* w₆ */
#define TAO_KL8_W2  0.19075471029623837995   /* w₅ */
#define TAO_KL8_W3 -0.57386247111608226666   /* w₄ */
#define TAO_KL8_W4  0.29906418130365592384   /* w₃ */
#define TAO_KL8_W5  0.33462491824529818378   /* w₂ */
#define TAO_KL8_W6  0.31529309239676659663   /* w₁ */
#define TAO_KL8_W7 -0.79688793935291635402   /* w₀ (center) */
#endif

/* ── Tao + Kahan-Li 8th-order step ────────────────────────── */

/* 15 Strang base steps composed with Kahan-Li s15odr8 optimal
 * 8th-order palindromic coefficients.
 * Cost: 15 × (4 geoRHS + 1 rotation) = 60 geoRHS + 15 rotations. */
__device__ void tao_kahan_li8_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double *rs, double *ths, double *phis,
    double *prs, double *pths,
    double a, double b, double Q2, double he
) {
    /* Precompute step durations and rotation angles for each stage.
     * δ_i = w_i · h,  angle_i = 2ωδ_i = 2·TAO_OMEGA_C · w_i. */
    double d0 = TAO_KL8_W0 * he, a0 = 2.0 * TAO_OMEGA_C * TAO_KL8_W0;
    double d1 = TAO_KL8_W1 * he, a1 = 2.0 * TAO_OMEGA_C * TAO_KL8_W1;
    double d2 = TAO_KL8_W2 * he, a2 = 2.0 * TAO_OMEGA_C * TAO_KL8_W2;
    double d3 = TAO_KL8_W3 * he, a3 = 2.0 * TAO_OMEGA_C * TAO_KL8_W3;
    double d4 = TAO_KL8_W4 * he, a4 = 2.0 * TAO_OMEGA_C * TAO_KL8_W4;
    double d5 = TAO_KL8_W5 * he, a5 = 2.0 * TAO_OMEGA_C * TAO_KL8_W5;
    double d6 = TAO_KL8_W6 * he, a6 = 2.0 * TAO_OMEGA_C * TAO_KL8_W6;
    double d7 = TAO_KL8_W7 * he, a7 = 2.0 * TAO_OMEGA_C * TAO_KL8_W7;

    /* Forward half: stages w₇ through w₁ */
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d0, a0)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1, a1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d2, a2)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d3, a3)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d4, a4)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d5, a5)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d6, a6)

    /* Center stage: w₀ */
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d7, a7)

    /* Reverse half: stages w₁ through w₇ (palindromic) */
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d6, a6)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d5, a5)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d4, a4)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d3, a3)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d2, a2)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1, a1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d0, a0)
}


/* ── RK4 single step ──────────────────────────────────────── */

__device__ void rk4_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double a, double b, double Q2, double he
) {
    double r0 = *r, th0 = *th, phi0 = *phi, pr0 = *pr, pth0 = *pth;
    double dr1, dth1, dphi1, dpr1, dpth1;
    double dr2, dth2, dphi2, dpr2, dpth2;
    double dr3, dth3, dphi3, dpr3, dpth3;
    double dr4, dth4, dphi4, dpr4, dpth4;

    geoRHS(r0, th0, pr0, pth0, a, b, Q2,
           &dr1, &dth1, &dphi1, &dpr1, &dpth1);
    geoRHS(r0 + 0.5*he*dr1, th0 + 0.5*he*dth1,
           pr0 + 0.5*he*dpr1, pth0 + 0.5*he*dpth1, a, b, Q2,
           &dr2, &dth2, &dphi2, &dpr2, &dpth2);
    geoRHS(r0 + 0.5*he*dr2, th0 + 0.5*he*dth2,
           pr0 + 0.5*he*dpr2, pth0 + 0.5*he*dpth2, a, b, Q2,
           &dr3, &dth3, &dphi3, &dpr3, &dpth3);
    geoRHS(r0 + he*dr3, th0 + he*dth3,
           pr0 + he*dpr3, pth0 + he*dpth3, a, b, Q2,
           &dr4, &dth4, &dphi4, &dpr4, &dpth4);

    *r   = r0   + he * (dr1   + 2.0*dr2   + 2.0*dr3   + dr4  ) / 6.0;
    *th  = th0  + he * (dth1  + 2.0*dth2  + 2.0*dth3  + dth4 ) / 6.0;
    *phi = phi0 + he * (dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4) / 6.0;
    *pr  = pr0  + he * (dpr1  + 2.0*dpr2  + 2.0*dpr3  + dpr4 ) / 6.0;
    *pth = pth0 + he * (dpth1 + 2.0*dpth2 + 2.0*dpth3 + dpth4) / 6.0;
}


#endif /* INTEGRATOR_STEPS_CU */
