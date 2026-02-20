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


/* ── Yoshida 4th-order (Forest-Ruth) coefficients ──────────── */

#ifndef Y4_W1
#define Y4_W1  1.3512071919596576
#define Y4_W0 -1.7024143839193153
#define Y4_D1  0.6756035959798288
#define Y4_D0 -0.1756035959798288
#endif

/* Yoshida 4th-order single step.
 * 3 symmetric drift-kick substeps. */
__device__ void yoshida4_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double a, double b, double Q2, double he
) {
    double dr_, dth_, dphi_, dpr_, dpth_;

    /* Substep 1: drift d1, kick w1 */
    geoRHS(*r, *th, *pr, *pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
    *r   += he * Y4_D1 * dr_;
    *th  += he * Y4_D1 * dth_;
    *phi += he * Y4_D1 * dphi_;
    geoRHS(*r, *th, *pr, *pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
    *pr  += he * Y4_W1 * dpr_;
    *pth += he * Y4_W1 * dpth_;

    /* Substep 2: drift d0, kick w0 */
    geoRHS(*r, *th, *pr, *pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
    *r   += he * Y4_D0 * dr_;
    *th  += he * Y4_D0 * dth_;
    *phi += he * Y4_D0 * dphi_;
    geoRHS(*r, *th, *pr, *pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
    *pr  += he * Y4_W0 * dpr_;
    *pth += he * Y4_W0 * dpth_;

    /* Substep 3: drift d1, kick w1 (symmetric) */
    geoRHS(*r, *th, *pr, *pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
    *r   += he * Y4_D1 * dr_;
    *th  += he * Y4_D1 * dth_;
    *phi += he * Y4_D1 * dphi_;
    geoRHS(*r, *th, *pr, *pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
    *pr  += he * Y4_W1 * dpr_;
    *pth += he * Y4_W1 * dpth_;
}


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
 *  (q, p, q̃, p̃) with the extended Hamiltonian:
 *    H_ext = H(q, p̃) + H(q̃, p) + ½ω²(|q-q̃|² + |p-p̃|²)
 *
 *  The three flows are:
 *    φ_A: advance (q, p̃) using H(q, p̃)  [real pos, shadow mom]
 *    φ_B: advance (q̃, p) using H(q̃, p)  [shadow pos, real mom]
 *    φ_C: harmonic rotation coupling q↔q̃ and p↔p̃
 *
 *  Each φ_A/φ_B is a single Euler-like step of the geodesic
 *  equations.  The Yoshida/Kahan-Li composition of the Strang
 *  splitting φ_C ∘ φ_A ∘ φ_B ∘ φ_A ∘ φ_C achieves the full
 *  nominal order (4th/6th/8th) on the non-separable system.
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

/* Exact solution of the harmonic coupling:
 *   q_new  =  q·cos(ωτ) + q̃·sin(ωτ)
 *   q̃_new = -q·sin(ωτ) + q̃·cos(ωτ)
 * Same for (p, p̃).
 *
 * For a Yoshida substep with coefficient W_i, the rotation
 * angle is ωτ = (TAO_OMEGA_C / h) · (W_i · h) = TAO_OMEGA_C · W_i.
 * Note: the step size h cancels out! The rotation angle depends
 * only on the Yoshida coefficient and the coupling constant.
 */
__device__ void tao_rotate(
    double *q, double *qs,   /* real and shadow variable */
    double c, double s       /* cos(ωτ) and sin(ωτ) */
) {
    double q0 = *q, qs0 = *qs;
    *q  =  c * q0 + s * qs0;
    *qs = -s * q0 + c * qs0;
}


/* ── Tao substep macro ────────────────────────────────────── */

/* A single Tao-Yoshida substep with drift coefficient D and
 * kick coefficient W:
 *
 *   1. φ_A(D·h): Evaluate geoRHS at (r, th, prs, pths) — real
 *      positions with shadow momenta.  Advance real positions
 *      and shadow momenta.
 *
 *   2. φ_B(D·h): Evaluate geoRHS at (rs, ths, pr, pth) — shadow
 *      positions with real momenta.  Advance shadow positions
 *      and real momenta.
 *
 *   3. φ_C(W·h): Harmonic rotation coupling real↔shadow.
 *      Angle = TAO_OMEGA_C · W (h cancels).
 */
#define _TAO_SUBSTEP(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                     a, b, Q2, he, D_COEFF, W_COEFF) \
    { \
        double dr_, dth_, dphi_, dpr_, dpth_; \
        double tau_d = he * (D_COEFF); \
        \
        /* φ_A: advance real positions (r,th,phi) and shadow momenta (prs,pths) */ \
        geoRHS(r, th, prs, pths, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
        r    += tau_d * dr_; \
        th   += tau_d * dth_; \
        phi  += tau_d * dphi_; \
        prs  += tau_d * dpr_; \
        pths += tau_d * dpth_; \
        \
        /* φ_B: advance shadow positions (rs,ths,phis) and real momenta (pr,pth) */ \
        geoRHS(rs, ths, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
        rs   += tau_d * dr_; \
        ths  += tau_d * dth_; \
        phis += tau_d * dphi_; \
        pr   += tau_d * dpr_; \
        pth  += tau_d * dpth_; \
        \
        /* φ_C: harmonic rotation with angle = TAO_OMEGA_C * W_COEFF */ \
        double angle_ = TAO_OMEGA_C * (W_COEFF); \
        double c_ = cos(angle_), s_ = sin(angle_); \
        tao_rotate(&r,   &rs,   c_, s_); \
        tao_rotate(&th,  &ths,  c_, s_); \
        tao_rotate(&phi, &phis, c_, s_); \
        tao_rotate(&pr,  &prs,  c_, s_); \
        tao_rotate(&pth, &pths, c_, s_); \
    }


/* ── Tao + Yoshida 4th-order step ─────────────────────────── */

/* 3 symmetric substeps using Forest-Ruth/Yoshida 4th-order
 * coefficients with Tao extended phase space splitting.
 * Cost: 6 geoRHS evaluations + 3 harmonic rotations per step. */
__device__ void tao_yoshida4_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double *rs, double *ths, double *phis,
    double *prs, double *pths,
    double a, double b, double Q2, double he
) {
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y4_D1, Y4_W1)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y4_D0, Y4_W0)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y4_D1, Y4_W1)
}


/* ── Yoshida 6th-order coefficients (Solution A) ──────────── */

#ifndef Y6_W1
#define Y6_W1  0.78451361047755726382
#define Y6_W2  0.23557321335935813368
#define Y6_W3 -1.17767998417887100695
#define Y6_W0  1.31518632068391121889
#define Y6_D1  0.39225680523877863191
#define Y6_D2  0.51004341191845769508
#define Y6_D3 -0.47105338540975643969
#define Y6_D0  0.06875316825252012625
#endif

/* ── Tao + Yoshida 6th-order step ─────────────────────────── */

/* 7 symmetric substeps using Yoshida Solution A 6th-order
 * coefficients with Tao extended phase space splitting.
 * Cost: 14 geoRHS evaluations + 7 harmonic rotations per step. */
__device__ void tao_yoshida6_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double *rs, double *ths, double *phis,
    double *prs, double *pths,
    double a, double b, double Q2, double he
) {
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y6_D1, Y6_W1)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y6_D2, Y6_W2)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y6_D3, Y6_W3)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y6_D0, Y6_W0)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y6_D3, Y6_W3)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y6_D2, Y6_W2)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, Y6_D1, Y6_W1)
}


/* ── Kahan-Li s15odr8 optimal 8th-order coefficients ──────── */

/* From: W. Kahan & R.-C. Li, Math. Comp. 66:1089–1099, 1997.
 * max |W_i| = 0.797 (vs 2.447 for Yoshida Solution D).
 *
 * 15-stage palindromic composition:
 *   W7 W6 W5 W4 W3 W2 W1 W0 W1 W2 W3 W4 W5 W6 W7
 *
 * Array convention: index 0 = outermost (w7), index 7 = center (w0). */

#ifndef TAO_KL8_W0
#define TAO_KL8_W0  0.74167036435061295345   /* w7 (outermost) */
#define TAO_KL8_W1 -0.40910082580003159400   /* w6 */
#define TAO_KL8_W2  0.19075471029623837995   /* w5 */
#define TAO_KL8_W3 -0.57386247111608226666   /* w4 */
#define TAO_KL8_W4  0.29906418130365592384   /* w3 */
#define TAO_KL8_W5  0.33462491824529818378   /* w2 */
#define TAO_KL8_W6  0.31529309239676659663   /* w1 */
#define TAO_KL8_W7 -0.79688793935291635402   /* w0 (center) */

#define TAO_KL8_D0  0.37083518217530647672   /* W0/2 */
#define TAO_KL8_D1  0.16628476927529067972   /* (W0+W1)/2 */
#define TAO_KL8_D2 -0.10917305775189660702   /* (W1+W2)/2 */
#define TAO_KL8_D3 -0.19155388040992194336   /* (W2+W3)/2 */
#define TAO_KL8_D4 -0.13739914490621317141   /* (W3+W4)/2 */
#define TAO_KL8_D5  0.31684454977447705381   /* (W4+W5)/2 */
#define TAO_KL8_D6  0.32495900532103239020   /* (W5+W6)/2 */
#define TAO_KL8_D7 -0.24079742347807487870   /* (W6+W7)/2 */
#endif

/* ── Tao + Kahan-Li 8th-order step ────────────────────────── */

/* 15 symmetric substeps using Kahan-Li s15odr8 optimal 8th-order
 * coefficients with Tao extended phase space splitting.
 * Cost: 30 geoRHS evaluations + 15 harmonic rotations per step. */
__device__ void tao_kahan_li8_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double *rs, double *ths, double *phis,
    double *prs, double *pths,
    double a, double b, double Q2, double he
) {
    /* Forward half: substeps 1-7 */
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D0, TAO_KL8_W0)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D1, TAO_KL8_W1)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D2, TAO_KL8_W2)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D3, TAO_KL8_W3)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D4, TAO_KL8_W4)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D5, TAO_KL8_W5)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D6, TAO_KL8_W6)

    /* Center substep 8 */
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D7, TAO_KL8_W7)

    /* Reverse half: substeps 9-15 (palindromic) */
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D6, TAO_KL8_W6)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D5, TAO_KL8_W5)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D4, TAO_KL8_W4)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D3, TAO_KL8_W3)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D2, TAO_KL8_W2)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D1, TAO_KL8_W1)
    _TAO_SUBSTEP(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                 a, b, Q2, he, TAO_KL8_D0, TAO_KL8_W0)
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
