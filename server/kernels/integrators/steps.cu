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

/* Macro for a single Yoshida drift-kick substep (used by Y6 and Y8) */
#define _YOSHIDA_SUBSTEP_IMPL(r, th, phi, pr, pth, a, b, Q2, he, D_COEFF, W_COEFF) \
    { \
        double dr_, dth_, dphi_, dpr_, dpth_; \
        geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
        r   += he * (D_COEFF) * dr_;   \
        th  += he * (D_COEFF) * dth_;  \
        phi += he * (D_COEFF) * dphi_; \
        geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
        pr  += he * (W_COEFF) * dpr_;  \
        pth += he * (W_COEFF) * dpth_; \
    }

/* Yoshida 6th-order single step.
 * 7 symmetric drift-kick substeps. */
__device__ void yoshida6_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double a, double b, double Q2, double he
) {
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y6_D1, Y6_W1)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y6_D2, Y6_W2)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y6_D3, Y6_W3)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y6_D0, Y6_W0)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y6_D3, Y6_W3)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y6_D2, Y6_W2)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y6_D1, Y6_W1)
}


/* ── Yoshida 8th-order coefficients (Solution D) ──────────── */

#ifndef Y8_W1
#define Y8_W1  1.04242620869991
#define Y8_W2  1.82020630970714
#define Y8_W3  0.157739928123617
#define Y8_W4  2.44002732616735
#define Y8_W5 -0.00716989419708120
#define Y8_W6 -2.44699182370524
#define Y8_W7 -1.61582374150097
#define Y8_W0 -1.7808286265894516

#define Y8_D1  0.52121310434996
#define Y8_D2  1.43131625920353
#define Y8_D3  0.98897311891538
#define Y8_D4  1.29888362714548
#define Y8_D5  1.21642871598513
#define Y8_D6 -1.22708085895116
#define Y8_D7 -2.03140778260311
#define Y8_D0 -1.69832618454521
#endif

/* Yoshida 8th-order single step.
 * 15 symmetric drift-kick substeps. */
__device__ void yoshida8_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double a, double b, double Q2, double he
) {
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D1, Y8_W1)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D2, Y8_W2)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D3, Y8_W3)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D4, Y8_W4)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D5, Y8_W5)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D6, Y8_W6)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D7, Y8_W7)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D0, Y8_W0)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D7, Y8_W7)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D6, Y8_W6)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D5, Y8_W5)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D4, Y8_W4)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D3, Y8_W3)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D2, Y8_W2)
    _YOSHIDA_SUBSTEP_IMPL(*r, *th, *phi, *pr, *pth, a, b, Q2, he, Y8_D1, Y8_W1)
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
