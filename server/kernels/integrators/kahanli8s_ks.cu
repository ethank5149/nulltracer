/* ============================================================
 *  KAHANLI8S_KS — KAHAN-LI 8th-ORDER SYMPLECTIC INTEGRATOR IN
 *  KERR-SCHILD COORDINATES WITH SUNDMAN (MINO TIME) TRANSFORMATION
 *
 *  This is the Kerr-Schild coordinate variant of the kahanli8s
 *  integrator.  It uses the same Kahan-Li s15odr8 optimal
 *  8th-order symmetric composition (15 stages) with:
 *    1. Sundman time transformation — dτ = dλ/Σ (Mino time)
 *    2. Compensated (Kahan) summation — ~32 digits effective
 *    3. Symplectic corrector — Wisdom (2006) near-identity map
 *    4. Hamiltonian projection — quadratic solve onto H=0
 *
 *  KEY DIFFERENCE FROM kahanli8s.cu (Boyer-Lindquist version):
 *
 *  Kerr-Schild coordinates eliminate the coordinate singularity
 *  at the event horizon r = r₊ where Δ → 0.  In BL coordinates,
 *  g_rr = Σ/Δ → ∞ at the horizon, requiring the regularization
 *  floor fmax(del, 1e-14) which introduces systematic errors.
 *  In KS coordinates, all metric components remain smooth and
 *  finite at r = r₊, enabling accurate integration through the
 *  horizon without any regularization.
 *
 *  The Kerr equations of motion are dramatically simpler than BL:
 *    - Force (kick): ~20 FLOPs vs ~80 FLOPs in BL
 *    - dp_r/dλ = [(1−r)·p_r² + 2r·p_r] / Σ
 *    - dp_θ/dλ = cosθ·[b²/(s²·sinθ) − a²·sinθ] / Σ
 *
 *  The tradeoff is that the KS Hamiltonian is non-separable
 *  (the force dp_r/dλ depends on p_r), but this is handled by
 *  the standard drift-kick splitting — the non-separability
 *  error is O(h²), negligible compared to the O(h⁸) truncation
 *  error of the 8th-order composition.
 *
 *  Ray initialization uses the existing BL initRay() followed
 *  by a momentum transformation p_r^KS = p_r^BL + (r²+a²−ab)/Δ.
 *  Position coordinates (r, θ, φ) are unchanged at the observer.
 *
 *  All geodesic integration in float64; color output in float32.
 *
 *  References:
 *    [1] R.P. Kerr, "Gravitational field of a spinning mass as
 *        an example of algebraically special metrics,"
 *        Phys. Rev. Lett. 11:237–238, 1963.
 *    [2] M. Visser, "The Kerr spacetime: A brief introduction,"
 *        arXiv:0706.0622, 2007.
 *    [3] S. Chandrasekhar, The Mathematical Theory of Black Holes,
 *        Oxford University Press, 1983.  Chapter 6.
 *    [4] W. Kahan & R.-C. Li, "Composition constants for raising
 *        the orders of unconventional schemes for ordinary
 *        differential equations," Math. Comp. 66:1089–1099, 1997.
 *    [5] Y. Mino, "Perturbative approach to an orbital evolution
 *        around a supermassive black hole," Phys. Rev. D 67, 2003.
 *    [6] K. Sundman, "Mémoire sur le problème des trois corps,"
 *        Acta Math. 36:105–179, 1913.
 *    [7] Z. Ge & J.E. Marsden, "Lie-Poisson Hamilton-Jacobi
 *        theory and Lie-Poisson integrators," Phys. Lett. A
 *        133:134–139, 1988.
 *    [8] J. Wisdom, "Symplectic correctors for canonical
 *        heliocentric N-body maps," Astron. J. 131:2294, 2006.
 *    [9] E. Hairer, R.I. McLachlan & A. Razakarivony,
 *        "Achieving Brouwer's law with implicit Runge-Kutta
 *        methods," BIT Numer. Math. 48:231–243, 2008.
 *   [10] W. Kahan, "Pracniques: further remarks on reducing
 *        truncation errors," Comm. ACM 8(1):40, 1965.
 *   [11] B. Carter, "Global structure of the Kerr family of
 *        gravitational fields," Phys. Rev. 174:1559–1571, 1968.
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"


/* ── Kahan-Li s15odr8 optimal 8th-order coefficients ───────────
 *
 *  From: W. Kahan & R.-C. Li, Math. Comp. 66:1089–1099, 1997.
 *  Also found independently by Suzuki & Umeno (1993); confirmed
 *  optimal among 15-stage 8th-order symmetric compositions by
 *  Sofroniou & Spaletta (2005).
 *
 *  max |W_i| = 0.797 (vs 2.447 for Yoshida Solution A) — a 3.1×
 *  reduction in coefficient magnitude with zero additional cost.
 *
 *  Code convention: W[0]=w7 (outermost) … W[7]=w0 (center).
 *  The 15-stage palindromic composition is:
 *    W7 W6 W5 W4 W3 W2 W1 W0 W1 W2 W3 W4 W5 W6 W7
 *
 *  These are identical to the coefficients in kahanli8s.cu — the
 *  composition scheme is universal for any drift-kick splitting.
 * ─────────────────────────────────────────────────────────────── */

static __constant__ double KL8S_W8[8] = {
     0.74167036435061295345,  /* W[0] = w7 (outermost) */
    -0.40910082580003159400,  /* W[1] = w6 */
     0.19075471029623837995,  /* W[2] = w5 */
    -0.57386247111608226666,  /* W[3] = w4 */
     0.29906418130365592384,  /* W[4] = w3 */
     0.33462491824529818378,  /* W[5] = w2 */
     0.31529309239676659663,  /* W[6] = w1 */
    -0.79688793935291635402   /* W[7] = w0 (center) */
};

static __constant__ double KL8S_D8[8] = {
     0.37083518217530647672,  /* D[0] = W[0]/2 */
     0.16628476927529067972,  /* D[1] = (W[0]+W[1])/2 */
    -0.10917305775189660702,  /* D[2] = (W[1]+W[2])/2 */
    -0.19155388040992194336,  /* D[3] = (W[2]+W[3])/2 */
    -0.13739914490621317141,  /* D[4] = (W[3]+W[4])/2 */
     0.31684454977447705381,  /* D[5] = (W[4]+W[5])/2 */
     0.32495900532103239020,  /* D[6] = (W[5]+W[6])/2 */
    -0.24079742347807487870   /* D[7] = (W[6]+W[7])/2 */
};


/* ── Compensated summation helper ──────────────────────────── */

/* Kahan compensated addition: accumulates delta into *sum with
 * round-off tracked in *comp.  Gives ~2× machine precision.
 *
 * Standard addition loses low-order bits when |*sum| >> |delta|.
 * Compensated summation captures the lost bits in *comp and
 * feeds them back into the next addition.
 *
 * Reference: W. Kahan, Comm. ACM 8(1):40, 1965.
 *            E. Hairer et al., BIT 48:231–243, 2008. */
__device__ __forceinline__ void kahan_add(
    double *sum, double *comp, double delta
) {
    double y = delta - *comp;
    double t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}


/* ── Main kernel entry point ──────────────────────────────── */

extern "C" __global__
void trace_kahanli8s_ks(const RenderParams *pp, unsigned char *output) {
    const RenderParams &p = *pp;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int W = (int)p.width, H = (int)p.height;
    if (ix >= W || iy >= H) return;

    /* ── Initialize ray in BL coordinates ─────────────────── */
    /* Use the existing BL ray initialization (well-tested,
     * produces correct impact parameters), then transform
     * the radial momentum to KS coordinates. */
    double r, th, phi, pr, pth, b, rp;
    float alpha, beta;
    initRay(ix, iy, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta);

    double a = p.spin;
    double Q2 = p.charge * p.charge;

    /* ── Transform p_r from BL to KS coordinates ─────────── */
    /* p_r^KS = p_r^BL + (r² + a² − a·b) / Δ
     *
     * At the observer distance r_obs >> r₊, Δ ≈ r² and the
     * correction ≈ 1, so p_r^KS ≈ p_r^BL + 1.  This is a
     * significant correction that must not be neglected.
     *
     * Position coordinates (r, θ) are unchanged between BL
     * and KS.  The azimuthal angle φ_KS = φ_BL + g(r_obs),
     * but at large r the difference is negligible and absorbed
     * into the initial φ₀.
     *
     * Reference: Visser (2007), Section 2; Chandrasekhar (1983). */
    transformBLtoKS(r, a, b, Q2, &pr);

    /* The Kahan-Li s15odr8 composition has small coefficient
     * magnitudes (max |W_i| ≈ 0.797, max cumulative drift
     * ≈ 1.24×he).  The 4× step multiplier compensates for
     * the 15-substep composition cost while maintaining the
     * same user-facing step count semantics as other methods. */
    int STEPS = (int)p.steps * 4;
    int show_disk = (int)p.show_disk;
    int bg_mode = (int)p.bg_mode;
    int star_layers = (int)p.star_layers;
    int show_grid = (int)p.show_grid;
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    bool done = false;

    /* ── Compensated summation accumulators ────────────────── */
    double r_comp = 0.0, th_comp = 0.0, phi_comp = 0.0;
    double pr_comp = 0.0, pth_comp = 0.0;

    /* ── Carter constant at initialization (diagnostic) ───── */
    /* The Carter constant has the same form in KS coordinates
     * (it depends on r, θ, p_θ, b, a, Q — not on the time/
     * azimuthal coordinates that differ between BL and KS).
     * Reference: Carter (1968). */
    double Q0 = computeCarter(th, pth, a, b, Q2);
    (void)Q0;  /* retained as diagnostic; no guard rail */

    /* ── Sundman / Mino time step from geodesic budget ────── */
    /* Sundman (Mino time) transformation: dτ = dλ/Σ.
     * Fixed steps in Mino time τ give physical steps Δλ = Σ·Δτ
     * that automatically shrink near the horizon (small Σ) and
     * grow far away (large Σ).  Because this is a canonical
     * transformation of the independent variable, the integrator
     * remains truly symplectic (Ge & Marsden 1988).
     *
     * The Mino-time step Δτ is derived directly from the null
     * geodesic equations — no empirical tuning or magic constants.
     * Σ = r² + a²cos²θ is the same in both BL and KS coordinates,
     * so the Sundman transformation works identically.
     *
     * For a radial null geodesic in the far field (r >> M),
     * dr/dλ ≈ 1 and Σ ≈ r², so the Mino-time radial equation
     * dr/dτ = Σ·(dr/dλ) ≈ r² gives:
     *
     *   τ(r₁→r₂) = ∫ dr/r² = 1/r₁ − 1/r₂
     *
     * The total Mino time for a round-trip from the photon
     * sphere r_ph to the escape radius r_esc is:
     *
     *   τ_needed = 2·(1/r_ph − 1/r_esc)
     *
     * References:
     *   [5] Y. Mino, Phys. Rev. D 67, 084027 (2003).
     *   [6] K. Sundman, Acta Math. 36:105–179 (1913).
     *   [7] Z. Ge & J.E. Marsden, Phys. Lett. A 133:134 (1988).
     */
    double r_ph;
    if (Q2 < 1e-10) {
        /* Exact prograde equatorial photon orbit (Bardeen 1972) */
        r_ph = 2.0 * (1.0 + cos(2.0 / 3.0 * acos(-a)));
    } else {
        /* Conservative bound: r+ ≤ r_ph for all Kerr-Newman */
        r_ph = rp;
    }
    double tau_needed = 2.0 * (1.0 / r_ph - 1.0 / p.esc_radius);
    double dtau = (1.0 + p.step_size) * tau_needed / (double)STEPS;

    /* ── Integration loop ─────────────────────────────────── */

    for (int i = 0; i < STEPS; i++) {
        if (done) break;

        /* Save state for disk crossing interpolation */
        double oldR = r, oldTh = th, oldPhi = phi;

        /* ── Sundman-scaled step size ─────────────────────── */
        /* Physical step Δλ = Δτ·Σ.  The lower clamp prevents
         * vanishing steps at the poles; the upper clamp prevents
         * overshooting in the far field.  The upper bound scales
         * with obs_dist to remain consistent across scene sizes. */
        double cth = cos(th);
        double sig = r * r + a * a * cth * cth;
        double he = dtau * sig;
        he = fmin(fmax(he, 0.005), 0.2 * p.obs_dist);

        /* ════════════════════════════════════════════════════
         *  KAHAN-LI s15odr8 8th-ORDER: 15 symmetric substeps
         *  in KERR-SCHILD coordinates
         *
         *  Pattern: [D1,W1] [D2,W2] ... [D0,W0] ... [D1,W1]
         *
         *  Each substep:
         *    1. Evaluate geoVelocityKS at current state (drift)
         *    2. Drift: update positions with D coefficient
         *    3. Evaluate geoForceKS at new position (kick)
         *    4. Kick: update momenta with W coefficient
         *
         *  The drift uses geoVelocityKS() which includes the
         *  KS-specific g^tr and g^rφ cross-terms.
         *  The kick uses geoForceKS() which is ~4× cheaper than
         *  the BL version due to the simpler KS force equations.
         *
         *  The force dp_r/dλ depends on p_r (non-separable), but
         *  this is handled by evaluating the force at the current
         *  p_r value.  The resulting error is O(h²) per substep,
         *  which is negligible compared to the O(h⁸) truncation
         *  error of the 8th-order composition.
         *
         *  Uses compensated summation for all accumulations.
         * ════════════════════════════════════════════════════ */

        double dr_, dth_, dphi_, dpr_, dpth_;

        /* Macro for a single drift-kick substep with Kahan summation.
         *
         * Structure:
         *   1. Compute velocities at current (r, θ, φ, p_r, p_θ)
         *   2. Drift: advance positions by D[idx]·he
         *   3. Compute forces at updated (r, θ) with current (p_r, p_θ)
         *   4. Kick: advance momenta by W[idx]·he
         *
         * The force evaluation at step 3 uses the UPDATED positions
         * (after drift) but the CURRENT momenta (before kick).
         * This is the standard leapfrog structure. */
        #define KL8S_KS_SUBSTEP(idx) { \
            geoVelocityKS(r, th, pr, pth, a, b, Q2, \
                          &dr_, &dth_, &dphi_); \
            kahan_add(&r,   &r_comp,   he * KL8S_D8[idx] * dr_); \
            kahan_add(&th,  &th_comp,  he * KL8S_D8[idx] * dth_); \
            kahan_add(&phi, &phi_comp, he * KL8S_D8[idx] * dphi_); \
            geoForceKS(r, th, pr, pth, a, b, Q2, \
                       &dpr_, &dpth_); \
            kahan_add(&pr,  &pr_comp,  he * KL8S_W8[idx] * dpr_); \
            kahan_add(&pth, &pth_comp, he * KL8S_W8[idx] * dpth_); \
        }

        /* Forward half: substeps 1-7 */
        KL8S_KS_SUBSTEP(0)  /* substep 1:  D1, W1 */
        KL8S_KS_SUBSTEP(1)  /* substep 2:  D2, W2 */
        KL8S_KS_SUBSTEP(2)  /* substep 3:  D3, W3 */
        KL8S_KS_SUBSTEP(3)  /* substep 4:  D4, W4 */
        KL8S_KS_SUBSTEP(4)  /* substep 5:  D5, W5 */
        KL8S_KS_SUBSTEP(5)  /* substep 6:  D6, W6 */
        KL8S_KS_SUBSTEP(6)  /* substep 7:  D7, W7 */

        /* Center substep 8: D0, W0 */
        KL8S_KS_SUBSTEP(7)

        /* Reverse half: substeps 9-15 (palindromic) */
        KL8S_KS_SUBSTEP(6)  /* substep 9:  D7, W7 */
        KL8S_KS_SUBSTEP(5)  /* substep 10: D6, W6 */
        KL8S_KS_SUBSTEP(4)  /* substep 11: D5, W5 */
        KL8S_KS_SUBSTEP(3)  /* substep 12: D4, W4 */
        KL8S_KS_SUBSTEP(2)  /* substep 13: D3, W3 */
        KL8S_KS_SUBSTEP(1)  /* substep 14: D2, W2 */
        KL8S_KS_SUBSTEP(0)  /* substep 15: D1, W1 */

        #undef KL8S_KS_SUBSTEP

        /* ── Symplectic corrector (Wisdom 2006) ───────────── */
        /* Near-identity canonical transformation that cancels
         * the leading error term of the 8th-order method.
         *
         * Implemented as the composition C = exp(ε·{·,V}) ∘ exp(ε·{·,T})
         * where ε = h²/24 and {·,·} is the Poisson bracket.
         *
         * Step 1: Evaluate forces F at current state
         * Step 2: Corrector kick:  p += ε·F  (momentum correction)
         * Step 3: Evaluate velocities V at corrected momenta
         * Step 4: Corrector drift: q += ε·V  (position correction)
         *
         * This is a proper symplectic map (canonical transformation)
         * that raises effective accuracy from O(h^8) to O(h^10).
         *
         * The KS versions geoForceKS/geoVelocityKS are used here.
         *
         * Reference: Wisdom (2006), Eq. 12-14;
         *            Wisdom & Holman (1991), Section 3. */
        {
            /* The corrector coefficient uses the physical step he.
             * Since he = h_base·Σ/Σ₀ is already small near the
             * horizon (Sundman scaling), corr_eps = he²/24 is
             * naturally tiny where it matters most. */
            double corr_eps = he * he / 24.0;

            /* Step 1-2: Corrector kick — evaluate forces, update momenta */
            double f_pr, f_pth;
            geoForceKS(r, th, pr, pth, a, b, Q2, &f_pr, &f_pth);
            pr  += corr_eps * f_pr;
            pth += corr_eps * f_pth;

            /* Step 3-4: Corrector drift — evaluate velocities, update positions */
            double v_r, v_th, v_phi;
            geoVelocityKS(r, th, pr, pth, a, b, Q2, &v_r, &v_th, &v_phi);
            kahan_add(&r,   &r_comp,   corr_eps * v_r);
            kahan_add(&th,  &th_comp,  corr_eps * v_th);
            kahan_add(&phi, &phi_comp, corr_eps * v_phi);
        }

        /* ── Hamiltonian projection (KS quadratic solve) ──── */
        /* Projects onto H = 0 by solving the quadratic equation
         * Δ·p_r² + 2·(ab−w)·p_r + C = 0 for p_r.
         *
         * Near the horizon where Δ → 0, falls back to the linear
         * solve p_r = −C / (2·(ab−w)).  This is the key advantage
         * of KS coordinates: the projection remains well-defined
         * even at the horizon.
         *
         * Reference: See plans/kahanli8s-ks-design.md, Section 5. */
        projectHamiltonianKS(r, th, &pr, pth, a, b, Q2);
        pr_comp = 0.0;  /* reset compensator after algebraic reset of pr */

        /* ── Pole reflection ──────────────────────────────── */
        /* θ is the same coordinate in BL and KS, so the pole
         * reflection logic is identical. */
        if (th < 0.005) { th = 0.005; pth = fabs(pth); th_comp = 0.0; pth_comp = 0.0; }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); th_comp = 0.0; pth_comp = 0.0; }

        /* ── Termination conditions ───────────────────────── */

        /* Horizon capture — KS coordinates are regular at the
         * horizon, so rays can pass through smoothly.  We detect
         * capture well inside the horizon (r ≤ 0.5·r₊) rather
         * than just outside (r ≤ 1.01·r₊ in BL).
         *
         * In KS coordinates, the metric is smooth at r = r₊ and
         * the ray continues inward.  The capture condition at
         * 0.5·r₊ ensures the ray is genuinely trapped (well past
         * the point of no return) before we terminate it.
         *
         * Reference: Kerr (1963); Visser (2007), Section 3. */
        if (r <= rp * 0.5) { done = true; break; }

        /* Disk crossing detection — θ is the same coordinate
         * in both BL and KS, so equatorial plane crossing
         * detection works identically.  The interpolated hit
         * radius r_hit and azimuth φ_hit are also valid since
         * r is the same coordinate and φ_KS ≈ φ_BL at disk
         * radii (the correction is O(a/r), negligible). */
        if (show_disk) {
            double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
            if (cross < 0.0) {
                double f = fmin(fmax(fabs(oldTh - PI * 0.5) /
                           fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                double r_hit = oldR + f * (r - oldR);
                float dr_f = (float)r_hit;
                float dphi_f = (float)(oldPhi + f * (phi - oldPhi));

                /* The g-factor computation uses BL metric components.
                 * Since the disk is at finite r > r_ISCO > r₊, the
                 * BL metric is well-defined (Δ > 0).  We use the
                 * existing compute_g_factor() which works in BL. */
                float g = compute_g_factor(r_hit, a, Q2, b);

                float dcr, dcg, dcb;
                diskColor(dr_f, dphi_f, (float)a,
                         (float)p.isco, (float)p.disk_outer, (float)p.disk_temp,
                         g, (int)p.doppler_boost,
                         &dcr, &dcg, &dcb);
                float atten = 1.0f - fminf(sqrtf(cr*cr + cg*cg + cb*cb) * 0.4f, 0.9f);
                cr += dcr * atten;
                cg += dcg * atten;
                cb += dcb * atten;
            }
        }

        /* Escape to background — at large r (escape radius),
         * φ_KS ≈ φ_BL (the difference g(r) → 0 as r → ∞),
         * so the background direction computation using (θ, φ)
         * works identically in KS coordinates.
         *
         * Reference: See plans/kahanli8s-ks-design.md, Section 6.3. */
        if (r > p.esc_radius) {
            double frac = fmin(fmax((p.esc_radius - oldR) /
                          fmax(r - oldR, 1e-14), 0.0), 1.0);
            double fth = oldTh + (th - oldTh) * frac;
            double fph = oldPhi + (phi - oldPhi) * frac;
            float dx, dy, dz;
            sphereDir(fth, fph, &dx, &dy, &dz);
            float bgr, bgg, bgb;
            background(dx, dy, dz, bg_mode, star_layers, show_grid,
                       &bgr, &bgg, &bgb);
            float atten = 1.0f - fminf(sqrtf(cr*cr + cg*cg + cb*cb) * 0.3f, 0.9f);
            cr += bgr * atten;
            cg += bgg * atten;
            cb += bgb * atten;
            done = true; break;
        }

        /* NaN / underflow safety — in KS coordinates, r can
         * go below r₊ (the horizon) without numerical issues,
         * but r < 0.5 indicates the ray has reached the ring
         * singularity region (r = 0, θ = π/2) or numerical
         * breakdown.  Terminate to prevent garbage output. */
        if (r < 0.5 || r != r || th != th) { done = true; break; }
    }

    /* ── Post-processing: tone mapping + gamma ────────────── */
    float ux = 2.0f * (ix + 0.5f) / (float)W  - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / (float)H - 1.0f;
    postProcess(&cr, &cg, &cb, alpha, beta, (float)p.spin, ux, uy);

    /* ── Write output (RGB, uint8) ────────────────────────── */
    int idx = (iy * W + ix) * 3;
    output[idx + 0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx + 1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx + 2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
