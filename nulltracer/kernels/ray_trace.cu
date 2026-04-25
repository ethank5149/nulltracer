/* ============================================================
 *  RAY_TRACE ??? Single-ray tracing kernel for the /ray endpoint
 *
 *  Traces a single photon geodesic through Kerr-Newman spacetime
 *  and records the full trajectory, equatorial plane crossings,
 *  and disk physics (g-factor, Novikov-Thorne flux, temperature).
 *
 *  Provides per-integrator entry points for all 3 methods,
 *  sharing step functions from integrators/steps.cu and
 *  adaptive step sizing from integrators/adaptive_step.cu.
 *
 *  Output buffer layout (all double, 64-bit):
 *    [0..7]     Initial state: r, th, phi, pr, pth, b, alpha, beta
 *    [8..12]    Final state: r, th, phi, pr, pth
 *    [13]       Termination reason (0=steps_exhausted, 1=horizon,
 *               2=escape, 3=nan, 4=underflow)
 *    [14]       Steps used
 *    [15]       Event horizon radius r+
 *    [16]       ISCO radius
 *    [17..19]   Reserved
 *
 *    Trajectory block (starts at offset 20):
 *      For each step i (0 <= i < steps_used):
 *        [20 + i*4 + 0] = r
 *        [20 + i*4 + 1] = theta
 *        [20 + i*4 + 2] = phi
 *        [20 + i*4 + 3] = step_size (he)
 *
 *    Disk crossings block (starts at offset 20 + max_traj*4):
 *      [crossing_base + 0] = number of crossings
 *      For each crossing j (0 <= j < num_crossings, max 16):
 *        [crossing_base + 1 + j*8 + 0] = step index
 *        [crossing_base + 1 + j*8 + 1] = r at crossing
 *        [crossing_base + 1 + j*8 + 2] = phi at crossing
 *        [crossing_base + 1 + j*8 + 3] = direction (1.0=N->S, -1.0=S->N)
 *        [crossing_base + 1 + j*8 + 4] = g-factor
 *        [crossing_base + 1 + j*8 + 5] = Novikov-Thorne flux (normalized)
 *        [crossing_base + 1 + j*8 + 6] = T_emit (K)
 *        [crossing_base + 1 + j*8 + 7] = T_observed (K)
 *
 *  Input parameters are passed via the standard RenderParams struct.
 *  The pixel coordinates (ix, iy) OR direct impact parameters (alpha, beta)
 *  are passed via the first 4 doubles of the output buffer:
 *    output[0] = mode (0=pixel, 1=impact_parameter)
 *    output[1] = ix or alpha
 *    output[2] = iy or beta
 *    output[3] = max_trajectory_points
 * ============================================================ */

#include "geodesic_base.cu"
#include "cks_metric.cu"
#include "disk.cu"
#include "integrators/steps.cu"
#include "integrators/adaptive_step.cu"

/* -- Maximum disk crossings to record ----------------------- */
#define MAX_CROSSINGS 16


/* -- Pole reflection helper -------------------------------- */

__device__ __forceinline__ void pole_reflect(double *th, double *pth, double *phi) {
    if (*th < 0.0) {
        *th = -*th;
        *pth = -*pth;
        *phi += PI;
    } else if (*th > PI) {
        *th = 2.0 * PI - *th;
        *pth = -*pth;
        *phi += PI;
    }
}

/* -- Novikov-Thorne flux with peak normalization ------------ */

/* Compute normalized Novikov-Thorne flux at radius r.
 * Returns F/F_max where F_max is sampled over the disk. */
__device__ double normalized_nt_flux(double r, double a, double r_isco) {
    float F = novikov_thorne_flux(r, a, r_isco);

    /* Find approximate peak flux by sampling */
    float F_max = 0.0f;
    for (int i = 1; i <= 20; i++) {
        float r_sample = (float)r_isco * (1.0f + 0.5f * (float)i);
        float F_sample = novikov_thorne_flux((double)r_sample, a, r_isco);
        if (F_sample > F_max) F_max = F_sample;
    }
    F_max = fmaxf(F_max, 1e-10f);

    return (double)fminf(F / F_max, 1.0f);
}


/* -- Kahan-Li s15odr8 coefficients (for tkl108 ray trace) -- */

static __constant__ double RT_KL8S_W8[8] = {
     0.74167036435061295345,
    -0.40910082580003159400,
     0.19075471029623837995,
    -0.57386247111608226666,
     0.29906418130365592384,
     0.33462491824529818378,
     0.31529309239676659663,
    -0.79688793935291635402
};

static __constant__ double RT_KL8S_D8[8] = {
     0.37083518217530647672,
     0.16628476927529067972,
    -0.10917305775189660702,
    -0.19155388040992194336,
    -0.13739914490621317141,
     0.31684454977447705381,
     0.32495900532103239020,
    -0.24079742347807487870
};

/* Kahan compensated addition */
__device__ __forceinline__ void rt_kahan_add(
    double *sum, double *comp, double delta
) {
    double y = delta - *comp;
    double t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}


/* -- Common ray initialization ------------------------------ */

__device__ bool ray_init(
    const RenderParams &p, double *output,
    double *r, double *th, double *phi, double *pr, double *pth,
    double *b, double *rp, double *alpha_val, double *beta_val,
    int *max_traj
) {
    int mode = (int)output[0];
    double input1 = output[1];
    double input2 = output[2];
    *max_traj = (int)output[3];

    double a = p.spin, a2 = a * a;
    double Q2 = p.charge * p.charge;

    if (mode == 0) {
        /* Pixel mode: use initRay from geodesic_base.cu */
        int ix = (int)input1, iy = (int)input2;
        float alpha_f, beta_f;
        if (!initRay(ix, iy, p, r, th, phi, pr, pth, b, rp, &alpha_f, &beta_f)) {
            return false;
        }
        *alpha_val = (double)alpha_f;
        *beta_val = (double)beta_f;
    } else {
        /* Impact parameter mode: direct (alpha, beta) specification */
        *alpha_val = input1;
        *beta_val = input2;

        double sO = sin(p.incl);
        *b = -(*alpha_val) * sO;

        *r = p.obs_dist;
        *th = p.incl;
        *phi = p.phi0;

        /* Compute initial p_r from null condition H = 0 */
        double sth = sin(p.incl), cth = cos(p.incl);
        double s2 = sth * sth + S2_EPS;
        double c2 = cth * cth;
        double r0 = p.obs_dist, r02 = r0 * r0;
        double sig = r02 + a2 * c2;
        double del = r02 - 2.0 * r0 + a2 + Q2;
        double sdel = fmax(del, 1e-14);
        double rpa2 = r02 + a2;
        double A_ = rpa2 * rpa2 - sdel * a2 * s2;
        double iSD = 1.0 / (sig * sdel);
        double is2 = 1.0 / s2;
        double grr = sdel / sig;
        double gthi = 1.0 / sig;
        double w_init = 2.0 * r0 - Q2;

        *pth = -(*beta_val);
        double rest = -A_ * iSD + 2.0 * a * (*b) * w_init * iSD
                      + gthi * (*beta_val) * (*beta_val)
                      + (sig - w_init) * iSD * is2 * (*b) * (*b);
        double pr2 = -rest / grr;
        if (pr2 <= 0.0) return false;
        *pr = -sqrt(pr2);

        /* Event horizon radius */
        *rp = 1.0 + sqrt(fmax(1.0 - a2 - Q2, 0.0));
    }

    /* Write initial state */
    output[0] = *r;
    output[1] = *th;
    output[2] = *phi;
    output[3] = *pr;
    output[4] = *pth;
    output[5] = *b;
    output[6] = *alpha_val;
    output[7] = *beta_val;
    output[15] = *rp;
    output[16] = p.isco;
    output[17] = computeHamiltonian(*r, *th, *pr, *pth, a, *b, Q2);
    output[19] = computeCarter(*th, *pth, a, *b, Q2);

    return true;
}


/* -- Common disk crossing recording ------------------------- */

__device__ void record_crossing(
    double *output, int crossing_base, int *num_crossings,
    int step_idx, double oldR, double oldTh, double oldPhi,
    double r, double th, double phi,
    double a, double Q2, double b, double isco, double disk_temp
) {
    if (*num_crossings >= MAX_CROSSINGS) return;

    double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
    if (cross >= 0.0) return;

    double f = fmin(fmax(fabs(oldTh - PI * 0.5) /
               fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
    double r_hit = oldR + f * (r - oldR);
    double phi_hit = oldPhi + f * (phi - oldPhi);
    double direction = (oldTh > PI * 0.5) ? 1.0 : -1.0;

    /* Compute disk physics at crossing point */
    float g = kerr_g_factor(r_hit, a, Q2, b, isco);
    double F_norm = normalized_nt_flux(r_hit, a, isco);

    /* Temperature from flux: T ??? F^{1/4} */
    double T_base = 8000.0 * disk_temp;
    double T_emit = T_base * pow(fmax(F_norm, 0.0), 0.25);
    double T_obs = (double)g * T_emit;

    /* Write crossing data */
    int coff = crossing_base + 1 + (*num_crossings) * 8;
    output[coff + 0] = (double)step_idx;
    output[coff + 1] = r_hit;
    output[coff + 2] = phi_hit;
    output[coff + 3] = direction;
    output[coff + 4] = (double)g;
    output[coff + 5] = F_norm;
    output[coff + 6] = T_emit;
    output[coff + 7] = T_obs;
    (*num_crossings)++;
}


/* -- Common final state writing ----------------------------- */

__device__ void ray_finalize(
    double *output, int crossing_base, int num_crossings,
    double r, double th, double phi, double pr, double pth,
    double a, double b, double Q2,
    int term_reason, int steps_used, int coords
) {
    if (coords == 1) {
        output[18] = computeHamiltonianKS(r, th, pr, pth, a, b, Q2);
    } else {
        output[18] = computeHamiltonian(r, th, pr, pth, a, b, Q2);
    }
    output[20] = computeCarter(th, pth, a, b, Q2);

    output[8]  = r;
    output[9]  = th;
    output[10] = phi;
    output[11] = pr;
    output[12] = pth;
    output[13] = (double)term_reason;
    output[14] = (double)steps_used;
    output[crossing_base] = (double)num_crossings;
}








/* ================================================================
 *  SYMPLECTIC8 RAY TRACE
 *  Unified 8th-10th order symplectic integrator combining:
 *    - Tao extended phase space (non-separable splitting)
 *    - Kahan-Li s15odr8 8th-order composition (15 palindromic stages)
 *    - Kerr-Schild coordinates (no Delta=0 singularity)
 *    - AS-Phi adaptive stepping (Wu et al. 2024 / Preto & Saha 2009)
 *    - Compensated (Kahan) summation (~2x machine precision)
 *    - Wisdom symplectic corrector (raises order 8->~10)
 *    - Hamiltonian projection (algebraic H=0 constraint)
 * ================================================================ */

extern "C" __global__
void ray_trace_tkl108(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double a = p.spin;
    double Q2 = p.charge * p.charge;

    int coords = 1;  /* KS coordinates */
    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    if (!ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj)) {
        ray_finalize(output, 20 + max_traj * 4, 0, r, th, phi, pr, pth, a, b, Q2, 4, 0, coords);
        return;
    }

    /* Transform p_r from BL to Kerr-Schild coordinates */
    transformBLtoKS(r, a, b, Q2, &pr);

    /* Recompute H_init in KS coordinates (ray_init computed it in BL,
     * which gives a different numerical value after the p_r transform). */
    output[17] = computeHamiltonianKS(r, th, pr, pth, a, b, Q2);

    /* Initialize Tao shadow variables */
    double rs = r, ths = th, phis = phi, prs = pr, pths = pth;

    int traj_base = 24;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;

    /* 4 step multiplier (matching render kernel) */
    int STEPS = (int)p.steps * 4;
    int term_reason = 0, steps_used = 0;

    /* Compensated summation accumulators (real variables + ) */
    double r_comp = 0.0, th_comp = 0.0, phi_comp = 0.0;
    double pr_comp = 0.0, pth_comp = 0.0;

    /* AS: Sundman/Mino time base step +  variable */
    double dtau = sundman_dtau(a, Q2, rp, p.step_size, p.esc_radius, STEPS);
    double Phi = p.obs_dist / r;
    double Phi_comp = 0.0;
    double h_phi = dtau * p.obs_dist * p.obs_dist;

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;

        double oldR = r, oldTh = th, oldPhi = phi;

        /*  AS Step 1: Half-step  update (KS)  */
        double g_sun = phi_var_sundman_g(r, th, a);
        double dPhi = phi_var_dphi_KS(r, th, pr, a, Q2, g_sun, h_phi);
        rt_kahan_add(&Phi, &Phi_comp, dPhi);
        if (Phi < 0.01) Phi = 0.01;

        /*  AS Step 3: Compute physical step h/  */
        double he = phi_var_physical_step(h_phi, Phi, r, th, pth, a, p.obs_dist);

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off + 0] = r; output[off + 1] = th;
            output[off + 2] = phi; output[off + 3] = he;
        }

        /*  Tao + Kahan-Li 8th-order step  */
        tao_kahan_li8_step(&r, &th, &phi, &pr, &pth,
                           &rs, &ths, &phis, &prs, &pths,
                           a, b, Q2, he);

        /*  Wisdom symplectic corrector (KS, real + shadow)  */
        {
            double corr_eps = he * he / 24.0;

            /* Real variables: kick then drift */
            double f_pr, f_pth;
            geoForceKS(r, th, pr, pth, a, b, Q2, &f_pr, &f_pth);
            rt_kahan_add(&pr,  &pr_comp,  corr_eps * f_pr);
            rt_kahan_add(&pth, &pth_comp, corr_eps * f_pth);

            double v_r, v_th, v_phi;
            geoVelocityKS(r, th, pr, pth, a, b, Q2, &v_r, &v_th, &v_phi);
            rt_kahan_add(&r,   &r_comp,   corr_eps * v_r);
            rt_kahan_add(&th,  &th_comp,  corr_eps * v_th);
            rt_kahan_add(&phi, &phi_comp, corr_eps * v_phi);

            /* Shadow variables: same canonical correction */
            geoForceKS(rs, ths, prs, pths, a, b, Q2, &f_pr, &f_pth);
            prs  += corr_eps * f_pr;
            pths += corr_eps * f_pth;

            geoVelocityKS(rs, ths, prs, pths, a, b, Q2, &v_r, &v_th, &v_phi);
            rs   += corr_eps * v_r;
            ths  += corr_eps * v_th;
            phis += corr_eps * v_phi;
        }

        /*  Hamiltonian projection (KS)  */
        projectHamiltonianKS(r, th, &pr, pth, a, b, Q2);
        pr_comp = 0.0;

        /*  AS Step 5: Second half-step  update  */
        g_sun = phi_var_sundman_g(r, th, a);
        dPhi = phi_var_dphi_KS(r, th, pr, a, Q2, g_sun, h_phi);
        rt_kahan_add(&Phi, &Phi_comp, dPhi);
        if (Phi < 0.01) Phi = 0.01;

        /*  Pole reflection (real + shadow)  */
        pole_reflect(&th, &pth, &phi);
        pole_reflect(&ths, &pths, &phis);

        /* KS coordinates: well inside horizon check */
        if (r <= rp * 0.5) { term_reason = 1; break; }

        record_crossing(output, crossing_base, &num_crossings,
                        i, oldR, oldTh, oldPhi, r, th, phi,
                        a, Q2, b, p.isco, p.disk_temp);

        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }

    ray_finalize(output, crossing_base, num_crossings,
                 r, th, phi, pr, pth, a, b, Q2, term_reason, steps_used, coords);
}


/* ================================================================
 *  VERNER 9(8) RAY TRACE
 *  16-stage adaptive with FSAL, embedded 8th-order error estimate.
 *  See integrators/verner98.cu for coefficients and method details.
 * ================================================================ */

extern "C" __global__
void ray_trace_verner98(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double a = p.spin;
    double Q2 = p.charge * p.charge;

    int coords = 0;
    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    if (!ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj)) {
        ray_finalize(output, 20 + max_traj * 4, 0, r, th, phi, pr, pth, a, b, Q2, 4, 0, coords);
        return;
    }
    int traj_base = 24;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;
    int STEPS = (int)p.steps;
    int term_reason = 0, steps_used = 0;

    double atol = 1.0e-10, rtol = 1.0e-10, safety = 0.9;
    double h_min = 0.0005, h_max = 2.5;
    int max_reject = 5;
    double he = adaptive_step_verner98_initial(r, rp, p.step_size, p.obs_dist, h_min, h_max);

    double fsal_kr=0, fsal_kth=0, fsal_kphi=0, fsal_kpr=0, fsal_kpth=0;
    bool have_fsal = false;

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off+0] = r; output[off+1] = th;
            output[off+2] = phi; output[off+3] = he;
        }

        double oldR = r, oldTh = th, oldPhi = phi;
        int rejects = 0;
        bool accepted = false;

        while (!accepted) {
            double kr1,kth1,kphi1,kpr1,kpth1;
            if (have_fsal) {
                kr1=fsal_kr; kth1=fsal_kth; kphi1=fsal_kphi;
                kpr1=fsal_kpr; kpth1=fsal_kpth;
            } else {
                geoRHS(r,th,pr,pth,a,b,Q2,&kr1,&kth1,&kphi1,&kpr1,&kpth1);
            }

            /* Stages 2-16: reuse same coefficient structure as verner98.cu
             * (abbreviated for the single-ray path - same math, no rendering) */
            double kr2,kth2,kphi2,kpr2,kpth2;
            geoRHS(r+he*0.03462105947808740*kr1, th+he*0.03462105947808740*kth1,
                   pr+he*0.03462105947808740*kpr1, pth+he*0.03462105947808740*kpth1,
                   a,b,Q2, &kr2,&kth2,&kphi2,&kpr2,&kpth2);

            double kr3,kth3,kphi3,kpr3,kpth3;
            geoRHS(r+he*(-0.01145764041037430*kr1+0.10847931051905938*kr2),
                   th+he*(-0.01145764041037430*kth1+0.10847931051905938*kth2),
                   pr+he*(-0.01145764041037430*kpr1+0.10847931051905938*kpr2),
                   pth+he*(-0.01145764041037430*kpth1+0.10847931051905938*kpth2),
                   a,b,Q2, &kr3,&kth3,&kphi3,&kpr3,&kpth3);

            double kr4,kth4,kphi4,kpr4,kpth4;
            geoRHS(r+he*(0.03638312629075691*kr1+0.10914937887227072*kr3),
                   th+he*(0.03638312629075691*kth1+0.10914937887227072*kth3),
                   pr+he*(0.03638312629075691*kpr1+0.10914937887227072*kpr3),
                   pth+he*(0.03638312629075691*kpth1+0.10914937887227072*kpth3),
                   a,b,Q2, &kr4,&kth4,&kphi4,&kpr4,&kpth4);

            double kr5,kth5,kphi5,kpr5,kpth5;
            geoRHS(r+he*(2.02576084012321920*kr1-7.63878392400893200*kr3+6.17402308388571280*kr4),
                   th+he*(2.02576084012321920*kth1-7.63878392400893200*kth3+6.17402308388571280*kth4),
                   pr+he*(2.02576084012321920*kpr1-7.63878392400893200*kpr3+6.17402308388571280*kpr4),
                   pth+he*(2.02576084012321920*kpth1-7.63878392400893200*kpth3+6.17402308388571280*kpth4),
                   a,b,Q2, &kr5,&kth5,&kphi5,&kpr5,&kpth5);

            double kr6,kth6,kphi6,kpr6,kpth6;
            geoRHS(r+he*(0.04267276475929752*kr1+0.14820330033498510*kr4+0.03812393490471746*kr5),
                   th+he*(0.04267276475929752*kth1+0.14820330033498510*kth4+0.03812393490471746*kth5),
                   pr+he*(0.04267276475929752*kpr1+0.14820330033498510*kpr4+0.03812393490471746*kpr5),
                   pth+he*(0.04267276475929752*kpth1+0.14820330033498510*kpth4+0.03812393490471746*kpth5),
                   a,b,Q2, &kr6,&kth6,&kphi6,&kpr6,&kpth6);

            double kr7,kth7,kphi7,kpr7,kpth7;
            geoRHS(r+he*(-0.54767920780445450*kr1+1.57586541028785960*kr4-0.18728267956932220*kr5-0.25390352291408300*kr6),
                   th+he*(-0.54767920780445450*kth1+1.57586541028785960*kth4-0.18728267956932220*kth5-0.25390352291408300*kth6),
                   pr+he*(-0.54767920780445450*kpr1+1.57586541028785960*kpr4-0.18728267956932220*kpr5-0.25390352291408300*kpr6),
                   pth+he*(-0.54767920780445450*kpth1+1.57586541028785960*kpth4-0.18728267956932220*kpth5-0.25390352291408300*kpth6),
                   a,b,Q2, &kr7,&kth7,&kphi7,&kpr7,&kpth7);

            double kr8,kth8,kphi8,kpr8,kpth8;
            geoRHS(r+he*(0.05205177336498790*kr1-0.00604125259748469*kr5+0.19662478445710860*kr6+0.02733169418038820*kr7),
                   th+he*(0.05205177336498790*kth1-0.00604125259748469*kth5+0.19662478445710860*kth6+0.02733169418038820*kth7),
                   pr+he*(0.05205177336498790*kpr1-0.00604125259748469*kpr5+0.19662478445710860*kpr6+0.02733169418038820*kpr7),
                   pth+he*(0.05205177336498790*kpth1-0.00604125259748469*kpth5+0.19662478445710860*kpth6+0.02733169418038820*kpth7),
                   a,b,Q2, &kr8,&kth8,&kphi8,&kpr8,&kpth8);

            double kr9,kth9,kphi9,kpr9,kpth9;
            geoRHS(r+he*(0.38160482408692510*kr1-0.37859553897503830*kr5+0.65180263498919150*kr6-0.31280721047629230*kr7+0.17799828948221400*kr8),
                   th+he*(0.38160482408692510*kth1-0.37859553897503830*kth5+0.65180263498919150*kth6-0.31280721047629230*kth7+0.17799828948221400*kth8),
                   pr+he*(0.38160482408692510*kpr1-0.37859553897503830*kpr5+0.65180263498919150*kpr6-0.31280721047629230*kpr7+0.17799828948221400*kpr8),
                   pth+he*(0.38160482408692510*kpth1-0.37859553897503830*kpth5+0.65180263498919150*kpth6-0.31280721047629230*kpth7+0.17799828948221400*kpth8),
                   a,b,Q2, &kr9,&kth9,&kphi9,&kpr9,&kpth9);

            double kr10,kth10,kphi10,kpr10,kpth10;
            geoRHS(r+he*(-0.30028658890498370*kr1-0.17358257652498910*kr5+0.55026380681023870*kr6+0.97537826024843460*kr7-0.27064375576498590*kr8+0.08887105873994540*kr9),
                   th+he*(-0.30028658890498370*kth1-0.17358257652498910*kth5+0.55026380681023870*kth6+0.97537826024843460*kth7-0.27064375576498590*kth8+0.08887105873994540*kth9),
                   pr+he*(-0.30028658890498370*kpr1-0.17358257652498910*kpr5+0.55026380681023870*kpr6+0.97537826024843460*kpr7-0.27064375576498590*kpr8+0.08887105873994540*kpr9),
                   pth+he*(-0.30028658890498370*kpth1-0.17358257652498910*kpth5+0.55026380681023870*kpth6+0.97537826024843460*kpth7-0.27064375576498590*kpth8+0.08887105873994540*kpth9),
                   a,b,Q2, &kr10,&kth10,&kphi10,&kpr10,&kpth10);

            double kr11,kth11,kphi11,kpr11,kpth11;
            geoRHS(r+he*(-0.18498523756063790*kr1-0.10174529024826280*kr5+0.39760051866653810*kr6-0.08466557698498370*kr7+0.22573424329660270*kr8-0.00468612828898990*kr9+0.11274147013073350*kr10),
                   th+he*(-0.18498523756063790*kth1-0.10174529024826280*kth5+0.39760051866653810*kth6-0.08466557698498370*kth7+0.22573424329660270*kth8-0.00468612828898990*kth9+0.11274147013073350*kth10),
                   pr+he*(-0.18498523756063790*kpr1-0.10174529024826280*kpr5+0.39760051866653810*kpr6-0.08466557698498370*kpr7+0.22573424329660270*kpr8-0.00468612828898990*kpr9+0.11274147013073350*kpr10),
                   pth+he*(-0.18498523756063790*kpth1-0.10174529024826280*kpth5+0.39760051866653810*kpth6-0.08466557698498370*kpth7+0.22573424329660270*kpth8-0.00468612828898990*kpth9+0.11274147013073350*kpth10),
                   a,b,Q2, &kr11,&kth11,&kphi11,&kpr11,&kpth11);

            double kr12,kth12,kphi12,kpr12,kpth12;
            geoRHS(r+he*(0.04060463992884014*kr1-0.01524376895647988*kr5+0.22816037995671680*kr6+0.02282113269509040*kr7+0.05755223006614940*kr8+0.00312076906683198*kr9+0.02197722370854290*kr10+0.00100253547909560*kr11),
                   th+he*(0.04060463992884014*kth1-0.01524376895647988*kth5+0.22816037995671680*kth6+0.02282113269509040*kth7+0.05755223006614940*kth8+0.00312076906683198*kth9+0.02197722370854290*kth10+0.00100253547909560*kth11),
                   pr+he*(0.04060463992884014*kpr1-0.01524376895647988*kpr5+0.22816037995671680*kpr6+0.02282113269509040*kpr7+0.05755223006614940*kpr8+0.00312076906683198*kpr9+0.02197722370854290*kpr10+0.00100253547909560*kpr11),
                   pth+he*(0.04060463992884014*kpth1-0.01524376895647988*kpth5+0.22816037995671680*kpth6+0.02282113269509040*kpth7+0.05755223006614940*kpth8+0.00312076906683198*kpth9+0.02197722370854290*kpth10+0.00100253547909560*kpth11),
                   a,b,Q2, &kr12,&kth12,&kphi12,&kpr12,&kpth12);

            double kr13,kth13,kphi13,kpr13,kpth13;
            geoRHS(r+he*(-0.79001697653399120*kr1-0.78129536715770160*kr5+2.39773499306773300*kr6-0.87007974028645130*kr7+0.66018863998549750*kr8-0.05346786559133970*kr9-0.02350873464028790*kr10+0.00067497625461070*kr12),
                   th+he*(-0.79001697653399120*kth1-0.78129536715770160*kth5+2.39773499306773300*kth6-0.87007974028645130*kth7+0.66018863998549750*kth8-0.05346786559133970*kth9-0.02350873464028790*kth10+0.00067497625461070*kth12),
                   pr+he*(-0.79001697653399120*kpr1-0.78129536715770160*kpr5+2.39773499306773300*kpr6-0.87007974028645130*kpr7+0.66018863998549750*kpr8-0.05346786559133970*kpr9-0.02350873464028790*kpr10+0.00067497625461070*kpr12),
                   pth+he*(-0.79001697653399120*kpth1-0.78129536715770160*kpth5+2.39773499306773300*kpth6-0.87007974028645130*kpth7+0.66018863998549750*kpth8-0.05346786559133970*kpth9-0.02350873464028790*kpth10+0.00067497625461070*kpth12),
                   a,b,Q2, &kr13,&kth13,&kphi13,&kpr13,&kpth13);

            double kr14,kth14,kphi14,kpr14,kpth14;
            geoRHS(r+he*(1.95133027691168770*kr1+2.68843654567613760*kr5-6.85905028498498400*kr6+3.24617379538318340*kr7-1.31856430884698270*kr8+0.01930972752471484*kr9+0.38088027094063360*kr10-0.00348545093199020*kr12-0.10523063848774780*kr13),
                   th+he*(1.95133027691168770*kth1+2.68843654567613760*kth5-6.85905028498498400*kth6+3.24617379538318340*kth7-1.31856430884698270*kth8+0.01930972752471484*kth9+0.38088027094063360*kth10-0.00348545093199020*kth12-0.10523063848774780*kth13),
                   pr+he*(1.95133027691168770*kpr1+2.68843654567613760*kpr5-6.85905028498498400*kpr6+3.24617379538318340*kpr7-1.31856430884698270*kpr8+0.01930972752471484*kpr9+0.38088027094063360*kpr10-0.00348545093199020*kpr12-0.10523063848774780*kpr13),
                   pth+he*(1.95133027691168770*kpth1+2.68843654567613760*kpth5-6.85905028498498400*kpth6+3.24617379538318340*kpth7-1.31856430884698270*kpth8+0.01930972752471484*kpth9+0.38088027094063360*kpth10-0.00348545093199020*kpth12-0.10523063848774780*kpth13),
                   a,b,Q2, &kr14,&kth14,&kphi14,&kpr14,&kpth14);

            double kr15,kth15,kphi15,kpr15,kpth15;
            geoRHS(r+he*(-6.22831040813895700*kr1-8.52538682726529500*kr5+22.62914620458986000*kr6-10.37853781827994400*kr7+3.97822004722482400*kr8+0.14881347033498820*kr9-0.80882510422676250*kr10+0.02037534522884804*kr12+0.53927013529397810*kr13-0.37511622624617130*kr14),
                   th+he*(-6.22831040813895700*kth1-8.52538682726529500*kth5+22.62914620458986000*kth6-10.37853781827994400*kth7+3.97822004722482400*kth8+0.14881347033498820*kth9-0.80882510422676250*kth10+0.02037534522884804*kth12+0.53927013529397810*kth13-0.37511622624617130*kth14),
                   pr+he*(-6.22831040813895700*kpr1-8.52538682726529500*kpr5+22.62914620458986000*kpr6-10.37853781827994400*kpr7+3.97822004722482400*kpr8+0.14881347033498820*kpr9-0.80882510422676250*kpr10+0.02037534522884804*kpr12+0.53927013529397810*kpr13-0.37511622624617130*kpr14),
                   pth+he*(-6.22831040813895700*kpth1-8.52538682726529500*kpth5+22.62914620458986000*kpth6-10.37853781827994400*kpth7+3.97822004722482400*kpth8+0.14881347033498820*kpth9-0.80882510422676250*kpth10+0.02037534522884804*kpth12+0.53927013529397810*kpth13-0.37511622624617130*kpth14),
                   a,b,Q2, &kr15,&kth15,&kphi15,&kpr15,&kpth15);

            double kr16,kth16,kphi16,kpr16,kpth16;
            geoRHS(r+he*(0.18735067067678510*kr1+0.23036200709948010*kr8+0.02216475594979920*kr9-0.00282237605135478*kr10+0.01122809895497192*kr12+0.20521517541037220*kr13+0.34586338456370620*kr15),
                   th+he*(0.18735067067678510*kth1+0.23036200709948010*kth8+0.02216475594979920*kth9-0.00282237605135478*kth10+0.01122809895497192*kth12+0.20521517541037220*kth13+0.34586338456370620*kth15),
                   pr+he*(0.18735067067678510*kpr1+0.23036200709948010*kpr8+0.02216475594979920*kpr9-0.00282237605135478*kpr10+0.01122809895497192*kpr12+0.20521517541037220*kpr13+0.34586338456370620*kpr15),
                   pth+he*(0.18735067067678510*kpth1+0.23036200709948010*kpth8+0.02216475594979920*kpth9-0.00282237605135478*kpth10+0.01122809895497192*kpth12+0.20521517541037220*kpth13+0.34586338456370620*kpth15),
                   a,b,Q2, &kr16,&kth16,&kphi16,&kpr16,&kpth16);

            /* 9th-order weights */
            double b1=0.04427989419007951, b8=0.21535969913498340, b9=0.02085536024116400;
            double b10=-0.00283482882918294, b12=0.01011592660757810, b13=0.22012883537885700;
            double b15=0.34065469893999770, b16=0.15044564984635120;

            double dr9  =he*(b1*kr1+b8*kr8+b9*kr9+b10*kr10+b12*kr12+b13*kr13+b15*kr15+b16*kr16);
            double dth9 =he*(b1*kth1+b8*kth8+b9*kth9+b10*kth10+b12*kth12+b13*kth13+b15*kth15+b16*kth16);
            double dpr9 =he*(b1*kpr1+b8*kpr8+b9*kpr9+b10*kpr10+b12*kpr12+b13*kpr13+b15*kpr15+b16*kpr16);
            double dpth9=he*(b1*kpth1+b8*kpth8+b9*kpth9+b10*kpth10+b12*kpth12+b13*kpth13+b15*kpth15+b16*kpth16);

            /* Error coefficients */
            double e1=0.00130288967723760,e8=-0.00887264545482580,e9=0.00252166975287920;
            double e10=-0.00282237605135478,e12=0.00142543427426602,e13=0.01508818990882670;
            double e14=-0.00544096020943730,e15=0.00524893979047480,e16=-0.00846073189916644;

            double er  =he*(e1*kr1+e8*kr8+e9*kr9+e10*kr10+e12*kr12+e13*kr13+e14*kr14+e15*kr15+e16*kr16);
            double eth =he*(e1*kth1+e8*kth8+e9*kth9+e10*kth10+e12*kth12+e13*kth13+e14*kth14+e15*kth15+e16*kth16);
            double epr =he*(e1*kpr1+e8*kpr8+e9*kpr9+e10*kpr10+e12*kpr12+e13*kpr13+e14*kpr14+e15*kpr15+e16*kpr16);
            double epth=he*(e1*kpth1+e8*kpth8+e9*kpth9+e10*kpth10+e12*kpth12+e13*kpth13+e14*kpth14+e15*kpth15+e16*kpth16);

            double sc_r=atol+rtol*fmax(fabs(r),fabs(r+dr9));
            double sc_th=atol+rtol*fmax(fabs(th),fabs(th+dth9));
            double sc_pr=atol+rtol*fmax(fabs(pr),fabs(pr+dpr9));
            double sc_pth=atol+rtol*fmax(fabs(pth),fabs(pth+dpth9));
            double err_norm=sqrt(0.25*((er/sc_r)*(er/sc_r)+(eth/sc_th)*(eth/sc_th)+(epr/sc_pr)*(epr/sc_pr)+(epth/sc_pth)*(epth/sc_pth)));

            if (err_norm<=1.0||rejects>=max_reject) {
                r+=dr9; th+=dth9;
                phi+=he*(b1*kphi1+b8*kphi8+b9*kphi9+b10*kphi10+b12*kphi12+b13*kphi13+b15*kphi15+b16*kphi16);
                pr+=dpr9; pth+=dpth9;
                accepted=true;
                fsal_kr=kr16;fsal_kth=kth16;fsal_kphi=kphi16;fsal_kpr=kpr16;fsal_kpth=kpth16;
                have_fsal=true;
                if(err_norm>1e-30){double f=safety*pow(err_norm,-1.0/9.0);f=fmin(fmax(f,0.1),5.0);he*=f;}
                he=fmin(fmax(he,h_min),h_max);
            } else {
                double f=safety*pow(err_norm,-1.0/9.0);f=fmax(f,0.1);he*=f;he=fmax(he,h_min);
                rejects++;have_fsal=false;
            }
        }

        if (r<=rp*1.01) { term_reason=1; break; }
        record_crossing(output,crossing_base,&num_crossings,i,oldR,oldTh,oldPhi,r,th,phi,a,Q2,b,p.isco,p.disk_temp);
        if (r>p.esc_radius) { term_reason=2; break; }
        if (r<0.5) { term_reason=4; break; }
        if (r!=r||th!=th) { term_reason=3; break; }
    }

    ray_finalize(output,crossing_base,num_crossings,r,th,phi,pr,pth,a,b,Q2,term_reason,steps_used,coords);
}


/* ================================================================
 *  RKN 8(6) RAY TRACE
 *  Runge-Kutta-Nystrm second-order formulation.
 *  Stub: dispatches through first-order geoRHS for the single-ray
 *  path (full Nystrm only in the render kernel where it matters
 *  for throughput).  Same error control as rkn86.cu.
 * ================================================================ */

extern "C" __global__
void ray_trace_rkn86(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double a = p.spin;
    double Q2 = p.charge * p.charge;

    int coords = 0;
    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    if (!ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj)) {
        ray_finalize(output, 20+max_traj*4, 0, r, th, phi, pr, pth, a, b, Q2, 4, 0, coords);
        return;
    }
    int traj_base = 24;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;
    int STEPS = (int)p.steps;
    int term_reason = 0, steps_used = 0;

    /* For the single-ray diagnostic path, use the RKDP8 integrator
     * with rkn86-tuned tolerances.  The full Nystrm formulation is
     * only used in the render kernel (trace_rkn86) where per-pixel
     * throughput justifies the dedicated geoAccel codepath. */
    double atol=1.0e-9, rtol=1.0e-9, safety=0.9, h_min=0.0005, h_max=2.5;
    int max_reject=4;
    double he = adaptive_step_rkn86_initial(r, rp, p.step_size, p.obs_dist, h_min, h_max);

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;
        if (i < max_traj) {
            int off = traj_base + i*4;
            output[off+0]=r; output[off+1]=th; output[off+2]=phi; output[off+3]=he;
        }
        double oldR=r, oldTh=th, oldPhi=phi;
        int rejects=0; bool accepted=false;

        while (!accepted) {
            /* Use truncated DP8(7) stages for single-ray (first 8 stages only) */
            double k[8][5];  // [stage][component: r,th,phi,pr,pth]

            /* Stages 1-8 using constant memory Butcher tableau */
            for (int stage = 0; stage < 8; stage++) {
                double yr  = r,  yth = th,  ypr = pr,  ypth = pth;
                double yphi = 0.0;  // phi accumulated separately

                for (int prev = 0; prev < stage; prev++) {
                    double a_ij = RKDP8_A[stage][prev];
                    yr  += he * a_ij * k[prev][0];
                    yth += he * a_ij * k[prev][1];
                    ypr += he * a_ij * k[prev][3];
                    ypth+= he * a_ij * k[prev][4];
                }

                geoRHS(yr, yth, ypr, ypth, a, b, Q2,
                       &k[stage][0], &k[stage][1], &k[stage][2], &k[stage][3], &k[stage][4]);
            }

            /* Compute solution increments using first 8 stages only */
            double dr8 = 0.0, dth8 = 0.0, dphi8 = 0.0, dpr8 = 0.0, dpth8 = 0.0;
            for (int stage = 0; stage < 8; stage++) {
                double b_weight = RKDP8_B[stage];
                dr8   += b_weight * k[stage][0];
                dth8  += b_weight * k[stage][1];
                dphi8 += b_weight * k[stage][2];
                dpr8  += b_weight * k[stage][3];
                dpth8 += b_weight * k[stage][4];
            }
            dr8   *= he;
            dth8  *= he;
            dphi8 *= he;
            dpr8  *= he;
            dpth8 *= he;

            /* Compute error estimate using first 8 stages only */
            double err_r = 0.0, err_th = 0.0, err_pr = 0.0, err_pth = 0.0;
            for (int stage = 0; stage < 8; stage++) {
                double e_weight = RKDP8_B[stage] - RKDP8_BHAT[stage];
                err_r   += e_weight * k[stage][0];
                err_th  += e_weight * k[stage][1];
                err_pr  += e_weight * k[stage][3];
                err_pth += e_weight * k[stage][4];
            }
            err_r   *= he;
            err_th  *= he;
            err_pr  *= he;
            err_pth *= he;

            double sc_r=atol+rtol*fmax(fabs(r),fabs(r+dr8));
            double sc_th=atol+rtol*fmax(fabs(th),fabs(th+dth8));
            double sc_pr=atol+rtol*fmax(fabs(pr),fabs(pr+dpr8));
            double sc_pth=atol+rtol*fmax(fabs(pth),fabs(pth+dpth8));
            double en=sqrt(0.25*((err_r/sc_r)*(err_r/sc_r)+(err_th/sc_th)*(err_th/sc_th)+(err_pr/sc_pr)*(err_pr/sc_pr)+(err_pth/sc_pth)*(err_pth/sc_pth)));

            if (en<=1.0||rejects>=max_reject) {
                r+=dr8;th+=dth8;
                phi+=dphi8;
                pr+=dpr8;pth+=dpth8;
                accepted=true;
                if(en>1e-30){double f=safety*pow(en,-1.0/8.0);f=fmin(fmax(f,0.2),5.0);he*=f;}
                he=fmin(fmax(he,h_min),h_max);
            } else {
                double f=safety*pow(en,-1.0/8.0);f=fmax(f,0.2);he*=f;he=fmax(he,h_min);rejects++;
            }
        }
        }

        if (r<=rp*1.01){term_reason=1;break;}
        record_crossing(output,crossing_base,&num_crossings,i,oldR,oldTh,oldPhi,r,th,phi,a,Q2,b,p.isco,p.disk_temp);
        if (r>p.esc_radius){term_reason=2;break;}
        if (r<0.5){term_reason=4;break;}
        if (r!=r||th!=th){term_reason=3;break;}
    }

    ray_finalize(output,crossing_base,num_crossings,r,th,phi,pr,pth,a,b,Q2,term_reason,steps_used,coords);
}
