/* ============================================================
 *  RAY_TRACE — Single-ray tracing kernel for the /ray endpoint
 *
 *  Traces a single photon geodesic through Kerr-Newman spacetime
 *  and records the full trajectory, equatorial plane crossings,
 *  and disk physics (g-factor, Novikov-Thorne flux, temperature).
 *
 *  Provides per-integrator entry points for ALL 7 methods,
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
 *        [crossing_base + 1 + j*8 + 3] = direction (1.0=N→S, -1.0=S→N)
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
#include "disk.cu"
#include "integrators/steps.cu"
#include "integrators/adaptive_step.cu"


/* ── Maximum disk crossings to record ─────────────────────── */
#define MAX_CROSSINGS 16


/* ── Novikov-Thorne flux with peak normalization ──────────── */

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


/* ── Kahan-Li s15odr8 coefficients (for kahanli8s/kahanli8s_ks) ── */

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


/* ── Common ray initialization ────────────────────────────── */

__device__ void ray_init(
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
        initRay(ix, iy, p, r, th, phi, pr, pth, b, rp, &alpha_f, &beta_f);
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
        *pr = (pr2 > 0.0) ? -sqrt(pr2) : 0.0;

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
}


/* ── Common disk crossing recording ───────────────────────── */

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
    float g = compute_g_factor(r_hit, a, Q2, b);
    double F_norm = normalized_nt_flux(r_hit, a, isco);

    /* Temperature from flux: T ∝ F^{1/4} */
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


/* ── Common final state writing ───────────────────────────── */

__device__ void ray_finalize(
    double *output, int crossing_base, int num_crossings,
    double r, double th, double phi, double pr, double pth,
    int term_reason, int steps_used
) {
    output[8]  = r;
    output[9]  = th;
    output[10] = phi;
    output[11] = pr;
    output[12] = pth;
    output[13] = (double)term_reason;
    output[14] = (double)steps_used;
    output[crossing_base] = (double)num_crossings;
}


/* ════════════════════════════════════════════════════════════
 *  YOSHIDA4 RAY TRACE
 * ════════════════════════════════════════════════════════════ */

extern "C" __global__
void ray_trace_yoshida4(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj);

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    int traj_base = 20;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;
    int STEPS = (int)p.steps;
    int term_reason = 0, steps_used = 0;

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;
        double he = adaptive_step_symplectic(r, rp, p.step_size, p.obs_dist);

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off + 0] = r; output[off + 1] = th;
            output[off + 2] = phi; output[off + 3] = he;
        }

        double oldR = r, oldTh = th, oldPhi = phi;
        yoshida4_step(&r, &th, &phi, &pr, &pth, a, b, Q2, he);

        /* Hamiltonian constraint projection */
        projectHamiltonian(r, th, &pr, pth, a, b, Q2);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) { term_reason = 1; break; }

        record_crossing(output, crossing_base, &num_crossings,
                        i, oldR, oldTh, oldPhi, r, th, phi,
                        a, Q2, b, p.isco, p.disk_temp);

        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }

    ray_finalize(output, crossing_base, num_crossings,
                 r, th, phi, pr, pth, term_reason, steps_used);
}


/* ════════════════════════════════════════════════════════════
 *  RK4 RAY TRACE
 * ════════════════════════════════════════════════════════════ */

extern "C" __global__
void ray_trace_rk4(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj);

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    int traj_base = 20;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;
    int STEPS = (int)p.steps;
    int term_reason = 0, steps_used = 0;

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;
        double he = adaptive_step_rk4(r, rp, p.step_size, p.obs_dist);

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off + 0] = r; output[off + 1] = th;
            output[off + 2] = phi; output[off + 3] = he;
        }

        double oldR = r, oldTh = th, oldPhi = phi;
        rk4_step(&r, &th, &phi, &pr, &pth, a, b, Q2, he);

        /* Hamiltonian constraint projection */
        projectHamiltonian(r, th, &pr, pth, a, b, Q2);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) { term_reason = 1; break; }

        record_crossing(output, crossing_base, &num_crossings,
                        i, oldR, oldTh, oldPhi, r, th, phi,
                        a, Q2, b, p.isco, p.disk_temp);

        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }

    ray_finalize(output, crossing_base, num_crossings,
                 r, th, phi, pr, pth, term_reason, steps_used);
}


/* ════════════════════════════════════════════════════════════
 *  RKDP8 RAY TRACE (Dormand-Prince 8th-order with adaptive step)
 * ════════════════════════════════════════════════════════════ */

extern "C" __global__
void ray_trace_rkdp8(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj);

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    int traj_base = 20;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;
    int STEPS = (int)p.steps;
    int term_reason = 0, steps_used = 0;

    /* Adaptive step control parameters (matching rkdp8.cu render kernel) */
    double atol = 1.0e-8;
    double rtol = 1.0e-8;
    double safety = 0.9;
    double h_min = 0.001;
    double h_max = 2.0;
    int max_reject = 4;

    double he = adaptive_step_rkdp8_initial(r, rp, p.step_size, p.obs_dist, h_min, h_max);

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off + 0] = r; output[off + 1] = th;
            output[off + 2] = phi; output[off + 3] = he;
        }

        double oldR = r, oldTh = th, oldPhi = phi;
        int rejects = 0;
        bool accepted = false;

        while (!accepted) {
            /* 13 stage variables */
            double kr1,kth1,kphi1,kpr1,kpth1;
            double kr2,kth2,kphi2,kpr2,kpth2;
            double kr3,kth3,kphi3,kpr3,kpth3;
            double kr4,kth4,kphi4,kpr4,kpth4;
            double kr5,kth5,kphi5,kpr5,kpth5;
            double kr6,kth6,kphi6,kpr6,kpth6;
            double kr7,kth7,kphi7,kpr7,kpth7;
            double kr8,kth8,kphi8,kpr8,kpth8;
            double kr9,kth9,kphi9,kpr9,kpth9;
            double kr10,kth10,kphi10,kpr10,kpth10;
            double kr11,kth11,kphi11,kpr11,kpth11;
            double kr12,kth12,kphi12,kpr12,kpth12;
            double kr13,kth13,kphi13,kpr13,kpth13;

            geoRHS(r, th, pr, pth, a, b, Q2, &kr1, &kth1, &kphi1, &kpr1, &kpth1);
            geoRHS(r+he*kr1/18.0, th+he*kth1/18.0, pr+he*kpr1/18.0, pth+he*kpth1/18.0, a, b, Q2, &kr2, &kth2, &kphi2, &kpr2, &kpth2);
            geoRHS(r+he*(kr1/48.0+kr2/16.0), th+he*(kth1/48.0+kth2/16.0), pr+he*(kpr1/48.0+kpr2/16.0), pth+he*(kpth1/48.0+kpth2/16.0), a, b, Q2, &kr3, &kth3, &kphi3, &kpr3, &kpth3);
            geoRHS(r+he*(kr1/32.0+kr3*3.0/32.0), th+he*(kth1/32.0+kth3*3.0/32.0), pr+he*(kpr1/32.0+kpr3*3.0/32.0), pth+he*(kpth1/32.0+kpth3*3.0/32.0), a, b, Q2, &kr4, &kth4, &kphi4, &kpr4, &kpth4);
            geoRHS(r+he*(kr1*5.0/16.0-kr3*75.0/64.0+kr4*75.0/64.0), th+he*(kth1*5.0/16.0-kth3*75.0/64.0+kth4*75.0/64.0), pr+he*(kpr1*5.0/16.0-kpr3*75.0/64.0+kpr4*75.0/64.0), pth+he*(kpth1*5.0/16.0-kpth3*75.0/64.0+kpth4*75.0/64.0), a, b, Q2, &kr5, &kth5, &kphi5, &kpr5, &kpth5);
            geoRHS(r+he*(kr1*3.0/80.0+kr4*3.0/16.0+kr5*3.0/20.0), th+he*(kth1*3.0/80.0+kth4*3.0/16.0+kth5*3.0/20.0), pr+he*(kpr1*3.0/80.0+kpr4*3.0/16.0+kpr5*3.0/20.0), pth+he*(kpth1*3.0/80.0+kpth4*3.0/16.0+kpth5*3.0/20.0), a, b, Q2, &kr6, &kth6, &kphi6, &kpr6, &kpth6);

            double a71=29443841.0/614563906.0, a74=77736538.0/692538347.0, a75=-28693883.0/1125000000.0, a76=23124283.0/1800000000.0;
            geoRHS(r+he*(a71*kr1+a74*kr4+a75*kr5+a76*kr6), th+he*(a71*kth1+a74*kth4+a75*kth5+a76*kth6), pr+he*(a71*kpr1+a74*kpr4+a75*kpr5+a76*kpr6), pth+he*(a71*kpth1+a74*kpth4+a75*kpth5+a76*kpth6), a, b, Q2, &kr7, &kth7, &kphi7, &kpr7, &kpth7);

            double a81=16016141.0/946692911.0, a84=61564180.0/158732637.0, a85=22789713.0/633445777.0, a86=545815736.0/2771057229.0, a87=-180193667.0/1043307555.0;
            geoRHS(r+he*(a81*kr1+a84*kr4+a85*kr5+a86*kr6+a87*kr7), th+he*(a81*kth1+a84*kth4+a85*kth5+a86*kth6+a87*kth7), pr+he*(a81*kpr1+a84*kpr4+a85*kpr5+a86*kpr6+a87*kpr7), pth+he*(a81*kpth1+a84*kpth4+a85*kpth5+a86*kpth6+a87*kpth7), a, b, Q2, &kr8, &kth8, &kphi8, &kpr8, &kpth8);

            double a91=39632708.0/573591083.0, a94=-433636366.0/683701615.0, a95=-421739975.0/2616292301.0, a96=100302831.0/723423059.0, a97=790204164.0/839813087.0, a98=800635310.0/3783071287.0;
            geoRHS(r+he*(a91*kr1+a94*kr4+a95*kr5+a96*kr6+a97*kr7+a98*kr8), th+he*(a91*kth1+a94*kth4+a95*kth5+a96*kth6+a97*kth7+a98*kth8), pr+he*(a91*kpr1+a94*kpr4+a95*kpr5+a96*kpr6+a97*kpr7+a98*kpr8), pth+he*(a91*kpth1+a94*kpth4+a95*kpth5+a96*kpth6+a97*kpth7+a98*kpth8), a, b, Q2, &kr9, &kth9, &kphi9, &kpr9, &kpth9);

            double a101=246121993.0/1340847787.0, a104=-37695042795.0/15268766246.0, a105=-309121744.0/1061227803.0, a106=-12992083.0/490766935.0, a107=6005943493.0/2108947869.0, a108=393006217.0/1396673457.0, a109=123872331.0/1001029789.0;
            geoRHS(r+he*(a101*kr1+a104*kr4+a105*kr5+a106*kr6+a107*kr7+a108*kr8+a109*kr9), th+he*(a101*kth1+a104*kth4+a105*kth5+a106*kth6+a107*kth7+a108*kth8+a109*kth9), pr+he*(a101*kpr1+a104*kpr4+a105*kpr5+a106*kpr6+a107*kpr7+a108*kpr8+a109*kpr9), pth+he*(a101*kpth1+a104*kpth4+a105*kpth5+a106*kpth6+a107*kpth7+a108*kpth8+a109*kpth9), a, b, Q2, &kr10, &kth10, &kphi10, &kpr10, &kpth10);

            double a111=-1028468189.0/846180014.0, a114=8478235783.0/508512852.0, a115=1311729495.0/1432422823.0, a116=-10304129995.0/1701304382.0, a117=-48777925059.0/3047939560.0, a118=15336726248.0/1032824649.0, a119=-45442868181.0/3398467696.0, a1110=3065993473.0/597172653.0;
            geoRHS(r+he*(a111*kr1+a114*kr4+a115*kr5+a116*kr6+a117*kr7+a118*kr8+a119*kr9+a1110*kr10), th+he*(a111*kth1+a114*kth4+a115*kth5+a116*kth6+a117*kth7+a118*kth8+a119*kth9+a1110*kth10), pr+he*(a111*kpr1+a114*kpr4+a115*kpr5+a116*kpr6+a117*kpr7+a118*kpr8+a119*kpr9+a1110*kpr10), pth+he*(a111*kpth1+a114*kpth4+a115*kpth5+a116*kpth6+a117*kpth7+a118*kpth8+a119*kpth9+a1110*kpth10), a, b, Q2, &kr11, &kth11, &kphi11, &kpr11, &kpth11);

            double a121=185892177.0/718116043.0, a124=-3185094517.0/667107341.0, a125=-477755414.0/1098053517.0, a126=-703635378.0/230739211.0, a127=5731566787.0/1027545527.0, a128=5232866602.0/850066563.0, a129=-4093664535.0/808688257.0, a1210=3962137247.0/1805957418.0, a1211=65686358.0/487910083.0;
            geoRHS(r+he*(a121*kr1+a124*kr4+a125*kr5+a126*kr6+a127*kr7+a128*kr8+a129*kr9+a1210*kr10+a1211*kr11), th+he*(a121*kth1+a124*kth4+a125*kth5+a126*kth6+a127*kth7+a128*kth8+a129*kth9+a1210*kth10+a1211*kth11), pr+he*(a121*kpr1+a124*kpr4+a125*kpr5+a126*kpr6+a127*kpr7+a128*kpr8+a129*kpr9+a1210*kpr10+a1211*kpr11), pth+he*(a121*kpth1+a124*kpth4+a125*kpth5+a126*kpth6+a127*kpth7+a128*kpth8+a129*kpth9+a1210*kpth10+a1211*kpth11), a, b, Q2, &kr12, &kth12, &kphi12, &kpr12, &kpth12);

            double a131=403863854.0/491063109.0, a134=-5068492393.0/434740067.0, a135=-411421997.0/543043805.0, a136=652783627.0/914296604.0, a137=11173962825.0/925320556.0, a138=-13158990841.0/6184727034.0, a139=3936647629.0/1978049680.0, a1310=-160528059.0/685178525.0, a1311=248638103.0/1413531060.0;
            geoRHS(r+he*(a131*kr1+a134*kr4+a135*kr5+a136*kr6+a137*kr7+a138*kr8+a139*kr9+a1310*kr10+a1311*kr11), th+he*(a131*kth1+a134*kth4+a135*kth5+a136*kth6+a137*kth7+a138*kth8+a139*kth9+a1310*kth10+a1311*kth11), pr+he*(a131*kpr1+a134*kpr4+a135*kpr5+a136*kpr6+a137*kpr7+a138*kpr8+a139*kpr9+a1310*kpr10+a1311*kpr11), pth+he*(a131*kpth1+a134*kpth4+a135*kpth5+a136*kpth6+a137*kpth7+a138*kpth8+a139*kpth9+a1310*kpth10+a1311*kpth11), a, b, Q2, &kr13, &kth13, &kphi13, &kpr13, &kpth13);

            /* 8th-order solution weights */
            double bw1=14005451.0/335480064.0, bw6=-59238493.0/1068277825.0, bw7=181606767.0/758867731.0, bw8=561292985.0/797845732.0, bw9=-1041891430.0/1371343529.0, bw10=760417239.0/1151165299.0, bw11=118820643.0/751138087.0, bw12=-528747749.0/2220607170.0, bw13=1.0/4.0;

            /* 7th-order embedded weights */
            double bhat1=13451932.0/455176623.0, bhat6=-808719846.0/976000145.0, bhat7=1757004468.0/5645159321.0, bhat8=656045339.0/265891186.0, bhat9=-3867574721.0/1518517206.0, bhat10=465885868.0/322736535.0, bhat11=53011238.0/667516719.0, bhat12=2.0/45.0;

            double dr8   = he*(bw1*kr1+bw6*kr6+bw7*kr7+bw8*kr8+bw9*kr9+bw10*kr10+bw11*kr11+bw12*kr12+bw13*kr13);
            double dth8  = he*(bw1*kth1+bw6*kth6+bw7*kth7+bw8*kth8+bw9*kth9+bw10*kth10+bw11*kth11+bw12*kth12+bw13*kth13);
            double dpr8  = he*(bw1*kpr1+bw6*kpr6+bw7*kpr7+bw8*kpr8+bw9*kpr9+bw10*kpr10+bw11*kpr11+bw12*kpr12+bw13*kpr13);
            double dpth8 = he*(bw1*kpth1+bw6*kpth6+bw7*kpth7+bw8*kpth8+bw9*kpth9+bw10*kpth10+bw11*kpth11+bw12*kpth12+bw13*kpth13);

            double err_r   = he*((bw1-bhat1)*kr1+(bw6-bhat6)*kr6+(bw7-bhat7)*kr7+(bw8-bhat8)*kr8+(bw9-bhat9)*kr9+(bw10-bhat10)*kr10+(bw11-bhat11)*kr11+(bw12-bhat12)*kr12+bw13*kr13);
            double err_th  = he*((bw1-bhat1)*kth1+(bw6-bhat6)*kth6+(bw7-bhat7)*kth7+(bw8-bhat8)*kth8+(bw9-bhat9)*kth9+(bw10-bhat10)*kth10+(bw11-bhat11)*kth11+(bw12-bhat12)*kth12+bw13*kth13);
            double err_pr  = he*((bw1-bhat1)*kpr1+(bw6-bhat6)*kpr6+(bw7-bhat7)*kpr7+(bw8-bhat8)*kpr8+(bw9-bhat9)*kpr9+(bw10-bhat10)*kpr10+(bw11-bhat11)*kpr11+(bw12-bhat12)*kpr12+bw13*kpr13);
            double err_pth = he*((bw1-bhat1)*kpth1+(bw6-bhat6)*kpth6+(bw7-bhat7)*kpth7+(bw8-bhat8)*kpth8+(bw9-bhat9)*kpth9+(bw10-bhat10)*kpth10+(bw11-bhat11)*kpth11+(bw12-bhat12)*kpth12+bw13*kpth13);

            double sc_r   = atol + rtol * fmax(fabs(r),   fabs(r   + dr8));
            double sc_th  = atol + rtol * fmax(fabs(th),  fabs(th  + dth8));
            double sc_pr  = atol + rtol * fmax(fabs(pr),  fabs(pr  + dpr8));
            double sc_pth = atol + rtol * fmax(fabs(pth), fabs(pth + dpth8));

            double err_norm = sqrt(0.25 * ((err_r/sc_r)*(err_r/sc_r) + (err_th/sc_th)*(err_th/sc_th) + (err_pr/sc_pr)*(err_pr/sc_pr) + (err_pth/sc_pth)*(err_pth/sc_pth)));

            if (err_norm <= 1.0 || rejects >= max_reject) {
                r   += dr8;
                th  += dth8;
                phi += he*(bw1*kphi1+bw6*kphi6+bw7*kphi7+bw8*kphi8+bw9*kphi9+bw10*kphi10+bw11*kphi11+bw12*kphi12+bw13*kphi13);
                pr  += dpr8;
                pth += dpth8;
                accepted = true;

                if (err_norm > 1e-30) {
                    double factor = safety * pow(err_norm, -1.0/8.0);
                    factor = fmin(fmax(factor, 0.2), 5.0);
                    he *= factor;
                }
                he = fmin(fmax(he, h_min), h_max);
            } else {
                double factor = safety * pow(err_norm, -1.0/8.0);
                factor = fmax(factor, 0.2);
                he *= factor;
                he = fmax(he, h_min);
                rejects++;
            }
        }

        /* Hamiltonian constraint projection */
        projectHamiltonian(r, th, &pr, pth, a, b, Q2);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) { term_reason = 1; break; }

        record_crossing(output, crossing_base, &num_crossings,
                        i, oldR, oldTh, oldPhi, r, th, phi,
                        a, Q2, b, p.isco, p.disk_temp);

        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }

    ray_finalize(output, crossing_base, num_crossings,
                 r, th, phi, pr, pth, term_reason, steps_used);
}


/* ════════════════════════════════════════════════════════════
 *  KAHANLI8S RAY TRACE (Boyer-Lindquist coordinates)
 *  Kahan-Li 8th-order with Sundman time + compensated summation
 *  + symplectic corrector + Hamiltonian projection
 * ════════════════════════════════════════════════════════════ */

extern "C" __global__
void ray_trace_kahanli8s(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj);

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    int traj_base = 20;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;

    /* 4× step multiplier (matching render kernel) */
    int STEPS = (int)p.steps * 4;
    int term_reason = 0, steps_used = 0;

    /* Compensated summation accumulators */
    double r_comp = 0.0, th_comp = 0.0, phi_comp = 0.0;
    double pr_comp = 0.0, pth_comp = 0.0;

    /* Sundman / Mino time step */
    double dtau = sundman_dtau(a, Q2, rp, p.step_size, p.esc_radius, STEPS);

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;

        double oldR = r, oldTh = th, oldPhi = phi;

        /* Sundman-scaled step size */
        double he = sundman_physical_step(dtau, r, th, a, p.obs_dist);

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off + 0] = r; output[off + 1] = th;
            output[off + 2] = phi; output[off + 3] = he;
        }

        /* Kahan-Li s15odr8: 15 symmetric substeps with compensated summation */
        double dr_, dth_, dphi_, dpr_, dpth_;

        #define RT_KL8S_SUBSTEP(idx) { \
            geoRHS(r, th, pr, pth, a, b, Q2, \
                   &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
            rt_kahan_add(&r,   &r_comp,   he * RT_KL8S_D8[idx] * dr_); \
            rt_kahan_add(&th,  &th_comp,  he * RT_KL8S_D8[idx] * dth_); \
            rt_kahan_add(&phi, &phi_comp, he * RT_KL8S_D8[idx] * dphi_); \
            geoRHS(r, th, pr, pth, a, b, Q2, \
                   &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
            rt_kahan_add(&pr,  &pr_comp,  he * RT_KL8S_W8[idx] * dpr_); \
            rt_kahan_add(&pth, &pth_comp, he * RT_KL8S_W8[idx] * dpth_); \
        }

        RT_KL8S_SUBSTEP(0) RT_KL8S_SUBSTEP(1) RT_KL8S_SUBSTEP(2)
        RT_KL8S_SUBSTEP(3) RT_KL8S_SUBSTEP(4) RT_KL8S_SUBSTEP(5)
        RT_KL8S_SUBSTEP(6) RT_KL8S_SUBSTEP(7) RT_KL8S_SUBSTEP(6)
        RT_KL8S_SUBSTEP(5) RT_KL8S_SUBSTEP(4) RT_KL8S_SUBSTEP(3)
        RT_KL8S_SUBSTEP(2) RT_KL8S_SUBSTEP(1) RT_KL8S_SUBSTEP(0)

        #undef RT_KL8S_SUBSTEP

        /* Symplectic corrector (Wisdom 2006) */
        {
            double corr_eps = he * he / 24.0;
            double f_pr, f_pth;
            geoForce(r, th, pr, pth, a, b, Q2, &f_pr, &f_pth);
            pr  += corr_eps * f_pr;
            pth += corr_eps * f_pth;

            double v_r, v_th, v_phi;
            geoVelocity(r, th, pr, pth, a, b, Q2, &v_r, &v_th, &v_phi);
            rt_kahan_add(&r,   &r_comp,   corr_eps * v_r);
            rt_kahan_add(&th,  &th_comp,  corr_eps * v_th);
            rt_kahan_add(&phi, &phi_comp, corr_eps * v_phi);
        }

        /* Hamiltonian projection */
        projectHamiltonian(r, th, &pr, pth, a, b, Q2);
        pr_comp = 0.0;

        /* Pole reflection */
        if (th < 0.005) { th = 0.005; pth = fabs(pth); th_comp = 0.0; pth_comp = 0.0; }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); th_comp = 0.0; pth_comp = 0.0; }

        if (r <= rp * 1.01) { term_reason = 1; break; }

        record_crossing(output, crossing_base, &num_crossings,
                        i, oldR, oldTh, oldPhi, r, th, phi,
                        a, Q2, b, p.isco, p.disk_temp);

        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }

    ray_finalize(output, crossing_base, num_crossings,
                 r, th, phi, pr, pth, term_reason, steps_used);
}


/* ════════════════════════════════════════════════════════════
 *  KAHANLI8S_KS RAY TRACE (Kerr-Schild coordinates)
 *  Kahan-Li 8th-order with Sundman time + compensated summation
 *  + symplectic corrector + KS Hamiltonian projection
 * ════════════════════════════════════════════════════════════ */

extern "C" __global__
void ray_trace_kahanli8s_ks(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj);

    double a = p.spin;
    double Q2 = p.charge * p.charge;

    /* Transform p_r from BL to KS coordinates */
    transformBLtoKS(r, a, b, Q2, &pr);

    int traj_base = 20;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;

    /* 4× step multiplier (matching render kernel) */
    int STEPS = (int)p.steps * 4;
    int term_reason = 0, steps_used = 0;

    /* Compensated summation accumulators */
    double r_comp = 0.0, th_comp = 0.0, phi_comp = 0.0;
    double pr_comp = 0.0, pth_comp = 0.0;

    /* Sundman / Mino time step */
    double dtau = sundman_dtau(a, Q2, rp, p.step_size, p.esc_radius, STEPS);

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;

        double oldR = r, oldTh = th, oldPhi = phi;

        /* Sundman-scaled step size */
        double he = sundman_physical_step(dtau, r, th, a, p.obs_dist);

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off + 0] = r; output[off + 1] = th;
            output[off + 2] = phi; output[off + 3] = he;
        }

        /* Kahan-Li s15odr8: 15 symmetric substeps in KS coordinates */
        double dr_, dth_, dphi_, dpr_, dpth_;

        #define RT_KL8S_KS_SUBSTEP(idx) { \
            geoVelocityKS(r, th, pr, pth, a, b, Q2, \
                          &dr_, &dth_, &dphi_); \
            rt_kahan_add(&r,   &r_comp,   he * RT_KL8S_D8[idx] * dr_); \
            rt_kahan_add(&th,  &th_comp,  he * RT_KL8S_D8[idx] * dth_); \
            rt_kahan_add(&phi, &phi_comp, he * RT_KL8S_D8[idx] * dphi_); \
            geoForceKS(r, th, pr, pth, a, b, Q2, \
                       &dpr_, &dpth_); \
            rt_kahan_add(&pr,  &pr_comp,  he * RT_KL8S_W8[idx] * dpr_); \
            rt_kahan_add(&pth, &pth_comp, he * RT_KL8S_W8[idx] * dpth_); \
        }

        RT_KL8S_KS_SUBSTEP(0) RT_KL8S_KS_SUBSTEP(1) RT_KL8S_KS_SUBSTEP(2)
        RT_KL8S_KS_SUBSTEP(3) RT_KL8S_KS_SUBSTEP(4) RT_KL8S_KS_SUBSTEP(5)
        RT_KL8S_KS_SUBSTEP(6) RT_KL8S_KS_SUBSTEP(7) RT_KL8S_KS_SUBSTEP(6)
        RT_KL8S_KS_SUBSTEP(5) RT_KL8S_KS_SUBSTEP(4) RT_KL8S_KS_SUBSTEP(3)
        RT_KL8S_KS_SUBSTEP(2) RT_KL8S_KS_SUBSTEP(1) RT_KL8S_KS_SUBSTEP(0)

        #undef RT_KL8S_KS_SUBSTEP

        /* Symplectic corrector (Wisdom 2006) — KS version */
        {
            double corr_eps = he * he / 24.0;
            double f_pr, f_pth;
            geoForceKS(r, th, pr, pth, a, b, Q2, &f_pr, &f_pth);
            pr  += corr_eps * f_pr;
            pth += corr_eps * f_pth;

            double v_r, v_th, v_phi;
            geoVelocityKS(r, th, pr, pth, a, b, Q2, &v_r, &v_th, &v_phi);
            rt_kahan_add(&r,   &r_comp,   corr_eps * v_r);
            rt_kahan_add(&th,  &th_comp,  corr_eps * v_th);
            rt_kahan_add(&phi, &phi_comp, corr_eps * v_phi);
        }

        /* KS Hamiltonian projection */
        projectHamiltonianKS(r, th, &pr, pth, a, b, Q2);
        pr_comp = 0.0;

        /* Pole reflection */
        if (th < 0.005) { th = 0.005; pth = fabs(pth); th_comp = 0.0; pth_comp = 0.0; }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); th_comp = 0.0; pth_comp = 0.0; }

        /* KS horizon capture: well inside horizon (r ≤ 0.5·r₊) */
        if (r <= rp * 0.5) { term_reason = 1; break; }

        record_crossing(output, crossing_base, &num_crossings,
                        i, oldR, oldTh, oldPhi, r, th, phi,
                        a, Q2, b, p.isco, p.disk_temp);

        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }

    ray_finalize(output, crossing_base, num_crossings,
                 r, th, phi, pr, pth, term_reason, steps_used);
}


/* ════════════════════════════════════════════════════════════
 *  TAO + YOSHIDA4 RAY TRACE
 * ════════════════════════════════════════════════════════════ */

extern "C" __global__
void ray_trace_tao_yoshida4(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj);

    double rs = r, ths = th, phis = phi, prs = pr, pths = pth;

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    int traj_base = 20;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;
    int STEPS = (int)p.steps;
    int term_reason = 0, steps_used = 0;

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;
        double he = adaptive_step_tao(r, rp, p.step_size, p.obs_dist);

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off + 0] = r; output[off + 1] = th;
            output[off + 2] = phi; output[off + 3] = he;
        }

        double oldR = r, oldTh = th, oldPhi = phi;
        tao_yoshida4_step(&r, &th, &phi, &pr, &pth,
                          &rs, &ths, &phis, &prs, &pths,
                          a, b, Q2, he);

        projectHamiltonian(r, th, &pr, pth, a, b, Q2);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) { term_reason = 1; break; }

        record_crossing(output, crossing_base, &num_crossings,
                        i, oldR, oldTh, oldPhi, r, th, phi,
                        a, Q2, b, p.isco, p.disk_temp);

        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }

    ray_finalize(output, crossing_base, num_crossings,
                 r, th, phi, pr, pth, term_reason, steps_used);
}


/* ════════════════════════════════════════════════════════════
 *  TAO + YOSHIDA6 RAY TRACE
 * ════════════════════════════════════════════════════════════ */

extern "C" __global__
void ray_trace_tao_yoshida6(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj);

    double rs = r, ths = th, phis = phi, prs = pr, pths = pth;

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    int traj_base = 20;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;
    int STEPS = (int)p.steps;
    int term_reason = 0, steps_used = 0;

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;
        double he = adaptive_step_tao(r, rp, p.step_size, p.obs_dist);

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off + 0] = r; output[off + 1] = th;
            output[off + 2] = phi; output[off + 3] = he;
        }

        double oldR = r, oldTh = th, oldPhi = phi;
        tao_yoshida6_step(&r, &th, &phi, &pr, &pth,
                          &rs, &ths, &phis, &prs, &pths,
                          a, b, Q2, he);

        projectHamiltonian(r, th, &pr, pth, a, b, Q2);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) { term_reason = 1; break; }

        record_crossing(output, crossing_base, &num_crossings,
                        i, oldR, oldTh, oldPhi, r, th, phi,
                        a, Q2, b, p.isco, p.disk_temp);

        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }

    ray_finalize(output, crossing_base, num_crossings,
                 r, th, phi, pr, pth, term_reason, steps_used);
}


/* ════════════════════════════════════════════════════════════
 *  TAO + KAHAN-LI 8th-ORDER RAY TRACE
 * ════════════════════════════════════════════════════════════ */

extern "C" __global__
void ray_trace_tao_kahan_li8(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const RenderParams &p = *pp;

    double r, th, phi, pr, pth, b, rp, alpha_val, beta_val;
    int max_traj;
    ray_init(p, output, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha_val, &beta_val, &max_traj);

    double rs = r, ths = th, phis = phi, prs = pr, pths = pth;

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    int traj_base = 20;
    int crossing_base = traj_base + max_traj * 4;
    int num_crossings = 0;
    int STEPS = (int)p.steps;
    int term_reason = 0, steps_used = 0;

    for (int i = 0; i < STEPS; i++) {
        steps_used = i + 1;
        double he = adaptive_step_tao(r, rp, p.step_size, p.obs_dist);

        if (i < max_traj) {
            int off = traj_base + i * 4;
            output[off + 0] = r; output[off + 1] = th;
            output[off + 2] = phi; output[off + 3] = he;
        }

        double oldR = r, oldTh = th, oldPhi = phi;
        tao_kahan_li8_step(&r, &th, &phi, &pr, &pth,
                           &rs, &ths, &phis, &prs, &pths,
                           a, b, Q2, he);

        projectHamiltonian(r, th, &pr, pth, a, b, Q2);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) { term_reason = 1; break; }

        record_crossing(output, crossing_base, &num_crossings,
                        i, oldR, oldTh, oldPhi, r, th, phi,
                        a, Q2, b, p.isco, p.disk_temp);

        if (r > p.esc_radius) { term_reason = 2; break; }
        if (r < 0.5) { term_reason = 4; break; }
        if (r != r || th != th) { term_reason = 3; break; }
    }

    ray_finalize(output, crossing_base, num_crossings,
                 r, th, phi, pr, pth, term_reason, steps_used);
}
