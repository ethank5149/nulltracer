/* ============================================================
 *  DORMAND-PRINCE 8th-ORDER (RK8(7)) INTEGRATOR
 *  with ADAPTIVE STEP SIZE CONTROL
 *
 *  13 stages per step with Dormand-Prince (1981) coefficients.
 *  Uses the embedded 7th-order solution for local error estimation
 *  and automatic step size adjustment (accept/reject).
 *
 *  All geodesic integration in float64; color output in float32.
 *
 *  Reference: P.J. Prince & J.R. Dormand (1981),
 *  "High order embedded Runge-Kutta formulae",
 *  J. Comp. Appl. Math. 7(1), pp. 67-75.
 *
 *  Uses shared adaptive step sizing from adaptive_step.cu for
 *  initial step estimate (subsequent steps use embedded error
 *  estimator).
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"
#include "adaptive_step.cu"


extern "C" __global__
void trace_rkdp8(const RenderParams *pp, unsigned char *output, const float *skymap) {
    const RenderParams &p = *pp;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int W = (int)p.width, H = (int)p.height;
    if (ix >= W || iy >= H) return;

    double r, th, phi, pr, pth, b, rp;
    float alpha, beta;
    initRay(ix, iy, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta);

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_a = 0.0f;
    int disk_crossings = 0;
    int max_crossings = (int)p.disk_max_crossings;
    float base_alpha = (float)p.disk_alpha;
    bool done = false;

    int STEPS = (int)p.steps;
    int show_disk = (int)p.show_disk;
    int bg_mode = (int)p.bg_mode;
    int star_layers = (int)p.star_layers;
    int show_grid = (int)p.show_grid;

    /* Adaptive step control parameters */
    double atol = 1.0e-8;   /* absolute tolerance */
    double rtol = 1.0e-8;   /* relative tolerance */
    double safety = 0.9;    /* safety factor for step adjustment */
    double h_min = 0.001;   /* minimum step size */
    double h_max = 2.0;     /* maximum step size */
    int max_reject = 4;     /* max consecutive rejections before forcing accept */

    /* Initial step size estimate (shared function) */
    double he = adaptive_step_rkdp8_initial(r, rp, p.step_size, p.obs_dist, h_min, h_max);

    for (int i = 0; i < STEPS; i++) {
        if (done) break;

        double oldTh = th, oldR = r, oldPhi = phi;
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

            /* Stage 1 */
            geoRHS(r, th, pr, pth, a, b, Q2, &kr1, &kth1, &kphi1, &kpr1, &kpth1);

            /* Stage 2: c2 = 1/18 */
            geoRHS(r+he*kr1/18.0, th+he*kth1/18.0,
                   pr+he*kpr1/18.0, pth+he*kpth1/18.0,
                   a, b, Q2, &kr2, &kth2, &kphi2, &kpr2, &kpth2);

            /* Stage 3: c3 = 1/12 */
            geoRHS(r+he*(kr1/48.0+kr2/16.0), th+he*(kth1/48.0+kth2/16.0),
                   pr+he*(kpr1/48.0+kpr2/16.0), pth+he*(kpth1/48.0+kpth2/16.0),
                   a, b, Q2, &kr3, &kth3, &kphi3, &kpr3, &kpth3);

            /* Stage 4: c4 = 1/8 */
            geoRHS(r+he*(kr1/32.0+kr3*3.0/32.0), th+he*(kth1/32.0+kth3*3.0/32.0),
                   pr+he*(kpr1/32.0+kpr3*3.0/32.0), pth+he*(kpth1/32.0+kpth3*3.0/32.0),
                   a, b, Q2, &kr4, &kth4, &kphi4, &kpr4, &kpth4);

            /* Stage 5: c5 = 5/16 */
            geoRHS(r+he*(kr1*5.0/16.0-kr3*75.0/64.0+kr4*75.0/64.0),
                   th+he*(kth1*5.0/16.0-kth3*75.0/64.0+kth4*75.0/64.0),
                   pr+he*(kpr1*5.0/16.0-kpr3*75.0/64.0+kpr4*75.0/64.0),
                   pth+he*(kpth1*5.0/16.0-kpth3*75.0/64.0+kpth4*75.0/64.0),
                   a, b, Q2, &kr5, &kth5, &kphi5, &kpr5, &kpth5);

            /* Stage 6: c6 = 3/8 */
            geoRHS(r+he*(kr1*3.0/80.0+kr4*3.0/16.0+kr5*3.0/20.0),
                   th+he*(kth1*3.0/80.0+kth4*3.0/16.0+kth5*3.0/20.0),
                   pr+he*(kpr1*3.0/80.0+kpr4*3.0/16.0+kpr5*3.0/20.0),
                   pth+he*(kpth1*3.0/80.0+kpth4*3.0/16.0+kpth5*3.0/20.0),
                   a, b, Q2, &kr6, &kth6, &kphi6, &kpr6, &kpth6);

            /* Stage 7: c7 = 59/400 */
            double a71=29443841.0/614563906.0, a74=77736538.0/692538347.0;
            double a75=-28693883.0/1125000000.0, a76=23124283.0/1800000000.0;
            geoRHS(r+he*(a71*kr1+a74*kr4+a75*kr5+a76*kr6),
                   th+he*(a71*kth1+a74*kth4+a75*kth5+a76*kth6),
                   pr+he*(a71*kpr1+a74*kpr4+a75*kpr5+a76*kpr6),
                   pth+he*(a71*kpth1+a74*kpth4+a75*kpth5+a76*kpth6),
                   a, b, Q2, &kr7, &kth7, &kphi7, &kpr7, &kpth7);

            /* Stage 8: c8 = 93/200 */
            double a81=16016141.0/946692911.0, a84=61564180.0/158732637.0;
            double a85=22789713.0/633445777.0, a86=545815736.0/2771057229.0;
            double a87=-180193667.0/1043307555.0;
            geoRHS(r+he*(a81*kr1+a84*kr4+a85*kr5+a86*kr6+a87*kr7),
                   th+he*(a81*kth1+a84*kth4+a85*kth5+a86*kth6+a87*kth7),
                   pr+he*(a81*kpr1+a84*kpr4+a85*kpr5+a86*kpr6+a87*kpr7),
                   pth+he*(a81*kpth1+a84*kpth4+a85*kpth5+a86*kpth6+a87*kpth7),
                   a, b, Q2, &kr8, &kth8, &kphi8, &kpr8, &kpth8);

            /* Stage 9 */
            double a91=39632708.0/573591083.0, a94=-433636366.0/683701615.0;
            double a95=-421739975.0/2616292301.0, a96=100302831.0/723423059.0;
            double a97=790204164.0/839813087.0, a98=800635310.0/3783071287.0;
            geoRHS(r+he*(a91*kr1+a94*kr4+a95*kr5+a96*kr6+a97*kr7+a98*kr8),
                   th+he*(a91*kth1+a94*kth4+a95*kth5+a96*kth6+a97*kth7+a98*kth8),
                   pr+he*(a91*kpr1+a94*kpr4+a95*kpr5+a96*kpr6+a97*kpr7+a98*kpr8),
                   pth+he*(a91*kpth1+a94*kpth4+a95*kpth5+a96*kpth6+a97*kpth7+a98*kpth8),
                   a, b, Q2, &kr9, &kth9, &kphi9, &kpr9, &kpth9);

            /* Stage 10: c10 = 13/20 */
            double a101=246121993.0/1340847787.0, a104=-37695042795.0/15268766246.0;
            double a105=-309121744.0/1061227803.0, a106=-12992083.0/490766935.0;
            double a107=6005943493.0/2108947869.0, a108=393006217.0/1396673457.0;
            double a109=123872331.0/1001029789.0;
            geoRHS(r+he*(a101*kr1+a104*kr4+a105*kr5+a106*kr6+a107*kr7+a108*kr8+a109*kr9),
                   th+he*(a101*kth1+a104*kth4+a105*kth5+a106*kth6+a107*kth7+a108*kth8+a109*kth9),
                   pr+he*(a101*kpr1+a104*kpr4+a105*kpr5+a106*kpr6+a107*kpr7+a108*kpr8+a109*kpr9),
                   pth+he*(a101*kpth1+a104*kpth4+a105*kpth5+a106*kpth6+a107*kpth7+a108*kpth8+a109*kpth9),
                   a, b, Q2, &kr10, &kth10, &kphi10, &kpr10, &kpth10);

            /* Stage 11 */
            double a111=-1028468189.0/846180014.0, a114=8478235783.0/508512852.0;
            double a115=1311729495.0/1432422823.0, a116=-10304129995.0/1701304382.0;
            double a117=-48777925059.0/3047939560.0, a118=15336726248.0/1032824649.0;
            double a119=-45442868181.0/3398467696.0, a1110=3065993473.0/597172653.0;
            geoRHS(r+he*(a111*kr1+a114*kr4+a115*kr5+a116*kr6+a117*kr7+a118*kr8+a119*kr9+a1110*kr10),
                   th+he*(a111*kth1+a114*kth4+a115*kth5+a116*kth6+a117*kth7+a118*kth8+a119*kth9+a1110*kth10),
                   pr+he*(a111*kpr1+a114*kpr4+a115*kpr5+a116*kpr6+a117*kpr7+a118*kpr8+a119*kpr9+a1110*kpr10),
                   pth+he*(a111*kpth1+a114*kpth4+a115*kpth5+a116*kpth6+a117*kpth7+a118*kpth8+a119*kpth9+a1110*kpth10),
                   a, b, Q2, &kr11, &kth11, &kphi11, &kpr11, &kpth11);

            /* Stage 12: c12 = 1 */
            double a121=185892177.0/718116043.0, a124=-3185094517.0/667107341.0;
            double a125=-477755414.0/1098053517.0, a126=-703635378.0/230739211.0;
            double a127=5731566787.0/1027545527.0, a128=5232866602.0/850066563.0;
            double a129=-4093664535.0/808688257.0, a1210=3962137247.0/1805957418.0;
            double a1211=65686358.0/487910083.0;
            geoRHS(r+he*(a121*kr1+a124*kr4+a125*kr5+a126*kr6+a127*kr7+a128*kr8+a129*kr9+a1210*kr10+a1211*kr11),
                   th+he*(a121*kth1+a124*kth4+a125*kth5+a126*kth6+a127*kth7+a128*kth8+a129*kth9+a1210*kth10+a1211*kth11),
                   pr+he*(a121*kpr1+a124*kpr4+a125*kpr5+a126*kpr6+a127*kpr7+a128*kpr8+a129*kpr9+a1210*kpr10+a1211*kpr11),
                   pth+he*(a121*kpth1+a124*kpth4+a125*kpth5+a126*kpth6+a127*kpth7+a128*kpth8+a129*kpth9+a1210*kpth10+a1211*kpth11),
                   a, b, Q2, &kr12, &kth12, &kphi12, &kpr12, &kpth12);

            /* Stage 13: c13 = 1 */
            double a131=403863854.0/491063109.0, a134=-5068492393.0/434740067.0;
            double a135=-411421997.0/543043805.0, a136=652783627.0/914296604.0;
            double a137=11173962825.0/925320556.0, a138=-13158990841.0/6184727034.0;
            double a139=3936647629.0/1978049680.0, a1310=-160528059.0/685178525.0;
            double a1311=248638103.0/1413531060.0;
            geoRHS(r+he*(a131*kr1+a134*kr4+a135*kr5+a136*kr6+a137*kr7+a138*kr8+a139*kr9+a1310*kr10+a1311*kr11),
                   th+he*(a131*kth1+a134*kth4+a135*kth5+a136*kth6+a137*kth7+a138*kth8+a139*kth9+a1310*kth10+a1311*kth11),
                   pr+he*(a131*kpr1+a134*kpr4+a135*kpr5+a136*kpr6+a137*kpr7+a138*kpr8+a139*kpr9+a1310*kpr10+a1311*kpr11),
                   pth+he*(a131*kpth1+a134*kpth4+a135*kpth5+a136*kpth6+a137*kpth7+a138*kpth8+a139*kpth9+a1310*kpth10+a1311*kpth11),
                   a, b, Q2, &kr13, &kth13, &kphi13, &kpr13, &kpth13);

            /* ── 8th-order solution weights ────────────────────── */
            double bw1=14005451.0/335480064.0;
            double bw6=-59238493.0/1068277825.0, bw7=181606767.0/758867731.0;
            double bw8=561292985.0/797845732.0, bw9=-1041891430.0/1371343529.0;
            double bw10=760417239.0/1151165299.0, bw11=118820643.0/751138087.0;
            double bw12=-528747749.0/2220607170.0, bw13=1.0/4.0;

            /* ── 7th-order embedded weights (Prince & Dormand 1981) ── */
            double bhat1=13451932.0/455176623.0;
            double bhat6=-808719846.0/976000145.0, bhat7=1757004468.0/5645159321.0;
            double bhat8=656045339.0/265891186.0, bhat9=-3867574721.0/1518517206.0;
            double bhat10=465885868.0/322736535.0, bhat11=53011238.0/667516719.0;
            double bhat12=2.0/45.0;
            /* bhat13 = 0 */

            /* ── 8th-order update (tentative) ──────────────────── */
            double dr8   = he*(bw1*kr1+bw6*kr6+bw7*kr7+bw8*kr8+bw9*kr9+bw10*kr10+bw11*kr11+bw12*kr12+bw13*kr13);
            double dth8  = he*(bw1*kth1+bw6*kth6+bw7*kth7+bw8*kth8+bw9*kth9+bw10*kth10+bw11*kth11+bw12*kth12+bw13*kth13);
            double dpr8  = he*(bw1*kpr1+bw6*kpr6+bw7*kpr7+bw8*kpr8+bw9*kpr9+bw10*kpr10+bw11*kpr11+bw12*kpr12+bw13*kpr13);
            double dpth8 = he*(bw1*kpth1+bw6*kpth6+bw7*kpth7+bw8*kpth8+bw9*kpth9+bw10*kpth10+bw11*kpth11+bw12*kpth12+bw13*kpth13);

            /* ── Local error estimate (8th - 7th order difference) ── */
            double err_r   = he*((bw1-bhat1)*kr1+(bw6-bhat6)*kr6+(bw7-bhat7)*kr7
                                +(bw8-bhat8)*kr8+(bw9-bhat9)*kr9+(bw10-bhat10)*kr10
                                +(bw11-bhat11)*kr11+(bw12-bhat12)*kr12+bw13*kr13);
            double err_th  = he*((bw1-bhat1)*kth1+(bw6-bhat6)*kth6+(bw7-bhat7)*kth7
                                +(bw8-bhat8)*kth8+(bw9-bhat9)*kth9+(bw10-bhat10)*kth10
                                +(bw11-bhat11)*kth11+(bw12-bhat12)*kth12+bw13*kth13);
            double err_pr  = he*((bw1-bhat1)*kpr1+(bw6-bhat6)*kpr6+(bw7-bhat7)*kpr7
                                +(bw8-bhat8)*kpr8+(bw9-bhat9)*kpr9+(bw10-bhat10)*kpr10
                                +(bw11-bhat11)*kpr11+(bw12-bhat12)*kpr12+bw13*kpr13);
            double err_pth = he*((bw1-bhat1)*kpth1+(bw6-bhat6)*kpth6+(bw7-bhat7)*kpth7
                                +(bw8-bhat8)*kpth8+(bw9-bhat9)*kpth9+(bw10-bhat10)*kpth10
                                +(bw11-bhat11)*kpth11+(bw12-bhat12)*kpth12+bw13*kpth13);

            /* Scaled error norm (mixed absolute/relative tolerance) */
            double sc_r   = atol + rtol * fmax(fabs(r),   fabs(r   + dr8));
            double sc_th  = atol + rtol * fmax(fabs(th),  fabs(th  + dth8));
            double sc_pr  = atol + rtol * fmax(fabs(pr),  fabs(pr  + dpr8));
            double sc_pth = atol + rtol * fmax(fabs(pth), fabs(pth + dpth8));

            double err_norm = sqrt(0.25 * (
                (err_r/sc_r)*(err_r/sc_r) + (err_th/sc_th)*(err_th/sc_th) +
                (err_pr/sc_pr)*(err_pr/sc_pr) + (err_pth/sc_pth)*(err_pth/sc_pth)
            ));

            if (err_norm <= 1.0 || rejects >= max_reject) {
                /* ── Accept step: advance state with 8th-order solution ── */
                r   += dr8;
                th  += dth8;
                phi += he*(bw1*kphi1+bw6*kphi6+bw7*kphi7+bw8*kphi8+bw9*kphi9+bw10*kphi10+bw11*kphi11+bw12*kphi12+bw13*kphi13);
                pr  += dpr8;
                pth += dpth8;
                accepted = true;

                /* Grow step for next iteration (PI controller) */
                if (err_norm > 1e-30) {
                    double factor = safety * pow(err_norm, -1.0/8.0);
                    factor = fmin(fmax(factor, 0.2), 5.0);  /* limit growth/shrink */
                    he *= factor;
                }
                he = fmin(fmax(he, h_min), h_max);
            } else {
                /* ── Reject step: shrink and retry ── */
                double factor = safety * pow(err_norm, -1.0/8.0);
                factor = fmax(factor, 0.2);  /* don't shrink too aggressively */
                he *= factor;
                he = fmax(he, h_min);
                rejects++;
            }
        } /* end while (!accepted) */

        /* Pole reflection */
        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        /* Volumetric emission: hot corona + relativistic jet */
        if (acc_a < 0.99f) {
            accumulate_volume_emission(r, th, he, a, (double)p.isco, p.disk_outer,
                                       &acc_r, &acc_g, &acc_b, &acc_a);
        }

        /* ── Termination conditions ──────────────────────── */

        if (r <= rp * 1.01) {
            blendColor(0.0f, 0.0f, 0.0f, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            done = true; break;
        }
        if (show_disk && acc_a < 0.99f) {
            double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
            if (cross < 0.0 && disk_crossings < max_crossings) {
                double f = fmin(fmax(fabs(oldTh - PI * 0.5) /
                           fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                double r_hit = oldR + f * (r - oldR);
                float dr_f = (float)r_hit;
                float dphi_f = (float)(oldPhi + f * (phi - oldPhi));

                float g = compute_g_factor_extended(r_hit, a, Q2, b, (double)p.isco);

                float dcr, dcg, dcb;
                diskColor(dr_f, dphi_f, (float)a,
                         (float)p.isco, (float)p.disk_outer, (float)p.disk_temp,
                         g, (int)p.doppler_boost,
                         &dcr, &dcg, &dcb);
                float crossing_alpha = base_alpha;
                blendColor(dcr, dcg, dcb, crossing_alpha, &acc_r, &acc_g, &acc_b, &acc_a);
                disk_crossings++;
            }
        }
        if (r > p.esc_radius) {
            double frac = fmin(fmax((p.esc_radius - oldR) /
                          fmax(r - oldR, 1e-14), 0.0), 1.0);
            double fth = oldTh + (th - oldTh) * frac;
            double fph = oldPhi + (phi - oldPhi) * frac;
            float dx, dy, dz;
            sphereDir(fth, fph, &dx, &dy, &dz);
            float bgr, bgg, bgb;
            background(dx, dy, dz, bg_mode, star_layers, show_grid,
                       skymap, (int)p.sky_width, (int)p.sky_height,
                       &bgr, &bgg, &bgb);
            if (acc_a < 1.0f) {
                blendColor(bgr, bgg, bgb, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            }
            done = true; break;
        }
        if (r < 0.5 || r != r || th != th) { done = true; break; }
    }

    float cr = acc_r, cg = acc_g, cb = acc_b;
    float ux = 2.0f * (ix + 0.5f) / (float)W  - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / (float)H - 1.0f;
    postProcess(&cr, &cg, &cb, alpha, beta, p, ux, uy);

    int idx = (iy * W + ix) * 3;
    output[idx + 0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx + 1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx + 2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
