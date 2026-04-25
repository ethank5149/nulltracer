/* ============================================================
 *  RKN 8(6) - Runge-Kutta-Nystrm 8th-order integrator
 *  with ADAPTIVE STEP SIZE CONTROL
 *
 *  Uses the second-order Nystrm formulation for the geodesic
 *  equations:
 *    q'' = a(q, q')
 *
 *  where q = (r, ) are non-cyclic coordinates and q' are
 *  their affine-parameter velocities.  The cyclic coordinate
 *   is integrated from accumulated velocity stages.
 *
 *  11 acceleration evaluations per step (via geoAccel), with
 *  embedded 6th-order error estimate.  The Nystrm structure
 *  gives O(h) position accuracy per force evaluation,
 *  achieving 8th order with fewer evaluations than a first-
 *  order RK method of equal order.
 *
 *  Reference: J.R. Dormand, M.E.A. El-Mikkawy, P.J. Prince
 *  (1987), "High-Order Embedded Runge-Kutta-Nystrm Formulae,"
 *  IMA J. Numer. Anal. 7, 423430.
 *
 *  Tableau adapted for y'' = f(y, y') by tracking velocity
 *  stages alongside position stages:
 *    Q_i  = q + c_i h v + h  _{ij} f_j    (positions)
 *    V_i  = v + h  a_{ij} f_j                (velocities)
 *    f_i  = geoAccel(Q_i, V_i)                (accelerations)
 *
 *  CUDA optimizations:
 *    - geoAccel reuses the same metric + derivative computation
 *      as geoForce but skips the separate velocity evaluation,
 *      saving ~15 FLOPs per evaluation vs geoRHS
 *    - 11 evaluations vs 13 for RKDP8 (15% fewer per step)
 *    -  accumulated from velocity stages only; excluded from
 *      error norm (b is conserved)
 *    - Velocity-conversion (vp) done only inside geoAccel;
 *      the main loop works entirely in (q, v) space
 *    - Error exponent -1/8 for 8th-order optimal step control
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"
#include "adaptive_step.cu"


extern "C" __global__
void trace_rkn86(const RenderParams *pp, unsigned char *output, const float *skymap, unsigned int *progress_counter) {
    const RenderParams &p = *pp;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int W = (int)p.width, H = (int)p.height;
    if (ix >= W || iy >= H) return;

    int SAMPLES = (int)p.aa_samples;
    if (SAMPLES < 1) SAMPLES = 1;

    float final_r = 0.0f, final_g = 0.0f, final_b = 0.0f;

    for (int s = 0; s < SAMPLES; s++) {
        float jitter_x = (SAMPLES == 1) ? 0.0f : (hash2((float)ix + s*1.3f, (float)iy + s*1.7f) - 0.5f);
        float jitter_y = (SAMPLES == 1) ? 0.0f : (hash2((float)ix + s*2.3f, (float)iy + s*2.9f) - 0.5f);

        double rr, th, phi, pr, pth, b, rp;
        float alpha, beta;
        if (!initRayJittered(ix, iy, jitter_x, jitter_y, p, &rr, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta)) {
            continue;
        }

        double a = p.spin;
        double Q2 = p.charge * p.charge;
        float F_peak = novikov_thorne_peak(a, (double)p.isco);
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

        double atol = 1.0e-9, rtol = 1.0e-9, safety = 0.9;
        double h_min = 0.0005, h_max = 2.5;
        int max_reject = 4;
        double he = adaptive_step_rkn86_initial(rr, rp, p.step_size, p.obs_dist, h_min, h_max);

        /* Convert initial momenta to velocities for Nystrm formulation */
        {
            double sth = sin(th), cth = cos(th);
            double s2 = sth * sth + S2_EPS;
            double a2 = a * a, r2 = rr * rr;
            double sig = r2 + a2 * cth * cth;
            double del = r2 - 2.0 * rr + a2 + Q2;
            double sdel = fmax(del, 1e-14);
            /* vr = g^rr * pr = (/) * pr */
            double vr  = (sdel / sig) * pr;
            /* vth = g^ * p = (1/) * p */
            double vth = pth / sig;
            pr = vr;   /* reuse pr/pth as velocity storage */
            pth = vth;
        }
        /* From this point: pr = vr, pth = vth (velocities, not momenta) */

        for (int i = 0; i < STEPS; i++) {
            if (done) break;

            double oldTh = th, oldR = rr, oldPhi = phi;
            double oldVr = pr, oldVth = pth;
            int rejects = 0;
            bool accepted = false;

            while (!accepted) {
                double h2 = he * he;

                /* DEP RKN 8(6) - 11 stages
                 *
                 * Nodes c[i]:
                 *   c1=0, c2=1/20, c3=1/10, c4=3/20, c5=297/1000,
                 *   c6=594/1000, c7=891/1000, c8=1, c9=1,
                 *   c10=297/1000, c11=594/1000
                 *
                 * Position: Q_i = q + c_i*h*v + h  _{ij} f_j
                 * Velocity: V_i = v + h  a_{ij} f_j
                 * f_i = geoAccel(Q_i, V_i)
                 */

                double fr1, fth1, vphi1;
                geoAccel(rr, th, pr, pth, a, b, Q2, &fr1, &fth1, &vphi1);

                /* Stage 2: c=1/20 */
                double fr2, fth2, vphi2;
                {
                    double c2 = 0.05;
                    double abar21 = 1.0/800.0;
                    double a21    = 1.0/20.0;
                    double Qr  = rr + c2*he*pr + h2*abar21*fr1;
                    double Qth = th + c2*he*pth + h2*abar21*fth1;
                    double Vr  = pr + he*a21*fr1;
                    double Vth = pth + he*a21*fth1;
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr2, &fth2, &vphi2);
                }

                /* Stage 3: c=1/10 */
                double fr3, fth3, vphi3;
                {
                    double c3 = 0.1;
                    double abar31 = 1.0/600.0, abar32 = 1.0/200.0;
                    double a31    = 1.0/60.0,  a32    = 1.0/20.0;
                    double Qr  = rr + c3*he*pr + h2*(abar31*fr1 + abar32*fr2);
                    double Qth = th + c3*he*pth + h2*(abar31*fth1 + abar32*fth2);
                    double Vr  = pr + he*(a31*fr1 + a32*fr2);
                    double Vth = pth + he*(a31*fth1 + a32*fth2);
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr3, &fth3, &vphi3);
                }

                /* Stage 4: c=3/20 */
                double fr4, fth4, vphi4;
                {
                    double c4 = 0.15;
                    double abar41 = 3.0/1600.0, abar43 = 9.0/1600.0;
                    double a41    = 3.0/80.0,    a43   = 9.0/80.0;
                    double Qr  = rr + c4*he*pr + h2*(abar41*fr1 + abar43*fr3);
                    double Qth = th + c4*he*pth + h2*(abar41*fth1 + abar43*fth3);
                    double Vr  = pr + he*(a41*fr1 + a43*fr3);
                    double Vth = pth + he*(a41*fth1 + a43*fth3);
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr4, &fth4, &vphi4);
                }

                /* Stage 5: c=297/1000 */
                double fr5, fth5, vphi5;
                {
                    double c5 = 0.297;
                    double abar51 =  102411552.0/2321577253.0;
                    double abar53 = -173403330.0/1386796801.0;
                    double abar54 =  333257550.0/1978595843.0;
                    double a51    =  5765175.0/28480768.0;
                    double a53    = -1221075.0/14240384.0;
                    double a54    =  1755675.0/14240384.0;
                    double Qr  = rr + c5*he*pr + h2*(abar51*fr1 + abar53*fr3 + abar54*fr4);
                    double Qth = th + c5*he*pth + h2*(abar51*fth1 + abar53*fth3 + abar54*fth4);
                    double Vr  = pr + he*(a51*fr1 + a53*fr3 + a54*fr4);
                    double Vth = pth + he*(a51*fth1 + a53*fth3 + a54*fth4);
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr5, &fth5, &vphi5);
                }

                /* Stage 6: c=594/1000 */
                double fr6, fth6, vphi6;
                {
                    double c6 = 0.594;
                    double abar61 = -13139384.0/75471875.0;
                    double abar63 =  76416012.0/150943750.0;
                    double abar64 = -13416708.0/150943750.0;
                    double abar65 =  14829580.0/75471875.0;
                    double a61 = -57487825.0/277700352.0;
                    double a63 =  242742375.0/277700352.0;
                    double a64 = -30223425.0/69425088.0;
                    double a65 =  81137500.0/138850176.0;
                    double Qr  = rr + c6*he*pr + h2*(abar61*fr1 + abar63*fr3 + abar64*fr4 + abar65*fr5);
                    double Qth = th + c6*he*pth + h2*(abar61*fth1 + abar63*fth3 + abar64*fth4 + abar65*fth5);
                    double Vr  = pr + he*(a61*fr1 + a63*fr3 + a64*fr4 + a65*fr5);
                    double Vth = pth + he*(a61*fth1 + a63*fth3 + a64*fth4 + a65*fth5);
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr6, &fth6, &vphi6);
                }

                /* Stage 7: c=891/1000 */
                double fr7, fth7, vphi7;
                {
                    double c7 = 0.891;
                    double abar71 =  40209893.0/100913500.0;
                    double abar73 = -280053102.0/201827000.0;
                    double abar74 =  111273.0/43790.0;
                    double abar75 = -20078613.0/100913500.0;
                    double abar76 =  53611040.0/201827000.0;
                    double a71 =  85327045.0/328477632.0;
                    double a73 = -351927465.0/328477632.0;
                    double a74 =  466043025.0/328477632.0;
                    double a75 = -29626250.0/82119408.0;
                    double a76 =  109755392.0/328477632.0;
                    double Qr  = rr + c7*he*pr + h2*(abar71*fr1 + abar73*fr3 + abar74*fr4 + abar75*fr5 + abar76*fr6);
                    double Qth = th + c7*he*pth + h2*(abar71*fth1 + abar73*fth3 + abar74*fth4 + abar75*fth5 + abar76*fth6);
                    double Vr  = pr + he*(a71*fr1 + a73*fr3 + a74*fr4 + a75*fr5 + a76*fr6);
                    double Vth = pth + he*(a71*fth1 + a73*fth3 + a74*fth4 + a75*fth5 + a76*fth6);
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr7, &fth7, &vphi7);
                }

                /* Stage 8: c=1 */
                double fr8, fth8, vphi8;
                {
                    double c8 = 1.0;
                    double abar81 =  23.0/320.0;
                    double abar85 =  11.0/320.0;
                    double abar86 =  297.0/1600.0;
                    double abar87 =  1000.0/4455.0;
                    double a81 =  13.0/288.0;
                    double a85 =  32.0/125.0;
                    double a86 =  31213.0/144000.0;
                    double a87 =  2401.0/12960.0;
                    double Qr  = rr + c8*he*pr + h2*(abar81*fr1 + abar85*fr5 + abar86*fr6 + abar87*fr7);
                    double Qth = th + c8*he*pth + h2*(abar81*fth1 + abar85*fth5 + abar86*fth6 + abar87*fth7);
                    double Vr  = pr + he*(a81*fr1 + a85*fr5 + a86*fr6 + a87*fr7);
                    double Vth = pth + he*(a81*fth1 + a85*fth5 + a86*fth6 + a87*fth7);
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr8, &fth8, &vphi8);
                }

                /* Stage 9: c=1 (for embedded error) */
                double fr9, fth9, vphi9;
                {
                    double c9 = 1.0;
                    double abar91 =  263.0/3780.0;
                    double abar95 =  176.0/5625.0;
                    double abar96 =  15551.0/75600.0;
                    double abar97 =  1000.0/4455.0;
                    double abar98 =  -1.0/5040.0;
                    double a91 =  263.0/2520.0;
                    double a95 =  176.0/3125.0;
                    double a96 =  15551.0/45360.0;
                    double a97 =  500.0/1323.0;
                    double a98 =  -1.0/3360.0;
                    double Qr  = rr + c9*he*pr + h2*(abar91*fr1 + abar95*fr5 + abar96*fr6 + abar97*fr7 + abar98*fr8);
                    double Qth = th + c9*he*pth + h2*(abar91*fth1 + abar95*fth5 + abar96*fth6 + abar97*fth7 + abar98*fth8);
                    double Vr  = pr + he*(a91*fr1 + a95*fr5 + a96*fr6 + a97*fr7 + a98*fr8);
                    double Vth = pth + he*(a91*fth1 + a95*fth5 + a96*fth6 + a97*fth7 + a98*fth8);
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr9, &fth9, &vphi9);
                }

                /* Stage 10: c=0.297 (for error) */
                double fr10, fth10, vphi10;
                {
                    double c10 = 0.297;
                    double abar101 =  0.008267990737899080;
                    double abar105 = -0.005034627335040000;
                    double abar106 =  0.019870189447230000;
                    double abar107 =  0.017633040233670000;
                    double abar108 = -0.000061736684755100;
                    double abar109 =  0.003405243131495600;
                    double a101 =  0.069000462476000000;
                    double a105 = -0.026419740190680000;
                    double a106 =  0.135085363779870000;
                    double a107 =  0.119058543697180000;
                    double a108 = -0.000330802289773200;
                    double a109 =  0.000505773087303000;
                    double Qr  = rr + c10*he*pr + h2*(abar101*fr1 + abar105*fr5 + abar106*fr6 + abar107*fr7 + abar108*fr8 + abar109*fr9);
                    double Qth = th + c10*he*pth + h2*(abar101*fth1 + abar105*fth5 + abar106*fth6 + abar107*fth7 + abar108*fth8 + abar109*fth9);
                    double Vr  = pr + he*(a101*fr1 + a105*fr5 + a106*fr6 + a107*fr7 + a108*fr8 + a109*fr9);
                    double Vth = pth + he*(a101*fth1 + a105*fth5 + a106*fth6 + a107*fth7 + a108*fth8 + a109*fth9);
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr10, &fth10, &vphi10);
                }

                /* Stage 11: c=0.594 (for error) */
                double fr11, fth11, vphi11;
                {
                    double c11 = 0.594;
                    double abar111 = -0.022587870192050000;
                    double abar115 =  0.052753600576000000;
                    double abar116 =  0.069527635459700000;
                    double abar117 =  0.005040533227900000;
                    double abar118 = -0.001234700696000000;
                    double abar119 =  0.013700604032450000;
                    double abar1110= 0.059345197592000000;
                    double a111 = -0.056780704825580000;
                    double a115 =  0.280498508928000000;
                    double a116 =  0.344254498697210000;
                    double a117 =  0.024474200117450000;
                    double a118 = -0.005851244640000000;
                    double a119 =  0.057614413076920000;
                    double a1110=  0.049790328646000000;
                    double Qr  = rr + c11*he*pr + h2*(abar111*fr1 + abar115*fr5 + abar116*fr6 + abar117*fr7 + abar118*fr8 + abar119*fr9 + abar1110*fr10);
                    double Qth = th + c11*he*pth + h2*(abar111*fth1 + abar115*fth5 + abar116*fth6 + abar117*fth7 + abar118*fth8 + abar119*fth9 + abar1110*fth10);
                    double Vr  = pr + he*(a111*fr1 + a115*fr5 + a116*fr6 + a117*fr7 + a118*fr8 + a119*fr9 + a1110*fr10);
                    double Vth = pth + he*(a111*fth1 + a115*fth5 + a116*fth6 + a117*fth7 + a118*fth8 + a119*fth9 + a1110*fth10);
                    geoAccel(Qr, Qth, Vr, Vth, a, b, Q2, &fr11, &fth11, &vphi11);
                }

                /* ---- 8th-order position and velocity update ---- */
                /* Position weights b[i] (O(h) Nystrm structure) */
                double bbar1 = 23.0/320.0;
                double bbar5 = 11.0/320.0;
                double bbar6 = 297.0/1600.0;
                double bbar7 = 1000.0/4455.0;
                /* bbar8..bbar11 = 0 for 8th order position */

                /* Velocity weights b[i] */
                double bv1 = 13.0/288.0;
                double bv5 = 32.0/125.0;
                double bv6 = 31213.0/144000.0;
                double bv7 = 2401.0/12960.0;
                /* bv8..bv11 = 0 for 8th order velocity */

                /* 8th-order solution */
                double dr8  = he * pr + h2 * (bbar1*fr1 + bbar5*fr5 + bbar6*fr6 + bbar7*fr7);
                double dth8 = he * pth + h2 * (bbar1*fth1 + bbar5*fth5 + bbar6*fth6 + bbar7*fth7);
                double dvr8  = he * (bv1*fr1 + bv5*fr5 + bv6*fr6 + bv7*fr7);
                double dvth8 = he * (bv1*fth1 + bv5*fth5 + bv6*fth6 + bv7*fth7);

                /* Accumulate  from velocity stages */
                double dphi8 = he * (bv1*vphi1 + bv5*vphi5 + bv6*vphi6 + bv7*vphi7);

                /* ---- 6th-order embedded velocity (for error) ---- */
                double bvhat1 = 263.0/2520.0;
                double bvhat5 = 176.0/3125.0;
                double bvhat6 = 15551.0/45360.0;
                double bvhat7 = 500.0/1323.0;
                double bvhat8 = -1.0/3360.0;
                /* bvhat9..bvhat11 have nonzero contribution through stages 10,11 */

                /* Error on velocity: (b8 - b6)  f */
                double evr  = he * ((bv1-bvhat1)*fr1 + (bv5-bvhat5)*fr5 + (bv6-bvhat6)*fr6 + (bv7-bvhat7)*fr7 - bvhat8*fr8);
                double evth = he * ((bv1-bvhat1)*fth1 + (bv5-bvhat5)*fth5 + (bv6-bvhat6)*fth6 + (bv7-bvhat7)*fth7 - bvhat8*fth8);

                /* Error on position (from embedded 6th-order) */
                double bbarhat1 = 263.0/3780.0;
                double bbarhat5 = 176.0/5625.0;
                double bbarhat6 = 15551.0/75600.0;
                double bbarhat7 = 1000.0/4455.0;
                double bbarhat8 = -1.0/5040.0;

                double epr = h2 * ((bbar1-bbarhat1)*fr1 + (bbar5-bbarhat5)*fr5 + (bbar6-bbarhat6)*fr6 + (bbar7-bbarhat7)*fr7 - bbarhat8*fr8);
                double epth = h2 * ((bbar1-bbarhat1)*fth1 + (bbar5-bbarhat5)*fth5 + (bbar6-bbarhat6)*fth6 + (bbar7-bbarhat7)*fth7 - bbarhat8*fth8);

                /* Scaled error norm (combine position and velocity errors) */
                double sc_r   = atol + rtol * fmax(fabs(rr), fabs(rr + dr8));
                double sc_th  = atol + rtol * fmax(fabs(th), fabs(th + dth8));
                double sc_vr  = atol + rtol * fmax(fabs(pr), fabs(pr + dvr8));
                double sc_vth = atol + rtol * fmax(fabs(pth), fabs(pth + dvth8));

                double err_norm = sqrt(0.25 * (
                    (epr/sc_r)*(epr/sc_r) + (epth/sc_th)*(epth/sc_th) +
                    (evr/sc_vr)*(evr/sc_vr) + (evth/sc_vth)*(evth/sc_vth)));

                if (err_norm <= 1.0 || rejects >= max_reject) {
                    /* Accept: advance with 8th-order solution */
                    rr  += dr8;
                    th  += dth8;
                    phi += dphi8;
                    pr  += dvr8;    /* velocity update */
                    pth += dvth8;
                    accepted = true;

                    /* Step control: exponent -1/8 for 8th order */
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
            } /* end accept/reject */

            /* Convert velocity back to momentum for physics computations */
            double mom_pr, mom_pth;
            {
                double sth = sin(th), cth = cos(th);
                double a2 = a * a, r2 = rr * rr;
                double sig = r2 + a2 * cth * cth;
                double del = r2 - 2.0 * rr + a2 + Q2;
                double sdel = fmax(del, 1e-14);
                mom_pr  = (sig / sdel) * pr;   /* p_r = (/) * vr */
                mom_pth = sig * pth;            /* p_ =  * v */
            }

            /* Volumetric emission */
            if (acc_a < 0.99f) {
                accumulate_volume_emission(rr, th, he, a, Q2, (double)p.isco, p.disk_outer,
                                           &acc_r, &acc_g, &acc_b, &acc_a);
            }

            /* Horizon capture */
            if (rr <= rp * 1.01) {
                float hr, hg, hb;
                hawking_glow_color(rr, a, Q2, p.hawking_boost, &hr, &hg, &hb);
                if (hr > 0 || hg > 0 || hb > 0) {
                    blendColor(hr, hg, hb, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
                }
                blendColor(0.0f, 0.0f, 0.0f, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
                done = true; break;
            }

            /* Disk crossing detection */
            if (show_disk && acc_a < 0.99f) {
                double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
                if (cross < 0.0 && disk_crossings < max_crossings) {
                    double t = fmin(fmax(fabs(oldTh - PI*0.5) / fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                    double r_hit = oldR + t * (rr - oldR);
                    float dr_f = (float)r_hit;
                    float dphi_f = (float)(oldPhi + t * (phi - oldPhi));
                    float g = (float)kerr_g_factor(r_hit, a, Q2, b, (double)p.isco);
                    float dcr, dcg, dcb;
                    diskColor(dr_f, dphi_f, (float)a, (float)Q2,
                             (float)p.isco, (float)p.disk_outer, (float)p.disk_temp,
                             g, (int)p.doppler_boost, F_peak, &dcr, &dcg, &dcb);

                    double p_total = sqrt(mom_pr*mom_pr + mom_pth*mom_pth + b*b);
                    float cos_em = (float)(fabs(mom_pth) / fmax(p_total, 1e-15));
                    float limb = limb_darkening(cos_em);
                    dcr *= limb; dcg *= limb; dcb *= limb;

                    float crossing_alpha;
                    if (disk_crossings == 0) crossing_alpha = base_alpha;
                    else if (disk_crossings == 1) crossing_alpha = base_alpha * 0.85f;
                    else {
                        float ring_brightness_boost = powf(2.71828f, (float)(disk_crossings - 1) * 0.5f);
                        crossing_alpha = fminf(base_alpha * ring_brightness_boost, 1.0f);
                    }
                    blendColor(dcr, dcg, dcb, crossing_alpha, &acc_r, &acc_g, &acc_b, &acc_a);
                    disk_crossings++;
                }
            }

            /* Escape */
            if (rr > p.esc_radius) {
                double frac = fmin(fmax((p.esc_radius - oldR) / fmax(rr - oldR, 1e-14), 0.0), 1.0);
                double fth = oldTh + (th - oldTh) * frac;
                double fph = oldPhi + (phi - oldPhi) * frac;
                float dx, dy, dz;
                sphereDir(fth, fph, &dx, &dy, &dz);
                float bgr, bgg, bgb;
                background(dx, dy, dz, bg_mode, star_layers, show_grid,
                           skymap, (int)p.sky_width, (int)p.sky_height, &bgr, &bgg, &bgb);
                if (acc_a < 1.0f) blendColor(bgr, bgg, bgb, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
                done = true; break;
            }

            /* Pole reflection */
            if (th < 0.0) { th = -th; pth = -pth; phi += PI; }
            else if (th > PI) { th = 2.0 * PI - th; pth = -pth; phi += PI; }

            if (rr < 0.5 || rr != rr || th != th) { done = true; break; }
        }

        float cr = acc_r, cg = acc_g, cb = acc_b;
        float ux = 2.0f * (ix + 0.5f + jitter_x) / (float)W  - 1.0f;
        float uy = 2.0f * (iy + 0.5f + jitter_y) / (float)H - 1.0f;
        postProcess(&cr, &cg, &cb, alpha, beta, p, ux, uy);
        final_r += cr; final_g += cg; final_b += cb;
    }

    final_r /= (float)SAMPLES;
    final_g /= (float)SAMPLES;
    final_b /= (float)SAMPLES;

    int idx = (iy * W + ix) * 3;
    output[idx + 0] = (unsigned char)(fminf(fmaxf(final_r * 255.0f, 0.0f), 255.0f));
    output[idx + 1] = (unsigned char)(fminf(fmaxf(final_g * 255.0f, 0.0f), 255.0f));
    output[idx + 2] = (unsigned char)(fminf(fmaxf(final_b * 255.0f, 0.0f), 255.0f));
    atomicAdd(progress_counter, 1);
}
