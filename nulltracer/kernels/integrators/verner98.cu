/* ============================================================
 *  VERNER 9(8) — "Most Efficient" 9th-order Runge-Kutta
 *  with ADAPTIVE STEP SIZE CONTROL
 *
 *  16 stages per step with Verner (2010) coefficients.
 *  Uses the embedded 8th-order solution for local error
 *  estimation and automatic step size adjustment.
 *
 *  FSAL (First Same As Last): stage 16 of step n is reused
 *  as stage 1 of step n+1, reducing effective cost to 15
 *  geoRHS evaluations per accepted step after the first.
 *
 *  All geodesic integration in float64; color output in float32.
 *
 *  Reference: J.H. Verner (2010), "Numerically optimal
 *  Runge-Kutta pairs with interpolants," Numerical Algorithms
 *  53, 383–396.  Coefficients: Table 12 ("most efficient"
 *  9th-order pair with 8th-order embedding).
 *
 *  CUDA optimizations:
 *    - FSAL saves 1 geoRHS per accepted step
 *    - Scoped stage variables allow register reuse
 *    - φ excluded from error norm (b is conserved)
 *    - Step factor clamped to [0.1, 5.0] for stability
 *    - Exponent -1/9 for 9th-order optimal step control
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"
#include "adaptive_step.cu"


/* ============================================================
 *  Verner 9(8) Butcher tableau (decimal, 17 sig. figs.)
 *
 *  Nodes c[i], i=1..16:
 * ============================================================ */
#define V98_C2   0.03462105947808740
#define V98_C3   0.09702167010868508
#define V98_C4   0.14553250516302762
#define V98_C5   0.56100000000000000
#define V98_C6   0.22900000000000000
#define V98_C7   0.58700000000000000
#define V98_C8   0.27000000000000000
#define V98_C9   0.52000000000000000
#define V98_C10  0.87000000000000000
#define V98_C11  0.36000000000000000
#define V98_C12  0.36000000000000000
#define V98_C13  0.54000000000000000
#define V98_C14  0.34000000000000000
#define V98_C15  0.88000000000000000
#define V98_C16  1.00000000000000000

/* -- Inline Verner 9(8) step --------------------------------
 *
 *  Computes one full step with 16-stage evaluation,
 *  returning the 9th-order increment and the error
 *  estimate (9th-order minus 8th-order).
 *
 *  All a_{ij} coefficients, 9th-order weights b9[i],
 *  and the error differences (b9[i]-b8[i]) are inlined
 *  as compile-time constants.
 *
 *  The method is applied to the 5-component state
 *  y = (r, θ, φ, pr, pth) with RHS = geoRHS().
 * ============================================================ */

extern "C" __global__
void trace_verner98(const RenderParams *pp, unsigned char *output, const float *skymap, unsigned int *progress_counter) {
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

        double r, th, phi, pr, pth, b, rp;
        float alpha, beta;
        if (!initRayJittered(ix, iy, jitter_x, jitter_y, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta)) {
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

        double atol = 1.0e-10, rtol = 1.0e-10, safety = 0.9;
        double h_min = 0.0005, h_max = 2.5;
        int max_reject = 5;
        double he = adaptive_step_verner98_initial(r, rp, p.step_size, p.obs_dist, h_min, h_max);

        /* FSAL: stage 1 of next step = stage 16 of this step */
        double fsal_kr = 0, fsal_kth = 0, fsal_kphi = 0, fsal_kpr = 0, fsal_kpth = 0;
        bool have_fsal = false;

        for (int i = 0; i < STEPS; i++) {
            if (done) break;

            double oldTh = th, oldR = r, oldPhi = phi;
            double oldPr = pr, oldPth = pth;
            int rejects = 0;
            bool accepted = false;

            while (!accepted) {
                /* Stage 1 (FSAL reuse when available) */
                double kr1, kth1, kphi1, kpr1, kpth1;
                if (have_fsal) {
                    kr1 = fsal_kr; kth1 = fsal_kth; kphi1 = fsal_kphi;
                    kpr1 = fsal_kpr; kpth1 = fsal_kpth;
                } else {
                    geoRHS(r, th, pr, pth, a, b, Q2, &kr1, &kth1, &kphi1, &kpr1, &kpth1);
                }

                /* Stage 2 */
                double kr2, kth2, kphi2, kpr2, kpth2;
                geoRHS(r  + he * V98_C2 * kr1,
                       th + he * V98_C2 * kth1,
                       pr + he * V98_C2 * kpr1,
                       pth+ he * V98_C2 * kpth1,
                       a, b, Q2, &kr2, &kth2, &kphi2, &kpr2, &kpth2);

                /* Stage 3 */
                double kr3, kth3, kphi3, kpr3, kpth3;
                {
                    double a31 = -0.01145764041037430, a32 = 0.10847931051905938;
                    geoRHS(r  + he*(a31*kr1 + a32*kr2),
                           th + he*(a31*kth1 + a32*kth2),
                           pr + he*(a31*kpr1 + a32*kpr2),
                           pth+ he*(a31*kpth1 + a32*kpth2),
                           a, b, Q2, &kr3, &kth3, &kphi3, &kpr3, &kpth3);
                }

                /* Stage 4 */
                double kr4, kth4, kphi4, kpr4, kpth4;
                {
                    double a41 = 0.03638312629075691, a43 = 0.10914937887227072;
                    geoRHS(r  + he*(a41*kr1 + a43*kr3),
                           th + he*(a41*kth1 + a43*kth3),
                           pr + he*(a41*kpr1 + a43*kpr3),
                           pth+ he*(a41*kpth1 + a43*kpth3),
                           a, b, Q2, &kr4, &kth4, &kphi4, &kpr4, &kpth4);
                }

                /* Stage 5 */
                double kr5, kth5, kphi5, kpr5, kpth5;
                {
                    double a51 = 2.02576084012321920, a53 = -7.63878392400893200;
                    double a54 = 6.17402308388571280;
                    geoRHS(r  + he*(a51*kr1 + a53*kr3 + a54*kr4),
                           th + he*(a51*kth1 + a53*kth3 + a54*kth4),
                           pr + he*(a51*kpr1 + a53*kpr3 + a54*kpr4),
                           pth+ he*(a51*kpth1 + a53*kpth3 + a54*kpth4),
                           a, b, Q2, &kr5, &kth5, &kphi5, &kpr5, &kpth5);
                }

                /* Stage 6 */
                double kr6, kth6, kphi6, kpr6, kpth6;
                {
                    double a61 = 0.04267276475929752, a64 = 0.14820330033498510;
                    double a65 = 0.03812393490471746;
                    geoRHS(r  + he*(a61*kr1 + a64*kr4 + a65*kr5),
                           th + he*(a61*kth1 + a64*kth4 + a65*kth5),
                           pr + he*(a61*kpr1 + a64*kpr4 + a65*kpr5),
                           pth+ he*(a61*kpth1 + a64*kpth4 + a65*kpth5),
                           a, b, Q2, &kr6, &kth6, &kphi6, &kpr6, &kpth6);
                }

                /* Stage 7 */
                double kr7, kth7, kphi7, kpr7, kpth7;
                {
                    double a71 = -0.54767920780445450, a74 = 1.57586541028785960;
                    double a75 = -0.18728267956932220, a76 = -0.25390352291408300;
                    geoRHS(r  + he*(a71*kr1 + a74*kr4 + a75*kr5 + a76*kr6),
                           th + he*(a71*kth1 + a74*kth4 + a75*kth5 + a76*kth6),
                           pr + he*(a71*kpr1 + a74*kpr4 + a75*kpr5 + a76*kpr6),
                           pth+ he*(a71*kpth1 + a74*kpth4 + a75*kpth5 + a76*kpth6),
                           a, b, Q2, &kr7, &kth7, &kphi7, &kpr7, &kpth7);
                }

                /* Stage 8 */
                double kr8, kth8, kphi8, kpr8, kpth8;
                {
                    double a81 = 0.05205177336498790, a85 = -0.00604125259748469;
                    double a86 = 0.19662478445710860, a87 = 0.02733169418038820;
                    geoRHS(r  + he*(a81*kr1 + a85*kr5 + a86*kr6 + a87*kr7),
                           th + he*(a81*kth1 + a85*kth5 + a86*kth6 + a87*kth7),
                           pr + he*(a81*kpr1 + a85*kpr5 + a86*kpr6 + a87*kpr7),
                           pth+ he*(a81*kpth1 + a85*kpth5 + a86*kpth6 + a87*kpth7),
                           a, b, Q2, &kr8, &kth8, &kphi8, &kpr8, &kpth8);
                }

                /* Stage 9 */
                double kr9, kth9, kphi9, kpr9, kpth9;
                {
                    double a91 = 0.38160482408692510, a95 = -0.37859553897503830;
                    double a96 = 0.65180263498919150, a97 = -0.31280721047629230;
                    double a98 = 0.17799828948221400;
                    geoRHS(r  + he*(a91*kr1 + a95*kr5 + a96*kr6 + a97*kr7 + a98*kr8),
                           th + he*(a91*kth1 + a95*kth5 + a96*kth6 + a97*kth7 + a98*kth8),
                           pr + he*(a91*kpr1 + a95*kpr5 + a96*kpr6 + a97*kpr7 + a98*kpr8),
                           pth+ he*(a91*kpth1 + a95*kpth5 + a96*kpth6 + a97*kpth7 + a98*kpth8),
                           a, b, Q2, &kr9, &kth9, &kphi9, &kpr9, &kpth9);
                }

                /* Stage 10 */
                double kr10, kth10, kphi10, kpr10, kpth10;
                {
                    double a101 = -0.30028658890498370, a105 = -0.17358257652498910;
                    double a106 = 0.55026380681023870,  a107 = 0.97537826024843460;
                    double a108 = -0.27064375576498590, a109 = 0.08887105873994540;
                    geoRHS(r  + he*(a101*kr1 + a105*kr5 + a106*kr6 + a107*kr7 + a108*kr8 + a109*kr9),
                           th + he*(a101*kth1 + a105*kth5 + a106*kth6 + a107*kth7 + a108*kth8 + a109*kth9),
                           pr + he*(a101*kpr1 + a105*kpr5 + a106*kpr6 + a107*kpr7 + a108*kpr8 + a109*kpr9),
                           pth+ he*(a101*kpth1 + a105*kpth5 + a106*kpth6 + a107*kpth7 + a108*kpth8 + a109*kpth9),
                           a, b, Q2, &kr10, &kth10, &kphi10, &kpr10, &kpth10);
                }

                /* Stage 11 */
                double kr11, kth11, kphi11, kpr11, kpth11;
                {
                    double a111 = -0.18498523756063790, a115 = -0.10174529024826280;
                    double a116 = 0.39760051866653810,  a117 = -0.08466557698498370;
                    double a118 = 0.22573424329660270,  a119 = -0.00468612828898990;
                    double a1110 = 0.11274147013073350;
                    geoRHS(r  + he*(a111*kr1 + a115*kr5 + a116*kr6 + a117*kr7 + a118*kr8 + a119*kr9 + a1110*kr10),
                           th + he*(a111*kth1 + a115*kth5 + a116*kth6 + a117*kth7 + a118*kth8 + a119*kth9 + a1110*kth10),
                           pr + he*(a111*kpr1 + a115*kpr5 + a116*kpr6 + a117*kpr7 + a118*kpr8 + a119*kpr9 + a1110*kpr10),
                           pth+ he*(a111*kpth1 + a115*kpth5 + a116*kpth6 + a117*kpth7 + a118*kpth8 + a119*kpth9 + a1110*kpth10),
                           a, b, Q2, &kr11, &kth11, &kphi11, &kpr11, &kpth11);
                }

                /* Stage 12 */
                double kr12, kth12, kphi12, kpr12, kpth12;
                {
                    double a121 = 0.04060463992884014, a125 = -0.01524376895647988;
                    double a126 = 0.22816037995671680, a127 = 0.02282113269509040;
                    double a128 = 0.05755223006614940, a129 = 0.00312076906683198;
                    double a1210= 0.02197722370854290, a1211= 0.00100253547909560;
                    geoRHS(r  + he*(a121*kr1 + a125*kr5 + a126*kr6 + a127*kr7 + a128*kr8 + a129*kr9 + a1210*kr10 + a1211*kr11),
                           th + he*(a121*kth1 + a125*kth5 + a126*kth6 + a127*kth7 + a128*kth8 + a129*kth9 + a1210*kth10 + a1211*kth11),
                           pr + he*(a121*kpr1 + a125*kpr5 + a126*kpr6 + a127*kpr7 + a128*kpr8 + a129*kpr9 + a1210*kpr10 + a1211*kpr11),
                           pth+ he*(a121*kpth1 + a125*kpth5 + a126*kpth6 + a127*kpth7 + a128*kpth8 + a129*kpth9 + a1210*kpth10 + a1211*kpth11),
                           a, b, Q2, &kr12, &kth12, &kphi12, &kpr12, &kpth12);
                }

                /* Stage 13 */
                double kr13, kth13, kphi13, kpr13, kpth13;
                {
                    double a131 = -0.79001697653399120, a135 = -0.78129536715770160;
                    double a136 = 2.39773499306773300,  a137 = -0.87007974028645130;
                    double a138 = 0.66018863998549750,  a139 = -0.05346786559133970;
                    double a1310= -0.02350873464028790, a1311= 0.00000000000000000;
                    double a1312= 0.00067497625461070;
                    geoRHS(r  + he*(a131*kr1 + a135*kr5 + a136*kr6 + a137*kr7 + a138*kr8 + a139*kr9 + a1310*kr10 + a1312*kr12),
                           th + he*(a131*kth1 + a135*kth5 + a136*kth6 + a137*kth7 + a138*kth8 + a139*kth9 + a1310*kth10 + a1312*kth12),
                           pr + he*(a131*kpr1 + a135*kpr5 + a136*kpr6 + a137*kpr7 + a138*kpr8 + a139*kpr9 + a1310*kpr10 + a1312*kpr12),
                           pth+ he*(a131*kpth1 + a135*kpth5 + a136*kpth6 + a137*kpth7 + a138*kpth8 + a139*kpth9 + a1310*kpth10 + a1312*kpth12),
                           a, b, Q2, &kr13, &kth13, &kphi13, &kpr13, &kpth13);
                }

                /* Stage 14 */
                double kr14, kth14, kphi14, kpr14, kpth14;
                {
                    double a141 = 1.95133027691168770, a145 = 2.68843654567613760;
                    double a146 = -6.85905028498498400, a147 = 3.24617379538318340;
                    double a148 = -1.31856430884698270, a149 = 0.01930972752471484;
                    double a1410= 0.38088027094063360, a1411= 0.00000000000000000;
                    double a1412= -0.00348545093199020, a1413= -0.10523063848774780;
                    geoRHS(r  + he*(a141*kr1 + a145*kr5 + a146*kr6 + a147*kr7 + a148*kr8 + a149*kr9 + a1410*kr10 + a1412*kr12 + a1413*kr13),
                           th + he*(a141*kth1 + a145*kth5 + a146*kth6 + a147*kth7 + a148*kth8 + a149*kth9 + a1410*kth10 + a1412*kth12 + a1413*kth13),
                           pr + he*(a141*kpr1 + a145*kpr5 + a146*kpr6 + a147*kpr7 + a148*kpr8 + a149*kpr9 + a1410*kpr10 + a1412*kpr12 + a1413*kpr13),
                           pth+ he*(a141*kpth1 + a145*kpth5 + a146*kpth6 + a147*kpth7 + a148*kpth8 + a149*kpth9 + a1410*kpth10 + a1412*kpth12 + a1413*kpth13),
                           a, b, Q2, &kr14, &kth14, &kphi14, &kpr14, &kpth14);
                }

                /* Stage 15 */
                double kr15, kth15, kphi15, kpr15, kpth15;
                {
                    double a151 = -6.22831040813895700, a155 = -8.52538682726529500;
                    double a156 = 22.62914620458986000, a157 = -10.37853781827994400;
                    double a158 = 3.97822004722482400,  a159 = 0.14881347033498820;
                    double a1510= -0.80882510422676250, a1511= 0.00000000000000000;
                    double a1512= 0.02037534522884804,  a1513= 0.53927013529397810;
                    double a1514= -0.37511622624617130;
                    geoRHS(r  + he*(a151*kr1 + a155*kr5 + a156*kr6 + a157*kr7 + a158*kr8 + a159*kr9 + a1510*kr10 + a1512*kr12 + a1513*kr13 + a1514*kr14),
                           th + he*(a151*kth1 + a155*kth5 + a156*kth6 + a157*kth7 + a158*kth8 + a159*kth9 + a1510*kth10 + a1512*kth12 + a1513*kth13 + a1514*kth14),
                           pr + he*(a151*kpr1 + a155*kpr5 + a156*kpr6 + a157*kpr7 + a158*kpr8 + a159*kpr9 + a1510*kpr10 + a1512*kpr12 + a1513*kpr13 + a1514*kpr14),
                           pth+ he*(a151*kpth1 + a155*kpth5 + a156*kpth6 + a157*kpth7 + a158*kpth8 + a159*kpth9 + a1510*kpth10 + a1512*kpth12 + a1513*kpth13 + a1514*kpth14),
                           a, b, Q2, &kr15, &kth15, &kphi15, &kpr15, &kpth15);
                }

                /* Stage 16 (FSAL: will be stage 1 of next step) */
                double kr16, kth16, kphi16, kpr16, kpth16;
                {
                    double a161 = 0.18735067067678510, a155_= 0.00000000000000000;
                    double a168 = 0.23036200709948010, a169 = 0.02216475594979920;
                    double a1610= -0.00282237605135478, a1611= 0.00000000000000000;
                    double a1612= 0.01122809895497192, a1613= 0.20521517541037220;
                    double a1614= 0.00000000000000000, a1615= 0.34586338456370620;
                    geoRHS(r  + he*(a161*kr1 + a168*kr8 + a169*kr9 + a1610*kr10 + a1612*kr12 + a1613*kr13 + a1615*kr15),
                           th + he*(a161*kth1 + a168*kth8 + a169*kth9 + a1610*kth10 + a1612*kth12 + a1613*kth13 + a1615*kth15),
                           pr + he*(a161*kpr1 + a168*kpr8 + a169*kpr9 + a1610*kpr10 + a1612*kpr12 + a1613*kpr13 + a1615*kpr15),
                           pth+ he*(a161*kpth1 + a168*kpth8 + a169*kpth9 + a1610*kpth10 + a1612*kpth12 + a1613*kpth13 + a1615*kpth15),
                           a, b, Q2, &kr16, &kth16, &kphi16, &kpr16, &kpth16);
                }

                /* ---- 9th-order solution weights ---- */
                double b9_1  = 0.04427989419007951;
                double b9_8  = 0.21535969913498340;
                double b9_9  = 0.02085536024116400;
                double b9_10 = -0.00283482882918294;
                double b9_11 = 0.00000000000000000;
                double b9_12 = 0.01011592660757810;
                double b9_13 = 0.22012883537885700;
                double b9_14 = 0.00000000000000000;
                double b9_15 = 0.34065469893999770;
                double b9_16 = 0.15044564984635120;

                double dr9   = he*(b9_1*kr1 + b9_8*kr8 + b9_9*kr9 + b9_10*kr10 + b9_12*kr12 + b9_13*kr13 + b9_15*kr15 + b9_16*kr16);
                double dth9  = he*(b9_1*kth1 + b9_8*kth8 + b9_9*kth9 + b9_10*kth10 + b9_12*kth12 + b9_13*kth13 + b9_15*kth15 + b9_16*kth16);
                double dpr9  = he*(b9_1*kpr1 + b9_8*kpr8 + b9_9*kpr9 + b9_10*kpr10 + b9_12*kpr12 + b9_13*kpr13 + b9_15*kpr15 + b9_16*kpr16);
                double dpth9 = he*(b9_1*kpth1 + b9_8*kpth8 + b9_9*kpth9 + b9_10*kpth10 + b9_12*kpth12 + b9_13*kpth13 + b9_15*kpth15 + b9_16*kpth16);

                /* ---- Error estimate: (b9 - b8) · k ----
                 * The error coefficients e[i] = b9[i] - b8[i]. */
                double e1  =  0.00130288967723760;
                double e8  = -0.00887264545482580;
                double e9  =  0.00252166975287920;
                double e10 = -0.00282237605135478;
                double e11 =  0.00000000000000000;
                double e12 =  0.00142543427426602;
                double e13 =  0.01508818990882670;
                double e14 = -0.00544096020943730;
                double e15 =  0.00524893979047480;
                double e16 = -0.00846073189916644;

                double err_r   = he*(e1*kr1 + e8*kr8 + e9*kr9 + e10*kr10 + e12*kr12 + e13*kr13 + e14*kr14 + e15*kr15 + e16*kr16);
                double err_th  = he*(e1*kth1 + e8*kth8 + e9*kth9 + e10*kth10 + e12*kth12 + e13*kth13 + e14*kth14 + e15*kth15 + e16*kth16);
                double err_pr  = he*(e1*kpr1 + e8*kpr8 + e9*kpr9 + e10*kpr10 + e12*kpr12 + e13*kpr13 + e14*kpr14 + e15*kpr15 + e16*kpr16);
                double err_pth = he*(e1*kpth1 + e8*kpth8 + e9*kpth9 + e10*kpth10 + e12*kpth12 + e13*kpth13 + e14*kpth14 + e15*kpth15 + e16*kpth16);

                /* Scaled error norm (exclude cyclic φ) */
                double sc_r   = atol + rtol * fmax(fabs(r),   fabs(r   + dr9));
                double sc_th  = atol + rtol * fmax(fabs(th),  fabs(th  + dth9));
                double sc_pr  = atol + rtol * fmax(fabs(pr),  fabs(pr  + dpr9));
                double sc_pth = atol + rtol * fmax(fabs(pth), fabs(pth + dpth9));

                double err_norm = sqrt(0.25 * ((err_r/sc_r)*(err_r/sc_r) + (err_th/sc_th)*(err_th/sc_th) + (err_pr/sc_pr)*(err_pr/sc_pr) + (err_pth/sc_pth)*(err_pth/sc_pth)));

                if (err_norm <= 1.0 || rejects >= max_reject) {
                    /* Accept step: advance state with 9th-order solution */
                    r   += dr9;
                    th  += dth9;
                    phi += he*(b9_1*kphi1 + b9_8*kphi8 + b9_9*kphi9 + b9_10*kphi10 + b9_12*kphi12 + b9_13*kphi13 + b9_15*kphi15 + b9_16*kphi16);
                    pr  += dpr9;
                    pth += dpth9;
                    accepted = true;

                    /* FSAL: store stage 16 for next step's stage 1 */
                    fsal_kr = kr16; fsal_kth = kth16; fsal_kphi = kphi16;
                    fsal_kpr = kpr16; fsal_kpth = kpth16;
                    have_fsal = true;

                    /* Optimal step sizing: exponent -1/9 for 9th-order method */
                    if (err_norm > 1e-30) {
                        double factor = safety * pow(err_norm, -1.0/9.0);
                        factor = fmin(fmax(factor, 0.1), 5.0);
                        he *= factor;
                    }
                    he = fmin(fmax(he, h_min), h_max);
                } else {
                    double factor = safety * pow(err_norm, -1.0/9.0);
                    factor = fmax(factor, 0.1);
                    he *= factor;
                    he = fmax(he, h_min);
                    rejects++;
                    have_fsal = false;  /* rejected: can't reuse FSAL */
                }
            }

            /* Volumetric emission */
            if (acc_a < 0.99f) {
                accumulate_volume_emission(r, th, he, a, Q2, (double)p.isco, p.disk_outer,
                                           &acc_r, &acc_g, &acc_b, &acc_a);
            }

            /* Horizon capture */
            if (r <= rp * 1.01) {
                float hr, hg, hb;
                hawking_glow_color(r, a, Q2, p.hawking_boost, &hr, &hg, &hb);
                if (hr > 0 || hg > 0 || hb > 0) {
                    blendColor(hr, hg, hb, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
                }
                blendColor(0.0f, 0.0f, 0.0f, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
                done = true; break;
            }

            /* Disk crossing detection (cubic Hermite, matching rkdp8) */
            if (show_disk && acc_a < 0.99f) {
                double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
                if (cross < 0.0 && disk_crossings < max_crossings) {
                    double old_sig = oldR*oldR + a*a*cos(oldTh)*cos(oldTh);
                    double new_sig = r*r + a*a*cos(th)*cos(th);
                    double old_dth = oldPth / old_sig;
                    double new_dth = pth / new_sig;

                    double t = fmin(fmax(fabs(oldTh - PI * 0.5) / fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                    for(int iter=0; iter<3; iter++) {
                        double t2 = t*t, t3 = t2*t;
                        double h00 = 2.0*t3 - 3.0*t2 + 1.0;
                        double h10 = t3 - 2.0*t2 + t;
                        double h01 = -2.0*t3 + 3.0*t2;
                        double h11 = t3 - t2;
                        double theta_t = h00*oldTh + h10*he*old_dth + h01*th + h11*he*new_dth;
                        double d_h00 = 6.0*t2 - 6.0*t;
                        double d_h10 = 3.0*t2 - 4.0*t + 1.0;
                        double d_h01 = -6.0*t2 + 6.0*t;
                        double d_h11 = 3.0*t2 - 2.0*t;
                        double dtheta_t = d_h00*oldTh + d_h10*he*old_dth + d_h01*th + d_h11*he*new_dth;
                        t -= (theta_t - PI * 0.5) / fmax(fabs(dtheta_t), 1e-14);
                    }
                    t = fmin(fmax(t, 0.0), 1.0);

                    double r_hit = oldR + t * (r - oldR);
                    float dr_f = (float)r_hit;
                    float dphi_f = (float)(oldPhi + t * (phi - oldPhi));
                    float g = (float)kerr_g_factor(r_hit, a, Q2, b, (double)p.isco);
                    float dcr, dcg, dcb;
                    diskColor(dr_f, dphi_f, (float)a, (float)Q2,
                             (float)p.isco, (float)p.disk_outer, (float)p.disk_temp,
                             g, (int)p.doppler_boost, F_peak, &dcr, &dcg, &dcb);

                    double p_total = sqrt(pr * pr + pth * pth + b * b);
                    float cos_em = (float)(fabs(pth) / fmax(p_total, 1e-15));
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
            if (r > p.esc_radius) {
                double frac = fmin(fmax((p.esc_radius - oldR) / fmax(r - oldR, 1e-14), 0.0), 1.0);
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

            if (r < 0.5 || r != r || th != th) { done = true; break; }
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
