/* ============================================================
 *  YOSHIDA 8th-ORDER SYMPLECTIC INTEGRATOR
 *
 *  15 symmetric substeps with Solution D coefficients (Table 2).
 *  All geodesic integration in float64; color output in float32.
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"

/* Yoshida 8th-order coefficients (Solution D from Table 2) */
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

/* Macro for a single Yoshida drift-kick substep */
#define YOSHIDA_SUBSTEP(D_COEFF, W_COEFF) \
    geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
    r   += he * (D_COEFF) * dr_;   \
    th  += he * (D_COEFF) * dth_;  \
    phi += he * (D_COEFF) * dphi_; \
    geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
    pr  += he * (W_COEFF) * dpr_;  \
    pth += he * (W_COEFF) * dpth_;


extern "C" __global__
void trace_yoshida8(const RenderParams *pp, unsigned char *output) {
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
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    bool done = false;

    int STEPS = (int)p.steps;
    int show_disk = (int)p.show_disk;
    int bg_mode = (int)p.bg_mode;
    int star_layers = (int)p.star_layers;
    int show_grid = (int)p.show_grid;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;

        /* Scale base step with observer distance (affine parameter budget ∝ R₀) */
        double h_scaled = p.step_size * (p.obs_dist / 30.0);
        double he = h_scaled * fmin(fmax((r - rp) * 0.4, 0.04), 1.0);
        he = fmin(fmax(he, 0.012), 1.0);
        double oldTh = th, oldR = r, oldPhi = phi;
        double dr_, dth_, dphi_, dpr_, dpth_;

        /* 15 symmetric substeps */
        YOSHIDA_SUBSTEP(Y8_D1, Y8_W1)  /* 1 */
        YOSHIDA_SUBSTEP(Y8_D2, Y8_W2)  /* 2 */
        YOSHIDA_SUBSTEP(Y8_D3, Y8_W3)  /* 3 */
        YOSHIDA_SUBSTEP(Y8_D4, Y8_W4)  /* 4 */
        YOSHIDA_SUBSTEP(Y8_D5, Y8_W5)  /* 5 */
        YOSHIDA_SUBSTEP(Y8_D6, Y8_W6)  /* 6 */
        YOSHIDA_SUBSTEP(Y8_D7, Y8_W7)  /* 7 */
        YOSHIDA_SUBSTEP(Y8_D0, Y8_W0)  /* 8 (center) */
        YOSHIDA_SUBSTEP(Y8_D7, Y8_W7)  /* 9 (symmetric) */
        YOSHIDA_SUBSTEP(Y8_D6, Y8_W6)  /* 10 */
        YOSHIDA_SUBSTEP(Y8_D5, Y8_W5)  /* 11 */
        YOSHIDA_SUBSTEP(Y8_D4, Y8_W4)  /* 12 */
        YOSHIDA_SUBSTEP(Y8_D3, Y8_W3)  /* 13 */
        YOSHIDA_SUBSTEP(Y8_D2, Y8_W2)  /* 14 */
        YOSHIDA_SUBSTEP(Y8_D1, Y8_W1)  /* 15 */

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) { done = true; break; }
        if (show_disk) {
            double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
            if (cross < 0.0) {
                double f = fmin(fmax(fabs(oldTh - PI * 0.5) /
                           fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                float dr_f = (float)(oldR + f * (r - oldR));
                float dphi_f = (float)(oldPhi + f * (phi - oldPhi));
                float dcr, dcg, dcb;
                diskColor(dr_f, dphi_f, (float)a,
                         (float)p.isco, (float)p.disk_outer, p.disk_temp,
                         &dcr, &dcg, &dcb);
                float atten = 1.0f - fminf(sqrtf(cr*cr + cg*cg + cb*cb) * 0.4f, 0.9f);
                cr += dcr * atten; cg += dcg * atten; cb += dcb * atten;
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
                       &bgr, &bgg, &bgb);
            float atten = 1.0f - fminf(sqrtf(cr*cr + cg*cg + cb*cb) * 0.3f, 0.9f);
            cr += bgr * atten; cg += bgg * atten; cb += bgb * atten;
            done = true; break;
        }
        if (r < 0.5 || r != r || th != th) { done = true; break; }
    }

    float ux = 2.0f * (ix + 0.5f) / (float)W  - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / (float)H - 1.0f;
    postProcess(&cr, &cg, &cb, alpha, beta, (float)p.spin, ux, uy);

    int idx = (iy * W + ix) * 3;
    output[idx + 0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx + 1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx + 2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}

#undef YOSHIDA_SUBSTEP
