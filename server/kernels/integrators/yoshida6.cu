/* ============================================================
 *  YOSHIDA 6th-ORDER SYMPLECTIC INTEGRATOR
 *
 *  7 symmetric substeps with Solution A triple-jump coefficients.
 *  All geodesic integration in float64; color output in float32.
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"

/* Yoshida 6th-order symmetric composition coefficients (Solution A) */
#define Y6_W1  0.78451361047755726382
#define Y6_W2  0.23557321335935813368
#define Y6_W3 -1.17767998417887100695
#define Y6_W0  1.31518632068391121889
#define Y6_D1  0.39225680523877863191
#define Y6_D2  0.51004341191845769508
#define Y6_D3 -0.47105338540975643969
#define Y6_D0  0.06875316825252012625

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
void trace_yoshida6(const RenderParams *pp, unsigned char *output) {
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

        /* 7 symmetric substeps: d1,w1 / d2,w2 / d3,w3 / d0,w0 / d3,w3 / d2,w2 / d1,w1 */
        YOSHIDA_SUBSTEP(Y6_D1, Y6_W1)
        YOSHIDA_SUBSTEP(Y6_D2, Y6_W2)
        YOSHIDA_SUBSTEP(Y6_D3, Y6_W3)
        YOSHIDA_SUBSTEP(Y6_D0, Y6_W0)
        YOSHIDA_SUBSTEP(Y6_D3, Y6_W3)
        YOSHIDA_SUBSTEP(Y6_D2, Y6_W2)
        YOSHIDA_SUBSTEP(Y6_D1, Y6_W1)

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
