/* ============================================================
 *  YOSHIDA 4th-ORDER (Forest-Ruth) SYMPLECTIC INTEGRATOR
 *
 *  3 symmetric substeps with the classic triple-jump coefficients.
 *  All geodesic integration in float64; color output in float32.
 *
 *  This file is loaded as a complete CUDA kernel source by
 *  renderer.py via CuPy RawKernel.
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"

/* Yoshida 4th-order (Forest-Ruth) coefficients
 * w1 = 1/(2 - 2^(1/3)), w0 = -2^(1/3)/(2 - 2^(1/3))
 * d1 = w1/2, d0 = (w0 + w1)/2 */
#define Y4_W1  1.3512071919596576
#define Y4_W0 -1.7024143839193153
#define Y4_D1  0.6756035959798288
#define Y4_D0 -0.1756035959798288


extern "C" __global__
void trace_yoshida4(const RenderParams *pp, unsigned char *output) {
    const RenderParams &p = *pp;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int W = (int)p.width, H = (int)p.height;
    if (ix >= W || iy >= H) return;

    /* ── Initialize ray ──────────────────────────────────── */
    double r, th, phi, pr, pth, b, rp;
    float alpha, beta;
    initRay(ix, iy, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta);

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    int STEPS = (int)p.steps;
    int show_disk = (int)p.show_disk;
    int bg_mode = (int)p.bg_mode;
    int star_layers = (int)p.star_layers;
    int show_grid = (int)p.show_grid;
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    bool done = false;

    /* ── Integration loop ────────────────────────────────── */
    for (int i = 0; i < STEPS; i++) {
        if (done) break;

        /* Adaptive step size: scale base step with observer distance
         * (affine parameter budget must grow with R₀ for the round trip) */
        double h_scaled = p.step_size * (p.obs_dist / 30.0);
        double he = h_scaled * fmin(fmax((r - rp) * 0.4, 0.04), 1.0);
        he = fmin(fmax(he, 0.012), 1.0);

        double oldTh = th, oldR = r, oldPhi = phi;

        /* Yoshida 4th-order symmetric composition: 3 substeps */
        double dr_, dth_, dphi_, dpr_, dpth_;

        /* --- Substep 1: drift d1, kick w1 --- */
        geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
        r   += he * Y4_D1 * dr_;
        th  += he * Y4_D1 * dth_;
        phi += he * Y4_D1 * dphi_;
        geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
        pr  += he * Y4_W1 * dpr_;
        pth += he * Y4_W1 * dpth_;

        /* --- Substep 2: drift d0, kick w0 --- */
        geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
        r   += he * Y4_D0 * dr_;
        th  += he * Y4_D0 * dth_;
        phi += he * Y4_D0 * dphi_;
        geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
        pr  += he * Y4_W0 * dpr_;
        pth += he * Y4_W0 * dpth_;

        /* --- Substep 3: drift d1, kick w1 (symmetric) --- */
        geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
        r   += he * Y4_D1 * dr_;
        th  += he * Y4_D1 * dth_;
        phi += he * Y4_D1 * dphi_;
        geoRHS(r, th, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_, &dpr_, &dpth_);
        pr  += he * Y4_W1 * dpr_;
        pth += he * Y4_W1 * dpth_;

        /* Pole reflection */
        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        /* ── Termination conditions ──────────────────────── */

        /* Horizon capture */
        if (r <= rp * 1.01) { done = true; break; }

        /* Disk crossing */
        if (show_disk) {
            double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
            if (cross < 0.0) {
                double f = fmin(fmax(fabs(oldTh - PI * 0.5) /
                           fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                double r_hit = oldR + f * (r - oldR);
                float dr_f = (float)r_hit;
                float dphi_f = (float)(oldPhi + f * (phi - oldPhi));

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

        /* Escape to background */
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

        /* NaN / underflow safety */
        if (r < 0.5 || r != r || th != th) { done = true; break; }
    }

    /* ── Post-processing ─────────────────────────────────── */
    float ux = 2.0f * (ix + 0.5f) / (float)W  - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / (float)H - 1.0f;
    postProcess(&cr, &cg, &cb, alpha, beta, (float)p.spin, ux, uy);

    /* ── Write output (RGB, uint8) ───────────────────────── */
    int idx = (iy * W + ix) * 3;
    output[idx + 0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx + 1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx + 2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
