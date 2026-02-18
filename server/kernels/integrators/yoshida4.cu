/* ============================================================
 *  YOSHIDA 4th-ORDER (Forest-Ruth) SYMPLECTIC INTEGRATOR
 *
 *  3 symmetric substeps with the classic triple-jump coefficients.
 *  All geodesic integration in float64; color output in float32.
 *
 *  This file is loaded as a complete CUDA kernel source by
 *  renderer_cuda.py via CuPy RawKernel.
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
void trace_yoshida4(const RenderParams p, unsigned char *output) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= p.width || iy >= p.height) return;

    /* ── Initialize ray ──────────────────────────────────── */
    double r, th, phi, pr, pth, b, rp;
    float alpha, beta;
    initRay(ix, iy, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta);

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    bool done = false;

    /* ── Integration loop ────────────────────────────────── */
    for (int i = 0; i < p.steps; i++) {
        if (done) break;

        /* Adaptive step size (near-horizon reduction) */
        double he = p.step_size * fmin(fmax((r - rp) * 0.4, 0.04), 1.0);
        he = fmin(fmax(he, 0.012), 0.6);

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
        if (p.show_disk) {
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
            background(dx, dy, dz, p.bg_mode, p.star_layers, p.show_grid,
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
    float ux = 2.0f * (ix + 0.5f) / p.width  - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / p.height - 1.0f;
    postProcess(&cr, &cg, &cb, alpha, beta, (float)p.spin, ux, uy);

    /* ── Write output (RGB, uint8) ───────────────────────── */
    int idx = (iy * p.width + ix) * 3;
    output[idx + 0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx + 1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx + 2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
