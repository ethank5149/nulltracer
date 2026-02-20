/* ============================================================
 *  TAO + YOSHIDA 4th-ORDER SYMPLECTIC INTEGRATOR
 *
 *  Tao extended phase space method (Tao 2016, Phys. Rev. E 94,
 *  043303) with Yoshida 4th-order (Forest-Ruth) composition.
 *
 *  Uses doubled phase space (10 variables per ray) to make the
 *  non-separable Kerr-Newman Hamiltonian amenable to symplectic
 *  splitting, achieving true 4th-order accuracy.
 *
 *  3 symmetric substeps × 2 geoRHS per substep = 6 geoRHS/step.
 *  All geodesic integration in float64; color output in float32.
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"
#include "steps.cu"
#include "adaptive_step.cu"


extern "C" __global__
void trace_tao_yoshida4(const RenderParams *pp, unsigned char *output) {
    const RenderParams &p = *pp;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int W = (int)p.width, H = (int)p.height;
    if (ix >= W || iy >= H) return;

    /* ── Initialize ray ──────────────────────────────────── */
    double r, th, phi, pr, pth, b, rp;
    float alpha, beta;
    initRay(ix, iy, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta);

    /* Initialize shadow variables = real variables (Tao 2016, Section II.A) */
    double rs = r, ths = th, phis = phi, prs = pr, pths = pth;

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

        double he = adaptive_step_tao(r, rp, p.step_size, p.obs_dist);
        double oldTh = th, oldR = r, oldPhi = phi;

        /* Tao + Yoshida 4th-order step (extended phase space) */
        tao_yoshida4_step(&r, &th, &phi, &pr, &pth,
                          &rs, &ths, &phis, &prs, &pths,
                          a, b, Q2, he);

        /* Hamiltonian constraint projection on real variables */
        projectHamiltonian(r, th, &pr, pth, a, b, Q2);

        /* Pole reflection */
        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        /* ── Termination conditions ──────────────────────── */

        if (r <= rp * 1.01) { done = true; break; }

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

    /* ── Post-processing ─────────────────────────────────── */
    float ux = 2.0f * (ix + 0.5f) / (float)W  - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / (float)H - 1.0f;
    postProcess(&cr, &cg, &cb, alpha, beta, (float)p.spin, ux, uy);

    int idx = (iy * W + ix) * 3;
    output[idx + 0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx + 1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx + 2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
