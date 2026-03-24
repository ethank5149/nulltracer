/* ============================================================
 *  TAO + KAHAN-LI 8th-ORDER SYMPLECTIC INTEGRATOR
 *  (Ingoing Kerr coordinates)
 *
 *  Tao extended phase space method (Tao 2016, Phys. Rev. E 94,
 *  043303) with Kahan-Li s15odr8 optimal 8th-order composition
 *  (Kahan & Li, Math. Comp. 66:1089–1099, 1997).
 *
 *  Uses doubled phase space (10 variables per ray) to make the
 *  non-separable Kerr-Newman Hamiltonian amenable to symplectic
 *  splitting, achieving true 8th-order accuracy.
 *
 *  Integration uses ingoing Kerr coordinates which eliminate the
 *  Boyer-Lindquist coordinate singularity at Δ = 0 (horizon).
 *  This prevents catastrophic force blowups when negative
 *  Kahan-Li substeps temporarily push r below the horizon.
 *
 *  Kahan-Li coefficients have max |W_i| = 0.797 (vs 2.447 for
 *  Yoshida Solution D), giving 3.1× smaller intermediate
 *  excursions and better numerical stability.
 *
 *  15 symmetric substeps × 2 force evals per substep = 30 evals/step.
 *  All geodesic integration in float64; color output in float32.
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"
#include "steps.cu"
#include "adaptive_step.cu"


extern "C" __global__
void trace_tao_kahan_li8(const RenderParams *pp, unsigned char *output, const float *skymap) {
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

    /* Transform p_r from BL to Kerr coordinates */
    transformBLtoKS(r, a, b, Q2, &pr);

    /* Initialize shadow variables = real variables */
    double rs = r, ths = th, phis = phi, prs = pr, pths = pth;
    int STEPS = (int)p.steps;
    int show_disk = (int)p.show_disk;
    int bg_mode = (int)p.bg_mode;
    int star_layers = (int)p.star_layers;
    int show_grid = (int)p.show_grid;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_a = 0.0f;
    int disk_crossings = 0;
    int max_crossings = (int)p.disk_max_crossings;
    float base_alpha = (float)p.disk_alpha;
    bool done = false;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;

        double he = adaptive_step_tao(r, rp, p.step_size, p.obs_dist);
        double oldTh = th, oldR = r, oldPhi = phi;

        /* Tao + Kahan-Li 8th-order step (extended phase space) */
        tao_kahan_li8_step(&r, &th, &phi, &pr, &pth,
                           &rs, &ths, &phis, &prs, &pths,
                           a, b, Q2, he);

        /* Hamiltonian constraint projection on real variables (KS) */
        projectHamiltonianKS(r, th, &pr, pth, a, b, Q2);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        /* KS coordinates are regular at the horizon */
        if (r <= rp * 0.5) {
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

                float g = compute_g_factor(r_hit, a, Q2, b);

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
