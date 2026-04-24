/* ============================================================
 *  RK4 ??? Classical 4th-order Runge-Kutta integrator
 *
 *  4 stages per step with 1/6, 1/3, 1/3, 1/6 weights.
 *  All geodesic integration in float64; color output in float32.
 *
 *  Uses shared step function from steps.cu and shared adaptive
 *  step sizing from adaptive_step.cu for modularity.
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"
#include "steps.cu"
#include "adaptive_step.cu"


extern "C" __global__
void trace_rk4(const RenderParams *pp, unsigned char *output, const float *skymap) {
    const RenderParams &p = *pp;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int W = (int)p.width, H = (int)p.height;
    if (ix >= W || iy >= H) return;

    double r, th, phi, pr, pth, b, rp;
    float alpha, beta;
    if (!initRay(ix, iy, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta)) {
        int idx = (iy * W + ix) * 3;
        output[idx + 0] = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 0;
        return;
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

    for (int i = 0; i < STEPS; i++) {
        if (done) break;
            double r_photon_sphere = 3.0;
            if (a != 0.0) {
                r_photon_sphere = 2.0 * (1.0 + cos(2.0/3.0 * acos(-a)));
            }
            double dist_to_photon_sphere = fabs(r - r_photon_sphere);
            double adaptive_factor = 1.0;
            if (dist_to_photon_sphere < 1.0) {
                adaptive_factor = 0.1 + 0.9 * dist_to_photon_sphere;
            }
            double effective_step = p.step_size * adaptive_factor;
            // Cap he if needed
            if (he > effective_step) he = effective_step;
    

        /* Adaptive step size (shared function) */
        double he = adaptive_step_rk4(r, rp, p.step_size, p.obs_dist);
        double oldTh = th, oldR = r, oldPhi = phi;

        /* RK4 step (shared function from steps.cu) */
        rk4_step(&r, &th, &phi, &pr, &pth, a, b, Q2, he);

        if (th < 0.0) {
            th = -th;
            pth = -pth;
            phi += PI;
        } else if (th > PI) {
            th = 2.0 * PI - th;
            pth = -pth;
            phi += PI;
        }




        /* Volumetric emission: hot corona + relativistic jet */
        if (acc_a < 0.99f) {
            accumulate_volume_emission(r, th, he, a, Q2, (double)p.isco, p.disk_outer,
                                       &acc_r, &acc_g, &acc_b, &acc_a);
        }

        if (r <= rp * 1.01) {
            /* Horizon capture: composite black */
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

                float g = (float)kerr_g_factor(r_hit, a, Q2, b, (double)p.isco);

                float dcr, dcg, dcb;
                diskColor(dr_f, dphi_f, (float)a, (float)Q2,
                         (float)p.isco, (float)p.disk_outer, (float)p.disk_temp,
                         g, (int)p.doppler_boost, F_peak,
                             &dcr, &dcg, &dcb);
                
                    double p_total = sqrt(pr * pr + pth * pth + b * b);
                    float cos_em = (float)(fabs(pth) / fmax(p_total, 1e-15));
                    float limb = limb_darkening(cos_em);
                    dcr *= limb;
                    dcg *= limb;
                    dcb *= limb;

                    float crossing_alpha;
                    if (disk_crossings == 0) {
                        crossing_alpha = base_alpha;
                    } else if (disk_crossings == 1) {
                        crossing_alpha = base_alpha * 0.85f;
                    } else {
                        float ring_brightness_boost = powf(2.71828f, (float)(disk_crossings - 1) * 0.5f);
                        crossing_alpha = fminf(base_alpha * ring_brightness_boost, 1.0f);
                    }

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
