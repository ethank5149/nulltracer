/* ============================================================
 *  RK4 — Classical 4th-order Runge-Kutta integrator
 *
 *  4 stages per step with 1/6, 1/3, 1/3, 1/6 weights.
 *  All geodesic integration in float64; color output in float32.
 * ============================================================ */

#include "../geodesic_base.cu"
#include "../backgrounds.cu"
#include "../disk.cu"


extern "C" __global__
void trace_rk4(const RenderParams p, unsigned char *output) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= p.width || iy >= p.height) return;

    double r, th, phi, pr, pth, b, rp;
    float alpha, beta;
    initRay(ix, iy, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta);

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    bool done = false;

    for (int i = 0; i < p.steps; i++) {
        if (done) break;

        double he = p.step_size * fmin(fmax((r - rp) * 0.4, 0.04), 1.0);
        he = fmin(fmax(he, 0.012), 0.6);
        double oldTh = th, oldR = r, oldPhi = phi;

        /* RK4: 4 stages */
        double dr1, dth1, dphi1, dpr1, dpth1;
        double dr2, dth2, dphi2, dpr2, dpth2;
        double dr3, dth3, dphi3, dpr3, dpth3;
        double dr4, dth4, dphi4, dpr4, dpth4;

        geoRHS(r, th, pr, pth, a, b, Q2,
               &dr1, &dth1, &dphi1, &dpr1, &dpth1);
        geoRHS(r + 0.5*he*dr1, th + 0.5*he*dth1,
               pr + 0.5*he*dpr1, pth + 0.5*he*dpth1, a, b, Q2,
               &dr2, &dth2, &dphi2, &dpr2, &dpth2);
        geoRHS(r + 0.5*he*dr2, th + 0.5*he*dth2,
               pr + 0.5*he*dpr2, pth + 0.5*he*dpth2, a, b, Q2,
               &dr3, &dth3, &dphi3, &dpr3, &dpth3);
        geoRHS(r + he*dr3, th + he*dth3,
               pr + he*dpr3, pth + he*dpth3, a, b, Q2,
               &dr4, &dth4, &dphi4, &dpr4, &dpth4);

        r   += he * (dr1   + 2.0*dr2   + 2.0*dr3   + dr4  ) / 6.0;
        th  += he * (dth1  + 2.0*dth2  + 2.0*dth3  + dth4 ) / 6.0;
        phi += he * (dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4) / 6.0;
        pr  += he * (dpr1  + 2.0*dpr2  + 2.0*dpr3  + dpr4 ) / 6.0;
        pth += he * (dpth1 + 2.0*dpth2 + 2.0*dpth3 + dpth4) / 6.0;

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) { done = true; break; }
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
            background(dx, dy, dz, p.bg_mode, p.star_layers, p.show_grid,
                       &bgr, &bgg, &bgb);
            float atten = 1.0f - fminf(sqrtf(cr*cr + cg*cg + cb*cb) * 0.3f, 0.9f);
            cr += bgr * atten; cg += bgg * atten; cb += bgb * atten;
            done = true; break;
        }
        if (r < 0.5 || r != r || th != th) { done = true; break; }
    }

    float ux = 2.0f * (ix + 0.5f) / p.width  - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / p.height - 1.0f;
    postProcess(&cr, &cg, &cb, alpha, beta, (float)p.spin, ux, uy);

    int idx = (iy * p.width + ix) * 3;
    output[idx + 0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx + 1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx + 2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
