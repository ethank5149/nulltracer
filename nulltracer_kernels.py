"""
nulltracer_kernels.py — Self-contained multi-integrator CUDA kernels
====================================================================

Usage in notebook:
    from nulltracer_kernels import compile_all, render_kerr, compare_integrators

Provides 4 integrators as separate CuPy RawKernels:
    rk4          — Classical 4th-order Runge-Kutta
    rkdp8        — Dormand-Prince 8(7) with adaptive step control
    tao_yoshida4 — Tao extended phase space + Yoshida 4th-order symplectic
    tao_kl8      — Tao extended phase space + Kahan-Li 8th-order symplectic

All kernels share identical:
    • geoRHS (general Hamiltonian, not Fuerst-Wu simplified)
    • Geometry-only adaptive stepping (no magic constants)
    • Initial conditions from exact null constraint H = 0
    • Output format: (class, r_disk, g_factor, xi) per pixel
"""

import math
import time as _time

import cupy as cp
import numpy as np

# ══════════════════════════════════════════════════════════════
#  SHARED CUDA HEADER
#  Included verbatim at the top of every kernel.
# ══════════════════════════════════════════════════════════════

_CUDA_HEADER = r"""
#define PI  3.14159265358979323846
#define S2_EPS 0.0004


/* ═══════════════════════════════════════════════════════════
 * Boyer-Lindquist geodesic functions
 * ═══════════════════════════════════════════════════════════ */

/* General Hamiltonian RHS in Boyer-Lindquist coordinates.
 * dp_r, dp_θ computed from ∂g^μν/∂r, ∂g^μν/∂θ — always correct
 * regardless of accumulated Hamiltonian drift. */
__device__ void geoRHS(
    double r, double th, double pr, double pth,
    double a, double b,
    double *dr, double *dth, double *dphi,
    double *dpr, double *dpth
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double c2 = cth * cth;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * c2;
    double del = r2 - 2.0 * r + a2;
    double sdel = fmax(del, 1e-14);
    double rpa2 = r2 + a2;
    double w = 2.0 * r;
    double A_ = rpa2 * rpa2 - sdel * a2 * s2;
    double isig = 1.0 / sig;
    double SD = sig * sdel;
    double iSD = 1.0 / SD;
    double is2 = 1.0 / s2;

    double grr   = sdel * isig;
    double gthth = isig;
    double gff   = (sig - w) * iSD * is2;
    double gtf   = -a * w * iSD;

    *dr   = grr * pr;
    *dth  = gthth * pth;
    *dphi = gff * b - gtf;

    double dsig_r = 2.0 * r;
    double ddel_r = 2.0 * r - 2.0;
    double dA_r   = 4.0 * r * rpa2 - ddel_r * a2 * s2;
    double dSD_r  = dsig_r * sdel + sig * ddel_r;
    double dgtt_r   = -(dA_r * SD - A_ * dSD_r) / (SD * SD);
    double dgtf_r   = -a * (2.0 * SD - w * dSD_r) / (SD * SD);
    double dgrr_r   = (ddel_r * sig - sdel * dsig_r) / (sig * sig);
    double dgthth_r = -dsig_r * isig * isig;
    double num_ff = sig - w;
    double den_ff = SD * s2;
    double dgff_r = ((dsig_r - 2.0) * den_ff - num_ff * dSD_r * s2) / (den_ff * den_ff);
    *dpr = -0.5 * (dgtt_r - 2.0 * b * dgtf_r + dgrr_r * pr * pr
                   + dgthth_r * pth * pth + dgff_r * b * b);

    double dsig_th = -2.0 * a2 * sth * cth;
    double ds2_th  = 2.0 * sth * cth;
    double dA_th   = -sdel * a2 * ds2_th;
    double dSD_th  = dsig_th * sdel;
    double dgtt_th   = -(dA_th * SD - A_ * dSD_th) / (SD * SD);
    double dgtf_th   =  a * w * dSD_th / (SD * SD);
    double dgrr_th   = -sdel * dsig_th / (sig * sig);
    double dgthth_th = -dsig_th * isig * isig;
    double dgff_th   = (dsig_th * den_ff - num_ff * (dsig_th * sdel * s2 + SD * ds2_th))
                       / (den_ff * den_ff);
    *dpth = -0.5 * (dgtt_th - 2.0 * b * dgtf_th + dgrr_th * pr * pr
                    + dgthth_th * pth * pth + dgff_th * b * b);
}


/* ═══════════════════════════════════════════════════════════
 * Ingoing Kerr coordinate functions (for symplectic integrators)
 * ═══════════════════════════════════════════════════════════ */

/* Velocity (drift) in Kerr coordinates */
__device__ void geoVelocityKS(
    double r, double th, double pr, double pth,
    double a, double b, double Q2,
    double *dr, double *dth, double *dphi
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * cth * cth;
    double del = r2 - 2.0 * r + a2 + Q2;
    double rpa2 = r2 + a2;
    double isig = 1.0 / sig;
    *dr   = (del * pr + a * b - rpa2) * isig;
    *dth  = pth * isig;
    *dphi = (a * pr - a + b / s2) * isig;
}

/* Force (kick) in Kerr coordinates */
__device__ void geoForceKS(
    double r, double th, double pr, double pth,
    double a, double b, double Q2,
    double *dpr, double *dpth
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * cth * cth;
    double isig = 1.0 / sig;
    *dpr  = ((1.0 - r) * pr * pr + 2.0 * r * pr) * isig;
    *dpth = cth * (b * b / (s2 * sth) - a2 * sth) * isig;
}

/* BL → Kerr-Schild momentum transform */
__device__ void transformBLtoKS(
    double r, double a, double b, double Q2, double *pr
) {
    double a2 = a * a, r2 = r * r;
    double del = r2 - 2.0 * r + a2 + Q2;
    *pr += (r2 + a2 - a * b) / del;
}

/* Hamiltonian projection in Kerr coordinates */
__device__ void projectHamiltonianKS(
    double r, double th, double *pr, double pth,
    double a, double b, double Q2
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double a2 = a * a, r2 = r * r;
    double del = r2 - 2.0 * r + a2 + Q2;
    double rpa2 = r2 + a2;
    double C = a2 * s2 - 2.0 * a * b + pth * pth + b * b / s2;
    double Bh = a * b - rpa2;

    if (fabs(del) < 1e-14) {
        if (fabs(Bh) > 1e-30) *pr = -C / (2.0 * Bh);
    } else {
        double disc = Bh * Bh - del * C;
        if (disc > 0.0) {
            double sq = sqrt(disc);
            double r1 = (-Bh + sq) / del;
            double r2 = (-Bh - sq) / del;
            *pr = (fabs(r1 - *pr) <= fabs(r2 - *pr)) ? r1 : r2;
        }
    }
}


/* ═══════════════════════════════════════════════════════════
 * Tao extended phase space infrastructure (Tao 2016)
 * ═══════════════════════════════════════════════════════════ */

#define TAO_OMEGA_C 2.0

__device__ void tao_rotate_coupled(
    double *q, double *p, double *qs, double *ps, double c, double s
) {
    double sq = *q + *qs, dq = *q - *qs;
    double sp = *p + *ps, dp = *p - *ps;
    double ndq = c * dq + s * dp;
    double ndp = -s * dq + c * dp;
    *q = 0.5 * (sq + ndq); *qs = 0.5 * (sq - ndq);
    *p = 0.5 * (sp + ndp); *ps = 0.5 * (sp - ndp);
}

__device__ void tao_rotate_cyclic(double *q, double *qs, double c) {
    double s = *q + *qs, d = *q - *qs;
    *q = 0.5 * (s + c * d); *qs = 0.5 * (s - c * d);
}

/* Tao φ_A, φ_B, φ_C flows and Strang base step — macros for zero overhead */
#define _TAO_PHI_A(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,tau) { \
    double _dr,_dth,_dphi,_dpr,_dpth; \
    geoVelocityKS(r,th,prs,pths,a,b,Q2,&_dr,&_dth,&_dphi); \
    geoForceKS(r,th,prs,pths,a,b,Q2,&_dpr,&_dpth); \
    pr+=tau*_dpr; pth+=tau*_dpth; \
    rs+=tau*_dr; ths+=tau*_dth; phis+=tau*_dphi; }

#define _TAO_PHI_B(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,tau) { \
    double _dr,_dth,_dphi,_dpr,_dpth; \
    geoVelocityKS(rs,ths,pr,pth,a,b,Q2,&_dr,&_dth,&_dphi); \
    geoForceKS(rs,ths,pr,pth,a,b,Q2,&_dpr,&_dpth); \
    r+=tau*_dr; th+=tau*_dth; phi+=tau*_dphi; \
    prs+=tau*_dpr; pths+=tau*_dpth; }

#define _TAO_PHI_C(r,th,phi,pr,pth,rs,ths,phis,prs,pths,ang) { \
    double _c=cos(ang), _s=sin(ang); \
    tao_rotate_coupled(&r,&pr,&rs,&prs,_c,_s); \
    tao_rotate_coupled(&th,&pth,&ths,&pths,_c,_s); \
    tao_rotate_cyclic(&phi,&phis,_c); }

#define _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d,ang) { \
    double _hd=0.5*(d); \
    _TAO_PHI_A(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,_hd) \
    _TAO_PHI_B(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,_hd) \
    _TAO_PHI_C(r,th,phi,pr,pth,rs,ths,phis,prs,pths,ang) \
    _TAO_PHI_B(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,_hd) \
    _TAO_PHI_A(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,_hd) }


/* ═══════════════════════════════════════════════════════════
 * Geometry-only adaptive step (shared by RK4 and symplectic)
 * ═══════════════════════════════════════════════════════════ */

__device__ double adaptive_step(double r, double rp, double h_base) {
    double x = fmax((r - rp) / rp, 0.0);
    double f = x / (1.0 + x);
    double h = h_base * fmax(f * (1.0 + 0.5 * x), 0.02);
    return fmin(h, r / 6.0);
}


/* Symplectic integrators need bounded steps — the Tao composition
 * substeps are w_i × h, and large h makes linear kicks inaccurate.
 * Cap at 1.0 (matching production adaptive_step_tao). */
__device__ double adaptive_step_symplectic(double r, double rp, double h_base) {
    double x = fmax((r - rp) / rp, 0.0);
    double f = x / (1.0 + x);
    double h = h_base * fmax(f * (1.0 + 0.5 * x), 0.02);
    return fmin(fmax(h, 0.012), 1.0);
}


/* ═══════════════════════════════════════════════════════════
 * Super-Hamiltonian diagnostic (BL coordinates)
 * ═══════════════════════════════════════════════════════════ */

__device__ double computeH_BL(
    double r, double th, double pr, double pth, double a, double b
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS, c2 = cth * cth;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * c2;
    double del = r2 - 2.0 * r + a2;
    double sdel = fmax(del, 1e-14);
    double rpa2 = r2 + a2;
    double w = 2.0 * r;
    double A_ = rpa2 * rpa2 - sdel * a2 * s2;
    double iSD = 1.0 / (sig * sdel);
    double is2 = 1.0 / s2;
    return 0.5 * (-A_ * iSD + 2.0 * b * (-a * w * iSD) + (sdel / sig) * pr * pr
                  + (1.0 / sig) * pth * pth + (sig - w) * iSD * is2 * b * b);
}


/* ═══════════════════════════════════════════════════════════
 * Shared ray initialization and output interface
 * ═══════════════════════════════════════════════════════════ */

/* Initialize ray from pixel coords. Returns BL state + impact param.
 * Solves H = 0 for p_r exactly at observer position. */
__device__ void initRayNotebook(
    int ix, int iy, int width, int height,
    double a, double theta_obs, double fov, double r_obs,
    double *r, double *th, double *phi, double *pr, double *pth,
    double *b_out, double *rp_out
) {
    double a2 = a * a;
    double aspect = (double)width / (double)height;
    double alpha = fov * aspect * (2.0 * ix / (double)(width - 1) - 1.0);
    double beta  = fov * (1.0 - 2.0 * iy / (double)(height - 1));
    double sobs = sin(theta_obs), cobs = cos(theta_obs);
    double b = -alpha * sobs;

    *r = r_obs;  *th = theta_obs;  *phi = 0.0;
    *pth = -beta;

    /* Solve H = 0 for p_r */
    double s2o = sobs * sobs + S2_EPS, c2o = cobs * cobs;
    double r02 = r_obs * r_obs;
    double sig0 = r02 + a2 * c2o;
    double del0 = r02 - 2.0 * r_obs + a2;
    double sdel0 = fmax(del0, 1e-14);
    double rpa2_0 = r02 + a2;
    double w0 = 2.0 * r_obs;
    double A_0 = rpa2_0 * rpa2_0 - sdel0 * a2 * s2o;
    double iSD0 = 1.0 / (sig0 * sdel0);
    double is2_0 = 1.0 / s2o;
    double grr0 = sdel0 / sig0;
    double gthth0 = 1.0 / sig0;

    double rest = -A_0 * iSD0 + 2.0 * a * b * w0 * iSD0
                  + gthth0 * beta * beta + (sig0 - w0) * iSD0 * is2_0 * b * b;
    double pr2 = -rest / grr0;
    *pr = (pr2 > 0.0) ? -sqrt(pr2) : 0.0;

    *rp_out = 1.0 + sqrt(fmax(1.0 - a2, 0.0));
    *b_out = b;
}
"""

# ══════════════════════════════════════════════════════════════
#  KERNEL BODIES (appended to header for each method)
# ══════════════════════════════════════════════════════════════

_KERNEL_RK4 = r"""
extern "C" __global__ void trace_kerr(
    double* out_class, double* out_rdisk, double* out_g, double* out_xi,
    int W, int H, double a, double thobs, double fov, double robs,
    double risco, int maxsteps, double hbase
) {
    int ix = blockIdx.x*blockDim.x+threadIdx.x;
    int iy = blockIdx.y*blockDim.y+threadIdx.y;
    if (ix>=W||iy>=H) return;
    int idx = iy*W+ix;

    double r,th,phi,pr,pth,b,rp;
    initRayNotebook(ix,iy,W,H,a,thobs,fov,robs,&r,&th,&phi,&pr,&pth,&b,&rp);
    double resc = robs + 12.0;
    int term=0; double rdh=0.0, gh=1.0, oldTh=th;

    for (int i=0; i<maxsteps; i++) {
        double he = adaptive_step(r, rp, hbase);
        double oldR=r; oldTh=th;
        double r0=r,th0=th,phi0=phi,pr0=pr,pth0=pth;
        double k1r,k1t,k1p,k1pr,k1pt, k2r,k2t,k2p,k2pr,k2pt;
        double k3r,k3t,k3p,k3pr,k3pt, k4r,k4t,k4p,k4pr,k4pt;
        geoRHS(r0,th0,pr0,pth0,a,b,&k1r,&k1t,&k1p,&k1pr,&k1pt);
        geoRHS(r0+.5*he*k1r,th0+.5*he*k1t,pr0+.5*he*k1pr,pth0+.5*he*k1pt,a,b,
               &k2r,&k2t,&k2p,&k2pr,&k2pt);
        geoRHS(r0+.5*he*k2r,th0+.5*he*k2t,pr0+.5*he*k2pr,pth0+.5*he*k2pt,a,b,
               &k3r,&k3t,&k3p,&k3pr,&k3pt);
        geoRHS(r0+he*k3r,th0+he*k3t,pr0+he*k3pr,pth0+he*k3pt,a,b,
               &k4r,&k4t,&k4p,&k4pr,&k4pt);
        r  =r0 +he/6.*(k1r +2.*k2r +2.*k3r +k4r);
        th =th0+he/6.*(k1t +2.*k2t +2.*k3t +k4t);
        phi=phi0+he/6.*(k1p+2.*k2p +2.*k3p +k4p);
        pr =pr0+he/6.*(k1pr+2.*k2pr+2.*k3pr+k4pr);
        pth=pth0+he/6.*(k1pt+2.*k2pt+2.*k3pt+k4pt);

        if(th<0.005){th=0.005;pth=fabs(pth);}
        if(th>PI-0.005){th=PI-0.005;pth=-fabs(pth);}
        if(r<=rp*1.01){term=1;break;}
        if(i>2&&rdh==0.0){
            double c=(oldTh-PI*.5)*(th-PI*.5);
            if(c<0.&&r>risco&&r<20.){
                double f=fabs(oldTh-PI*.5)/fmax(fabs(th-oldTh),1e-14);
                f=fmin(fmax(f,0.),1.); rdh=oldR+f*(r-oldR);
                double rh=rdh,sq=sqrt(rh),Om=1./(rh*sq+a);
                gh=sqrt(fmax(1.-3./rh+2.*a/(rh*sq),1e-12))/(fabs(1.-Om*b)+1e-30);
            }
        }
        if(r>resc){term=2;break;}
        if(r<0.5||r!=r||th!=th){term=1;break;}
    }
    if(term==0) term=1;
    out_class[idx]=(double)term; out_rdisk[idx]=rdh; out_g[idx]=gh; out_xi[idx]=b;
}
"""

_KERNEL_RKDP8 = r"""
extern "C" __global__ void trace_kerr(
    double* out_class, double* out_rdisk, double* out_g, double* out_xi,
    int W, int H, double a, double thobs, double fov, double robs,
    double risco, int maxsteps, double hbase
) {
    int ix = blockIdx.x*blockDim.x+threadIdx.x;
    int iy = blockIdx.y*blockDim.y+threadIdx.y;
    if (ix>=W||iy>=H) return;
    int idx = iy*W+ix;

    double r,th,phi,pr,pth,b,rp;
    initRayNotebook(ix,iy,W,H,a,thobs,fov,robs,&r,&th,&phi,&pr,&pth,&b,&rp);
    double resc = robs + 12.0;
    int term=0; double rdh=0.0, gh=1.0, oldTh=th;

    /* Adaptive step control */
    double atol=1e-8, rtol=1e-8, safety=0.9;
    double hmin=0.001, hmax=3.0;
    int max_reject=4;

    /* Initial step from geometry */
    double he = adaptive_step(r, rp, hbase);

    for (int i=0; i<maxsteps; i++) {
        double oldR=r; oldTh=th; double oldPhi=phi;
        int rejects=0; bool accepted=false;

        while (!accepted) {
            /* 13 stages — Dormand-Prince 8(7) coefficients */
            double k1r,k1t,k1p,k1pr,k1pt; double k2r,k2t,k2p,k2pr,k2pt;
            double k3r,k3t,k3p,k3pr,k3pt; double k4r,k4t,k4p,k4pr,k4pt;
            double k5r,k5t,k5p,k5pr,k5pt; double k6r,k6t,k6p,k6pr,k6pt;
            double k7r,k7t,k7p,k7pr,k7pt; double k8r,k8t,k8p,k8pr,k8pt;
            double k9r,k9t,k9p,k9pr,k9pt; double k10r,k10t,k10p,k10pr,k10pt;
            double k11r,k11t,k11p,k11pr,k11pt; double k12r,k12t,k12p,k12pr,k12pt;
            double k13r,k13t,k13p,k13pr,k13pt;

            geoRHS(r,th,pr,pth,a,b,&k1r,&k1t,&k1p,&k1pr,&k1pt);
            geoRHS(r+he*k1r/18.,th+he*k1t/18.,pr+he*k1pr/18.,pth+he*k1pt/18.,a,b,&k2r,&k2t,&k2p,&k2pr,&k2pt);
            geoRHS(r+he*(k1r/48.+k2r/16.),th+he*(k1t/48.+k2t/16.),pr+he*(k1pr/48.+k2pr/16.),pth+he*(k1pt/48.+k2pt/16.),a,b,&k3r,&k3t,&k3p,&k3pr,&k3pt);
            geoRHS(r+he*(k1r/32.+k3r*3./32.),th+he*(k1t/32.+k3t*3./32.),pr+he*(k1pr/32.+k3pr*3./32.),pth+he*(k1pt/32.+k3pt*3./32.),a,b,&k4r,&k4t,&k4p,&k4pr,&k4pt);
            geoRHS(r+he*(k1r*5./16.-k3r*75./64.+k4r*75./64.),th+he*(k1t*5./16.-k3t*75./64.+k4t*75./64.),pr+he*(k1pr*5./16.-k3pr*75./64.+k4pr*75./64.),pth+he*(k1pt*5./16.-k3pt*75./64.+k4pt*75./64.),a,b,&k5r,&k5t,&k5p,&k5pr,&k5pt);
            geoRHS(r+he*(k1r*3./80.+k4r*3./16.+k5r*3./20.),th+he*(k1t*3./80.+k4t*3./16.+k5t*3./20.),pr+he*(k1pr*3./80.+k4pr*3./16.+k5pr*3./20.),pth+he*(k1pt*3./80.+k4pt*3./16.+k5pt*3./20.),a,b,&k6r,&k6t,&k6p,&k6pr,&k6pt);

            double a71=29443841./614563906.,a74=77736538./692538347.,a75=-28693883./1125000000.,a76=23124283./1800000000.;
            geoRHS(r+he*(a71*k1r+a74*k4r+a75*k5r+a76*k6r),th+he*(a71*k1t+a74*k4t+a75*k5t+a76*k6t),pr+he*(a71*k1pr+a74*k4pr+a75*k5pr+a76*k6pr),pth+he*(a71*k1pt+a74*k4pt+a75*k5pt+a76*k6pt),a,b,&k7r,&k7t,&k7p,&k7pr,&k7pt);

            double a81=16016141./946692911.,a84=61564180./158732637.,a85=22789713./633445777.,a86=545815736./2771057229.,a87=-180193667./1043307555.;
            geoRHS(r+he*(a81*k1r+a84*k4r+a85*k5r+a86*k6r+a87*k7r),th+he*(a81*k1t+a84*k4t+a85*k5t+a86*k6t+a87*k7t),pr+he*(a81*k1pr+a84*k4pr+a85*k5pr+a86*k6pr+a87*k7pr),pth+he*(a81*k1pt+a84*k4pt+a85*k5pt+a86*k6pt+a87*k7pt),a,b,&k8r,&k8t,&k8p,&k8pr,&k8pt);

            double a91=39632708./573591083.,a94=-433636366./683701615.,a95=-421739975./2616292301.,a96=100302831./723423059.,a97=790204164./839813087.,a98=800635310./3783071287.;
            geoRHS(r+he*(a91*k1r+a94*k4r+a95*k5r+a96*k6r+a97*k7r+a98*k8r),th+he*(a91*k1t+a94*k4t+a95*k5t+a96*k6t+a97*k7t+a98*k8t),pr+he*(a91*k1pr+a94*k4pr+a95*k5pr+a96*k6pr+a97*k7pr+a98*k8pr),pth+he*(a91*k1pt+a94*k4pt+a95*k5pt+a96*k6pt+a97*k7pt+a98*k8pt),a,b,&k9r,&k9t,&k9p,&k9pr,&k9pt);

            double a101=246121993./1340847787.,a104=-37695042795./15268766246.,a105=-309121744./1061227803.,a106=-12992083./490766935.,a107=6005943493./2108947869.,a108=393006217./1396673457.,a109=123872331./1001029789.;
            geoRHS(r+he*(a101*k1r+a104*k4r+a105*k5r+a106*k6r+a107*k7r+a108*k8r+a109*k9r),th+he*(a101*k1t+a104*k4t+a105*k5t+a106*k6t+a107*k7t+a108*k8t+a109*k9t),pr+he*(a101*k1pr+a104*k4pr+a105*k5pr+a106*k6pr+a107*k7pr+a108*k8pr+a109*k9pr),pth+he*(a101*k1pt+a104*k4pt+a105*k5pt+a106*k6pt+a107*k7pt+a108*k8pt+a109*k9pt),a,b,&k10r,&k10t,&k10p,&k10pr,&k10pt);

            double a111=-1028468189./846180014.,a114=8478235783./508512852.,a115=1311729495./1432422823.,a116=-10304129995./1701304382.,a117=-48777925059./3047939560.,a118=15336726248./1032824649.,a119=-45442868181./3398467696.,a1110=3065993473./597172653.;
            geoRHS(r+he*(a111*k1r+a114*k4r+a115*k5r+a116*k6r+a117*k7r+a118*k8r+a119*k9r+a1110*k10r),th+he*(a111*k1t+a114*k4t+a115*k5t+a116*k6t+a117*k7t+a118*k8t+a119*k9t+a1110*k10t),pr+he*(a111*k1pr+a114*k4pr+a115*k5pr+a116*k6pr+a117*k7pr+a118*k8pr+a119*k9pr+a1110*k10pr),pth+he*(a111*k1pt+a114*k4pt+a115*k5pt+a116*k6pt+a117*k7pt+a118*k8pt+a119*k9pt+a1110*k10pt),a,b,&k11r,&k11t,&k11p,&k11pr,&k11pt);

            double a121=185892177./718116043.,a124=-3185094517./667107341.,a125=-477755414./1098053517.,a126=-703635378./230739211.,a127=5731566787./1027545527.,a128=5232866602./850066563.,a129=-4093664535./808688257.,a1210=3962137247./1805957418.,a1211=65686358./487910083.;
            geoRHS(r+he*(a121*k1r+a124*k4r+a125*k5r+a126*k6r+a127*k7r+a128*k8r+a129*k9r+a1210*k10r+a1211*k11r),th+he*(a121*k1t+a124*k4t+a125*k5t+a126*k6t+a127*k7t+a128*k8t+a129*k9t+a1210*k10t+a1211*k11t),pr+he*(a121*k1pr+a124*k4pr+a125*k5pr+a126*k6pr+a127*k7pr+a128*k8pr+a129*k9pr+a1210*k10pr+a1211*k11pr),pth+he*(a121*k1pt+a124*k4pt+a125*k5pt+a126*k6pt+a127*k7pt+a128*k8pt+a129*k9pt+a1210*k10pt+a1211*k11pt),a,b,&k12r,&k12t,&k12p,&k12pr,&k12pt);

            double a131=403863854./491063109.,a134=-5068492393./434740067.,a135=-411421997./543043805.,a136=652783627./914296604.,a137=11173962825./925320556.,a138=-13158990841./6184727034.,a139=3936647629./1978049680.,a1310=-160528059./685178525.,a1311=248638103./1413531060.;
            geoRHS(r+he*(a131*k1r+a134*k4r+a135*k5r+a136*k6r+a137*k7r+a138*k8r+a139*k9r+a1310*k10r+a1311*k11r),th+he*(a131*k1t+a134*k4t+a135*k5t+a136*k6t+a137*k7t+a138*k8t+a139*k9t+a1310*k10t+a1311*k11t),pr+he*(a131*k1pr+a134*k4pr+a135*k5pr+a136*k6pr+a137*k7pr+a138*k8pr+a139*k9pr+a1310*k10pr+a1311*k11pr),pth+he*(a131*k1pt+a134*k4pt+a135*k5pt+a136*k6pt+a137*k7pt+a138*k8pt+a139*k9pt+a1310*k10pt+a1311*k11pt),a,b,&k13r,&k13t,&k13p,&k13pr,&k13pt);

            /* 8th-order weights */
            double bw1=14005451./335480064.,bw6=-59238493./1068277825.,bw7=181606767./758867731.,bw8=561292985./797845732.,bw9=-1041891430./1371343529.,bw10=760417239./1151165299.,bw11=118820643./751138087.,bw12=-528747749./2220607170.,bw13=1./4.;
            /* 7th-order weights */
            double bh1=13451932./455176623.,bh6=-808719846./976000145.,bh7=1757004468./5645159321.,bh8=656045339./265891186.,bh9=-3867574721./1518517206.,bh10=465885868./322736535.,bh11=53011238./667516719.,bh12=2./45.;

            double dr8=he*(bw1*k1r+bw6*k6r+bw7*k7r+bw8*k8r+bw9*k9r+bw10*k10r+bw11*k11r+bw12*k12r+bw13*k13r);
            double dt8=he*(bw1*k1t+bw6*k6t+bw7*k7t+bw8*k8t+bw9*k9t+bw10*k10t+bw11*k11t+bw12*k12t+bw13*k13t);
            double dpr8=he*(bw1*k1pr+bw6*k6pr+bw7*k7pr+bw8*k8pr+bw9*k9pr+bw10*k10pr+bw11*k11pr+bw12*k12pr+bw13*k13pr);
            double dpt8=he*(bw1*k1pt+bw6*k6pt+bw7*k7pt+bw8*k8pt+bw9*k9pt+bw10*k10pt+bw11*k11pt+bw12*k12pt+bw13*k13pt);

            double er=he*((bw1-bh1)*k1r+(bw6-bh6)*k6r+(bw7-bh7)*k7r+(bw8-bh8)*k8r+(bw9-bh9)*k9r+(bw10-bh10)*k10r+(bw11-bh11)*k11r+(bw12-bh12)*k12r+bw13*k13r);
            double et=he*((bw1-bh1)*k1t+(bw6-bh6)*k6t+(bw7-bh7)*k7t+(bw8-bh8)*k8t+(bw9-bh9)*k9t+(bw10-bh10)*k10t+(bw11-bh11)*k11t+(bw12-bh12)*k12t+bw13*k13t);
            double epr=he*((bw1-bh1)*k1pr+(bw6-bh6)*k6pr+(bw7-bh7)*k7pr+(bw8-bh8)*k8pr+(bw9-bh9)*k9pr+(bw10-bh10)*k10pr+(bw11-bh11)*k11pr+(bw12-bh12)*k12pr+bw13*k13pr);
            double ept=he*((bw1-bh1)*k1pt+(bw6-bh6)*k6pt+(bw7-bh7)*k7pt+(bw8-bh8)*k8pt+(bw9-bh9)*k9pt+(bw10-bh10)*k10pt+(bw11-bh11)*k11pt+(bw12-bh12)*k12pt+bw13*k13pt);

            double sr=atol+rtol*fmax(fabs(r),fabs(r+dr8));
            double st=atol+rtol*fmax(fabs(th),fabs(th+dt8));
            double spr=atol+rtol*fmax(fabs(pr),fabs(pr+dpr8));
            double spt=atol+rtol*fmax(fabs(pth),fabs(pth+dpt8));
            double en=sqrt(0.25*((er/sr)*(er/sr)+(et/st)*(et/st)+(epr/spr)*(epr/spr)+(ept/spt)*(ept/spt)));

            if (en<=1.0||rejects>=max_reject) {
                r+=dr8; th+=dt8; pr+=dpr8; pth+=dpt8;
                phi+=he*(bw1*k1p+bw6*k6p+bw7*k7p+bw8*k8p+bw9*k9p+bw10*k10p+bw11*k11p+bw12*k12p+bw13*k13p);
                accepted=true;
                if(en>1e-30){double f=safety*pow(en,-1./8.); f=fmin(fmax(f,0.2),5.); he*=f;}
                he=fmin(fmax(he,hmin),hmax);
            } else {
                double f=fmax(safety*pow(en,-1./8.),0.2); he*=f; he=fmax(he,hmin);
                rejects++;
            }
        }
        if(th<0.005){th=0.005;pth=fabs(pth);}
        if(th>PI-0.005){th=PI-0.005;pth=-fabs(pth);}
        if(r<=rp*1.01){term=1;break;}
        if(i>2&&rdh==0.0){
            double c=(oldTh-PI*.5)*(th-PI*.5);
            if(c<0.&&r>risco&&r<20.){
                double f=fabs(oldTh-PI*.5)/fmax(fabs(th-oldTh),1e-14);
                f=fmin(fmax(f,0.),1.); rdh=oldR+f*(r-oldR);
                double rh=rdh,sq=sqrt(rh),Om=1./(rh*sq+a);
                gh=sqrt(fmax(1.-3./rh+2.*a/(rh*sq),1e-12))/(fabs(1.-Om*b)+1e-30);
            }
        }
        if(r>resc){term=2;break;}
        if(r<0.5||r!=r||th!=th){term=1;break;}
    }
    if(term==0) term=1;
    out_class[idx]=(double)term; out_rdisk[idx]=rdh; out_g[idx]=gh; out_xi[idx]=b;
}
"""

_KERNEL_TAO_Y4 = r"""
/* Yoshida 4th-order triple-jump coefficients */
#define Y4G  1.3512071919596576340
#define Y4G2 (-1.7024143839193152681)

extern "C" __global__ void trace_kerr(
    double* out_class, double* out_rdisk, double* out_g, double* out_xi,
    int W, int H, double a, double thobs, double fov, double robs,
    double risco, int maxsteps, double hbase
) {
    int ix = blockIdx.x*blockDim.x+threadIdx.x;
    int iy = blockIdx.y*blockDim.y+threadIdx.y;
    if (ix>=W||iy>=H) return;
    int idx = iy*W+ix;

    double r,th,phi,pr,pth,b,rp;
    initRayNotebook(ix,iy,W,H,a,thobs,fov,robs,&r,&th,&phi,&pr,&pth,&b,&rp);
    double Q2 = 0.0;  /* Kerr (no charge) */

    /* Transform BL → Kerr-Schild */
    transformBLtoKS(r, a, b, Q2, &pr);

    /* Shadow variables (Tao 2016) */
    double rs=r,ths=th,phis=phi,prs=pr,pths=pth;

    double resc = robs + 12.0;
    int term=0; double rdh=0.0, gh=1.0, oldTh=th;

    for (int i=0; i<maxsteps; i++) {
        double he = adaptive_step_symplectic(r, rp, hbase);
        double oldR=r; oldTh=th;

        /* Yoshida 4th: 3 Strang substeps */
        double d1=Y4G*he, d0=Y4G2*he;
        double a1=2.*TAO_OMEGA_C*Y4G, a0=2.*TAO_OMEGA_C*Y4G2;
        _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d1,a1)
        _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d0,a0)
        _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d1,a1)

        /* Project onto H=0 */
        projectHamiltonianKS(r, th, &pr, pth, a, b, Q2);

        if(th<0.005){th=0.005;pth=fabs(pth);}
        if(th>PI-0.005){th=PI-0.005;pth=-fabs(pth);}
        if(r<=rp*1.01){term=1;break;}
        if(i>2&&rdh==0.0){
            double c=(oldTh-PI*.5)*(th-PI*.5);
            if(c<0.&&r>risco&&r<20.){
                double f=fabs(oldTh-PI*.5)/fmax(fabs(th-oldTh),1e-14);
                f=fmin(fmax(f,0.),1.); rdh=oldR+f*(r-oldR);
                double rh=rdh,sq=sqrt(rh),Om=1./(rh*sq+a);
                gh=sqrt(fmax(1.-3./rh+2.*a/(rh*sq),1e-12))/(fabs(1.-Om*b)+1e-30);
            }
        }
        if(r>resc){term=2;break;}
        if(r<0.5||r!=r||th!=th){term=1;break;}
    }
    if(term==0) term=1;
    out_class[idx]=(double)term; out_rdisk[idx]=rdh; out_g[idx]=gh; out_xi[idx]=b;
}
"""

_KERNEL_TAO_KL8 = r"""
/* Kahan-Li s15odr8 optimal 8th-order palindromic coefficients */
#define KL0  0.74167036435061295345
#define KL1 -0.40910082580003159400
#define KL2  0.19075471029623837995
#define KL3 -0.57386247111608226666
#define KL4  0.29906418130365592384
#define KL5  0.33462491824529818378
#define KL6  0.31529309239676659663
#define KL7 -0.79688793935291635402

extern "C" __global__ void trace_kerr(
    double* out_class, double* out_rdisk, double* out_g, double* out_xi,
    int W, int H, double a, double thobs, double fov, double robs,
    double risco, int maxsteps, double hbase
) {
    int ix = blockIdx.x*blockDim.x+threadIdx.x;
    int iy = blockIdx.y*blockDim.y+threadIdx.y;
    if (ix>=W||iy>=H) return;
    int idx = iy*W+ix;

    double r,th,phi,pr,pth,b,rp;
    initRayNotebook(ix,iy,W,H,a,thobs,fov,robs,&r,&th,&phi,&pr,&pth,&b,&rp);
    double Q2 = 0.0;
    transformBLtoKS(r, a, b, Q2, &pr);
    double rs=r,ths=th,phis=phi,prs=pr,pths=pth;

    double resc = robs + 12.0;
    int term=0; double rdh=0.0, gh=1.0, oldTh=th;

    /* Precompute rotation angles (constant per ray) */
    double w[8] = {KL0,KL1,KL2,KL3,KL4,KL5,KL6,KL7};

    for (int i=0; i<maxsteps; i++) {
        double he = adaptive_step_symplectic(r, rp, hbase);
        double oldR=r; oldTh=th;

        /* 15-stage palindromic composition */
        for (int j=0; j<8; j++) {
            double d = w[j]*he;
            double ang = 2.0*TAO_OMEGA_C*w[j];
            _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d,ang)
        }
        for (int j=6; j>=0; j--) {
            double d = w[j]*he;
            double ang = 2.0*TAO_OMEGA_C*w[j];
            _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d,ang)
        }

        projectHamiltonianKS(r, th, &pr, pth, a, b, Q2);

        if(th<0.005){th=0.005;pth=fabs(pth);}
        if(th>PI-0.005){th=PI-0.005;pth=-fabs(pth);}
        if(r<=rp*1.01){term=1;break;}
        if(i>2&&rdh==0.0){
            double c=(oldTh-PI*.5)*(th-PI*.5);
            if(c<0.&&r>risco&&r<20.){
                double f=fabs(oldTh-PI*.5)/fmax(fabs(th-oldTh),1e-14);
                f=fmin(fmax(f,0.),1.); rdh=oldR+f*(r-oldR);
                double rh=rdh,sq=sqrt(rh),Om=1./(rh*sq+a);
                gh=sqrt(fmax(1.-3./rh+2.*a/(rh*sq),1e-12))/(fabs(1.-Om*b)+1e-30);
            }
        }
        if(r>resc){term=2;break;}
        if(r<0.5||r!=r||th!=th){term=1;break;}
    }
    if(term==0) term=1;
    out_class[idx]=(double)term; out_rdisk[idx]=rdh; out_g[idx]=gh; out_xi[idx]=b;
}
"""

# ══════════════════════════════════════════════════════════════
#  KERNEL REGISTRY
# ══════════════════════════════════════════════════════════════

METHODS = {
    'rk4':          ('Runge-Kutta 4th-order',              _KERNEL_RK4),
    'rkdp8':        ('Dormand-Prince 8(7) adaptive',       _KERNEL_RKDP8),
    'tao_yoshida4': ('Tao + Yoshida 4th symplectic',       _KERNEL_TAO_Y4),
    'tao_kl8':      ('Tao + Kahan-Li 8th symplectic',      _KERNEL_TAO_KL8),
}

_compiled = {}


def compile_all():
    """Compile all integrator kernels. Returns dict of method → RawKernel."""
    global _compiled
    for name, (label, body) in METHODS.items():
        if name not in _compiled:
            src = _CUDA_HEADER + body
            _compiled[name] = cp.RawKernel(src, 'trace_kerr')
            print(f"  Compiled: {label} ({name})")
    print(f"All {len(_compiled)} integrators ready.")
    return _compiled


def compile_one(method='rk4'):
    """Compile a single integrator kernel."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(METHODS.keys())}")
    if method not in _compiled:
        label, body = METHODS[method]
        _compiled[method] = cp.RawKernel(_CUDA_HEADER + body, 'trace_kerr')
    return _compiled[method]


# ══════════════════════════════════════════════════════════════
#  AUTO STEP BUDGET
# ══════════════════════════════════════════════════════════════

def auto_steps(obs_dist, h_base=0.3, rp=2.0, safety=3.0, symplectic=False):
    N_near = 20.0 / h_base
    if symplectic:
        # Symplectic steps capped at 1.0, so flat-space transit is O(r_obs)
        N_far = obs_dist / 1.0
    else:
        N_far = (2 * rp / h_base) * math.log(max(obs_dist / rp, 2.0))
    return max(int((N_near + N_far) * safety), 400)


# ══════════════════════════════════════════════════════════════
#  RENDER FUNCTION
# ══════════════════════════════════════════════════════════════

def render_kerr(spin, inclination_deg, width=512, height=512, fov=7.0,
                obs_dist=40.0, max_steps=None, step_size=0.3,
                method='rk4', show_disk=True, colormap='inferno'):
    """
    Render a Kerr black hole shadow and accretion disk.

    Parameters
    ----------
    method : str
        Integration method: 'rk4', 'rkdp8', 'tao_yoshida4', 'tao_kl8'.
    obs_dist : float
        Observer distance in M. Works at any distance (auto step budget).
    max_steps : int or None
        If None, computed automatically from obs_dist and step_size.

    Returns
    -------
    img : ndarray (H, W, 3) uint8
    shadow_mask : ndarray (H, W) bool
    info : dict with render_ms, max_steps, obs_dist, method
    """
    import matplotlib.pyplot as plt

    kern = compile_one(method)

    if max_steps is None:
        rp_est = 1.0 + np.sqrt(max(1.0 - spin**2, 0.0))
        max_steps = auto_steps(obs_dist, h_base=step_size, rp=rp_est, symplectic=method.startswith('tao'))

    incl_rad = np.radians(inclination_deg)
    # Import isco_kerr from notebook scope — or define inline
    a = spin
    z1 = 1.0 + (1.0 - a**2)**(1/3) * ((1.0 + a)**(1/3) + max(1.0 - a, 0.0)**(1/3))
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    r_isco_val = 3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))

    n = width * height
    d_class  = cp.zeros(n, dtype=cp.float64)
    d_rdisk  = cp.zeros(n, dtype=cp.float64)
    d_g      = cp.zeros(n, dtype=cp.float64)
    d_xi     = cp.zeros(n, dtype=cp.float64)

    block = (16, 16, 1)
    grid  = ((width + 15) // 16, (height + 15) // 16)

    t0 = _time.time()
    kern(grid, block, (
        d_class, d_rdisk, d_g, d_xi,
        np.int32(width), np.int32(height),
        np.float64(spin), np.float64(incl_rad),
        np.float64(fov), np.float64(obs_dist),
        np.float64(r_isco_val),
        np.int32(max_steps), np.float64(step_size)
    ))
    cp.cuda.Device(0).synchronize()
    ms = (_time.time() - t0) * 1000

    h_class  = d_class.get().reshape(height, width)
    h_rdisk  = d_rdisk.get().reshape(height, width)
    h_g      = d_g.get().reshape(height, width)

    shadow = (h_class == 1)
    escape = (h_class == 2)
    disk   = (h_rdisk > 0)

    img = np.zeros((height, width, 3), dtype=np.float64)
    img[escape] = [0.005, 0.005, 0.015]

    if show_disk and disk.any():
        cmap = plt.get_cmap(colormap)
        rv = h_rdisk[disk]
        gv = np.clip(h_g[disk], 0.05, 4.0)
        T = np.maximum(1.0 - np.sqrt(r_isco_val / rv), 0) * (r_isco_val / rv)**0.75
        T /= (T.max() + 1e-30)
        br = T * gv**4
        br = np.clip(br / (np.percentile(br, 99.5) + 1e-30), 0, 1)
        colors = cmap(br)[:, :3]
        img[disk] = colors * np.clip(br, 0.05, 1.0)[:, None]

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8), shadow, {
        'render_ms': ms, 'max_steps': max_steps,
        'obs_dist': obs_dist, 'method': method,
    }


# ══════════════════════════════════════════════════════════════
#  COMPARISON INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════

def compare_integrators(spin=0.6, inclination=80, obs_dist=40, step_size=0.2,
                        width=384, height=384, fov=7.0,
                        methods=None, fit_fn=None):
    """
    Run all integrators on the same scene and produce a comparison.

    Parameters
    ----------
    fit_fn : callable or None
        fit_ellipse_to_shadow function from notebook. If provided,
        measures shadow diameter and circularity for each method.

    Returns
    -------
    results : list of dicts
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if methods is None:
        methods = list(METHODS.keys())

    compile_all()
    results = []

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for ax, m in zip(axes, methods):
        img, shadow, info = render_kerr(
            spin, inclination, width, height, fov=fov,
            obs_dist=obs_dist, step_size=step_size, method=m,
            show_disk=True
        )
        ax.imshow(img)
        ax.set_title(f"{METHODS[m][0]}\n{info['render_ms']:.0f} ms, "
                     f"{info['max_steps']} steps", fontsize=10)
        ax.axis('off')

        row = {
            'method': m,
            'label': METHODS[m][0],
            'render_ms': info['render_ms'],
            'max_steps': info['max_steps'],
        }
        if fit_fn is not None:
            obs = fit_fn(shadow, fov=fov, img_size=width)
            if obs:
                row['diameter_M'] = obs['diameter_M']
                row['delta_C'] = obs['delta_C']
                row['circularity'] = obs['circularity']
        results.append(row)

    fig.suptitle(f'Integrator Comparison — $a={spin}$, '
                 f'$\\theta={inclination}°$, $r_{{obs}}={obs_dist}\\,M$',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    # Print table
    print(f"\n{'Method':<30} {'Time':>8} {'Steps':>7}", end='')
    if fit_fn:
        print(f" {'Diameter':>10} {'ΔC':>8}", end='')
    print()
    print("─" * 70)
    for r in results:
        print(f"{r['label']:<30} {r['render_ms']:>7.0f}ms {r['max_steps']:>7d}", end='')
        if 'diameter_M' in r:
            print(f" {r['diameter_M']:>9.4f}M {r['delta_C']:>7.4f}", end='')
        print()

    return results, fig
