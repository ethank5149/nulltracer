"""
nulltracer_kernels.py — Production-grade multi-integrator CUDA rendering
Auto-generated from production source files.
"""

import math
import time as _time

import cupy as cp
import numpy as np

_CUDA_HEADER = r"""
/* Auto-generated from production source files */
#define PI  3.14159265358979323846
#define TAU 6.28318530717958647693
#define S2_EPS 0.0004

/* ── Geodesic RHS ── */
__device__ void geoRHS(
    double r, double th, double pr, double pth,
    double a, double b, double Q2,
    double *dr, double *dth, double *dphi,
    double *dpr, double *dpth
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double c2 = cth * cth;
    double a2 = a * a;
    double r2 = r * r;
    double sig = r2 + a2 * c2;
    double del = r2 - 2.0 * r + a2 + Q2;
    double sdel = fmax(del, 1e-14);  /* float64: tighter floor than GLSL 1e-6 */
    double rpa2 = r2 + a2;
    double w = 2.0 * r - Q2;
    double A_ = rpa2 * rpa2 - sdel * a2 * s2;
    double isig = 1.0 / sig;
    double SD = sig * sdel;
    double iSD = 1.0 / SD;
    double is2 = 1.0 / s2;

    double grr = sdel * isig;
    double gthth = isig;
    double gff = (sig - w) * iSD * is2;
    double gtf = -a * w * iSD;

    *dr   = grr * pr;
    *dth  = gthth * pth;
    *dphi = gff * b - gtf;

    /* ∂/∂r derivatives for dp_r */
    double dsig_r = 2.0 * r;
    double ddel_r = 2.0 * r - 2.0;
    double dA_r = 4.0 * r * rpa2 - ddel_r * a2 * s2;
    double dSD_r = dsig_r * sdel + sig * ddel_r;
    double dgtt_r = -(dA_r * SD - A_ * dSD_r) / (SD * SD);
    double dgtf_r = -a * (2.0 * SD - w * dSD_r) / (SD * SD);
    double dgrr_r = (ddel_r * sig - sdel * dsig_r) / (sig * sig);
    double dgthth_r = -dsig_r * isig * isig;
    double num_ff = sig - w;
    double den_ff = SD * s2;
    double dgff_r = ((dsig_r - 2.0) * den_ff - num_ff * dSD_r * s2) / (den_ff * den_ff);
    *dpr = -0.5 * (dgtt_r - 2.0 * b * dgtf_r + dgrr_r * pr * pr
                   + dgthth_r * pth * pth + dgff_r * b * b);

    /* ∂/∂θ derivatives for dp_θ */
    double dsig_th = -2.0 * a2 * sth * cth;
    double ds2_th = 2.0 * sth * cth;
    double dA_th = -sdel * a2 * ds2_th;
    double dSD_th = dsig_th * sdel;
    double dgtt_th = -(dA_th * SD - A_ * dSD_th) / (SD * SD);
    double dgtf_th = a * w * dSD_th / (SD * SD);
    double dgrr_th = -sdel * dsig_th / (sig * sig);
    double dgthth_th = -dsig_th * isig * isig;
    double dgff_th = (dsig_th * den_ff - num_ff * (dsig_th * sdel * s2 + SD * ds2_th))
                     / (den_ff * den_ff);
    *dpth = -0.5 * (dgtt_th - 2.0 * b * dgtf_th + dgrr_th * pr * pr
                    + dgthth_th * pth * pth + dgff_th * b * b);
}


/* ── Kerr coordinate functions ── */
__device__ void geoVelocityKS(
    double r, double th, double pr, double pth,
    double a, double b, double Q2,
    double *dr, double *dth, double *dphi
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * cth * cth;          /* Σ = r² + a²cos²θ */
    double del = r2 - 2.0 * r + a2 + Q2;       /* Δ = r² - 2r + a² + Q² */
    double rpa2 = r2 + a2;                      /* r² + a² */
    double isig = 1.0 / sig;

    /* dr/dλ = [Δ·p_r + (a·b - (r²+a²))] / Σ
     * From ∂F/∂p_r = 2Δ·p_r + 2[ab - (r²+a²)], divided by 2Σ.
     * The (ab - rpa2) term comes from g^Vr·p_V + g^rφ·p_φ:
     *   (r²+a²)/Σ · (-1) + (a/Σ) · b = [ab - (r²+a²)]/Σ */
    *dr  = (del * pr + a * b - rpa2) * isig;

    /* dθ/dλ = p_θ / Σ  (identical to BL) */
    *dth = pth * isig;

    /* dφ/dλ = [a·p_r - a + b/sin²θ] / Σ
     * From ∂F/∂b = -2a + 2a·p_r + 2b/s², divided by 2Σ.
     * The -a term comes from g^Vφ·p_V = (a/Σ)·(-1) = -a/Σ */
    *dphi = (a * pr - a + b / s2) * isig;
}


__device__ void geoForceKS(
    double r, double th, double pr, double pth,
    double a, double b, double Q2,
    double *dpr, double *dpth
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * cth * cth;          /* Σ = r² + a²cos²θ */
    double isig = 1.0 / sig;

    /* dp_r/dλ = [(1-r)·p_r² + 2r·p_r] / Σ
     *
     * This is the complete radial force in Kerr coordinates.
     * Note: at large r with p_r ≈ 0, the force ≈ 0 (correct for
     * nearly-free propagation in the far field).
     * The p_r dependence (linear and quadratic terms) is the
     * signature of non-separability. */
    *dpr = ((1.0 - r) * pr * pr + 2.0 * r * pr) * isig;

    /* dp_θ/dλ = cosθ · [b²/(s²·sinθ) - a²·sinθ] / Σ
     *
     * The b²/sin³θ term is regularized using s² = sin²θ + ε
     * to prevent divergence at the poles.  The pole reflection
     * at θ = 0.005 (in the integrator) provides additional safety. */
    *dpth = cth * (b * b / (s2 * sth) - a2 * sth) * isig;
}


__device__ void transformBLtoKS(
    double r, double a, double b, double Q2,
    double *pr
) {
    double a2 = a * a, r2 = r * r;
    double del = r2 - 2.0 * r + a2 + Q2;       /* Δ at observer */

    /* p_r^KS = p_r^BL + (r² + a² - a·b) / Δ */
    *pr += (r2 + a2 - a * b) / del;
}


__device__ void projectHamiltonianKS(
    double r, double th, double *pr, double pth,
    double a, double b, double Q2
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * cth * cth;          /* Σ */
    double del = r2 - 2.0 * r + a2 + Q2;       /* Δ */
    double rpa2 = r2 + a2;                      /* r² + a² */

    /* Constant term: everything except p_r terms in F */
    double C = a2 * s2 - 2.0 * a * b + pth * pth + b * b / s2;

    /* Half of the linear coefficient: B = a·b - (r²+a²) */
    double Bhalf = a * b - rpa2;

    if (fabs(del) < 1e-14) {
        /* Near horizon: Δ ≈ 0, equation is linear in p_r.
         * 2·B·p_r + C = 0  →  p_r = -C / (2·B)
         * B = ab - (r²+a²) is large and negative, so well-defined. */
        if (fabs(Bhalf) > 1e-30) {
            *pr = -C / (2.0 * Bhalf);
        }
        /* If both Δ ≈ 0 and B ≈ 0, leave p_r unchanged */
    } else {
        /* Standard quadratic solve:
         * Δ·p_r² + 2·B·p_r + C = 0
         * disc = B² - Δ·C
         * p_r = (-B ± √disc) / Δ */
        double disc = Bhalf * Bhalf - del * C;
        if (disc > 0.0) {
            double sqrt_disc = sqrt(disc);
            /* Pick the root closest to the current p_r.
             *
             * In Kerr coordinates, p_r ≈ 0 for ingoing rays at
             * large r (unlike BL where p_r ≈ -1).  The simple
             * copysign(√disc, p_r) strategy fails when p_r ≈ 0
             * because tiny floating-point noise can flip the sign
             * and select the wrong (outgoing) root.
             *
             * Instead, compute both roots and pick the nearest one.
             * This is robust regardless of the sign of p_r. */
            double root_plus  = (-Bhalf + sqrt_disc) / del;
            double root_minus = (-Bhalf - sqrt_disc) / del;
            double d_plus  = fabs(root_plus  - *pr);
            double d_minus = fabs(root_minus - *pr);
            *pr = (d_plus <= d_minus) ? root_plus : root_minus;
        }
        /* If disc ≤ 0, no real solution (near turning point);
         * leave p_r unchanged — the error is small and transient. */
    }
}


/* ── Utility functions ── */
__device__ float hash2(float inx, float iny) {
    /* p = fract(p * vec2(443.8, 441.4)) */
    float px = inx * 443.8f;
    px = px - floorf(px);
    float py = iny * 441.4f;
    py = py - floorf(py);

    /* dot(p, p + 19.19) = p.x*(p.x+19.19) + p.y*(p.y+19.19) */
    float d = px * (px + 19.19f) + py * (py + 19.19f);

    /* p += dot(p, p+19.19)  — adds scalar to both components */
    px += d;
    py += d;

    /* return fract(p.x * p.y) */
    float prod = px * py;
    return prod - floorf(prod);
}


__device__ void cubeMap(float dx, float dy, float dz,
                        float *face, float *uv_x, float *uv_y) {
    float ax = fabsf(dx), ay = fabsf(dy), az = fabsf(dz);
    if (az >= ax && az >= ay) {
        *face = (dz > 0.0f) ? 0.0f : 1.0f;
        *uv_x = dx / az;
        *uv_y = dy / az;
    } else if (ax >= ay) {
        *face = (dx > 0.0f) ? 2.0f : 3.0f;
        *uv_x = dy / ax;
        *uv_y = dz / ax;
    } else {
        *face = (dy > 0.0f) ? 4.0f : 5.0f;
        *uv_x = dx / ay;
        *uv_y = dz / ay;
    }
}


__device__ float cubeChecker(float uv_x, float uv_y, float div) {
    float cx = floorf((uv_x * 0.5f + 0.5f) * div);
    float cy = floorf((uv_y * 0.5f + 0.5f) * div);
    float sum = cx + cy;
    return sum - 2.0f * floorf(sum * 0.5f);  /* mod(sum, 2) */
}


__device__ float cubeGrid(float uv_x, float uv_y, float div) {
    float fx = (uv_x * 0.5f + 0.5f) * div;
    float fy = (uv_y * 0.5f + 0.5f) * div;
    fx = fx - floorf(fx);
    fy = fy - floorf(fy);
    float mx = fabsf(fx - 0.5f) * 2.0f;
    float my = fabsf(fy - 0.5f) * 2.0f;
    float m = fmaxf(mx, my);
    /* smoothstep(0.88, 0.96, m) */
    if (m <= 0.88f) return 0.0f;
    if (m >= 0.96f) return 1.0f;
    float t = (m - 0.88f) / (0.96f - 0.88f);
    return t * t * (3.0f - 2.0f * t);
}


__device__ void faceColor(float face, float *r, float *g, float *b) {
    if (face < 0.5f)      { *r = 0.14f; *g = 0.08f; *b = 0.04f; }
    else if (face < 1.5f) { *r = 0.06f; *g = 0.05f; *b = 0.14f; }
    else if (face < 2.5f) { *r = 0.04f; *g = 0.12f; *b = 0.07f; }
    else if (face < 3.5f) { *r = 0.12f; *g = 0.04f; *b = 0.08f; }
    else if (face < 4.5f) { *r = 0.04f; *g = 0.08f; *b = 0.14f; }
    else                  { *r = 0.12f; *g = 0.10f; *b = 0.04f; }
}


__device__ float smoothstepf(float edge0, float edge1, float x) {
    float t = (x - edge0) / (edge1 - edge0);
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}


__device__ void sphereDir(double th, double ph,
                          float *dx, float *dy, float *dz) {
    double sth = sin(th);
    *dx = (float)(sth * cos(ph));
    *dy = (float)(sth * sin(ph));
    *dz = (float)(cos(th));
}


/* ── Color pipeline ── */
__device__ float linear_to_srgb(float c) {
    if (c <= 0.0031308f)
        return 12.92f * c;
    else
        return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}


__device__ float aces_curve(float x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return fminf(fmaxf((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f), 1.0f);
}


__device__ void blendColor(float sr, float sg, float sb, float sa,
                           float *dr, float *dg, float *db, float *da) {
    float one_minus_da = 1.0f - *da;
    float contrib = sa * one_minus_da;
    *dr += sr * contrib;
    *dg += sg * contrib;
    *db += sb * contrib;
    *da += contrib;
}


/* ── Disk ── */
/* ============================================================
 *  DISK — Accretion disk emission and color (float32)
 *
 *  Page-Thorne (1974) thin disk model with:
 *    - Novikov-Thorne radial flux F(r) with zero-torque ISCO BC
 *    - CIE 1931 Planck spectrum → linear sRGB via 256-entry LUT
 *    - Full GR redshift g-factor for Kerr-Newman metric
 *
 *  Requires geodesic_base.cu to be included first.
 * ============================================================ */



/* ── Planck spectrum → linear sRGB lookup table ───────────── */

/* 256 entries covering 1000K to 40000K.
 * Generated from CIE 1931 2-degree standard observer color
 * matching functions integrated against the Planck distribution,
 * then converted to linear sRGB via the D65 color matrix.
 * Normalized so max(R,G,B) = 1.0 at each temperature
 * (chromaticity only — intensity is handled separately). */

#define PLANCK_LUT_SIZE 256
#define PLANCK_T_MIN 1000.0f
#define PLANCK_T_MAX 40000.0f

__constant__ float planck_lut_r[PLANCK_LUT_SIZE] = {
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 0.972800f, 0.941644f, 0.912898f,
    0.886310f, 0.861662f, 0.838762f, 0.817443f, 0.797555f, 0.778970f, 0.761570f, 0.745254f,
    0.729928f, 0.715510f, 0.701928f, 0.689114f, 0.677009f, 0.665559f, 0.654716f, 0.644435f,
    0.634675f, 0.625401f, 0.616580f, 0.608179f, 0.600173f, 0.592534f, 0.585240f, 0.578269f,
    0.571601f, 0.565217f, 0.559102f, 0.553238f, 0.547612f, 0.542211f, 0.537021f, 0.532030f,
    0.527229f, 0.522608f, 0.518156f, 0.513865f, 0.509726f, 0.505733f, 0.501878f, 0.498154f,
    0.494555f, 0.491075f, 0.487709f, 0.484450f, 0.481295f, 0.478239f, 0.475277f, 0.472405f,
    0.469619f, 0.466916f, 0.464292f, 0.461743f, 0.459268f, 0.456862f, 0.454523f, 0.452249f,
    0.450036f, 0.447882f, 0.445786f, 0.443744f, 0.441756f, 0.439818f, 0.437929f, 0.436088f,
    0.434292f, 0.432540f, 0.430831f, 0.429162f, 0.427533f, 0.425942f, 0.424389f, 0.422870f,
    0.421386f, 0.419936f, 0.418518f, 0.417131f, 0.415774f, 0.414447f, 0.413148f, 0.411877f,
    0.410632f, 0.409413f, 0.408219f, 0.407050f, 0.405904f, 0.404781f, 0.403681f, 0.402602f,
    0.401544f, 0.400506f, 0.399488f, 0.398490f, 0.397510f, 0.396549f, 0.395605f, 0.394679f,
    0.393770f, 0.392877f, 0.391999f, 0.391138f, 0.390291f, 0.389460f, 0.388642f, 0.387839f,
    0.387049f, 0.386273f, 0.385510f, 0.384759f, 0.384021f, 0.383295f, 0.382580f, 0.381878f,
    0.381186f, 0.380505f, 0.379836f, 0.379176f, 0.378527f, 0.377888f, 0.377258f, 0.376639f,
    0.376028f, 0.375427f, 0.374835f, 0.374251f, 0.373676f, 0.373109f, 0.372551f, 0.372000f,
    0.371458f, 0.370923f, 0.370396f, 0.369876f, 0.369363f, 0.368857f, 0.368358f, 0.367867f,
    0.367381f, 0.366903f, 0.366430f, 0.365964f, 0.365504f, 0.365050f, 0.364602f, 0.364160f,
    0.363724f, 0.363293f, 0.362868f, 0.362448f, 0.362033f, 0.361623f, 0.361219f, 0.360820f,
    0.360425f, 0.360035f, 0.359650f, 0.359270f, 0.358894f, 0.358523f, 0.358156f, 0.357793f,
    0.357435f, 0.357081f, 0.356731f, 0.356385f, 0.356043f, 0.355705f, 0.355370f, 0.355040f,
    0.354713f, 0.354390f, 0.354070f, 0.353754f, 0.353441f, 0.353132f, 0.352826f, 0.352524f,
    0.352225f, 0.351929f, 0.351636f, 0.351346f, 0.351059f, 0.350775f, 0.350495f, 0.350217f,
    0.349942f, 0.349670f, 0.349400f, 0.349134f, 0.348870f, 0.348608f, 0.348350f, 0.348094f,
    0.347840f, 0.347589f, 0.347341f, 0.347094f, 0.346851f, 0.346609f, 0.346370f, 0.346134f,
    0.345899f, 0.345667f, 0.345437f, 0.345209f, 0.344983f, 0.344760f, 0.344538f, 0.344318f,
    0.344101f, 0.343886f, 0.343672f, 0.343461f, 0.343251f, 0.343043f, 0.342837f, 0.342633f,
};

__constant__ float planck_lut_g[PLANCK_LUT_SIZE] = {
    0.008653f, 0.045885f, 0.084078f, 0.122635f, 0.161132f, 0.199263f, 0.236808f, 0.273609f,
    0.309554f, 0.344565f, 0.378590f, 0.411597f, 0.443567f, 0.474496f, 0.504386f, 0.533246f,
    0.561092f, 0.587941f, 0.613815f, 0.638738f, 0.662736f, 0.685835f, 0.708063f, 0.729449f,
    0.750021f, 0.769807f, 0.788836f, 0.807136f, 0.824736f, 0.841661f, 0.857940f, 0.873597f,
    0.888660f, 0.903151f, 0.917096f, 0.930517f, 0.943437f, 0.929876f, 0.911376f, 0.894089f,
    0.877905f, 0.862726f, 0.848465f, 0.835045f, 0.822397f, 0.810458f, 0.799174f, 0.788495f,
    0.778375f, 0.768774f, 0.759654f, 0.750982f, 0.742726f, 0.734860f, 0.727357f, 0.720194f,
    0.713349f, 0.706803f, 0.700537f, 0.694535f, 0.688780f, 0.683259f, 0.677958f, 0.672865f,
    0.667968f, 0.663257f, 0.658722f, 0.654353f, 0.650143f, 0.646082f, 0.642163f, 0.638380f,
    0.634726f, 0.631194f, 0.627779f, 0.624475f, 0.621277f, 0.618180f, 0.615180f, 0.612273f,
    0.609454f, 0.606719f, 0.604066f, 0.601490f, 0.598988f, 0.596558f, 0.594196f, 0.591899f,
    0.589666f, 0.587493f, 0.585379f, 0.583321f, 0.581316f, 0.579363f, 0.577460f, 0.575606f,
    0.573798f, 0.572034f, 0.570314f, 0.568635f, 0.566996f, 0.565396f, 0.563834f, 0.562308f,
    0.560817f, 0.559360f, 0.557935f, 0.556542f, 0.555180f, 0.553848f, 0.552544f, 0.551268f,
    0.550020f, 0.548797f, 0.547600f, 0.546427f, 0.545279f, 0.544153f, 0.543051f, 0.541970f,
    0.540910f, 0.539871f, 0.538852f, 0.537852f, 0.536872f, 0.535909f, 0.534965f, 0.534039f,
    0.533129f, 0.532236f, 0.531359f, 0.530497f, 0.529651f, 0.528820f, 0.528003f, 0.527200f,
    0.526411f, 0.525636f, 0.524874f, 0.524124f, 0.523387f, 0.522662f, 0.521949f, 0.521248f,
    0.520558f, 0.519879f, 0.519210f, 0.518553f, 0.517905f, 0.517268f, 0.516640f, 0.516022f,
    0.515414f, 0.514815f, 0.514224f, 0.513643f, 0.513070f, 0.512505f, 0.511949f, 0.511401f,
    0.510861f, 0.510328f, 0.509803f, 0.509285f, 0.508775f, 0.508272f, 0.507775f, 0.507286f,
    0.506803f, 0.506327f, 0.505857f, 0.505393f, 0.504936f, 0.504485f, 0.504039f, 0.503600f,
    0.503166f, 0.502738f, 0.502315f, 0.501898f, 0.501485f, 0.501079f, 0.500677f, 0.500280f,
    0.499888f, 0.499501f, 0.499119f, 0.498741f, 0.498368f, 0.497999f, 0.497635f, 0.497275f,
    0.496919f, 0.496568f, 0.496220f, 0.495877f, 0.495538f, 0.495202f, 0.494870f, 0.494543f,
    0.494218f, 0.493898f, 0.493581f, 0.493267f, 0.492957f, 0.492651f, 0.492348f, 0.492048f,
    0.491751f, 0.491458f, 0.491167f, 0.490880f, 0.490596f, 0.490315f, 0.490037f, 0.489761f,
    0.489489f, 0.489219f, 0.488953f, 0.488689f, 0.488427f, 0.488168f, 0.487912f, 0.487659f,
    0.487408f, 0.487159f, 0.486913f, 0.486669f, 0.486428f, 0.486189f, 0.485953f, 0.485718f,
    0.485486f, 0.485257f, 0.485029f, 0.484803f, 0.484580f, 0.484359f, 0.484140f, 0.483923f,
    0.483708f, 0.483494f, 0.483283f, 0.483074f, 0.482867f, 0.482661f, 0.482458f, 0.482256f,
    0.482056f, 0.481858f, 0.481662f, 0.481467f, 0.481274f, 0.481083f, 0.480893f, 0.480705f,
};

__constant__ float planck_lut_b[PLANCK_LUT_SIZE] = {
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.001311f, 0.014628f,
    0.030921f, 0.050063f, 0.071885f, 0.096190f, 0.122767f, 0.151396f, 0.181859f, 0.213940f,
    0.247432f, 0.282137f, 0.317870f, 0.354457f, 0.391734f, 0.429553f, 0.467776f, 0.506279f,
    0.544946f, 0.583676f, 0.622374f, 0.660959f, 0.699356f, 0.737499f, 0.775331f, 0.812802f,
    0.849866f, 0.886487f, 0.922632f, 0.958274f, 0.993390f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
};


/* ── Planck spectrum color lookup with linear interpolation ── */

__device__ void blackbody(float T, float *out_r, float *out_g, float *out_b) {
    /* Clamp temperature to LUT range */
    float t = fminf(fmaxf(T, PLANCK_T_MIN), PLANCK_T_MAX);

    /* Compute fractional index into LUT */
    float frac = (t - PLANCK_T_MIN) / (PLANCK_T_MAX - PLANCK_T_MIN) * (float)(PLANCK_LUT_SIZE - 1);
    int idx = (int)frac;
    float w = frac - (float)idx;

    /* Clamp index to valid range */
    idx = min(max(idx, 0), PLANCK_LUT_SIZE - 2);

    /* Linear interpolation */
    *out_r = planck_lut_r[idx] * (1.0f - w) + planck_lut_r[idx + 1] * w;
    *out_g = planck_lut_g[idx] * (1.0f - w) + planck_lut_g[idx + 1] * w;
    *out_b = planck_lut_b[idx] * (1.0f - w) + planck_lut_b[idx + 1] * w;
}


/* ── Gravitational redshift / g-factor (float64) ──────────── */

/* Compute the gravitational redshift factor g = ν_obs / ν_emit
 * for a photon emitted from a circular Keplerian orbit in the
 * equatorial plane of a Kerr-Newman black hole.
 *
 * All intermediate computation in float64 for numerical stability
 * near the ISCO and horizon.  Returns float for use in color pipeline.
 *
 * Parameters:
 *   r   — radial coordinate of emission point (double)
 *   a   — dimensionless spin parameter (double)
 *   Q2  — charge squared, Q^2 (double)
 *   b   — photon impact parameter = -α sin(θ_obs) (double)
 */
__device__ float compute_g_factor(double r, double a, double Q2, double b) {
    /* All computation in float64 */
    double r2 = r * r;
    double a2 = a * a;
    double w = 2.0 * r - Q2;  /* w = 2Mr - Q^2, with M=1 */

    /* Covariant metric at equator (theta = pi/2, so Sigma = r^2) */
    double gtt = -(1.0 - w / r2);
    double gtph = -a * w / r2;
    double gphph = (r2 * r2 + a2 * r2 + a2 * w) / r2;

    /* Keplerian angular velocity from circular orbit condition
     * d/dr(g_tt) + 2*Omega*d/dr(g_tphi) + Omega^2*d/dr(g_phiphi) = 0 */
    double dgtt_dr = 2.0 * (Q2 - r) / (r2 * r);
    double dgtph_dr = 2.0 * a * (r - Q2) / (r2 * r);
    double dgphph_dr = 2.0 * r + a2 * (-2.0 / r2 + 2.0 * Q2 / (r2 * r));

    double disc = dgtph_dr * dgtph_dr - dgtt_dr * dgphph_dr;
    double Omega = (-dgtph_dr + sqrt(fmax(disc, 0.0))) / fmax(dgphph_dr, 1e-30);

    /* Four-velocity normalization: u^t = 1/sqrt(-(g_tt + 2*Omega*g_tphi + Omega^2*g_phiphi)) */
    double denom = -(gtt + 2.0 * Omega * gtph + Omega * Omega * gphph);
    double ut = 1.0 / sqrt(fmax(denom, 1e-30));

    /* g = 1 / (u^t * (1 - b * Omega)) */
    double one_minus_bOmega = 1.0 - b * Omega;
    double g = 1.0 / (ut * fmax(fabs(one_minus_bOmega), 1e-30));
    g = fmin(fmax(g, 0.01), 10.0);  // Clamp to physical range [0.01, 10.0]

    return (float)g;
}


/* ── Novikov-Thorne radial flux (float64) ─────────────────── */

/* Compute the Page-Thorne (1974) radial flux F(r) for a thin
 * accretion disk around a Kerr black hole with spin parameter a.
 *
 * Uses the Novikov-Thorne form:
 *   F(r) ∝ 1/(r² (r^{3/2} - a)) × [integral correction terms]
 *
 * The integral is evaluated analytically using the three roots
 * of x³ - 3x + 2a = 0 (where x = √r in M=1 units).
 *
 * Returns the normalized flux (peak = 1.0).
 *
 * Reference: Page & Thorne, ApJ 191, 499 (1974), Eq. 15n
 */
__device__ float novikov_thorne_flux(double r, double a, double r_isco) {
    /* x = sqrt(r), x_isco = sqrt(r_isco) */
    double x = sqrt(r);
    double x_i = sqrt(r_isco);
    double a2 = a * a;

    /* The three roots of x^2 (x^3 - 3x + 2a) = 0 that matter:
     * x0 = sqrt(r_ms) [= x_i, the ISCO]
     * x1, x2, x3 are roots of x^3 - 3x + 2a = 0
     *
     * For the Novikov-Thorne integral, we need the three roots
     * of y^3 - 3y + 2a = 0.  Using the trigonometric solution: */
    double cos_val = a;  /* cos(theta/3) where cos(theta) = a for |a| <= 1 */
    double theta = acos(fmin(fmax(cos_val, -1.0), 1.0));

    double y1 = 2.0 * cos(theta / 3.0);                    /* largest root */
    double y2 = 2.0 * cos((theta + 2.0 * PI) / 3.0);       /* middle root */
    double y3 = 2.0 * cos((theta + 4.0 * PI) / 3.0);       /* smallest root */

    /* Prefactor: 1 / (x^2 * (x^3 - 3x + 2a)) */
    double x3_term = x * x * x - 3.0 * x + 2.0 * a;
    if (fabs(x3_term) < 1e-14) return 0.0f;
    double prefactor = 1.0 / (x * x * x3_term);

    /* The Novikov-Thorne integral I(x) from x_isco to x:
     *
     * I(x) = x - x_i
     *       - (3/2) a ln(x / x_i)
     *       - 3(y1-a)^2 / (y1(y1-y2)(y1-y3)) * ln((x-y1)/(x_i-y1))
     *       - 3(y2-a)^2 / (y2(y2-y1)(y2-y3)) * ln((x-y2)/(x_i-y2))
     *       - 3(y3-a)^2 / (y3(y3-y1)(y3-y2)) * ln((x-y3)/(x_i-y3))
     */
    double I = x - x_i - 1.5 * a * log(fmax(x / x_i, 1e-30));

    /* Correction terms from the three roots */
    double roots[3] = {y1, y2, y3};
    for (int k = 0; k < 3; k++) {
        double yk = roots[k];
        double yk_minus_a = yk - a;

        /* Denominator: yk * product of (yk - yj) for j != k */
        double denom = yk;
        for (int j = 0; j < 3; j++) {
            if (j != k) denom *= (yk - roots[j]);
        }
        if (fabs(denom) < 1e-30) continue;

        double arg_x  = fabs(x - yk);
        double arg_xi = fabs(x_i - yk);
        if (arg_x < 1e-14 || arg_xi < 1e-14) continue;

        I -= 3.0 * yk_minus_a * yk_minus_a / denom * log(arg_x / arg_xi);
    }

    /* Full flux: F = prefactor * I */
    double F = prefactor * I;

    /* F should be non-negative for r > r_isco */
    return (float)fmax(F, 0.0);
}


/* ── Accretion disk color with configurable Doppler boost ──── */

__device__ void diskColor(float r, float ph, float a,
                          float isco, float disk_outer, float disk_temp,
                          float g_factor, int doppler_boost,
                          float *cr, float *cg, float *cb) {
    float ri = isco;

    /* Outside disk bounds → no emission */
    if (r < ri * 0.98f || r > disk_outer) {
        *cr = 0.0f; *cg = 0.0f; *cb = 0.0f;
        return;
    }

    /* --- Novikov-Thorne flux profile --- */
    float F = novikov_thorne_flux((double)r, (double)a, (double)ri);

    /* Compute peak flux for normalization (at ~1.36 * r_isco for Schwarzschild).
     * We sample a few points to find the approximate peak. */
    float F_max = 0.0f;
    for (int i = 1; i <= 20; i++) {
        float r_sample = ri * (1.0f + 0.5f * (float)i);
        float F_sample = novikov_thorne_flux((double)r_sample, (double)a, (double)ri);
        if (F_sample > F_max) F_max = F_sample;
    }
    F_max = fmaxf(F_max, 1e-10f);

    /* Normalized flux [0, 1] */
    float F_norm = fminf(F / F_max, 1.0f);

    /* --- Temperature from flux: T ∝ F^{1/4} --- */
    float T_base = 8000.0f * disk_temp;  /* Peak temperature scale */
    float T_emit = T_base * powf(fmaxf(F_norm, 0.0f), 0.25f);

    /* --- Intensity proportional to flux --- */
    float I = F_norm * 3.0f / fmaxf(r * 0.15f, 0.01f);

    /* --- Edge smoothing at outer boundary --- */
    I *= smoothstepf(disk_outer, disk_outer * 0.55f, r);

    /* --- Smooth inner edge (just outside ISCO) --- */
    I *= smoothstepf(ri * 0.98f, ri * 1.05f, r);

    /* --- Apply redshift based on doppler_boost mode --- */
    float g = g_factor;
    float T_obs, I_adjusted;
    if (doppler_boost == 0) {
        /* No beaming — use emitted values directly */
        T_obs = T_emit;
        I_adjusted = I;
    } else if (doppler_boost == 1) {
        /* Optically thin: g^3 */
        T_obs = g * T_emit;
        float g3 = g * g * g;
        I_adjusted = I * g3;
    } else {
        /* Optically thick (default): g^4 */
        T_obs = g * T_emit;
        float g4 = g * g * g * g;
        I_adjusted = I * g4;
    }

    /* --- Blackbody color from observed temperature (Planck LUT) --- */
    float col_r, col_g, col_b;
    blackbody(T_obs, &col_r, &col_g, &col_b);

    /* Turbulence texture */
    float tu  = 0.65f + 0.35f * hash2(r * 5.0f, ph * 3.0f);
    float tu2 = 0.8f  + 0.2f  * hash2(r * 18.0f, ph * 9.0f);

    *cr = col_r * I_adjusted * tu * tu2 * 3.2f;
    *cg = col_g * I_adjusted * tu * tu2 * 3.2f;
    *cb = col_b * I_adjusted * tu * tu2 * 3.2f;
}




/* ── Backgrounds ── */
/* ============================================================
 *  BACKGROUNDS — Procedural background rendering (float32)
 *
 *  Originally ported from GLSL background rendering functions.
 *  All functions take a Cartesian direction vector (dx, dy, dz)
 *  and never use θ or φ directly — pole-safe by construction.
 *
 *  Requires geodesic_base.cu to be included first (for hash,
 *  cubeMap, cubeChecker, cubeGrid, faceColor, smoothstepf).
 * ============================================================ */



/* ── Star field background ────────────────────────────────── */

__device__ void bgStars(float dx, float dy, float dz,
                        int star_layers,
                        float *cr, float *cg, float *cb) {
    float face, uv_x, uv_y;
    cubeMap(dx, dy, dz, &face, &uv_x, &uv_y);
    float fuv_x = uv_x * 0.5f + 0.5f;
    float fuv_y = uv_y * 0.5f + 0.5f;

    /* Milky Way band: bright near equator (dz ≈ 0) */
    float mw = expf(-8.0f * dz * dz);

    /* Base nebula color */
    *cr = 0.007f;  *cg = 0.008f;  *cb = 0.018f;
    *cr += 0.018f * mw;  *cg += 0.014f * mw;  *cb += 0.028f * mw;

    /* Nebula noise — cube-face-local */
    float ns1x = floorf(fuv_x * 5.0f + face * 7.0f);
    float ns1y = floorf(fuv_y * 5.0f + face * 7.0f);
    float h1 = hash2(ns1x, ns1y);
    *cr += 0.018f * h1 * mw * 0.5f;
    *cg += 0.006f * h1 * mw * 0.5f;
    *cb += 0.022f * h1 * mw * 0.5f;

    float ns2x = floorf(fuv_x * 3.0f + face * 13.0f + 40.0f);
    float ns2y = floorf(fuv_y * 3.0f + face * 13.0f + 40.0f);
    float h2 = hash2(ns2x, ns2y);
    *cr += 0.004f * h2 * 0.25f;
    *cg += 0.012f * h2 * 0.25f;
    *cb += 0.022f * h2 * 0.25f;

    /* Stars — cube-face-local cells */
    for (int L = 0; L < star_layers; L++) {
        float sc = 10.0f + (float)L * 14.0f;
        float cellx = floorf(fuv_x * sc);
        float celly = floorf(fuv_y * sc);
        float seedx = cellx + face * 100.0f + (float)L * 47.0f;
        float seedy = celly + face * 100.0f + (float)L * 47.0f;
        float h = hash2(seedx, seedy);
        if (h > 0.88f) {
            float spx = (cellx + 0.3f + 0.4f * hash2(seedx + 0.5f, seedy + 0.5f)) / sc;
            float spy = (celly + 0.3f + 0.4f * hash2(seedx + 1.5f, seedy + 1.5f)) / sc;
            float ddx = fuv_x - spx;
            float ddy = fuv_y - spy;
            float dist = sqrtf(ddx * ddx + ddy * ddy) * sc;
            float s = expf(-dist * dist * 5.0f);
            float t = hash2(seedx + 77.0f, seedy + 77.0f);
            float scr, scg, scb;
            if (t < 0.2f)       { scr = 1.0f; scg = 0.7f;  scb = 0.4f; }
            else if (t < 0.55f) { scr = 1.0f; scg = 0.95f; scb = 0.8f; }
            else if (t < 0.8f)  { scr = 0.8f; scg = 0.9f;  scb = 1.0f; }
            else                { scr = 0.6f; scg = 0.75f; scb = 1.0f; }
            float bright = 0.4f + 2.0f * hash2(seedx + 33.0f, seedy + 33.0f);
            *cr += scr * s * bright;
            *cg += scg * s * bright;
            *cb += scb * s * bright;
        }
    }
}


/* ── Checker background ───────────────────────────────────── */

__device__ void bgChecker(float dx, float dy, float dz,
                          float *cr, float *cg, float *cb) {
    float face, uv_x, uv_y;
    cubeMap(dx, dy, dz, &face, &uv_x, &uv_y);
    float check = cubeChecker(uv_x, uv_y, 6.0f);

    /* Base color: mix dark and light by checker */
    float dark_r = 0.05f, dark_g = 0.045f, dark_b = 0.065f;
    float lite_r = 0.09f, lite_g = 0.07f,  lite_b = 0.045f;
    *cr = dark_r + (lite_r - dark_r) * check;
    *cg = dark_g + (lite_g - dark_g) * check;
    *cb = dark_b + (lite_b - dark_b) * check;

    /* Face tint */
    float fr, fg, fb;
    faceColor(face, &fr, &fg, &fb);
    float tint = 0.4f + 0.15f * check;
    *cr += fr * tint;
    *cg += fg * tint;
    *cb += fb * tint;

    /* Grid lines */
    float grid = cubeGrid(uv_x, uv_y, 6.0f);
    *cr += 0.18f * grid;
    *cg += 0.15f * grid;
    *cb += 0.10f * grid;

    /* Face-edge seams */
    float edgeDist = 1.0f - fmaxf(fabsf(uv_x), fabsf(uv_y));
    float edgeFade = 1.0f - smoothstepf(0.0f, 0.04f, edgeDist);
    *cr += 0.06f * edgeFade;
    *cg += 0.05f * edgeFade;
    *cb += 0.04f * edgeFade;

    /* Equator highlight (dz ≈ 0) */
    float eqFade = 1.0f - smoothstepf(0.0f, 0.04f, fabsf(dz));
    *cr += 0.22f * eqFade;
    *cg += 0.14f * eqFade;
    *cb += 0.05f * eqFade;
}


/* ── Color-map background ─────────────────────────────────── */

__device__ void bgColorMap(float dx, float dy, float dz,
                           float *cr, float *cg, float *cb) {
    /* Direct axis → channel mapping. Inherently pole-safe. */
    *cr = 0.08f + 0.35f * (dx * 0.5f + 0.5f);
    *cg = 0.08f + 0.35f * (dy * 0.5f + 0.5f);
    *cb = 0.08f + 0.35f * fmaxf(-dz, 0.0f);
    *cr += 0.08f * fmaxf(dz, 0.0f);
    *cg += 0.04f * fmaxf(dz, 0.0f);
    *cb += 0.02f * fmaxf(dz, 0.0f);

    /* Gamma-like adjustment: pow(col, 0.8) */
    *cr = powf(fmaxf(*cr, 0.0f), 0.8f);
    *cg = powf(fmaxf(*cg, 0.0f), 0.8f);
    *cb = powf(fmaxf(*cb, 0.0f), 0.8f);

    /* Grid overlay */
    float face, uv_x, uv_y;
    cubeMap(dx, dy, dz, &face, &uv_x, &uv_y);
    float grid = cubeGrid(uv_x, uv_y, 6.0f);
    *cr += 0.12f * grid;
    *cg += 0.10f * grid;
    *cb += 0.08f * grid;

    /* Equator */
    float eqFade = 1.0f - smoothstepf(0.0f, 0.04f, fabsf(dz));
    *cr += 0.15f * eqFade;
    *cg += 0.12f * eqFade;
    *cb += 0.05f * eqFade;
}


/* ── Background dispatcher ────────────────────────────────── */

__device__ void background(float dx, float dy, float dz,
                           int bg_mode, int star_layers, int show_grid,
                           float *cr, float *cg, float *cb) {
    if (bg_mode == 0) {
        bgStars(dx, dy, dz, star_layers, cr, cg, cb);
    } else if (bg_mode == 1) {
        bgChecker(dx, dy, dz, cr, cg, cb);
    } else {
        bgColorMap(dx, dy, dz, cr, cg, cb);
    }

    /* Extra grid for stars mode */
    if (show_grid && bg_mode == 0) {
        float face, uv_x, uv_y;
        cubeMap(dx, dy, dz, &face, &uv_x, &uv_y);
        float grid = cubeGrid(uv_x, uv_y, 6.0f);
        *cr += 0.055f * grid;
        *cg += 0.04f  * grid;
        *cb += 0.028f * grid;
        float eqFade = 1.0f - smoothstepf(0.0f, 0.03f, fabsf(dz));
        *cr += 0.07f  * eqFade;
        *cg += 0.035f * eqFade;
        *cb += 0.02f  * eqFade;
    }
}




/* ── Tao infrastructure ── */
#define TAO_OMEGA_C 2.0

__device__ void tao_rotate_coupled(
    double *q, double *p,    /* real position and momentum */
    double *qs, double *ps,  /* shadow position and momentum */
    double c, double s       /* cos(2ωδ) and sin(2ωδ) */
) {
    double sum_q = *q + *qs, diff_q = *q - *qs;
    double sum_p = *p + *ps, diff_p = *p - *ps;
    double new_diff_q = c * diff_q + s * diff_p;
    double new_diff_p = -s * diff_q + c * diff_p;
    *q  = 0.5 * (sum_q + new_diff_q);
    *qs = 0.5 * (sum_q - new_diff_q);
    *p  = 0.5 * (sum_p + new_diff_p);
    *ps = 0.5 * (sum_p - new_diff_p);
}


__device__ void tao_rotate_cyclic(
    double *q, double *qs,   /* real and shadow cyclic coordinate */
    double c                 /* cos(2ωδ) */
) {
    double sum = *q + *qs, diff = *q - *qs;
    *q  = 0.5 * (sum + c * diff);
    *qs = 0.5 * (sum - c * diff);
}


#define _TAO_PHI_A(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, tau) \
    { \
        double dr_, dth_, dphi_, dpr_, dpth_; \
        geoVelocityKS(r, th, prs, pths, a, b, Q2, &dr_, &dth_, &dphi_); \
        geoForceKS(r, th, prs, pths, a, b, Q2, &dpr_, &dpth_); \
        /* Update real momenta (p) using -∂_q H(q,y) */ \
        pr   += (tau) * dpr_; \
        pth  += (tau) * dpth_; \
        /* Update shadow positions (x) using ∂_y H(q,y) */ \
        rs   += (tau) * dr_; \
        ths  += (tau) * dth_; \
        phis += (tau) * dphi_; \
    }


#define _TAO_PHI_B(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, tau) \
    { \
        double dr_, dth_, dphi_, dpr_, dpth_; \
        geoVelocityKS(rs, ths, pr, pth, a, b, Q2, &dr_, &dth_, &dphi_); \
        geoForceKS(rs, ths, pr, pth, a, b, Q2, &dpr_, &dpth_); \
        /* Update real positions (q) using ∂_p H(x,p) */ \
        r    += (tau) * dr_; \
        th   += (tau) * dth_; \
        phi  += (tau) * dphi_; \
        /* Update shadow momenta (y) using -∂_x H(x,p) */ \
        prs  += (tau) * dpr_; \
        pths += (tau) * dpth_; \
    }


#define _TAO_PHI_C(r, th, phi, pr, pth, rs, ths, phis, prs, pths, angle2) \
    { \
        double c_ = cos(angle2), s_ = sin(angle2); \
        tao_rotate_coupled(&r,  &pr,  &rs,  &prs,  c_, s_); \
        tao_rotate_coupled(&th, &pth, &ths, &pths, c_, s_); \
        tao_rotate_cyclic(&phi, &phis, c_); \
    }


#define _TAO_STRANG(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                    a, b, Q2, delta, angle2) \
    { \
        double half_delta_ = 0.5 * (delta); \
        _TAO_PHI_A(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, half_delta_) \
        _TAO_PHI_B(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, half_delta_) \
        _TAO_PHI_C(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   angle2) \
        _TAO_PHI_B(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, half_delta_) \
        _TAO_PHI_A(r, th, phi, pr, pth, rs, ths, phis, prs, pths, \
                   a, b, Q2, half_delta_) \
    }


/* ── Integrator steps ── */
#define TAO_Y4_GAMMA  1.3512071919596576340
#define TAO_Y4_GAMMA2 (-1.7024143839193152681)

__device__ void tao_yoshida4_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double *rs, double *ths, double *phis,
    double *prs, double *pths,
    double a, double b, double Q2, double he
) {
    /* ω = TAO_OMEGA_C / h.  For each Strang step with duration δ_i,
     * the rotation angle is 2ωδ_i = 2·TAO_OMEGA_C · (δ_i / h).
     * Since δ_i = γ·h or (1-2γ)·h, the angle is 2·TAO_OMEGA_C · γ
     * or 2·TAO_OMEGA_C · (1-2γ). */
    double d1 = TAO_Y4_GAMMA * he;
    double d0 = TAO_Y4_GAMMA2 * he;
    double angle1 = 2.0 * TAO_OMEGA_C * TAO_Y4_GAMMA;
    double angle0 = 2.0 * TAO_OMEGA_C * TAO_Y4_GAMMA2;

    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1, angle1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d0, angle0)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1, angle1)
}


#ifndef TAO_KL8_W0
#define TAO_KL8_W0  0.74167036435061295345   /* w₇ (outermost) */
#define TAO_KL8_W1 -0.40910082580003159400   /* w₆ */
#define TAO_KL8_W2  0.19075471029623837995   /* w₅ */
#define TAO_KL8_W3 -0.57386247111608226666   /* w₄ */
#define TAO_KL8_W4  0.29906418130365592384   /* w₃ */
#define TAO_KL8_W5  0.33462491824529818378   /* w₂ */
#define TAO_KL8_W6  0.31529309239676659663   /* w₁ */
#define TAO_KL8_W7 -0.79688793935291635402   /* w₀ (center) */
#endif

__device__ void tao_kahan_li8_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double *rs, double *ths, double *phis,
    double *prs, double *pths,
    double a, double b, double Q2, double he
) {
    /* Precompute step durations and rotation angles for each stage.
     * δ_i = w_i · h,  angle_i = 2ωδ_i = 2·TAO_OMEGA_C · w_i. */
    double d0 = TAO_KL8_W0 * he, a0 = 2.0 * TAO_OMEGA_C * TAO_KL8_W0;
    double d1 = TAO_KL8_W1 * he, a1 = 2.0 * TAO_OMEGA_C * TAO_KL8_W1;
    double d2 = TAO_KL8_W2 * he, a2 = 2.0 * TAO_OMEGA_C * TAO_KL8_W2;
    double d3 = TAO_KL8_W3 * he, a3 = 2.0 * TAO_OMEGA_C * TAO_KL8_W3;
    double d4 = TAO_KL8_W4 * he, a4 = 2.0 * TAO_OMEGA_C * TAO_KL8_W4;
    double d5 = TAO_KL8_W5 * he, a5 = 2.0 * TAO_OMEGA_C * TAO_KL8_W5;
    double d6 = TAO_KL8_W6 * he, a6 = 2.0 * TAO_OMEGA_C * TAO_KL8_W6;
    double d7 = TAO_KL8_W7 * he, a7 = 2.0 * TAO_OMEGA_C * TAO_KL8_W7;

    /* Forward half: stages w₇ through w₁ */
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d0, a0)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1, a1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d2, a2)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d3, a3)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d4, a4)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d5, a5)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d6, a6)

    /* Center stage: w₀ */
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d7, a7)

    /* Reverse half: stages w₁ through w₇ (palindromic) */
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d6, a6)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d5, a5)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d4, a4)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d3, a3)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d2, a2)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d1, a1)
    _TAO_STRANG(*r, *th, *phi, *pr, *pth, *rs, *ths, *phis, *prs, *pths,
                a, b, Q2, d0, a0)
}


__device__ void rk4_step(
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double a, double b, double Q2, double he
) {
    double r0 = *r, th0 = *th, phi0 = *phi, pr0 = *pr, pth0 = *pth;
    double dr1, dth1, dphi1, dpr1, dpth1;
    double dr2, dth2, dphi2, dpr2, dpth2;
    double dr3, dth3, dphi3, dpr3, dpth3;
    double dr4, dth4, dphi4, dpr4, dpth4;

    geoRHS(r0, th0, pr0, pth0, a, b, Q2,
           &dr1, &dth1, &dphi1, &dpr1, &dpth1);
    geoRHS(r0 + 0.5*he*dr1, th0 + 0.5*he*dth1,
           pr0 + 0.5*he*dpr1, pth0 + 0.5*he*dpth1, a, b, Q2,
           &dr2, &dth2, &dphi2, &dpr2, &dpth2);
    geoRHS(r0 + 0.5*he*dr2, th0 + 0.5*he*dth2,
           pr0 + 0.5*he*dpr2, pth0 + 0.5*he*dpth2, a, b, Q2,
           &dr3, &dth3, &dphi3, &dpr3, &dpth3);
    geoRHS(r0 + he*dr3, th0 + he*dth3,
           pr0 + he*dpr3, pth0 + he*dpth3, a, b, Q2,
           &dr4, &dth4, &dphi4, &dpr4, &dpth4);

    *r   = r0   + he * (dr1   + 2.0*dr2   + 2.0*dr3   + dr4  ) / 6.0;
    *th  = th0  + he * (dth1  + 2.0*dth2  + 2.0*dth3  + dth4 ) / 6.0;
    *phi = phi0 + he * (dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4) / 6.0;
    *pr  = pr0  + he * (dpr1  + 2.0*dpr2  + 2.0*dpr3  + dpr4 ) / 6.0;
    *pth = pth0 + he * (dpth1 + 2.0*dpth2 + 2.0*dpth3 + dpth4) / 6.0;
}



/* ── Adaptive step (geometry-only) ── */
__device__ double adaptive_step(double r, double rp, double h_base) {
    double x = fmax((r - rp) / rp, 0.0);
    double f = x / (1.0 + x);
    double h = h_base * fmax(f * (1.0 + 0.5 * x), 0.02);
    return fmin(h, r / 6.0);
}
__device__ double adaptive_step_symplectic(double r, double rp, double h_base) {
    double x = fmax((r - rp) / rp, 0.0);
    double f = x / (1.0 + x);
    double h = h_base * fmax(f * (1.0 + 0.5 * x), 0.02);
    return fmin(fmax(h, 0.012), 1.0);
}

/* ── Ray initialization (notebook interface) ── */
__device__ void initRayNotebook(
    int ix, int iy, int width, int height,
    double a, double theta_obs, double fov, double r_obs,
    double *r, double *th, double *phi, double *pr, double *pth,
    double *b_out, double *rp_out
) {
    double a2 = a * a;
    double Q2 = 0.0;
    double asp = (double)width / (double)height;
    double ux = (2.0 * (ix + 0.5) / (double)width  - 1.0);
    double uy = (2.0 * (iy + 0.5) / (double)height - 1.0);
    double alpha = ux * fov * asp;
    double beta  = uy * fov;
    double sO = sin(theta_obs), cO = cos(theta_obs);
    double b = -alpha * sO;
    *r = r_obs;  *th = theta_obs;  *phi = 0.0;
    double s2 = sO * sO + S2_EPS;
    double c2 = cO * cO;
    double r0 = r_obs, r02 = r0 * r0;
    double sig = r02 + a2 * c2;
    double del = r02 - 2.0 * r0 + a2 + Q2;
    double sdel = fmax(del, 1e-14);
    double rpa2 = r02 + a2;
    double A_ = rpa2 * rpa2 - sdel * a2 * s2;
    double iSD = 1.0 / (sig * sdel);
    double is2 = 1.0 / s2;
    double grr = sdel / sig;
    double gthi = 1.0 / sig;
    double w_init = 2.0 * r0 - Q2;
    *pth = -beta;
    double rest = -A_ * iSD + 2.0 * a * b * w_init * iSD
                  + gthi * beta * beta + (sig - w_init) * iSD * is2 * b * b;
    double pr2 = -rest / grr;
    *pr = (pr2 > 0.0) ? -sqrt(pr2) : 0.0;
    *rp_out = 1.0 + sqrt(fmax(1.0 - a2 - Q2, 0.0));
    *b_out = b;
}
"""

_KERNEL_RK4 = r"""

extern "C" __global__ void trace_kerr(
    unsigned char* output,
    int W, int H, double a, double thobs, double fov, double robs,
    double risco, int maxsteps, double hbase
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;

    double r, th, phi, pr, pth, b, rp;
    initRayNotebook(ix, iy, W, H, a, thobs, fov, robs,
                    &r, &th, &phi, &pr, &pth, &b, &rp);

    double Q2 = 0.0;
    float isco_f = (float)risco;
    float disk_outer = 14.0f;
    float disk_temp = 1.0f;
    int doppler_boost = 2;
    int bg_mode = 0;
    int star_layers = 3;
    int show_grid = 0;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_a = 0.0f;
    int disk_crossings = 0;
    int max_crossings = 5;
    float base_alpha = 0.95f;
    bool done = false;
    double sobs_r = sin(thobs);
    float alpha_f = (sobs_r != 0.0) ? (float)(-b / sobs_r) : 0.0f;
    float beta_f = (float)(-(pth));
    double resc = robs + 12.0;

    for (int i = 0; i < maxsteps; i++) {
        if (done) break;
        double he = adaptive_step(r, rp, hbase);
        double oldR = r, oldTh = th, oldPhi = phi;
        rk4_step(&r, &th, &phi, &pr, &pth, a, b, Q2, he);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) {
            blendColor(0.0f, 0.0f, 0.0f, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            done = true; break;
        }
        if (acc_a < 0.99f) {
            double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
            if (cross < 0.0 && disk_crossings < max_crossings) {
                double f = fmin(fmax(fabs(oldTh - PI * 0.5) /
                           fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                double r_hit = oldR + f * (r - oldR);
                float g = compute_g_factor(r_hit, a, Q2, b);
                float dcr, dcg, dcb;
                diskColor((float)r_hit, (float)(oldPhi + f * (phi - oldPhi)), (float)a,
                         isco_f, disk_outer, disk_temp, g, doppler_boost, &dcr, &dcg, &dcb);
                blendColor(dcr, dcg, dcb, base_alpha, &acc_r, &acc_g, &acc_b, &acc_a);
                disk_crossings++;
            }
        }
        if (r > resc) {
            double frac = fmin(fmax((resc - oldR) / fmax(r - oldR, 1e-14), 0.0), 1.0);
            double fth = oldTh + (th - oldTh) * frac;
            double fph = oldPhi + (phi - oldPhi) * frac;
            float dx, dy, dz;
            sphereDir(fth, fph, &dx, &dy, &dz);
            float bgr, bgg, bgb;
            background(dx, dy, dz, bg_mode, star_layers, show_grid, &bgr, &bgg, &bgb);
            if (acc_a < 1.0f) blendColor(bgr, bgg, bgb, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            done = true; break;
        }
        if (r < 0.5 || r != r || th != th) { done = true; break; }
    }

    float cr = acc_r, cg = acc_g, cb = acc_b;
    float ux = 2.0f * (ix + 0.5f) / (float)W - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / (float)H - 1.0f;
    float imp = sqrtf(alpha_f * alpha_f + beta_f * beta_f);
    float rc = 5.2f - 1.0f * (float)a;
    float dd = (imp - rc) / 0.3f;
    float glow = expf(-dd * dd) * 0.06f;
    cr += 0.1f * glow; cg += 0.07f * glow; cb += 0.04f * glow;
    float vig = 1.0f - 0.3f * (ux * ux + uy * uy);
    cr *= vig; cg *= vig; cb *= vig;
    float L = 0.2126f * cr + 0.7152f * cg + 0.0722f * cb;
    if (L > 1e-6f) { float Lm = aces_curve(L); float sc = Lm / L; cr *= sc; cg *= sc; cb *= sc; }
    cr = fminf(cr, 1.0f); cg = fminf(cg, 1.0f); cb = fminf(cb, 1.0f);
    cr = linear_to_srgb(fmaxf(cr, 0.0f));
    cg = linear_to_srgb(fmaxf(cg, 0.0f));
    cb = linear_to_srgb(fmaxf(cb, 0.0f));
    int idx = (iy * W + ix) * 3;
    output[idx+0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx+1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx+2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
"""

_KERNEL_RKDP8 = r"""

extern "C" __global__ void trace_kerr(
    unsigned char* output,
    int W, int H, double a, double thobs, double fov, double robs,
    double risco, int maxsteps, double hbase
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;

    double r, th, phi, pr, pth, b, rp;
    initRayNotebook(ix, iy, W, H, a, thobs, fov, robs,
                    &r, &th, &phi, &pr, &pth, &b, &rp);

    double Q2 = 0.0;
    float isco_f = (float)risco;
    float disk_outer = 14.0f;
    float disk_temp = 1.0f;
    int doppler_boost = 2;
    int bg_mode = 0;
    int star_layers = 3;
    int show_grid = 0;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_a = 0.0f;
    int disk_crossings = 0;
    int max_crossings = 5;
    float base_alpha = 0.95f;
    bool done = false;
    double sobs_r = sin(thobs);
    float alpha_f = (sobs_r != 0.0) ? (float)(-b / sobs_r) : 0.0f;
    float beta_f = (float)(-(pth));
    double resc = robs + 12.0;

    double atol = 1e-8, rtol = 1e-8, safety = 0.9;
    double hmin = 0.001, hmax = 3.0;
    int max_reject = 4;
    double he = adaptive_step(r, rp, hbase);

    for (int i = 0; i < maxsteps; i++) {
        if (done) break;
        double oldR = r, oldTh = th, oldPhi = phi;
        int rejects = 0; bool accepted = false;

        while (!accepted) {
            double kr1,kt1,kp1,kpr1,kpt1, kr2,kt2,kp2,kpr2,kpt2;
            double kr3,kt3,kp3,kpr3,kpt3, kr4,kt4,kp4,kpr4,kpt4;
            double kr5,kt5,kp5,kpr5,kpt5, kr6,kt6,kp6,kpr6,kpt6;
            double kr7,kt7,kp7,kpr7,kpt7, kr8,kt8,kp8,kpr8,kpt8;
            double kr9,kt9,kp9,kpr9,kpt9, kr10,kt10,kp10,kpr10,kpt10;
            double kr11,kt11,kp11,kpr11,kpt11, kr12,kt12,kp12,kpr12,kpt12;
            double kr13,kt13,kp13,kpr13,kpt13;

            geoRHS(r,th,pr,pth,a,b,Q2,&kr1,&kt1,&kp1,&kpr1,&kpt1);
            geoRHS(r+he*kr1/18.,th+he*kt1/18.,pr+he*kpr1/18.,pth+he*kpt1/18.,a,b,Q2,&kr2,&kt2,&kp2,&kpr2,&kpt2);
            geoRHS(r+he*(kr1/48.+kr2/16.),th+he*(kt1/48.+kt2/16.),pr+he*(kpr1/48.+kpr2/16.),pth+he*(kpt1/48.+kpt2/16.),a,b,Q2,&kr3,&kt3,&kp3,&kpr3,&kpt3);
            geoRHS(r+he*(kr1/32.+kr3*3./32.),th+he*(kt1/32.+kt3*3./32.),pr+he*(kpr1/32.+kpr3*3./32.),pth+he*(kpt1/32.+kpt3*3./32.),a,b,Q2,&kr4,&kt4,&kp4,&kpr4,&kpt4);
            geoRHS(r+he*(kr1*5./16.-kr3*75./64.+kr4*75./64.),th+he*(kt1*5./16.-kt3*75./64.+kt4*75./64.),pr+he*(kpr1*5./16.-kpr3*75./64.+kpr4*75./64.),pth+he*(kpt1*5./16.-kpt3*75./64.+kpt4*75./64.),a,b,Q2,&kr5,&kt5,&kp5,&kpr5,&kpt5);
            geoRHS(r+he*(kr1*3./80.+kr4*3./16.+kr5*3./20.),th+he*(kt1*3./80.+kt4*3./16.+kt5*3./20.),pr+he*(kpr1*3./80.+kpr4*3./16.+kpr5*3./20.),pth+he*(kpt1*3./80.+kpt4*3./16.+kpt5*3./20.),a,b,Q2,&kr6,&kt6,&kp6,&kpr6,&kpt6);
            double a71=29443841./614563906.,a74=77736538./692538347.,a75=-28693883./1125000000.,a76=23124283./1800000000.;
            geoRHS(r+he*(a71*kr1+a74*kr4+a75*kr5+a76*kr6),th+he*(a71*kt1+a74*kt4+a75*kt5+a76*kt6),pr+he*(a71*kpr1+a74*kpr4+a75*kpr5+a76*kpr6),pth+he*(a71*kpt1+a74*kpt4+a75*kpt5+a76*kpt6),a,b,Q2,&kr7,&kt7,&kp7,&kpr7,&kpt7);
            double a81=16016141./946692911.,a84=61564180./158732637.,a85=22789713./633445777.,a86=545815736./2771057229.,a87=-180193667./1043307555.;
            geoRHS(r+he*(a81*kr1+a84*kr4+a85*kr5+a86*kr6+a87*kr7),th+he*(a81*kt1+a84*kt4+a85*kt5+a86*kt6+a87*kt7),pr+he*(a81*kpr1+a84*kpr4+a85*kpr5+a86*kpr6+a87*kpr7),pth+he*(a81*kpt1+a84*kpt4+a85*kpt5+a86*kpt6+a87*kpt7),a,b,Q2,&kr8,&kt8,&kp8,&kpr8,&kpt8);
            double a91=39632708./573591083.,a94=-433636366./683701615.,a95=-421739975./2616292301.,a96=100302831./723423059.,a97=790204164./839813087.,a98=800635310./3783071287.;
            geoRHS(r+he*(a91*kr1+a94*kr4+a95*kr5+a96*kr6+a97*kr7+a98*kr8),th+he*(a91*kt1+a94*kt4+a95*kt5+a96*kt6+a97*kt7+a98*kt8),pr+he*(a91*kpr1+a94*kpr4+a95*kpr5+a96*kpr6+a97*kpr7+a98*kpr8),pth+he*(a91*kpt1+a94*kpt4+a95*kpt5+a96*kpt6+a97*kpt7+a98*kpt8),a,b,Q2,&kr9,&kt9,&kp9,&kpr9,&kpt9);
            double a101=246121993./1340847787.,a104=-37695042795./15268766246.,a105=-309121744./1061227803.,a106=-12992083./490766935.,a107=6005943493./2108947869.,a108=393006217./1396673457.,a109=123872331./1001029789.;
            geoRHS(r+he*(a101*kr1+a104*kr4+a105*kr5+a106*kr6+a107*kr7+a108*kr8+a109*kr9),th+he*(a101*kt1+a104*kt4+a105*kt5+a106*kt6+a107*kt7+a108*kt8+a109*kt9),pr+he*(a101*kpr1+a104*kpr4+a105*kpr5+a106*kpr6+a107*kpr7+a108*kpr8+a109*kpr9),pth+he*(a101*kpt1+a104*kpt4+a105*kpt5+a106*kpt6+a107*kpt7+a108*kpt8+a109*kpt9),a,b,Q2,&kr10,&kt10,&kp10,&kpr10,&kpt10);
            double a111=-1028468189./846180014.,a114=8478235783./508512852.,a115=1311729495./1432422823.,a116=-10304129995./1701304382.,a117=-48777925059./3047939560.,a118=15336726248./1032824649.,a119=-45442868181./3398467696.,a1110=3065993473./597172653.;
            geoRHS(r+he*(a111*kr1+a114*kr4+a115*kr5+a116*kr6+a117*kr7+a118*kr8+a119*kr9+a1110*kr10),th+he*(a111*kt1+a114*kt4+a115*kt5+a116*kt6+a117*kt7+a118*kt8+a119*kt9+a1110*kt10),pr+he*(a111*kpr1+a114*kpr4+a115*kpr5+a116*kpr6+a117*kpr7+a118*kpr8+a119*kpr9+a1110*kpr10),pth+he*(a111*kpt1+a114*kpt4+a115*kpt5+a116*kpt6+a117*kpt7+a118*kpt8+a119*kpt9+a1110*kpt10),a,b,Q2,&kr11,&kt11,&kp11,&kpr11,&kpt11);
            double a121=185892177./718116043.,a124=-3185094517./667107341.,a125=-477755414./1098053517.,a126=-703635378./230739211.,a127=5731566787./1027545527.,a128=5232866602./850066563.,a129=-4093664535./808688257.,a1210=3962137247./1805957418.,a1211=65686358./487910083.;
            geoRHS(r+he*(a121*kr1+a124*kr4+a125*kr5+a126*kr6+a127*kr7+a128*kr8+a129*kr9+a1210*kr10+a1211*kr11),th+he*(a121*kt1+a124*kt4+a125*kt5+a126*kt6+a127*kt7+a128*kt8+a129*kt9+a1210*kt10+a1211*kt11),pr+he*(a121*kpr1+a124*kpr4+a125*kpr5+a126*kpr6+a127*kpr7+a128*kpr8+a129*kpr9+a1210*kpr10+a1211*kpr11),pth+he*(a121*kpt1+a124*kpt4+a125*kpt5+a126*kpt6+a127*kpt7+a128*kpt8+a129*kpt9+a1210*kpt10+a1211*kpt11),a,b,Q2,&kr12,&kt12,&kp12,&kpr12,&kpt12);
            double a131=403863854./491063109.,a134=-5068492393./434740067.,a135=-411421997./543043805.,a136=652783627./914296604.,a137=11173962825./925320556.,a138=-13158990841./6184727034.,a139=3936647629./1978049680.,a1310=-160528059./685178525.,a1311=248638103./1413531060.;
            geoRHS(r+he*(a131*kr1+a134*kr4+a135*kr5+a136*kr6+a137*kr7+a138*kr8+a139*kr9+a1310*kr10+a1311*kr11),th+he*(a131*kt1+a134*kt4+a135*kt5+a136*kt6+a137*kt7+a138*kt8+a139*kt9+a1310*kt10+a1311*kt11),pr+he*(a131*kpr1+a134*kpr4+a135*kpr5+a136*kpr6+a137*kpr7+a138*kpr8+a139*kpr9+a1310*kpr10+a1311*kpr11),pth+he*(a131*kpt1+a134*kpt4+a135*kpt5+a136*kpt6+a137*kpt7+a138*kpt8+a139*kpt9+a1310*kpt10+a1311*kpt11),a,b,Q2,&kr13,&kt13,&kp13,&kpr13,&kpt13);

            double bw1=14005451./335480064.,bw6=-59238493./1068277825.,bw7=181606767./758867731.,bw8=561292985./797845732.,bw9=-1041891430./1371343529.,bw10=760417239./1151165299.,bw11=118820643./751138087.,bw12=-528747749./2220607170.,bw13=1./4.;
            double bh1=13451932./455176623.,bh6=-808719846./976000145.,bh7=1757004468./5645159321.,bh8=656045339./265891186.,bh9=-3867574721./1518517206.,bh10=465885868./322736535.,bh11=53011238./667516719.,bh12=2./45.;

            double dr8=he*(bw1*kr1+bw6*kr6+bw7*kr7+bw8*kr8+bw9*kr9+bw10*kr10+bw11*kr11+bw12*kr12+bw13*kr13);
            double dt8=he*(bw1*kt1+bw6*kt6+bw7*kt7+bw8*kt8+bw9*kt9+bw10*kt10+bw11*kt11+bw12*kt12+bw13*kt13);
            double dpr8=he*(bw1*kpr1+bw6*kpr6+bw7*kpr7+bw8*kpr8+bw9*kpr9+bw10*kpr10+bw11*kpr11+bw12*kpr12+bw13*kpr13);
            double dpt8=he*(bw1*kpt1+bw6*kpt6+bw7*kpt7+bw8*kpt8+bw9*kpt9+bw10*kpt10+bw11*kpt11+bw12*kpt12+bw13*kpt13);

            double er=he*((bw1-bh1)*kr1+(bw6-bh6)*kr6+(bw7-bh7)*kr7+(bw8-bh8)*kr8+(bw9-bh9)*kr9+(bw10-bh10)*kr10+(bw11-bh11)*kr11+(bw12-bh12)*kr12+bw13*kr13);
            double et=he*((bw1-bh1)*kt1+(bw6-bh6)*kt6+(bw7-bh7)*kt7+(bw8-bh8)*kt8+(bw9-bh9)*kt9+(bw10-bh10)*kt10+(bw11-bh11)*kt11+(bw12-bh12)*kt12+bw13*kt13);
            double epr_v=he*((bw1-bh1)*kpr1+(bw6-bh6)*kpr6+(bw7-bh7)*kpr7+(bw8-bh8)*kpr8+(bw9-bh9)*kpr9+(bw10-bh10)*kpr10+(bw11-bh11)*kpr11+(bw12-bh12)*kpr12+bw13*kpr13);
            double ept=he*((bw1-bh1)*kpt1+(bw6-bh6)*kpt6+(bw7-bh7)*kpt7+(bw8-bh8)*kpt8+(bw9-bh9)*kpt9+(bw10-bh10)*kpt10+(bw11-bh11)*kpt11+(bw12-bh12)*kpt12+bw13*kpt13);

            double sr=atol+rtol*fmax(fabs(r),fabs(r+dr8));
            double st=atol+rtol*fmax(fabs(th),fabs(th+dt8));
            double spr_v=atol+rtol*fmax(fabs(pr),fabs(pr+dpr8));
            double spt=atol+rtol*fmax(fabs(pth),fabs(pth+dpt8));
            double en=sqrt(0.25*((er/sr)*(er/sr)+(et/st)*(et/st)+(epr_v/spr_v)*(epr_v/spr_v)+(ept/spt)*(ept/spt)));

            if (en<=1.0||rejects>=max_reject) {
                r+=dr8; th+=dt8; pr+=dpr8; pth+=dpt8;
                phi+=he*(bw1*kp1+bw6*kp6+bw7*kp7+bw8*kp8+bw9*kp9+bw10*kp10+bw11*kp11+bw12*kp12+bw13*kp13);
                accepted=true;
                if(en>1e-30){double f=safety*pow(en,-1./8.); f=fmin(fmax(f,0.2),5.); he*=f;}
                he=fmin(fmax(he,hmin),hmax);
            } else {
                double f=fmax(safety*pow(en,-1./8.),0.2); he*=f; he=fmax(he,hmin);
                rejects++;
            }
        }

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) {
            blendColor(0.0f, 0.0f, 0.0f, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            done = true; break;
        }
        if (acc_a < 0.99f) {
            double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
            if (cross < 0.0 && disk_crossings < max_crossings) {
                double f = fmin(fmax(fabs(oldTh - PI * 0.5) /
                           fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                double r_hit = oldR + f * (r - oldR);
                float g = compute_g_factor(r_hit, a, Q2, b);
                float dcr, dcg, dcb;
                diskColor((float)r_hit, (float)(oldPhi + f * (phi - oldPhi)), (float)a,
                         isco_f, disk_outer, disk_temp, g, doppler_boost, &dcr, &dcg, &dcb);
                blendColor(dcr, dcg, dcb, base_alpha, &acc_r, &acc_g, &acc_b, &acc_a);
                disk_crossings++;
            }
        }
        if (r > resc) {
            double frac = fmin(fmax((resc - oldR) / fmax(r - oldR, 1e-14), 0.0), 1.0);
            double fth = oldTh + (th - oldTh) * frac;
            double fph = oldPhi + (phi - oldPhi) * frac;
            float dx, dy, dz;
            sphereDir(fth, fph, &dx, &dy, &dz);
            float bgr, bgg, bgb;
            background(dx, dy, dz, bg_mode, star_layers, show_grid, &bgr, &bgg, &bgb);
            if (acc_a < 1.0f) blendColor(bgr, bgg, bgb, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            done = true; break;
        }
        if (r < 0.5 || r != r || th != th) { done = true; break; }
    }

    float cr = acc_r, cg = acc_g, cb = acc_b;
    float ux = 2.0f * (ix + 0.5f) / (float)W - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / (float)H - 1.0f;
    float imp = sqrtf(alpha_f * alpha_f + beta_f * beta_f);
    float rc = 5.2f - 1.0f * (float)a;
    float dd = (imp - rc) / 0.3f;
    float glow = expf(-dd * dd) * 0.06f;
    cr += 0.1f * glow; cg += 0.07f * glow; cb += 0.04f * glow;
    float vig = 1.0f - 0.3f * (ux * ux + uy * uy);
    cr *= vig; cg *= vig; cb *= vig;
    float L = 0.2126f * cr + 0.7152f * cg + 0.0722f * cb;
    if (L > 1e-6f) { float Lm = aces_curve(L); float sc = Lm / L; cr *= sc; cg *= sc; cb *= sc; }
    cr = fminf(cr, 1.0f); cg = fminf(cg, 1.0f); cb = fminf(cb, 1.0f);
    cr = linear_to_srgb(fmaxf(cr, 0.0f));
    cg = linear_to_srgb(fmaxf(cg, 0.0f));
    cb = linear_to_srgb(fmaxf(cb, 0.0f));
    int idx = (iy * W + ix) * 3;
    output[idx+0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx+1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx+2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
"""

_KERNEL_TAO_Y4 = r"""

extern "C" __global__ void trace_kerr(
    unsigned char* output,
    int W, int H, double a, double thobs, double fov, double robs,
    double risco, int maxsteps, double hbase
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;

    double r, th, phi, pr, pth, b, rp;
    initRayNotebook(ix, iy, W, H, a, thobs, fov, robs,
                    &r, &th, &phi, &pr, &pth, &b, &rp);

    double Q2 = 0.0;
    transformBLtoKS(r, a, b, Q2, &pr);
    double rs=r, ths=th, phis=phi, prs=pr, pths=pth;

    double Q2 = 0.0;
    float isco_f = (float)risco;
    float disk_outer = 14.0f;
    float disk_temp = 1.0f;
    int doppler_boost = 2;
    int bg_mode = 0;
    int star_layers = 3;
    int show_grid = 0;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_a = 0.0f;
    int disk_crossings = 0;
    int max_crossings = 5;
    float base_alpha = 0.95f;
    bool done = false;
    double sobs_r = sin(thobs);
    float alpha_f = (sobs_r != 0.0) ? (float)(-b / sobs_r) : 0.0f;
    float beta_f = (float)(-(pth));
    double resc = robs + 12.0;

    for (int i = 0; i < maxsteps; i++) {
        if (done) break;
        double he = adaptive_step_symplectic(r, rp, hbase);
        double oldR = r, oldTh = th, oldPhi = phi;
        double d1=TAO_Y4_GAMMA*he, d0=TAO_Y4_GAMMA2*he;
        double ang1=2.*TAO_OMEGA_C*TAO_Y4_GAMMA, ang0=2.*TAO_OMEGA_C*TAO_Y4_GAMMA2;
        _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d1,ang1)
        _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d0,ang0)
        _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d1,ang1)
        projectHamiltonianKS(r, th, &pr, pth, a, b, Q2);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) {
            blendColor(0.0f, 0.0f, 0.0f, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            done = true; break;
        }
        if (acc_a < 0.99f) {
            double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
            if (cross < 0.0 && disk_crossings < max_crossings) {
                double f = fmin(fmax(fabs(oldTh - PI * 0.5) /
                           fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                double r_hit = oldR + f * (r - oldR);
                float g = compute_g_factor(r_hit, a, Q2, b);
                float dcr, dcg, dcb;
                diskColor((float)r_hit, (float)(oldPhi + f * (phi - oldPhi)), (float)a,
                         isco_f, disk_outer, disk_temp, g, doppler_boost, &dcr, &dcg, &dcb);
                blendColor(dcr, dcg, dcb, base_alpha, &acc_r, &acc_g, &acc_b, &acc_a);
                disk_crossings++;
            }
        }
        if (r > resc) {
            double frac = fmin(fmax((resc - oldR) / fmax(r - oldR, 1e-14), 0.0), 1.0);
            double fth = oldTh + (th - oldTh) * frac;
            double fph = oldPhi + (phi - oldPhi) * frac;
            float dx, dy, dz;
            sphereDir(fth, fph, &dx, &dy, &dz);
            float bgr, bgg, bgb;
            background(dx, dy, dz, bg_mode, star_layers, show_grid, &bgr, &bgg, &bgb);
            if (acc_a < 1.0f) blendColor(bgr, bgg, bgb, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            done = true; break;
        }
        if (r < 0.5 || r != r || th != th) { done = true; break; }
    }

    float cr = acc_r, cg = acc_g, cb = acc_b;
    float ux = 2.0f * (ix + 0.5f) / (float)W - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / (float)H - 1.0f;
    float imp = sqrtf(alpha_f * alpha_f + beta_f * beta_f);
    float rc = 5.2f - 1.0f * (float)a;
    float dd = (imp - rc) / 0.3f;
    float glow = expf(-dd * dd) * 0.06f;
    cr += 0.1f * glow; cg += 0.07f * glow; cb += 0.04f * glow;
    float vig = 1.0f - 0.3f * (ux * ux + uy * uy);
    cr *= vig; cg *= vig; cb *= vig;
    float L = 0.2126f * cr + 0.7152f * cg + 0.0722f * cb;
    if (L > 1e-6f) { float Lm = aces_curve(L); float sc = Lm / L; cr *= sc; cg *= sc; cb *= sc; }
    cr = fminf(cr, 1.0f); cg = fminf(cg, 1.0f); cb = fminf(cb, 1.0f);
    cr = linear_to_srgb(fmaxf(cr, 0.0f));
    cg = linear_to_srgb(fmaxf(cg, 0.0f));
    cb = linear_to_srgb(fmaxf(cb, 0.0f));
    int idx = (iy * W + ix) * 3;
    output[idx+0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx+1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx+2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
"""

_KERNEL_TAO_KL8 = r"""

extern "C" __global__ void trace_kerr(
    unsigned char* output,
    int W, int H, double a, double thobs, double fov, double robs,
    double risco, int maxsteps, double hbase
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;

    double r, th, phi, pr, pth, b, rp;
    initRayNotebook(ix, iy, W, H, a, thobs, fov, robs,
                    &r, &th, &phi, &pr, &pth, &b, &rp);

    double Q2 = 0.0;
    transformBLtoKS(r, a, b, Q2, &pr);
    double rs=r, ths=th, phis=phi, prs=pr, pths=pth;

    double Q2 = 0.0;
    float isco_f = (float)risco;
    float disk_outer = 14.0f;
    float disk_temp = 1.0f;
    int doppler_boost = 2;
    int bg_mode = 0;
    int star_layers = 3;
    int show_grid = 0;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_a = 0.0f;
    int disk_crossings = 0;
    int max_crossings = 5;
    float base_alpha = 0.95f;
    bool done = false;
    double sobs_r = sin(thobs);
    float alpha_f = (sobs_r != 0.0) ? (float)(-b / sobs_r) : 0.0f;
    float beta_f = (float)(-(pth));
    double resc = robs + 12.0;

    double kl_w[8] = {
        0.74167036435061295345, -0.40910082580003159400,
        0.19075471029623837995, -0.57386247111608226666,
        0.29906418130365592384,  0.33462491824529818378,
        0.31529309239676659663, -0.79688793935291635402
    };
    for (int i = 0; i < maxsteps; i++) {
        if (done) break;
        double he = adaptive_step_symplectic(r, rp, hbase);
        double oldR = r, oldTh = th, oldPhi = phi;
        for (int j = 0; j < 8; j++) {
            double d = kl_w[j]*he, ang = 2.0*TAO_OMEGA_C*kl_w[j];
            _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d,ang)
        }
        for (int j = 6; j >= 0; j--) {
            double d = kl_w[j]*he, ang = 2.0*TAO_OMEGA_C*kl_w[j];
            _TAO_STRANG(r,th,phi,pr,pth,rs,ths,phis,prs,pths,a,b,Q2,d,ang)
        }
        projectHamiltonianKS(r, th, &pr, pth, a, b, Q2);

        if (th < 0.005) { th = 0.005; pth = fabs(pth); }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); }

        if (r <= rp * 1.01) {
            blendColor(0.0f, 0.0f, 0.0f, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            done = true; break;
        }
        if (acc_a < 0.99f) {
            double cross = (oldTh - PI * 0.5) * (th - PI * 0.5);
            if (cross < 0.0 && disk_crossings < max_crossings) {
                double f = fmin(fmax(fabs(oldTh - PI * 0.5) /
                           fmax(fabs(th - oldTh), 1e-14), 0.0), 1.0);
                double r_hit = oldR + f * (r - oldR);
                float g = compute_g_factor(r_hit, a, Q2, b);
                float dcr, dcg, dcb;
                diskColor((float)r_hit, (float)(oldPhi + f * (phi - oldPhi)), (float)a,
                         isco_f, disk_outer, disk_temp, g, doppler_boost, &dcr, &dcg, &dcb);
                blendColor(dcr, dcg, dcb, base_alpha, &acc_r, &acc_g, &acc_b, &acc_a);
                disk_crossings++;
            }
        }
        if (r > resc) {
            double frac = fmin(fmax((resc - oldR) / fmax(r - oldR, 1e-14), 0.0), 1.0);
            double fth = oldTh + (th - oldTh) * frac;
            double fph = oldPhi + (phi - oldPhi) * frac;
            float dx, dy, dz;
            sphereDir(fth, fph, &dx, &dy, &dz);
            float bgr, bgg, bgb;
            background(dx, dy, dz, bg_mode, star_layers, show_grid, &bgr, &bgg, &bgb);
            if (acc_a < 1.0f) blendColor(bgr, bgg, bgb, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
            done = true; break;
        }
        if (r < 0.5 || r != r || th != th) { done = true; break; }
    }

    float cr = acc_r, cg = acc_g, cb = acc_b;
    float ux = 2.0f * (ix + 0.5f) / (float)W - 1.0f;
    float uy = 2.0f * (iy + 0.5f) / (float)H - 1.0f;
    float imp = sqrtf(alpha_f * alpha_f + beta_f * beta_f);
    float rc = 5.2f - 1.0f * (float)a;
    float dd = (imp - rc) / 0.3f;
    float glow = expf(-dd * dd) * 0.06f;
    cr += 0.1f * glow; cg += 0.07f * glow; cb += 0.04f * glow;
    float vig = 1.0f - 0.3f * (ux * ux + uy * uy);
    cr *= vig; cg *= vig; cb *= vig;
    float L = 0.2126f * cr + 0.7152f * cg + 0.0722f * cb;
    if (L > 1e-6f) { float Lm = aces_curve(L); float sc = Lm / L; cr *= sc; cg *= sc; cb *= sc; }
    cr = fminf(cr, 1.0f); cg = fminf(cg, 1.0f); cb = fminf(cb, 1.0f);
    cr = linear_to_srgb(fmaxf(cr, 0.0f));
    cg = linear_to_srgb(fmaxf(cg, 0.0f));
    cb = linear_to_srgb(fmaxf(cb, 0.0f));
    int idx = (iy * W + ix) * 3;
    output[idx+0] = (unsigned char)(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
    output[idx+1] = (unsigned char)(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    output[idx+2] = (unsigned char)(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));
}
"""


# ══════════════════════════════════════════════════════════════
#  KERNEL REGISTRY AND PYTHON API
# ══════════════════════════════════════════════════════════════

METHODS = {
    'rk4':          ('Runge-Kutta 4th-order',         _KERNEL_RK4),
    'rkdp8':        ('Dormand-Prince 8(7) adaptive',  _KERNEL_RKDP8),
    'tao_yoshida4': ('Tao + Yoshida 4th symplectic',  _KERNEL_TAO_Y4),
    'tao_kl8':      ('Tao + Kahan-Li 8th symplectic', _KERNEL_TAO_KL8),
}

_compiled = {}


def compile_all():
    global _compiled
    for name, (label, body) in METHODS.items():
        if name not in _compiled:
            src = _CUDA_HEADER + body
            _compiled[name] = cp.RawKernel(src, 'trace_kerr')
            print(f"  Compiled: {label} ({name})")
    print(f"All {len(_compiled)} integrators ready.")
    return _compiled


def compile_one(method='rk4'):
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(METHODS.keys())}")
    if method not in _compiled:
        label, body = METHODS[method]
        _compiled[method] = cp.RawKernel(_CUDA_HEADER + body, 'trace_kerr')
    return _compiled[method]


def auto_steps(obs_dist, h_base=0.3, rp=2.0, safety=3.0, symplectic=False):
    N_near = 20.0 / h_base
    if symplectic:
        N_transit = obs_dist
        N_strong = 200.0 / h_base
        return max(int((N_transit + N_strong) * safety), 400)
    else:
        N_far = (2 * rp / h_base) * math.log(max(obs_dist / rp, 2.0))
        return max(int((N_near + N_far) * safety), 400)


def render_kerr(spin, inclination_deg, width=512, height=512, fov=7.0,
                obs_dist=40.0, max_steps=None, step_size=0.3,
                method='rk4', **kwargs):
    """Render a Kerr black hole with production-quality visuals."""
    kern = compile_one(method)

    if max_steps is None:
        rp_est = 1.0 + np.sqrt(max(1.0 - spin**2, 0.0))
        max_steps = auto_steps(obs_dist, h_base=step_size, rp=rp_est,
                               symplectic=method.startswith('tao'))

    incl_rad = np.radians(inclination_deg)
    a = spin
    z1 = 1.0 + (1.0 - a**2)**(1/3) * ((1.0 + a)**(1/3) + max(1.0 - a, 0.0)**(1/3))
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    r_isco = 3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))

    n = width * height
    d_output = cp.zeros(n * 3, dtype=cp.uint8)

    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16)

    t0 = _time.time()
    kern(grid, block, (
        d_output,
        np.int32(width), np.int32(height),
        np.float64(spin), np.float64(incl_rad),
        np.float64(fov), np.float64(obs_dist),
        np.float64(r_isco),
        np.int32(max_steps), np.float64(step_size)
    ))
    cp.cuda.Device(0).synchronize()
    ms = (_time.time() - t0) * 1000

    img = d_output.get().reshape(height, width, 3)
    img = np.flipud(img)
    shadow_mask = (img.max(axis=2) < 5)

    return img, shadow_mask, {
        'render_ms': ms, 'max_steps': max_steps,
        'obs_dist': obs_dist, 'method': method,
    }


def compare_integrators(spin=0.6, inclination=80, obs_dist=40, step_size=0.3,
                        width=512, height=512, fov=7.0,
                        methods=None, fit_fn=None):
    import matplotlib.pyplot as plt
    if methods is None:
        methods = list(METHODS.keys())
    compile_all()
    results = []
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1: axes = [axes]
    for ax, m in zip(axes, methods):
        img, shadow, info = render_kerr(
            spin, inclination, width, height, fov=fov,
            obs_dist=obs_dist, step_size=step_size, method=m)
        ax.imshow(img)
        ax.set_title(f"{METHODS[m][0]}\n{info['render_ms']:.0f} ms, "
                     f"{info['max_steps']} steps", fontsize=10)
        ax.axis('off')
        row = {'method': m, 'label': METHODS[m][0],
               'render_ms': info['render_ms'], 'max_steps': info['max_steps']}
        if fit_fn is not None:
            obs = fit_fn(shadow, fov=fov, img_size=width)
            if obs:
                row['diameter_M'] = obs['diameter_M']
                row['delta_C'] = obs['delta_C']
                row['circularity'] = obs['circularity']
        results.append(row)
    fig.suptitle(f'Integrator Comparison — $a={spin}$, '
                 rf'$\theta={inclination}°$, $r_{{\rm obs}}={obs_dist}\,M$',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    hdr = f"\n{'Method':<30} {'Time':>8} {'Steps':>7}"
    if fit_fn: hdr += f" {'Diameter':>10} {'ΔC':>8}"
    print(hdr)
    print("─" * 70)
    for r in results:
        line = f"{r['label']:<30} {r['render_ms']:>7.0f}ms {r['max_steps']:>7d}"
        if 'diameter_M' in r:
            line += f" {r['diameter_M']:>9.4f}M {r['delta_C']:>7.4f}"
        print(line)
    return results, fig
