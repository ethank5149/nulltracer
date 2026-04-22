/* ============================================================
 *  GEODESIC BASE — Kerr-Newman metric functions (float64)
 * ============================================================ */

#define PI  3.14159265358979323846
#define TAU 6.28318530717958647693

/* Smooth regularization epsilon for sin²θ.
 * Prevents dφ/dλ divergence at poles.
 * Physics error confined to θ < arcsin(√ε) ≈ 1.1° from poles. */
#define S2_EPS 0.0004


/* ── Parameter struct passed to kernel ─────────────────────── */

/* All fields are double to guarantee identical layout between
 * Python ctypes and CUDA compiler (no alignment padding issues).
 * Integer values are stored as double and cast to int in the kernel. */
struct RenderParams {
    /* Resolution */
    double width;
    double height;

    /* Black hole parameters */
    double spin;        /* a: dimensionless spin parameter */
    double charge;      /* Q: dimensionless electric charge */
    double incl;        /* observer inclination in radians */
    double fov;         /* field of view */
    double phi0;        /* rotation angle */
    double isco;        /* ISCO radius (precomputed) */

    /* Integration parameters */
    double steps;       /* max integration steps */
    double obs_dist;    /* observer distance R0 */
    double esc_radius;  /* escape radius RESC = R0 + 12 */
    double disk_outer;  /* outer disk radius RDISK */
    double step_size;   /* base step size H_BASE */

    /* Rendering options */
    double bg_mode;     /* 0=stars, 1=checker, 2=colormap */
    double star_layers; /* number of star layers */
    double show_disk;   /* 1=show accretion disk */
    double show_grid;   /* 1=show grid overlay */
    double disk_temp;   /* disk temperature multiplier */
    double doppler_boost; /* 0=off, 1=g^3 (thin), 2=g^4 (thick) */
    double srgb_output;   /* >0.5 = apply IEC 61966-2-1 sRGB transfer */
    double disk_alpha;          /* base opacity per disk crossing (0.0–1.0) */
    double disk_max_crossings;  /* max disk crossings to accumulate (as double, cast to int) */
    double bloom_enabled;       /* 1.0 = output float32 linear for bloom, 0.0 = normal uint8 sRGB */

    /* Skymap texture (equirectangular projection) */
    double sky_width;           /* skymap pixel width (0 = no skymap, use procedural) */
    double sky_height;          /* skymap pixel height */
};


/* ── Kerr-Newman geodesic RHS (double precision) ──────────── */

/* Computes the right-hand side of the geodesic equations:
 *   dr/dλ, dθ/dλ, dφ/dλ, dp_r/dλ, dp_θ/dλ
 *
 * Uses smooth sin²θ + ε regularization for pole safety.
 * All intermediate quantities in float64 for maximum accuracy.
 */
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


/* ── Ray initialization ───────────────────────────────────── */

/* Initialize a ray from pixel coordinates (ix, iy).
 * Returns the initial state: r, th, phi, pr, pth, and the
 * impact parameter b and event horizon radius rp.
 *
 * This is the common setup shared by all integrators.
 */
__device__ void initRay(
    int ix, int iy, const RenderParams &p,
    double *r, double *th, double *phi,
    double *pr, double *pth,
    double *b_out, double *rp_out,
    float *alpha_out, float *beta_out
) {
    double asp = p.width / p.height;
    /* Map pixel to normalized [-1, 1] coordinates.
     * ux: left=-1, right=+1.
     * uy: bottom=-1, top=+1 (bottom-to-top row order).
     * iy=0 maps to uy≈-1 (bottom of image).
     * The caller (renderer.py) applies np.flipud() to produce standard
     * top-to-bottom image order. */
    double ux = (2.0 * (ix + 0.5) / p.width  - 1.0);
    double uy = (2.0 * (iy + 0.5) / p.height - 1.0);

    double alpha = ux * p.fov * asp;
    double beta  = uy * p.fov;

    double a = p.spin;
    double a2 = a * a;
    double Q2 = p.charge * p.charge;
    double thObs = p.incl;
    double sO = sin(thObs), cO = cos(thObs);

    double sin_i = sO;
    double cos_i = cO;

    double Lz = -alpha * sin_i;

    *r   = p.obs_dist;
    *th  = thObs;
    *phi = p.phi0;

    double sth = sin(thObs), cth = cos(thObs);
    double s2 = sth * sth + S2_EPS;
    double c2 = cth * cth;
    double r0 = p.obs_dist;
    double r02 = r0 * r0;
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

    double Q = beta * beta + c2 * (a2 - Lz * Lz / s2);
    double pth2 = fmax(Q - c2 * (a2 - Lz * Lz / s2), 0.0);
    *pth = sqrt(pth2);
    if (beta < 0.0) *pth = -*pth;

    double rest = -A_ * iSD + 2.0 * a * Lz * w_init * iSD
                  + gthi * (*pth) * (*pth) + (sig - w_init) * iSD * is2 * Lz * Lz;
    double pr2 = -rest / grr;
    *pr = (pr2 > 0.0) ? -sqrt(pr2) : 0.0;

    /* Event horizon radius */
    *rp_out = 1.0 + sqrt(fmax(1.0 - a2 - Q2, 0.0));
    *b_out = Lz;

    *alpha_out = (float)alpha;
    *beta_out  = (float)beta;
}


/* ── Hash function (for procedural backgrounds) ───────────── */

/* Exact port of the GLSL hash function:
 *   float hash(vec2 p){
 *       p=fract(p*vec2(443.8,441.4));
 *       p+=dot(p,p+19.19);
 *       return fract(p.x*p.y);
 *   }
 */
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


/* ── Cube-map projection (Cartesian direction → face + UV) ── */

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


/* ── Smoothstep utility ───────────────────────────────────── */

__device__ float smoothstepf(float edge0, float edge1, float x) {
    float t = (x - edge0) / (edge1 - edge0);
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}


/* ── Sphere direction from (θ, φ) ─────────────────────────── */

__device__ void sphereDir(double th, double ph,
                          float *dx, float *dy, float *dz) {
    double sth = sin(th);
    *dx = (float)(sth * cos(ph));
    *dy = (float)(sth * sin(ph));
    *dz = (float)(cos(th));
}


/* ── Post-processing: tone mapping + gamma ────────────────── */

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

__device__ void postProcess(float *cr, float *cg, float *cb,
                            float alpha, float beta,
                            const RenderParams &p,
                            float ux, float uy) {
    // Artificial photon-ring glow removed for physical accuracy.
    // The true ring now emerges purely from geodesic lensing + disk emission.

    /* Vignette */
    float vig = 1.0f - 0.3f * (ux * ux + uy * uy);
    *cr *= vig;
    *cg *= vig;
    *cb *= vig;

    /* Luminance-preserving ACES filmic tone mapping. */
    float L = 0.2126f * (*cr) + 0.7152f * (*cg) + 0.0722f * (*cb);
    if (L > 1e-6f) {
        float L_mapped = aces_curve(L);
        float scale = L_mapped / L;
        *cr *= scale;
        *cg *= scale;
        *cb *= scale;
    }

    /* Clamp (luminance-based scaling can slightly exceed 1.0
     * for highly saturated colors) */
    *cr = fminf(*cr, 1.0f);
    *cg = fminf(*cg, 1.0f);
    *cb = fminf(*cb, 1.0f);

    /* sRGB transfer function (IEC 61966-2-1) or simple gamma.
     * When srgb_output is enabled, apply the standard piecewise
     * sRGB OETF for correct display on sRGB monitors.
     * Otherwise, fall back to the simple 1/2.2 power curve. */
    if (p.srgb_output > 0.5) {
        *cr = linear_to_srgb(fmaxf(*cr, 0.0f));
        *cg = linear_to_srgb(fmaxf(*cg, 0.0f));
        *cb = linear_to_srgb(fmaxf(*cb, 0.0f));
    } else {
        float inv_gamma = 1.0f / 2.2f;
        *cr = powf(fmaxf(*cr, 0.0f), inv_gamma);
        *cg = powf(fmaxf(*cg, 0.0f), inv_gamma);
        *cb = powf(fmaxf(*cb, 0.0f), inv_gamma);
    }
}


/* ============================================================
 *  SYMPLECTIC SPLITTING FUNCTIONS for Kahan-Li S10 integrator
 * ============================================================ */

__device__ void geoVelocity(
    double r, double th, double pr, double pth,
    double a, double b, double Q2,
    double *dr, double *dth, double *dphi
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * cth * cth;
    double del = r2 - 2.0 * r + a2 + Q2;
    double sdel = fmax(del, 1e-14);
    double w = 2.0 * r - Q2;
    double isig = 1.0 / sig;
    double iSD = 1.0 / (sig * sdel);
    double is2 = 1.0 / s2;

    /* g^rr · p_r */
    *dr  = sdel * isig * pr;
    /* g^θθ · p_θ */
    *dth = isig * pth;
    /* g^φφ · b + g^tφ · E  (E = 1 by affine normalization) */
    *dphi = (sig - w) * iSD * is2 * b - (-a * w * iSD);
}


__device__ void geoForce(
    double r, double th, double pr, double pth,
    double a, double b, double Q2,
    double *dpr, double *dpth
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double c2 = cth * cth;
    double a2 = a * a;
    double r2 = r * r;
    double sig = r2 + a2 * c2;
    double del = r2 - 2.0 * r + a2 + Q2;
    double sdel = fmax(del, 1e-14);
    double rpa2 = r2 + a2;
    double w = 2.0 * r - Q2;
    double A_ = rpa2 * rpa2 - sdel * a2 * s2;
    double isig = 1.0 / sig;
    double SD = sig * sdel;
    double iSD = 1.0 / SD;
    double is2 = 1.0 / s2;

    /* ∂/∂r derivatives for dp_r/dλ */
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

    /* ∂/∂θ derivatives for dp_θ/dλ */
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


__device__ void projectHamiltonian(
    double r, double th, double *pr, double pth,
    double a, double b, double Q2
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double c2 = cth * cth;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * c2;
    double del = r2 - 2.0 * r + a2 + Q2;
    double sdel = fmax(del, 1e-14);
    double rpa2 = r2 + a2;
    double w = 2.0 * r - Q2;
    double A_ = rpa2 * rpa2 - sdel * a2 * s2;
    double iSD = 1.0 / (sig * sdel);
    double is2 = 1.0 / s2;

    double gtt   = -A_ * iSD;
    double gtf   = -a * w * iSD;
    double grr   = sdel / sig;
    double gthth = 1.0 / sig;
    double gff   = (sig - w) * iSD * is2;

    /* R = ½(g^tt + 2b·g^tφ + g^θθ·p_θ² + g^φφ·b²)
     * — all Hamiltonian terms except the g^rr·p_r² piece */
    double R = 0.5 * (gtt + 2.0 * b * gtf + gthth * pth * pth + gff * b * b);

    /* Solve: 0 = ½ g^rr · p_r² + R  →  p_r² = -2R / g^rr */
    double pr2_new = -2.0 * R / grr;
    if (pr2_new > 0.0) {
        *pr = copysign(sqrt(pr2_new), *pr);  /* preserve radial direction */
    }
    /* If pr2_new <= 0, no real solution exists (near turning point);
     * leave p_r unchanged — the error is small and transient. */
}

/* ============================================================
 *  BACKGROUNDS — Procedural background rendering (float32)
 * ============================================================ */

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


__device__ void bgSkymap(float dx, float dy, float dz,
                         const float *skymap,
                         int sky_w, int sky_h,
                         float *cr, float *cg, float *cb) {
    /* Direction → spherical coordinates */
    double th = acos(fmax(fmin((double)dz, 1.0), -1.0));  /* [0, π] */
    double ph = atan2((double)dy, (double)dx);             /* [-π, π] */
    if (ph < 0.0) ph += TAU;                               /* [0, 2π] */

    /* Spherical → equirectangular UV */
    float u = (float)(ph / TAU);           /* [0, 1] */
    float v = (float)(th / PI);            /* [0, 1] */

    /* UV → fractional pixel coordinates */
    float fx = u * (float)(sky_w - 1);
    float fy = v * (float)(sky_h - 1);

    /* Bilinear interpolation */
    int ix0 = (int)fx;
    int iy0 = (int)fy;
    int ix1 = min(ix0 + 1, sky_w - 1);
    int iy1 = min(iy0 + 1, sky_h - 1);
    float wx = fx - (float)ix0;
    float wy = fy - (float)iy0;

    /* Fetch 4 texels (float32 RGB, already linear light) */
    int idx00 = (iy0 * sky_w + ix0) * 3;
    int idx10 = (iy0 * sky_w + ix1) * 3;
    int idx01 = (iy1 * sky_w + ix0) * 3;
    int idx11 = (iy1 * sky_w + ix1) * 3;

    float r00 = skymap[idx00], g00 = skymap[idx00+1], b00 = skymap[idx00+2];
    float r10 = skymap[idx10], g10 = skymap[idx10+1], b10 = skymap[idx10+2];
    float r01 = skymap[idx01], g01 = skymap[idx01+1], b01 = skymap[idx01+2];
    float r11 = skymap[idx11], g11 = skymap[idx11+1], b11 = skymap[idx11+2];

    /* Bilinear blend */
    float w00 = (1-wx)*(1-wy), w10 = wx*(1-wy), w01 = (1-wx)*wy, w11 = wx*wy;
    *cr = r00*w00 + r10*w10 + r01*w01 + r11*w11;
    *cg = g00*w00 + g10*w10 + g01*w01 + g11*w11;
    *cb = b00*w00 + b10*w10 + b01*w01 + b11*w11;
}


__device__ void background(float dx, float dy, float dz,
                           int bg_mode, int star_layers, int show_grid,
                           const float *skymap, int sky_w, int sky_h,
                           float *cr, float *cg, float *cb) {
    if (bg_mode == 3 && skymap != 0 && sky_w > 0 && sky_h > 0) {
        bgSkymap(dx, dy, dz, skymap, sky_w, sky_h, cr, cg, cb);
        return;
    }
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

/* ============================================================
 *  DISK — Accretion disk emission and color (float32)
 * ============================================================ */

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
};


__device__ void blackbody(float T, float *out_r, float *out_g, float *out_b) {
    float t = fminf(fmaxf(T, PLANCK_T_MIN), PLANCK_T_MAX);
    float frac = (t - PLANCK_T_MIN) / (PLANCK_T_MAX - PLANCK_T_MIN) * (float)(PLANCK_LUT_SIZE - 1);
    int idx = (int)frac;
    float w = frac - (float)idx;
    idx = min(max(idx, 0), PLANCK_LUT_SIZE - 2);
    *out_r = planck_lut_r[idx] * (1.0f - w) + planck_lut_r[idx + 1] * w;
    *out_g = planck_lut_g[idx] * (1.0f - w) + planck_lut_g[idx + 1] * w;
    *out_b = planck_lut_b[idx] * (1.0f - w) + planck_lut_b[idx + 1] * w;
}


__device__ float compute_g_factor(double r, double a, double Q2, double b) {
    double r2 = r * r;
    double a2 = a * a;
    double w = 2.0 * r - Q2;
    double gtt = -(1.0 - w / r2);
    double gtph = -a * w / r2;
    double gphph = (r2 * r2 + a2 * r2 + a2 * w) / r2;
    double dgtt_dr = 2.0 * (Q2 - r) / (r2 * r);
    double dgtph_dr = 2.0 * a * (r - Q2) / (r2 * r);
    double dgphph_dr = 2.0 * r + a2 * (-2.0 / r2 + 2.0 * Q2 / (r2 * r));
    double disc = dgtph_dr * dgtph_dr - dgtt_dr * dgphph_dr;
    double Omega = (-dgtph_dr + sqrt(fmax(disc, 0.0))) / fmax(dgphph_dr, 1e-30);
    double denom = -(gtt + 2.0 * Omega * gtph + Omega * Omega * gphph);
    double ut = 1.0 / sqrt(fmax(denom, 1e-30));
    double one_minus_bOmega = 1.0 - b * Omega;
    double g = 1.0 / (ut * fmax(fabs(one_minus_bOmega), 1e-30));
    g = fmin(fmax(g, 0.01), 10.0);
    return (float)g;
}


__device__ float compute_g_factor_extended(double r, double a, double Q2,
                                          double b, double r_isco) {
    double r_horizon = 1.0 + sqrt(fmax(1.0 - a * a - Q2, 0.0));
    if (r >= r_isco) {
        return compute_g_factor(r, a, Q2, b);
    }
    float g_isco = compute_g_factor(r_isco, a, Q2, b);
    double x = (r - r_horizon) / fmax(r_isco - r_horizon, 1e-10);
    x = fmax(fmin(x, 1.0), 0.0);
    return g_isco * (float)(x * x);
}


__device__ float novikov_thorne_flux(double r, double a, double r_isco) {
    double x = sqrt(r);
    double x_i = sqrt(r_isco);
    double a2 = a * a;
    double cos_val = a;
    double theta = acos(fmin(fmax(cos_val, -1.0), 1.0));
    double y1 = 2.0 * cos(theta / 3.0);
    double y2 = 2.0 * cos((theta + 2.0 * PI) / 3.0);
    double y3 = 2.0 * cos((theta + 4.0 * PI) / 3.0);
    double x3_term = x * x * x - 3.0 * x + 2.0 * a;
    if (fabs(x3_term) < 1e-14) return 0.0f;
    double prefactor = 1.0 / (x * x * x3_term);
    double I = x - x_i - 1.5 * a * log(fmax(x / x_i, 1e-30));
    double roots[3] = {y1, y2, y3};
    for (int k = 0; k < 3; k++) {
        double yk = roots[k];
        double yk_minus_a = yk - a;
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
    double F = prefactor * I;
    return (float)fmax(F, 0.0);
}


__device__ void diskColor(float r, float ph, float a,
                          float isco, float disk_outer, float disk_temp,
                          float g_factor, int doppler_boost,
                          float *cr, float *cg, float *cb) {
    float ri = isco;
    float r_horizon = 1.0f + sqrtf(fmaxf(1.0f - a * a, 0.0f));
    if (r < r_horizon * 1.02f || r > disk_outer) {
        *cr = 0.0f; *cg = 0.0f; *cb = 0.0f;
        return;
    }
    float F_norm;
    float F_max = 0.0f;
    for (int i = 1; i <= 20; i++) {
        float r_sample = ri * (1.0f + 0.5f * (float)i);
        float F_sample = novikov_thorne_flux((double)r_sample, (double)a, (double)ri);
        if (F_sample > F_max) F_max = F_sample;
    }
    F_max = fmaxf(F_max, 1e-10f);
    if (r >= ri) {
        float F = novikov_thorne_flux((double)r, (double)a, (double)ri);
        F_norm = fminf(F / F_max, 1.0f);
    } else {
        float F_isco = novikov_thorne_flux((double)ri, (double)a, (double)ri);
        float x = (r - r_horizon) / fmaxf(ri - r_horizon, 1e-6f);
        x = fmaxf(x, 0.0f);
        F_norm = fminf(F_isco / F_max, 1.0f) * x * x;
    }
    float T_base = 8000.0f * disk_temp;
    float T_emit = T_base * powf(fmaxf(F_norm, 0.0f), 0.25f);
    float I = F_norm * 3.0f / fmaxf(r * 0.15f, 0.01f);
    I *= smoothstepf(disk_outer, disk_outer * 0.55f, r);
    I *= smoothstepf(r_horizon * 1.02f, r_horizon * 1.5f, r);
    float g = g_factor;
    float T_obs, I_adjusted;
    if (doppler_boost == 0) {
        T_obs = T_emit;
        I_adjusted = I;
    } else if (doppler_boost == 1) {
        T_obs = g * T_emit;
        float g3 = g * g * g;
        I_adjusted = I * g3;
    } else {
        T_obs = g * T_emit;
        float g4 = g * g * g * g;
        I_adjusted = I * g4;
    }
    float col_r, col_g, col_b;
    blackbody(T_obs, &col_r, &col_g, &col_b);
    float tu  = 0.65f + 0.35f * hash2(r * 5.0f, ph * 3.0f);
    float tu2 = 0.8f  + 0.2f  * hash2(r * 18.0f, ph * 9.0f);
    *cr = col_r * I_adjusted * tu * tu2 * 3.2f;
    *cg = col_g * I_adjusted * tu * tu2 * 3.2f;
    *cb = col_b * I_adjusted * tu * tu2 * 3.2f;
}

/* ============================================================
 *  ADAPTIVE_STEP — Shared adaptive step size functions
 * ============================================================ */

__device__ double sundman_dtau(double a, double Q2, double rp,
                              double step_size, double esc_radius,
                              int STEPS) {
    double r_ph;
    if (Q2 < 1e-10) {
        r_ph = 2.0 * (1.0 + cos(2.0 / 3.0 * acos(-a)));
    } else {
        r_ph = rp;
    }
    double tau_needed = 2.0 * (1.0 / r_ph - 1.0 / esc_radius);
    return (1.0 + step_size) * tau_needed / (double)STEPS;
}

__device__ double phi_var_sundman_g(double r, double th, double a) {
    double cth = cos(th);
    double sig = r * r + a * a * cth * cth;
    return sig / (r * r);
}

__device__ double phi_var_dphi_BL(double r, double th, double pr,
                                  double a, double Q2,
                                  double g, double h) {
    double cth = cos(th);
    double sig = r * r + a * a * cth * cth;
    double del = r * r - 2.0 * r + a * a + Q2;
    if (del < 1e-14) del = 1e-14;
    double grr = del / sig;
    double v_r = grr * pr;
    return -g * h * v_r / (2.0 * r);
}

__device__ double phi_var_physical_step(double h, double Phi,
                                        double r, double th,
                                        double pth, double a,
                                        double obs_dist) {
    double he = h / Phi;
    double sth = sin(th);
    if (fabs(sth) > 1e-8) {
        double sig = r * r + a * a * cos(th) * cos(th);
        double dth_rate = fabs(pth / sig);
        double max_dth = 0.3;
        if (fabs(he) * dth_rate > max_dth) {
            he = copysign(max_dth / dth_rate, he);
        }
    }
    double pole_dist = fmin(th, PI - th);
    if (pole_dist < 0.1) {
        double pole_factor = pole_dist / 0.1;
        pole_factor = fmax(pole_factor, 0.05);
        he *= pole_factor;
    }
    double abs_he = fabs(he);
    abs_he = fmax(abs_he, 0.005);
    abs_he = fmin(abs_he, 0.2 * obs_dist);
    he = copysign(abs_he, he);
    return he;
}

/* ═══════════════════════════════════════════════════════════
 * Volumetric emission: hot corona + relativistic jet
 * ═══════════════════════════════════════════════════════════ */

__device__ void accumulate_volume_emission(
    double r, double th, double he, double a,
    double r_isco, double disk_outer,
    float *acc_r, float *acc_g, float *acc_b, float *acc_a
) {
    double cth = cos(th), sth = sin(th);
    double r_cyl = r * fabs(sth);
    double z = r * cth;
    double r_horizon = 1.0 + sqrt(fmax(1.0 - a * a, 0.0));
    if (r_cyl > r_horizon * 1.5 && r_cyl < disk_outer * 0.7 && r > r_horizon * 1.3) {
        double scale_h = 0.3 * r_cyl;
        double rho = exp(-z * z / (2.0 * scale_h * scale_h));
        float opacity = (float)(rho * he * 0.03 / r);
        opacity = fminf(opacity, 0.005f);
        blendColor(0.80f, 0.50f, 0.25f, opacity,
                   acc_r, acc_g, acc_b, acc_a);
    }
    if (fabs(cth) > 0.90 && r > r_horizon * 1.5 && r < 30.0) {
        double axis_dist = 1.0 - fabs(cth);
        double jet_profile = exp(-axis_dist * axis_dist / 0.003);
        float opacity = (float)(jet_profile * he * 0.015 / r);
        opacity = fminf(opacity, 0.004f);
        blendColor(0.30f, 0.50f, 1.00f, opacity,
                   acc_r, acc_g, acc_b, acc_a);
    }
}

/* ============================================================
 *  KAHANLI8S — KAHAN-LI 8th-ORDER SYMPLECTIC INTEGRATOR WITH
 *  SUNDMAN (MINO TIME) TRANSFORMATION
 * ============================================================ */

static __constant__ double KL8S_W8[8] = {
     0.74167036435061295345,
    -0.40910082580003159400,
     0.19075471029623837995,
    -0.57386247111608226666,
     0.29906418130365592384,
     0.33462491824529818378,
     0.31529309239676659663,
    -0.79688793935291635402
};

static __constant__ double KL8S_D8[8] = {
     0.37083518217530647672,
     0.16628476927529067972,
    -0.10917305775189660702,
    -0.19155388040992194336,
    -0.13739914490621317141,
     0.31684454977447705381,
     0.32495900532103239020,
    -0.24079742347807487870
};

__device__ __forceinline__ void kahan_add(
    double *sum, double *comp, double delta
) {
    double y = delta - *comp;
    double t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

extern "C" __global__
void trace_geodesics(const RenderParams *pp, unsigned char *output, const float *skymap) {
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
    int STEPS = (int)p.steps * 4;
    int show_disk = (int)p.show_disk;
    int bg_mode = (int)p.bg_mode;
    int star_layers = (int)p.star_layers;
    int show_grid = (int)p.show_grid;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_a = 0.0f;
    int disk_crossings = 0;
    int max_crossings = (int)p.disk_max_crossings;
    float base_alpha = (float)p.disk_alpha;
    bool done = false;

    double r_comp = 0.0, th_comp = 0.0, phi_comp = 0.0;
    double pr_comp = 0.0, pth_comp = 0.0;

    double dtau = sundman_dtau(a, Q2, rp, p.step_size, p.esc_radius, STEPS);

    double Phi = p.obs_dist / r;
    double Phi_comp = 0.0;
    double h_phi = dtau * p.obs_dist * p.obs_dist;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;

        double oldR = r, oldTh = th, oldPhi = phi;

        double g_sun = phi_var_sundman_g(r, th, a);
        double dPhi = phi_var_dphi_BL(r, th, pr, a, Q2, g_sun, h_phi);
        kahan_add(&Phi, &Phi_comp, dPhi);
        if (Phi < 0.01) Phi = 0.01;

        double he = phi_var_physical_step(h_phi, Phi, r, th, pth, a, p.obs_dist);

        double dr_, dth_, dphi_, dpr_, dpth_;

        #define KL8S_SUBSTEP(idx) { \
            geoRHS(r, th, pr, pth, a, b, Q2, \
                   &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
            kahan_add(&r,   &r_comp,   he * KL8S_D8[idx] * dr_); \
            kahan_add(&th,  &th_comp,  he * KL8S_D8[idx] * dth_); \
            kahan_add(&phi, &phi_comp, he * KL8S_D8[idx] * dphi_); \
            geoRHS(r, th, pr, pth, a, b, Q2, \
                   &dr_, &dth_, &dphi_, &dpr_, &dpth_); \
            kahan_add(&pr,  &pr_comp,  he * KL8S_W8[idx] * dpr_); \
            kahan_add(&pth, &pth_comp, he * KL8S_W8[idx] * dpth_); \
        }

        KL8S_SUBSTEP(0)
        KL8S_SUBSTEP(1)
        KL8S_SUBSTEP(2)
        KL8S_SUBSTEP(3)
        KL8S_SUBSTEP(4)
        KL8S_SUBSTEP(5)
        KL8S_SUBSTEP(6)
        KL8S_SUBSTEP(7)
        KL8S_SUBSTEP(6)
        KL8S_SUBSTEP(5)
        KL8S_SUBSTEP(4)
        KL8S_SUBSTEP(3)
        KL8S_SUBSTEP(2)
        KL8S_SUBSTEP(1)
        KL8S_SUBSTEP(0)

        #undef KL8S_SUBSTEP

        {
            double corr_eps = he * he / 24.0;
            double f_pr, f_pth;
            geoForce(r, th, pr, pth, a, b, Q2, &f_pr, &f_pth);
            pr  += corr_eps * f_pr;
            pth += corr_eps * f_pth;
            double v_r, v_th, v_phi;
            geoVelocity(r, th, pr, pth, a, b, Q2, &v_r, &v_th, &v_phi);
            kahan_add(&r,   &r_comp,   corr_eps * v_r);
            kahan_add(&th,  &th_comp,  corr_eps * v_th);
            kahan_add(&phi, &phi_comp, corr_eps * v_phi);
        }

        projectHamiltonian(r, th, &pr, pth, a, b, Q2);
        pr_comp = 0.0;

        g_sun = phi_var_sundman_g(r, th, a);
        dPhi = phi_var_dphi_BL(r, th, pr, a, Q2, g_sun, h_phi);
        kahan_add(&Phi, &Phi_comp, dPhi);
        if (Phi < 0.01) Phi = 0.01;

        if (th < 0.005) { th = 0.005; pth = fabs(pth); th_comp = 0.0; pth_comp = 0.0; }
        if (th > PI - 0.005) { th = PI - 0.005; pth = -fabs(pth); th_comp = 0.0; pth_comp = 0.0; }

        if (acc_a < 0.99f) {
            accumulate_volume_emission(r, th, he, a, (double)p.isco, p.disk_outer,
                                      &acc_r, &acc_g, &acc_b, &acc_a);
        }

        if (r <= rp * 1.01) {
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

                float g = compute_g_factor_extended(r_hit, a, Q2, b, (double)p.isco);

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
