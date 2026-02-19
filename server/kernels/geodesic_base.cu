/* ============================================================
 *  GEODESIC BASE — Kerr-Newman metric functions (float64)
 *
 *  This file is #include'd by each integrator kernel.
 *  It provides:
 *    - Constants and parameter struct
 *    - geoRHS(): geodesic right-hand side in Boyer-Lindquist coords
 *    - Ray initialization from pixel coordinates
 *    - Termination conditions (horizon, escape, NaN)
 *
 *  All metric computations use double precision (float64).
 *  Color/visual output uses float (float32).
 * ============================================================ */

#ifndef GEODESIC_BASE_CU
#define GEODESIC_BASE_CU

/* ── Constants ─────────────────────────────────────────────── */

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

    double b = -alpha * sO;

    *r   = p.obs_dist;
    *th  = thObs;
    *phi = p.phi0;

    /* Compute initial p_r from the null condition H = 0 */
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

    *pth = -beta;
    double rest = -A_ * iSD + 2.0 * a * b * w_init * iSD
                  + gthi * beta * beta + (sig - w_init) * iSD * is2 * b * b;
    double pr2 = -rest / grr;
    *pr = (pr2 > 0.0) ? -sqrt(pr2) : 0.0;

    /* Event horizon radius */
    *rp_out = 1.0 + sqrt(fmax(1.0 - a2 - Q2, 0.0));
    *b_out = b;

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

/* Matches GLSL smoothstep semantics: works correctly even when
 * edge0 > edge1 (reversed edges for fade-out ramps).
 * Computes t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
 * then returns t² × (3 - 2t). */
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

__device__ void postProcess(float *cr, float *cg, float *cb,
                            float alpha, float beta,
                            float spin, float ux, float uy) {
    /* Photon ring glow */
    float imp = sqrtf(alpha * alpha + beta * beta);
    float rc = 5.2f - 1.0f * spin;
    float d = (imp - rc) / 0.3f;
    float glow = expf(-d * d) * 0.06f;
    *cr += 0.1f * glow;
    *cg += 0.07f * glow;
    *cb += 0.04f * glow;

    /* Vignette */
    float vig = 1.0f - 0.3f * (ux * ux + uy * uy);
    *cr *= vig;
    *cg *= vig;
    *cb *= vig;

    /* Reinhard tone mapping */
    *cr = *cr / (1.0f + *cr);
    *cg = *cg / (1.0f + *cg);
    *cb = *cb / (1.0f + *cb);

    /* Gamma correction (1/2.2) */
    float inv_gamma = 1.0f / 2.2f;
    *cr = powf(fmaxf(*cr, 0.0f), inv_gamma);
    *cg = powf(fmaxf(*cg, 0.0f), inv_gamma);
    *cb = powf(fmaxf(*cb, 0.0f), inv_gamma);
}


#endif /* GEODESIC_BASE_CU */
