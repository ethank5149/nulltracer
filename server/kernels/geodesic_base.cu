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

/* Standard sRGB transfer function (IEC 61966-2-1).
 * Maps linear-light [0,1] to sRGB-encoded [0,1].
 * The piecewise function avoids the infinite slope at zero
 * that a pure gamma curve would have. */
__device__ float linear_to_srgb(float c) {
    if (c <= 0.0031308f)
        return 12.92f * c;
    else
        return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

/* ACES filmic tone mapping curve (Narkowicz 2015 approximation).
 * Maps HDR luminance [0, ∞) to SDR [0, ~1.0) with a natural
 * film-like S-curve: deep toe (rich shadows), linear mid-range,
 * and smooth shoulder (highlight rolloff without flat clipping). */
__device__ float aces_curve(float x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return fminf(fmaxf((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f), 1.0f);
}

/* Front-to-back alpha compositing.
 * Blends source color (sr,sg,sb) with opacity sa onto accumulated
 * destination (dr,dg,db) with accumulated opacity da.
 * Updates destination in-place. */
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
    /* Photon ring glow */
    float spin = (float)p.spin;
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

    /* Luminance-preserving ACES filmic tone mapping.
     *
     * We compute the scene luminance, apply the ACES curve to it,
     * then scale all channels by the same ratio.  This preserves
     * the hue (R:G:B ratio) while giving a cinematic S-curve
     * response — deep shadows, smooth highlight rolloff, and
     * natural color rendering across the full dynamic range.
     *
     * This is critical for relativistic beaming: the approaching
     * (blueshifted) side gets a smooth highlight rolloff instead
     * of clipping to flat white, while the receding (redshifted)
     * side retains rich warm tones in the shadows. */
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
 *
 *  These functions decompose the geodesic equations into separate
 *  "drift" (velocity/position update) and "kick" (force/momentum
 *  update) components for use in symplectic composition methods.
 *
 *  In the Hamiltonian formulation of geodesic motion, the super-
 *  Hamiltonian H = ½ g^μν p_μ p_ν generates the equations of motion:
 *
 *    dq^i/dλ = ∂H/∂p_i   (velocities — computed by geoVelocity)
 *    dp_i/dλ = -∂H/∂q^i  (forces    — computed by geoForce)
 *
 *  The symplectic leapfrog alternates between these two updates,
 *  preserving the symplectic 2-form to machine precision.
 *
 *  References:
 *    - W. Kahan & R.-C. Li, "Composition constants for raising the
 *      orders of unconventional schemes for ordinary differential
 *      equations," Math. Comp. 66(219):1089–1099, 1997.
 *    - J. Wisdom & M. Holman, "Symplectic maps for the N-body
 *      problem," Astron. J. 102:1528–1538, 1991.
 *    - H. Yoshida, "Construction of higher order symplectic
 *      integrators," Phys. Lett. A 150(5–7):262–268, 1990.
 * ============================================================ */


/* ── geoVelocity: drift (position update) ─────────────────── */

/* Computes only the velocity part of the geodesic equations:
 *   dr/dλ   = g^rr · p_r        = (Δ/Σ) · p_r
 *   dθ/dλ   = g^θθ · p_θ        = (1/Σ) · p_θ
 *   dφ/dλ   = g^φφ · b + g^tφ   (from conserved L_z = b, E = 1)
 *
 * This is the "cheap" half of the geodesic RHS (~15 FLOPs):
 * it requires only the metric components, not their derivatives.
 *
 * The velocity terms depend on the current position (r, θ) and
 * momenta (p_r, p_θ), so a fresh call is needed after each kick
 * (which changes the momenta).
 *
 * Metric components in Boyer-Lindquist coordinates:
 *   Σ = r² + a²cos²θ
 *   Δ = r² - 2r + a² + Q²  (clamped to 1e-14 for horizon safety)
 *   w = 2r - Q²
 *   g^rr  = Δ/Σ
 *   g^θθ  = 1/Σ
 *   g^tφ  = -a·w/(Σ·Δ)
 *   g^φφ  = (Σ - w)/(Σ·Δ·sin²θ)
 */
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


/* ── geoForce: kick (momentum update) ─────────────────────── */

/* Computes only the force/momentum part of the geodesic equations:
 *   dp_r/dλ  = -½ ∂(g^μν)/∂r · p_μ p_ν
 *   dp_θ/dλ  = -½ ∂(g^μν)/∂θ · p_μ p_ν
 *
 * This is the "expensive" half of the geodesic RHS (~80 FLOPs):
 * it requires the metric derivatives ∂g^μν/∂r and ∂g^μν/∂θ.
 *
 * The force terms depend on position (r, θ) and momenta (p_r, p_θ),
 * but in the symplectic splitting the kick is evaluated at the
 * *current* position (after the preceding drift has updated q).
 *
 * The computation is identical to the dp_r, dp_θ portion of geoRHS()
 * but avoids computing the velocity terms (saving ~15 FLOPs).
 */
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


/* ── computeHamiltonian: null geodesic constraint ─────────── */

/* Computes the super-Hamiltonian for null geodesic motion:
 *
 *   H = ½ g^μν p_μ p_ν
 *     = ½ [ g^tt·E² + 2·g^tφ·E·L_z + g^rr·p_r² + g^θθ·p_θ² + g^φφ·L_z² ]
 *
 * With the affine normalization E = 1 and L_z = b (impact parameter):
 *
 *   H = ½ [ g^tt + 2b·g^tφ + g^rr·p_r² + g^θθ·p_θ² + g^φφ·b² ]
 *
 * For a perfect null geodesic, H = 0 identically. Any numerical
 * drift in H indicates accumulated integration error.
 *
 * The inverse metric components in Boyer-Lindquist coordinates are:
 *   g^tt  = -A/(Σ·Δ)           where A = (r²+a²)² - Δ·a²·sin²θ
 *   g^tφ  = -a·w/(Σ·Δ)         where w = 2r - Q²
 *   g^rr  = Δ/Σ
 *   g^θθ  = 1/Σ
 *   g^φφ  = (Σ - w)/(Σ·Δ·sin²θ)
 *
 * Cost: ~30 FLOPs (dominated by sin, cos, and one division).
 * This is roughly 1/3 the cost of a full geoRHS() call.
 *
 * Reference: B. Carter, "Global structure of the Kerr family of
 * gravitational fields," Phys. Rev. 174:1559–1571, 1968.
 */


/* ── computeCarter: Carter constant diagnostic ────────────── */

/* Computes the Carter constant Q for Kerr-Newman spacetime:
 *
 *   Q = p_θ² + cos²θ · (b²/sin²θ - a² + Q_charge²)
 *
 * The Carter constant is the fourth integral of motion for geodesic
 * motion in Kerr(-Newman) spacetime, arising from a hidden symmetry
 * encoded in the Killing tensor K_μν:
 *
 *   Q = K^μν p_μ p_ν - (aE - L_z)²
 *
 * For Kerr spacetime (Q_charge = 0), this reduces to:
 *   Q = p_θ² + cos²θ · (b²/sin²θ - a²)
 *
 * The Carter constant should remain exactly conserved along each
 * geodesic. Its drift |ΔQ/Q₀| provides an independent measure of
 * integration quality complementary to the Hamiltonian:
 *   - H monitors constraint satisfaction (are we on the null cone?)
 *   - Q monitors phase-space accuracy (are we on the correct geodesic?)
 *
 * Cost: ~10 FLOPs (one sin, one cos, a few multiplies).
 *
 * References:
 *   - B. Carter, "Global structure of the Kerr family of gravitational
 *     fields," Phys. Rev. 174:1559–1571, 1968.
 *   - S. Chandrasekhar, The Mathematical Theory of Black Holes,
 *     Oxford University Press, 1983, Chapter 7.
 */
__device__ double computeCarter(
    double th, double pth,
    double a, double b, double Q2  /* Q2 = Q_charge² */
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double c2 = cth * cth;
    return pth * pth + c2 * (b * b / s2 - a * a + Q2);
}


/* ── projectHamiltonian: constraint projection ────────────── */

/* Projects the state back onto the H = 0 constraint surface by
 * solving for p_r algebraically.
 *
 * The Hamiltonian is quadratic in p_r with no cross-terms (the
 * Boyer-Lindquist metric is block-diagonal in {t,φ} and {r,θ}):
 *
 *   H = ½ g^rr · p_r² + R
 *
 * where R = ½(g^tt + 2b·g^tφ + g^θθ·p_θ² + g^φφ·b²) collects
 * all terms independent of p_r.
 *
 * Setting H = 0:
 *   0 = ½ g^rr · p_r² + R
 *   p_r² = -2R / g^rr
 *   p_r  = ±√(-2R / g^rr)   [preserving the sign of the current p_r]
 *
 * This is NOT Newton iteration — it is an exact algebraic solve.
 * The projection is a canonical transformation that maps the
 * slightly-off-shell trajectory back to the constraint surface.
 *
 * If -2R/g^rr < 0 (no real solution), the state is left unchanged.
 * This can happen transiently near turning points where p_r ≈ 0.
 *
 * Cost: ~35 FLOPs (metric computation + one sqrt).
 *
 * References:
 *   - J. Wisdom & M. Holman, "Symplectic maps for the N-body
 *     problem," Astron. J. 102:1528–1538, 1991.
 *   - J. Wisdom, "Symplectic correctors for canonical heliocentric
 *     N-body maps," Astron. J. 131:2294–2298, 2006.
 */
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
 *  KERR (INGOING) COORDINATE FUNCTIONS
 *
 *  These functions implement geodesic integration in ingoing Kerr
 *  coordinates (V, r, θ, φ̃) for the Kerr-Newman spacetime.
 *
 *  Ingoing Kerr coordinates (MTW Box 33.2, Eq. 4-5) eliminate
 *  the coordinate singularity at the event horizon that plagues
 *  Boyer-Lindquist coordinates (where Δ → 0 causes g_rr → ∞).
 *  All metric components remain smooth and finite at r = r₊.
 *
 *  The coordinate transformation from BL is (MTW Eq. 4):
 *    dV  = dt + (r²+a²)/Δ · dr
 *    dφ̃  = dφ + a/Δ · dr
 *
 *  The inverse metric in Kerr coordinates, derived by
 *  transforming the BL inverse metric via the Jacobian
 *  ∂x^μ_Kerr/∂x^ν_BL, is:
 *
 *    g^VV  = a²sin²θ/Σ
 *    g^Vr  = (r²+a²)/Σ     ← absent in BL
 *    g^Vφ  = a/Σ            ← absent in BL
 *    g^rr  = Δ/Σ
 *    g^rφ  = a/Σ            ← absent in BL
 *    g^θθ  = 1/Σ
 *    g^φφ  = 1/(Σ sin²θ)
 *
 *  where Σ = r² + a²cos²θ, Δ = r² − 2r + a² + Q².
 *
 *  The super-Hamiltonian F = 2Σ·H with p_V = −1, p_φ = b is:
 *
 *    F = a²s² − 2(r²+a²)·p_r − 2ab
 *        + Δ·p_r² + 2ab·p_r + p_θ² + b²/s²
 *      = a²s² − 2ab + Δ·p_r² + 2[ab − (r²+a²)]·p_r
 *        + p_θ² + b²/s²
 *
 *  References:
 *    [1] C.W. Misner, K.S. Thorne & J.A. Wheeler, "Gravitation,"
 *        W.H. Freeman, 1973.  Box 33.2, Eqs. (4)–(5).
 *    [2] R.P. Kerr, "Gravitational field of a spinning mass as
 *        an example of algebraically special metrics,"
 *        Phys. Rev. Lett. 11:237–238, 1963.
 *    [3] S. Chandrasekhar, The Mathematical Theory of Black Holes,
 *        Oxford University Press, 1983.  Chapter 6.
 *    [4] B. Carter, "Global structure of the Kerr family of
 *        gravitational fields," Phys. Rev. 174:1559–1571, 1968.
 * ============================================================ */


/* ── geoVelocityKS: drift (position update) in Kerr coords ── */

/* Computes the velocity (drift) part of the geodesic equations
 * in ingoing Kerr coordinates:
 *
 *   dr/dλ  = [Δ·p_r + a·b − (r²+a²)] / Σ
 *   dθ/dλ  = p_θ / Σ
 *   dφ/dλ  = [a·p_r − a + b/sin²θ] / Σ
 *
 * Derived from dq^i/dλ = ∂H/∂p_i with the Kerr inverse metric.
 *
 * The radial velocity:
 *   dr/dλ = ∂(2ΣH)/∂p_r / (2Σ)
 *         = [2Δ·p_r + 2(ab − (r²+a²))] / (2Σ)
 *         = [Δ·p_r + ab − (r²+a²)] / Σ
 *
 * The azimuthal velocity:
 *   dφ/dλ = ∂(2ΣH)/∂b / (2Σ)
 *         = [−2a + 2a·p_r + 2b/s²] / (2Σ)
 *         = [a·p_r − a + b/s²] / Σ
 *
 * Key differences from BL (geoVelocity):
 *   - dr/dλ has (ab − (r²+a²))/Σ instead of BL's g^tφ term
 *   - dφ/dλ has a·(p_r − 1)/Σ instead of BL's −a·w/(ΣΔ)
 *   - No Δ-clamping needed: the equations are regular at r = r₊
 *
 * Cost: ~15 FLOPs (comparable to BL version).
 */
__device__ void geoVelocityKS(
    double r, double th, double pr, double pth,
    double a, double b, double Q2,
    double *dr, double *dth, double *dphi
) {
    double sth = sin(th), cth = cos(th);
    double s2 = sth * sth + S2_EPS;
    double a2 = a * a, r2 = r * r;
    double sig = r2 + a2 * cth * cth;          /* Σ = r² + a²cos²θ */
    double del = r2 - 2.0 * r + a2 + Q2;       /* Δ = r² − 2r + a² + Q² */
    double rpa2 = r2 + a2;                      /* r² + a² */
    double isig = 1.0 / sig;

    /* dr/dλ = [Δ·p_r + (a·b − (r²+a²))] / Σ
     * From ∂F/∂p_r = 2Δ·p_r + 2[ab − (r²+a²)], divided by 2Σ.
     * The (ab − rpa2) term comes from g^Vr·p_V + g^rφ·p_φ:
     *   (r²+a²)/Σ · (−1) + (a/Σ) · b = [ab − (r²+a²)]/Σ */
    *dr  = (del * pr + a * b - rpa2) * isig;

    /* dθ/dλ = p_θ / Σ  (identical to BL) */
    *dth = pth * isig;

    /* dφ/dλ = [a·p_r − a + b/sin²θ] / Σ
     * From ∂F/∂b = −2a + 2a·p_r + 2b/s², divided by 2Σ.
     * The −a term comes from g^Vφ·p_V = (a/Σ)·(−1) = −a/Σ */
    *dphi = (a * pr - a + b / s2) * isig;
}


/* ── geoForceKS: kick (momentum update) in Kerr coords ────── */

/* Computes the force (kick) part of the geodesic equations
 * in ingoing Kerr coordinates:
 *
 *   dp_r/dλ  = [(1−r)·p_r² + 2r·p_r] / Σ
 *   dp_θ/dλ  = cosθ · [b²/(sin²θ · sinθ) − a²·sinθ] / Σ
 *
 * Derived from dp_i/dλ = −∂H/∂q^i evaluated on the H = 0
 * constraint surface (null geodesic).  On-shell, F = 2Σ·H = 0,
 * so dp_r/dλ = −∂F/(2Σ·∂r).
 *
 * The p_r force equation:
 *   ∂F/∂r = (2r−2)·p_r² − 4r·p_r
 *   dp_r/dλ = −∂F/(2Σ) = [(1−r)·p_r² + 2r·p_r] / Σ
 *
 *   Derivation of ∂F/∂r:
 *     ∂(a²s²)/∂r = 0
 *     ∂(−2ab)/∂r = 0
 *     ∂(Δ·p_r²)/∂r = (2r−2)·p_r²     [∂Δ/∂r = 2r−2]
 *     ∂(2[ab−(r²+a²)]·p_r)/∂r = −4r·p_r
 *     ∂(p_θ²)/∂r = 0
 *     ∂(b²/s²)/∂r = 0
 *
 * The p_θ force equation (unchanged from previous version):
 *   ∂F/∂θ = 2a²sinθcosθ − 2b²cosθ/sin³θ
 *   dp_θ/dλ = −∂F/(2Σ) = cosθ·[b²/(s²·sinθ) − a²·sinθ] / Σ
 *
 * Key features:
 *   - Simpler than BL: ~20 FLOPs vs ~80 FLOPs
 *   - dp_r/dλ depends on p_r (non-separable)
 *   - dp_θ/dλ depends only on positions (fully separable)
 *   - Independent of Q² (charge terms cancel in ∂F/∂r)
 *
 * Reference: MTW (1973), Box 33.2; derivation from F = 2ΣH.
 */
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

    /* dp_r/dλ = [(1−r)·p_r² + 2r·p_r] / Σ
     *
     * This is the complete radial force in Kerr coordinates.
     * Note: at large r with p_r ≈ 0, the force ≈ 0 (correct for
     * nearly-free propagation in the far field).
     * The p_r dependence (linear and quadratic terms) is the
     * signature of non-separability. */
    *dpr = ((1.0 - r) * pr * pr + 2.0 * r * pr) * isig;

    /* dp_θ/dλ = cosθ · [b²/(s²·sinθ) − a²·sinθ] / Σ
     *
     * The b²/sin³θ term is regularized using s² = sin²θ + ε
     * to prevent divergence at the poles.  The pole reflection
     * at θ = 0.005 (in the integrator) provides additional safety. */
    *dpth = cth * (b * b / (s2 * sth) - a2 * sth) * isig;
}


/* ── computeHamiltonianKS: null constraint in Kerr coords ──── */

/* Computes the super-Hamiltonian for null geodesic motion in
 * ingoing Kerr coordinates:
 *
 *   H = F / (2Σ)
 *
 * where F = a²s² − 2ab + Δ·p_r² + 2[ab − (r²+a²)]·p_r
 *           + p_θ² + b²/s²
 *
 * With the affine normalization E = −p_V = 1 and L_z = p_φ = b.
 *
 * The Kerr inverse metric components are (MTW Box 33.2):
 *   g^VV = a²s²/Σ,  g^Vr = (r²+a²)/Σ,  g^Vφ = a/Σ,
 *   g^rr = Δ/Σ,  g^rφ = a/Σ,  g^θθ = 1/Σ,  g^φφ = 1/(Σs²)
 *
 * Expanding 2ΣH = g^μν p_μ p_ν · Σ:
 *   g^VV·p_V² = a²s²/Σ · 1 · Σ = a²s²
 *   2g^Vr·p_V·p_r = 2(r²+a²)/Σ · (−1) · p_r · Σ = −2(r²+a²)·p_r
 *   2g^Vφ·p_V·p_φ = 2(a/Σ) · (−1) · b · Σ = −2ab
 *   g^rr·p_r² = Δ/Σ · p_r² · Σ = Δ·p_r²
 *   2g^rφ·p_r·p_φ = 2(a/Σ) · p_r · b · Σ = 2ab·p_r
 *   g^θθ·p_θ² = 1/Σ · p_θ² · Σ = p_θ²
 *   g^φφ·p_φ² = 1/(Σs²) · b² · Σ = b²/s²
 *
 * For a perfect null geodesic, H = 0 identically.
 *
 * Cost: ~25 FLOPs.
 *
 * Reference: MTW (1973), Box 33.2; Carter (1968).
 */


/* ── projectHamiltonianKS: constraint projection in Kerr ───── */

/* Projects the state back onto the H = 0 constraint surface by
 * solving for p_r algebraically in ingoing Kerr coordinates.
 *
 * Setting F = 0 gives a quadratic in p_r:
 *
 *   Δ·p_r² + 2·B·p_r + C = 0
 *
 * where:
 *   B = a·b − (r²+a²)     (half the linear coefficient)
 *   C = a²sin²θ − 2ab + p_θ² + b²/sin²θ
 *
 * Using the quadratic formula:
 *   disc = B² − Δ·C
 *   p_r  = (−B ± √disc) / Δ
 *
 * The sign of p_r is preserved from the current value.
 *
 * Near the horizon where Δ → 0, the quadratic degenerates to
 * a linear equation: 2·B·p_r + C = 0, giving:
 *   p_r = −C / (2·B)
 *
 * This linear fallback is well-defined since B = ab − (r²+a²)
 * is generically large and negative (dominated by −(r²+a²)).
 *
 * Cost: ~30 FLOPs (metric computation + discriminant + sqrt).
 *
 * References:
 *   - MTW (1973), Box 33.2.
 *   - Wisdom & Holman (1991), Section 3.
 *   - Wisdom (2006), Eq. 12–14.
 */
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

    /* Half of the linear coefficient: B = a·b − (r²+a²) */
    double Bhalf = a * b - rpa2;

    if (fabs(del) < 1e-14) {
        /* Near horizon: Δ ≈ 0, equation is linear in p_r.
         * 2·B·p_r + C = 0  →  p_r = −C / (2·B)
         * B = ab − (r²+a²) is large and negative, so well-defined. */
        if (fabs(Bhalf) > 1e-30) {
            *pr = -C / (2.0 * Bhalf);
        }
        /* If both Δ ≈ 0 and B ≈ 0, leave p_r unchanged */
    } else {
        /* Standard quadratic solve:
         * Δ·p_r² + 2·B·p_r + C = 0
         * disc = B² − Δ·C
         * p_r = (−B ± √disc) / Δ */
        double disc = Bhalf * Bhalf - del * C;
        if (disc > 0.0) {
            double sqrt_disc = sqrt(disc);
            /* Pick the root closest to the current p_r.
             *
             * In Kerr coordinates, p_r ≈ 0 for ingoing rays at
             * large r (unlike BL where p_r ≈ −1).  The simple
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


/* ── transformBLtoKS: momentum transformation BL → KS ──────── */

/* Transforms the radial covariant momentum from Boyer-Lindquist
 * to Kerr-Schild coordinates:
 *
 *   p_r^KS = p_r^BL + (r² + a² − a·b) / Δ
 *
 * Derived from the Jacobian of the coordinate transformation:
 *   t_BL = t_KS − f(r),  φ_BL = φ_KS − g(r)
 *   where df/dr = (r²+a²)/Δ,  dg/dr = a/Δ
 *
 * The covariant momentum transforms as:
 *   p_r^KS = (∂t_BL/∂r_KS)·p_t^BL + p_r^BL + (∂φ_BL/∂r_KS)·p_φ^BL
 *          = −(r²+a²)/Δ·(−1) + p_r^BL + (−a/Δ)·b
 *          = p_r^BL + (r² + a² − a·b) / Δ
 *
 * Position coordinates (r, θ) are unchanged between BL and KS.
 * The azimuthal angle φ_KS = φ_BL + g(r), but at the observer
 * distance (r >> r₊) the difference is negligible and absorbed
 * into the initial φ₀.
 *
 * At large r_obs, Δ ≈ r² and the correction ≈ 1, so
 * p_r^KS ≈ p_r^BL + 1.  This is a significant correction
 * that must not be neglected.
 *
 * Reference: See plans/kahanli8s-ks-design.md, Section 3.3–3.4.
 */
__device__ void transformBLtoKS(
    double r, double a, double b, double Q2,
    double *pr
) {
    double a2 = a * a, r2 = r * r;
    double del = r2 - 2.0 * r + a2 + Q2;       /* Δ at observer */

    /* p_r^KS = p_r^BL + (r² + a² − a·b) / Δ */
    *pr += (r2 + a2 - a * b) / del;
}


/* ═══════════════════════════════════════════════════════════
 * Volumetric emission: hot corona + relativistic jet
 * ═══════════════════════════════════════════════════════════
 *
 * Called once per integration step to accumulate optically thin
 * emission from the diffuse environment around the black hole.
 * This fills in the dark regions between the disk images with
 * a physically motivated glow, matching the appearance of
 * GRMHD simulation renders (e.g. Moscibrodzka et al. 2016).
 *
 * Two components:
 *   1. Hot corona — exponential atmosphere above/below the disk
 *      plane. Emits thermal bremsstrahlung (warm white glow).
 *   2. Relativistic jet — collimated emission along the spin
 *      axis (|cos θ| > 0.85). Blue-shifted, concentrated near
 *      the BH. Based on force-free jet models (Blandford &
 *      Znajek 1977).
 *
 * Both are optically thin: emission accumulates proportionally
 * to path length (step size he) with no self-absorption.
 */

__device__ void accumulate_volume_emission(
    double r, double th, double he, double a,
    double r_isco, double disk_outer,
    float *acc_r, float *acc_g, float *acc_b, float *acc_a
) {
    double cth = cos(th), sth = sin(th);
    double r_cyl = r * fabs(sth);          /* cylindrical radius */
    double z = r * cth;                     /* height above equator */
    double r_horizon = 1.0 + sqrt(fmax(1.0 - a * a, 0.0));

    /* ── Hot corona (disk atmosphere) ─────────────────────── */
    /* Exponential density profile above/below the equatorial plane.
     * Scale height h = 0.3 × r_cyl (flared disk geometry).
     * Emission ∝ ρ × ds / r where ds = he is the path length.
     * Using 1/r (column density through a slab) not 1/r²
     * (point source), since we're integrating through an
     * extended atmosphere.  Warm orange-white color (~10^7 K).
     *
     * NOTE: blendColor(R,G,B, alpha) applies alpha internally as
     * R * alpha * (1 - acc_a), so pass UN-premultiplied colors. */
    if (r_cyl > r_horizon * 1.5 && r_cyl < disk_outer * 0.7 && r > r_horizon * 1.3) {
        double scale_h = 0.3 * r_cyl;
        double rho = exp(-z * z / (2.0 * scale_h * scale_h));
        float opacity = (float)(rho * he * 0.5 / r);
        opacity = fminf(opacity, 0.08f);
        /* Pass bright emission color; blendColor handles the alpha multiply */
        blendColor(0.80f, 0.50f, 0.25f, opacity,
                   acc_r, acc_g, acc_b, acc_a);
    }

    /* ── Relativistic jet (polar funnel) ──────────────────── */
    /* Collimated synchrotron emission along the spin axis.
     * Half-opening angle ~10° (|cos θ| > 0.985).
     * Gaussian intensity profile in angle from axis.
     * Blue-white color (relativistic electrons).
     * Falls as 1/r (conical expansion). */
    if (fabs(cth) > 0.90 && r > r_horizon * 1.5 && r < 30.0) {
        double axis_dist = 1.0 - fabs(cth);
        double jet_profile = exp(-axis_dist * axis_dist / 0.003);
        float opacity = (float)(jet_profile * he * 0.3 / r);
        opacity = fminf(opacity, 0.06f);
        blendColor(0.30f, 0.50f, 1.00f, opacity,
                   acc_r, acc_g, acc_b, acc_a);
    }
}


#endif /* GEODESIC_BASE_CU */
