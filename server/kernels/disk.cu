/* ============================================================
 *  DISK — Accretion disk emission and color (float32)
 *
 *  Originally ported from GLSL disk() / blackbody()
 *  functions.  Uses proper blackbody spectrum coloring and full
 *  GR redshift for Kerr-Newman metric.
 *
 *  Requires geodesic_base.cu to be included first.
 * ============================================================ */

#ifndef DISK_CU
#define DISK_CU


/* ── Blackbody spectrum approximation ─────────────────────── */

/* Helland-style approximation for blackbody → linear sRGB.
 * Input: T in Kelvin (1000K to 40000K range).
 * Output: linear sRGB color (NOT gamma corrected), normalized
 *         so white ≈ (1,1,1) at ~6500K.
 */
__device__ void blackbody(float T, float *out_r, float *out_g, float *out_b) {
    float t = T / 100.0f;
    float r, g, b;

    /* Red */
    if (t <= 66.0f) {
        r = 1.0f;
    } else {
        r = 1.292936186f * powf(t - 60.0f, -0.1332047592f);
        r = fminf(fmaxf(r, 0.0f), 1.0f);
    }

    /* Green */
    if (t <= 66.0f) {
        g = 0.3900815788f * logf(t) - 0.6318414438f;
        g = fminf(fmaxf(g, 0.0f), 1.0f);
    } else {
        g = 1.129890861f * powf(t - 60.0f, -0.0755148492f);
        g = fminf(fmaxf(g, 0.0f), 1.0f);
    }

    /* Blue */
    if (t >= 66.0f) {
        b = 1.0f;
    } else if (t <= 19.0f) {
        b = 0.0f;
    } else {
        b = 0.5432067891f * logf(t - 10.0f) - 1.19625409f;
        b = fminf(fmaxf(b, 0.0f), 1.0f);
    }

    *out_r = r;
    *out_g = g;
    *out_b = b;
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


/* ── Accretion disk color with configurable Doppler boost ──── */

__device__ void diskColor(float r, float ph, float a,
                          float isco, float disk_outer, float disk_temp,
                          float g_factor, int doppler_boost,
                          float *cr, float *cg, float *cb) {
    float ri = isco;

    /* Outside disk bounds → no emission */
    if (r < ri * 0.85f || r > disk_outer) {
        *cr = 0.0f; *cg = 0.0f; *cb = 0.0f;
        return;
    }

    /* --- Novikov-Thorne temperature profile --- */
    float x = r / ri;
    float T_base = 8000.0f * disk_temp;  /* Base temperature at ISCO */
    float T_emit = T_base * powf(x, -0.75f);

    /* --- Intensity: Stefan-Boltzmann I ∝ T^4 / r --- */
    float I = powf(T_emit / T_base, 4.0f) / (r * 0.3f);

    /* --- Edge smoothing --- */
    I *= smoothstepf(ri * 0.85f, ri * 1.3f, r);
    I *= smoothstepf(disk_outer, disk_outer * 0.55f, r);

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

    /* --- Blackbody color from observed temperature --- */
    float col_r, col_g, col_b;
    blackbody(T_obs, &col_r, &col_g, &col_b);

    /* Turbulence texture */
    float tu  = 0.65f + 0.35f * hash2(r * 5.0f, ph * 3.0f);
    float tu2 = 0.8f  + 0.2f  * hash2(r * 18.0f, ph * 9.0f);

    *cr = col_r * I_adjusted * tu * tu2 * 3.2f;
    *cg = col_g * I_adjusted * tu * tu2 * 3.2f;
    *cb = col_b * I_adjusted * tu * tu2 * 3.2f;
}


#endif /* DISK_CU */
