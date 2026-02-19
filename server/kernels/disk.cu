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


/* ── Accretion disk color with full GR redshift ───────────── */

__device__ void diskColor(float r, float ph, float a,
                          float isco, float disk_outer, float disk_temp,
                          float b_impact, float charge,
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

    /* --- Full GR redshift factor for Kerr-Newman --- */
    /* Metric components at equator (theta = pi/2, so Sigma = r^2) */
    float r2 = r * r;
    float a2 = a * a;
    float Q2 = charge * charge;
    float Delta = r2 - 2.0f * r + a2 + Q2;
    float w = 2.0f * r - Q2;  /* = 2Mr - Q^2 with M=1 */

    /* Covariant metric at equator (Sigma = r^2):
     * g_tt = -(1 - w/r^2)
     * g_tphi = -a * w / r^2
     * g_phiphi = (r^4 + a^2*r^2 + a^2*w) / r^2 */
    float gtt = -(1.0f - w / r2);
    float gtph = -a * w / r2;
    float gphph = (r2 * r2 + a2 * r2 + a2 * w) / r2;

    /* Angular velocity of prograde circular orbit */
    float dgtt_dr = 2.0f * (Q2 - r) / (r2 * r);
    float dgtph_dr = 2.0f * a * (r - Q2) / (r2 * r);
    float dgphph_dr = 2.0f * r + a2 * (-2.0f / r2 + 2.0f * Q2 / (r2 * r));

    /* Solve quadratic: dgphph*Omega^2 + 2*dgtph*Omega + dgtt = 0 */
    float disc = dgtph_dr * dgtph_dr - dgtt_dr * dgphph_dr;
    float Omega = (-dgtph_dr + sqrtf(fmaxf(disc, 0.0f))) / fmaxf(dgphph_dr, 1e-10f);

    /* Four-velocity normalization:
     * u^t = 1/sqrt(-(g_tt + 2*Omega*g_tphi + Omega^2*g_phiphi)) */
    float denom = -(gtt + 2.0f * Omega * gtph + Omega * Omega * gphph);
    float ut = 1.0f / sqrtf(fmaxf(denom, 1e-10f));

    /* Redshift factor: g = 1 / (u^t * (1 - b * Omega))
     * where b is the photon impact parameter = -alpha * sin(theta_obs) */
    float g = 1.0f / (ut * fmaxf(fabsf(1.0f - b_impact * Omega), 1e-6f));
    /* g should always be positive for physical photons */
    g = fabsf(g);

    /* Apply relativistic beaming: T_obs = g * T_emit, I_obs = g^4 * I_emit */
    float T_obs = g * T_emit;
    float g4 = g * g * g * g;
    I *= g4;

    /* --- Blackbody color from observed temperature --- */
    float col_r, col_g, col_b;
    blackbody(T_obs, &col_r, &col_g, &col_b);

    /* Turbulence texture */
    float tu  = 0.65f + 0.35f * hash2(r * 5.0f, ph * 3.0f);
    float tu2 = 0.8f  + 0.2f  * hash2(r * 18.0f, ph * 9.0f);

    *cr = col_r * I * tu * tu2 * 3.2f;
    *cg = col_g * I * tu * tu2 * 3.2f;
    *cb = col_b * I * tu * tu2 * 3.2f;
}


#endif /* DISK_CU */
