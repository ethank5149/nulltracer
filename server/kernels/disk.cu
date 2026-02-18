/* ============================================================
 *  DISK — Accretion disk emission and color (float32)
 *
 *  Ported from server/backgrounds.py GLSL disk() function.
 *  Uses ad-hoc Doppler boosting (to be replaced by physics-based
 *  g-factor in Phase 1A).
 *
 *  Requires geodesic_base.cu to be included first.
 * ============================================================ */

#ifndef DISK_CU
#define DISK_CU


/* ── Accretion disk color ─────────────────────────────────── */

__device__ void diskColor(float r, float ph, float a,
                          float isco, float disk_outer, float disk_temp,
                          float *cr, float *cg, float *cb) {
    float ri = isco;

    /* Outside disk bounds → no emission */
    if (r < ri * 0.85f || r > disk_outer) {
        *cr = 0.0f; *cg = 0.0f; *cb = 0.0f;
        return;
    }

    float x = r / ri;
    float tp = powf(x, -0.75f) * disk_temp;
    float I = powf(tp, 4.0f) / (r * 0.3f);

    /* Smooth edges */
    I *= smoothstepf(ri * 0.85f, ri * 1.3f, r);
    I *= smoothstepf(disk_outer, disk_outer * 0.55f, r);

    /* Ad-hoc Doppler boosting (Phase 1A will replace with physics g-factor) */
    float vo = 1.0f / sqrtf(r);
    float dop = 1.0f + 0.65f * vo * sinf(ph);
    float boost = powf(fmaxf(dop, 0.1f), 3.0f);
    I *= boost;

    /* Temperature-based color ramp */
    float t = fminf(fmaxf(tp * boost * 0.45f, 0.0f), 3.5f);
    float col_r, col_g, col_b;

    if (t < 0.4f) {
        float f = t * 2.5f;
        col_r = 0.25f + (0.85f - 0.25f) * f;
        col_g = 0.03f + (0.15f - 0.03f) * f;
        col_b = 0.0f  + (0.01f - 0.0f)  * f;
    } else if (t < 0.9f) {
        float f = (t - 0.4f) * 2.0f;
        col_r = 0.85f + (1.0f  - 0.85f) * f;
        col_g = 0.15f + (0.55f - 0.15f) * f;
        col_b = 0.01f + (0.08f - 0.01f) * f;
    } else if (t < 1.7f) {
        float f = (t - 0.9f) / 0.8f;
        col_r = 1.0f  + (1.0f  - 1.0f)  * f;
        col_g = 0.55f + (0.92f - 0.55f) * f;
        col_b = 0.08f + (0.6f  - 0.08f) * f;
    } else if (t < 2.5f) {
        float f = (t - 1.7f) / 0.8f;
        col_r = 1.0f  + (1.0f  - 1.0f)  * f;
        col_g = 0.92f + (1.0f  - 0.92f) * f;
        col_b = 0.6f  + (0.95f - 0.6f)  * f;
    } else {
        col_r = 1.0f; col_g = 1.0f; col_b = 1.0f;
    }

    /* Turbulence texture */
    float tu  = 0.65f + 0.35f * hash2(r * 5.0f, ph * 3.0f);
    float tu2 = 0.8f  + 0.2f  * hash2(r * 18.0f, ph * 9.0f);

    *cr = col_r * I * tu * tu2 * 3.2f;
    *cg = col_g * I * tu * tu2 * 3.2f;
    *cb = col_b * I * tu * tu2 * 3.2f;
}


#endif /* DISK_CU */
