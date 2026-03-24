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

#ifndef DISK_CU
#define DISK_CU


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


/* ── Extended g-factor with plunging region support ────────── */

/* Inside the ISCO, no stable circular orbits exist. Gas follows
 * plunging geodesics with approximately the ISCO angular momentum.
 * We interpolate g smoothly from its ISCO value toward gravitational
 * redshift at the horizon. This avoids the discontinuity at r=ISCO
 * and produces physically reasonable (if approximate) Doppler shifts
 * for the plunging region.
 *
 * Reference: Cunningham (1975), ApJ 202, 788 — plunging region spectra */

__device__ float compute_g_factor_extended(double r, double a, double Q2,
                                           double b, double r_isco) {
    double r_horizon = 1.0 + sqrt(fmax(1.0 - a * a - Q2, 0.0));

    if (r >= r_isco) {
        return compute_g_factor(r, a, Q2, b);
    }

    /* Plunging region: smooth fade from g(ISCO) toward 0 at horizon */
    float g_isco = compute_g_factor(r_isco, a, Q2, b);
    double x = (r - r_horizon) / fmax(r_isco - r_horizon, 1e-10);
    x = fmax(fmin(x, 1.0), 0.0);
    /* Quadratic profile: steepens near horizon, gentle near ISCO */
    return g_isco * (float)(x * x);
}

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


/* ── Accretion disk color with plunging region ─────────────── */

/* Extended disk model:
 *   r > ISCO:    Standard Novikov-Thorne (Page & Thorne 1974)
 *   ISCO > r > horizon: Plunging region — gas on infall geodesics
 *                 with decaying emission (Cunningham 1975)
 *
 * The plunging region fills the visual gap between the ISCO
 * and the shadow boundary, producing a more realistic appearance
 * that matches GRMHD simulations (Moscibrodzka et al. 2016). */

__device__ void diskColor(float r, float ph, float a,
                          float isco, float disk_outer, float disk_temp,
                          float g_factor, int doppler_boost,
                          float *cr, float *cg, float *cb) {
    float ri = isco;
    float r_horizon = 1.0f + sqrtf(fmaxf(1.0f - a * a, 0.0f));

    /* Outside disk bounds → no emission */
    if (r < r_horizon * 1.02f || r > disk_outer) {
        *cr = 0.0f; *cg = 0.0f; *cb = 0.0f;
        return;
    }

    /* --- Flux profile with plunging region --- */
    float F_norm;

    /* Compute peak NT flux for normalization */
    float F_max = 0.0f;
    for (int i = 1; i <= 20; i++) {
        float r_sample = ri * (1.0f + 0.5f * (float)i);
        float F_sample = novikov_thorne_flux((double)r_sample, (double)a, (double)ri);
        if (F_sample > F_max) F_max = F_sample;
    }
    F_max = fmaxf(F_max, 1e-10f);

    if (r >= ri) {
        /* Standard Novikov-Thorne region */
        float F = novikov_thorne_flux((double)r, (double)a, (double)ri);
        F_norm = fminf(F / F_max, 1.0f);
    } else {
        /* Plunging region: flux decays from ISCO value toward horizon.
         * Gas retains roughly the ISCO angular momentum but loses
         * energy as it spirals inward. Emission ∝ x² gives a
         * smooth fade that avoids a hard edge at the ISCO. */
        float F_isco = novikov_thorne_flux((double)ri, (double)a, (double)ri);
        float x = (r - r_horizon) / fmaxf(ri - r_horizon, 1e-6f);
        x = fmaxf(x, 0.0f);
        F_norm = fminf(F_isco / F_max, 1.0f) * x * x;
    }

    /* --- Temperature from flux: T ∝ F^{1/4} --- */
    float T_base = 8000.0f * disk_temp;
    float T_emit = T_base * powf(fmaxf(F_norm, 0.0f), 0.25f);

    /* --- Intensity proportional to flux --- */
    float I = F_norm * 3.0f / fmaxf(r * 0.15f, 0.01f);

    /* --- Edge smoothing --- */
    /* Outer: broad fade starting at 55% of disk_outer */
    I *= smoothstepf(disk_outer, disk_outer * 0.55f, r);

    /* Inner: smooth fade from horizon to slightly beyond ISCO.
     * This creates a gentle transition instead of a hard ISCO edge. */
    I *= smoothstepf(r_horizon * 1.02f, r_horizon * 1.5f, r);

    /* --- Apply redshift based on doppler_boost mode --- */
    float g = g_factor;
    float T_obs, I_adjusted;
    if (doppler_boost == 0) {
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
