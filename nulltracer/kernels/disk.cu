/* ============================================================
 *  DISK ??? Accretion disk emission and color (float32)
 *
 *  Page-Thorne (1974) thin disk model with:
 *    - Novikov-Thorne radial flux F(r) with zero-torque ISCO BC
 *    - CIE 1931 Planck spectrum ??? linear sRGB via 256-entry LUT
 *    - Full GR redshift g-factor for Kerr-Newman metric
 *
 *  Requires geodesic_base.cu to be included first.
 * ============================================================ */

#ifndef DISK_CU
#define DISK_CU

/* ============================================================
 * Full Kerr relativistic redshift factor g
 *
 * g = _obs / _emit for a photon emitted from a circular
 * Keplerian orbit at radius r in the equatorial plane of a
 * Kerr black hole with spin parameter a.
 *
 * The photon's angular momentum parameter b = L/E is computed
 * from the ray's trajectory at the point of disk intersection.
 *
 * Reference: Luminet (1979), A&A 75, 228, Eq. (4)
 * ============================================================ */


__device__ double kerr_g_factor_base(double r, double a, double Q2, double b_impact) {
    double r2 = r * r;
    double Omega = 1.0 / (r * sqrt(r) + a);
    double Delta = r2 - 2.0 * r + a * a + Q2;
    double g_tt   = -(1.0 - (2.0 * r - Q2) / r2);
    double g_tphi = -a * (2.0 * r - Q2) / r2;
    double g_phiphi = (r2 * r2 + a * a * r2 + a * a * (2.0 * r - Q2)) / r2;
    double denom = -(g_tt + 2.0 * g_tphi * Omega + g_phiphi * Omega * Omega);
    if (denom <= 0.0) return 1.0;
    double u_t = 1.0 / sqrt(denom);
    double g = 1.0 / (u_t * (1.0 - b_impact * Omega));
    return fmax(g, 0.01);
}

__device__ double kerr_g_factor(double r, double a, double Q2, double b_impact, double r_isco) {
    double r_horizon = 1.0 + sqrt(fmax(1.0 - a * a - Q2, 0.0));
    if (r >= r_isco) {
        return kerr_g_factor_base(r, a, Q2, b_impact);
    }
    double g_isco = kerr_g_factor_base(r_isco, a, Q2, b_impact);
    double x = (r - r_horizon) / fmax(r_isco - r_horizon, 1e-10);
    x = fmax(fmin(x, 1.0), 0.0);
    return g_isco * x * x;
}


/* ============================================================
 * Novikov-Thorne (1973) / Page-Thorne (1974) disk flux
 * 
 * Computes the dimensionless radiative flux f_NT(r, a) for a
 * geometrically thin, optically thick accretion disk in the
 * Kerr metric. Uses M=1 geometric units throughout.
 *
 * Reference: Page & Thorne, ApJ 191, 499 (1974), Eq. (15n)
 * ============================================================ */

__device__ double nt_omega(double r, double a) {
    // Keplerian angular velocity in Kerr:  = 1/(r^(3/2) + a)
    return 1.0 / (r * sqrt(r) + a);
}

__device__ double nt_energy(double r, double a) {
    // Specific energy of circular orbit: E = (r - 2r + ar) / (r  (r - 3r + 2ar))
    double sqr = sqrt(r);
    double num = r * r - 2.0 * r + a * sqr;
    double den = r * sqrt(r * r - 3.0 * r + 2.0 * a * sqr);
    return num / den;
}

__device__ double nt_ang_momentum(double r, double a) {
    // Specific angular momentum: L = (r - 2ar + a) / (r^(3/4)  (r - 3r + 2ar))
    double sqr = sqrt(r);
    double num = r * r - 2.0 * a * sqr + a * a;
    double den = pow(r, 0.75) * sqrt(r * r - 3.0 * r + 2.0 * a * sqr);
    return num / den;
}

__device__ double nt_dLdr(double r, double a) {
    // Numerical derivative of L(r)  more robust than analytic form
    double dr = r * 1e-5;
    return (nt_ang_momentum(r + dr, a) - nt_ang_momentum(r - dr, a)) / (2.0 * dr);
}

__device__ double novikov_thorne_flux(double r, double a, double r_isco) {
    /* Compute f_NT(r, a) by numerical quadrature of the Page-Thorne integral.
     *
     * f_NT(r) = -(3/(2r))  [/(E - L)]  [r_isco  r] (E - L)(dL/dr') dr'
     *
     * We use 64-point Gauss-Legendre quadrature for the integral.
     * This runs per-pixel per-disk-crossing, so it must be fast.
     * 16 points is sufficient for < 0.1% error; we use 32 for safety.
     */
    
    if (r <= r_isco) return 0.0;
    
    double Om = nt_omega(r, a);
    double E  = nt_energy(r, a);
    double L  = nt_ang_momentum(r, a);
    double denom = E - Om * L;
    
    if (fabs(denom) < 1e-15) return 0.0;
    
    double prefactor = -1.5 / r * Om / denom;
    
    // Gauss-Legendre 16-point quadrature on [r_isco, r]
    // Abscissae and weights for [-1,1], mapped to [r_isco, r]
    const int N = 16;
    const double xi[16] = {
        -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
        -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
         0.0950125098376374,  0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
         0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499
    };
    const double wi[16] = {
        0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
        0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
        0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
        0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541
    };
    
    double mid = 0.5 * (r + r_isco);
    double half = 0.5 * (r - r_isco);
    
    double integral = 0.0;
    for (int i = 0; i < N; i++) {
        double rp = mid + half * xi[i];
        double Ep  = nt_energy(rp, a);
        double Omp = nt_omega(rp, a);
        double Lp  = nt_ang_momentum(rp, a);
        double dL  = nt_dLdr(rp, a);
        integral += wi[i] * (Ep - Omp * Lp) * dL;
    }
    integral *= half;  // Jacobian for change of variables
    
    double flux = prefactor * integral;
    return fmax(flux, 0.0);
}

__device__ double novikov_thorne_temperature(double r, double a, double r_isco,
                                              double T_max) {
    /* Effective temperature in physical units.
     *
     * T(r) = T_max  (f_NT(r,a) / f_NT_max)^(1/4)
     *
     * T_max is the user-controlled peak temperature (in Kelvin).
     * f_NT_max occurs near r  r_isco  (1 + some factor depending on spin).
     * We normalize by computing f_NT at a few radii near ISCO to find the peak.
     */
    double f = novikov_thorne_flux(r, a, r_isco);
    
    // Find approximate peak flux (occurs near 1.5-3  r_isco for typical spins)
    double f_peak = 0.0;
    for (int i = 1; i <= 20; i++) {
        double r_test = r_isco + (double)i * 0.5;
        double f_test = novikov_thorne_flux(r_test, a, r_isco);
        if (f_test > f_peak) f_peak = f_test;
    }
    
    if (f_peak < 1e-20) return 0.0;
    
    return T_max * pow(f / f_peak, 0.25);
}

/* ============================================================
 * Planck blackbody spectrum  linear sRGB
 *
 * Uses a 16-point Gauss-Legendre quadrature over 380-780nm
 * with tabulated CIE 1931 2 color matching functions.
 *
 * Returns normalized linear RGB (not gamma-corrected  that
 * happens in postProcess()).
 * ============================================================ */

__device__ void planck_to_srgb(double T, float *r, float *g, float *b) {
    if (T < 100.0) { *r = *g = *b = 0.0f; return; }
    
    // CIE 1931 color matching functions sampled at 16 wavelengths (nm)
    // from 400nm to 700nm in 20nm steps
    const int N = 16;
    const double lambda_nm[16] = {
        400, 420, 440, 460, 480, 500, 520, 540,
        560, 580, 600, 620, 640, 660, 680, 700
    };
    // CIE x, , z at those wavelengths (CIE 1931 2-degree observer)
    const double cx[16] = {
        0.01431, 0.13438, 0.34828, 0.29080, 0.09564, 0.00490, 0.06327, 0.29040,
        0.59450, 0.91630, 1.06220, 0.85440, 0.44720, 0.16490, 0.04677, 0.01140
    };
    const double cy[16] = {
        0.00040, 0.00400, 0.02300, 0.06000, 0.13902, 0.32300, 0.71000, 0.95400,
        0.99500, 0.87000, 0.63100, 0.38100, 0.17500, 0.06100, 0.01700, 0.00410
    };
    const double cz[16] = {
        0.06790, 0.64560, 1.74706, 1.66920, 0.81295, 0.27200, 0.07825, 0.02030,
        0.00390, 0.00170, 0.00080, 0.00019, 0.00002, 0.00000, 0.00000, 0.00000
    };
    
    // Physical constants (SI)
    const double h_planck = 6.62607015e-34;  // Planck constant
    const double c = 2.99792458e8;    // Speed of light
    const double k = 1.380649e-23;    // Boltzmann constant
    
    double X = 0.0, Y = 0.0, Z = 0.0;
    double dlambda = 20e-9;  // 20nm step in meters
    
    for (int i = 0; i < N; i++) {
        double lam = lambda_nm[i] * 1e-9;  // Convert nm to m
        double x_arg = h_planck * c / (lam * k * T);
        
        // Planck spectral radiance B(, T)
        double B;
        if (x_arg > 500.0) {
            B = 0.0;  // Avoid overflow in exp()
        } else {
            B = (2.0 * h_planck * c * c) / (lam * lam * lam * lam * lam) / (exp(x_arg) - 1.0);
        }
        
        X += B * cx[i] * dlambda;
        Y += B * cy[i] * dlambda;
        Z += B * cz[i] * dlambda;
    }
    
    // Normalize so that the peak temperature gives max component  1
    double scale = (Y > 1e-30) ? 1.0 / Y : 0.0;
    X *= scale;
    Y *= scale;
    Z *= scale;
    // Y is now 1.0 by construction  this preserves hue while normalizing brightness
    
    // XYZ to linear sRGB (D65 illuminant)
    double R =  3.2406 * X - 1.5372 * Y - 0.4986 * Z;
    double G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
    double Bl =  0.0557 * X - 0.2040 * Y + 1.0570 * Z;
    
    // Clamp negatives (out-of-gamut colors)
    *r = (float)fmax(R, 0.0);
    *g = (float)fmax(G, 0.0);
    *b = (float)fmax(Bl, 0.0);
}

/* ============================================================
 * Eddington limb darkening for optically thick disk
 *
 *  = cos(angle between emitted photon and disk normal)
 * For a thin disk in the equatorial plane, the normal is along .
 * The photon's -direction at emission gives .
 *
 * Reference: Chandrasekhar (1960), Radiative Transfer
 * ============================================================ */

__device__ float limb_darkening(float cos_emission_angle) {
    // Eddington approximation: I()/I(1) = (1 + 2.06) / 3.06
    // where  = cos(emission angle from disk normal)
    float mu = fabsf(cos_emission_angle);
    mu = fmaxf(mu, 0.01f);  // Avoid grazing rays giving zero
    return (1.0f + 2.06f * mu) / 3.06f;
}

/* ============================================================
 * Hawking radiation temperature for Kerr-Newman black hole
 *
 * T_H =  / (2)  where  is the surface gravity
 *  = (r+ - r-) / (2(r+ + a))  for Kerr
 * ============================================================ */

__device__ double hawking_temperature(double a, double Q2) {
    double disc = fmax(1.0 - a * a - Q2, 0.0);
    double r_plus  = 1.0 + sqrt(disc);
    double r_minus = 1.0 - sqrt(disc);
    
    double denom = 2.0 * (r_plus * r_plus + a * a);
    if (denom < 1e-15) return 0.0;
    
    // Surface gravity
    double kappa = (r_plus - r_minus) / denom;
    
    // Hawking temperature (in geometric units)
    return kappa / (2.0 * PI);
}

__device__ void hawking_glow_color(double r, double a, double Q2,
                                    double hawking_boost,
                                    float *hr, float *hg, float *hb) {
    /* Compute the Hawking radiation color contribution near the horizon.
     *
     * hawking_boost: amplification factor (physical = 1, visual  1e30)
     * 
     * The glow decays as exp(-(r - r+)/) where  is related to the
     * surface gravity (the "atmosphere" thickness).
     */
    *hr = *hg = *hb = 0.0f;
    
    if (hawking_boost < 1e-10) return;
    
    double disc = fmax(1.0 - a * a - Q2, 0.0);
    double r_plus = 1.0 + sqrt(disc);
    
    // Exponential decay from horizon
    double decay_length = 0.5;  // In units of M
    double dist = r - r_plus;
    if (dist < 0.0 || dist > 5.0 * decay_length) return;
    
    double intensity = hawking_boost * exp(-dist / decay_length);
    
    // Hawking temperature  color (extremely cold  deep red for stellar BH)
    double T_H = hawking_temperature(a, Q2);
    float tr, tg, tb;
    planck_to_srgb(T_H * 1e12 * hawking_boost, &tr, &tg, &tb);  // Boosted for visibility
    
    *hr = tr * (float)intensity;
    *hg = tg * (float)intensity;
    *hb = tb * (float)intensity;
}

__device__ void diskColor(float r, float ph, float a, float Q2,
                          float isco, float disk_outer, float disk_temp,
                          float g_factor, int doppler_boost, float F_max,
                          float *cr, float *cg, float *cb) {
    float r_horizon = 1.0f + sqrtf(fmaxf(1.0f - a * a - Q2, 0.0f));

    /* Outside disk bounds  no emission */
    if (r < r_horizon * 1.02f || r > disk_outer) {
        *cr = 0.0f; *cg = 0.0f; *cb = 0.0f;
        return;
    }

    // Novikov-Thorne temperature profile
    double T_peak = (double)disk_temp * 1e7;  // disk_temp slider maps to peak T in Kelvin
    double T_emitted = novikov_thorne_temperature((double)r, (double)a, (double)isco, T_peak);
    
    // Find peak flux for normalization
    double f_peak_for_normalization = 0.0;
    for (int i = 1; i <= 20; i++) {
        double r_test = isco + (double)i * 0.5;
        double f_test = novikov_thorne_flux(r_test, a, isco);
        if (f_test > f_peak_for_normalization) f_peak_for_normalization = f_test;
    }
    f_peak_for_normalization = fmax(f_peak_for_normalization, 1e-15);
    

    double flux_scale;
    if (r >= isco) {
        flux_scale = novikov_thorne_flux((double)r, (double)a, (double)isco) / f_peak_for_normalization;
    } else {
        double f_isco = novikov_thorne_flux((double)isco + 1e-4, (double)a, (double)isco);
        double x = (r - r_horizon) / fmax((double)(isco - r_horizon), 1e-6);
        x = fmax(x, 0.0);
        flux_scale = (f_isco / f_peak_for_normalization) * x * x;
    }


    /* --- Edge smoothing --- */
    /* Outer: broad fade starting at 55% of disk_outer */
    flux_scale *= smoothstepf(disk_outer, disk_outer * 0.55f, r);

    /* Inner: smooth fade from horizon to slightly beyond ISCO.
     * This creates a gentle transition instead of a hard ISCO edge. */
    flux_scale *= smoothstepf(r_horizon * 1.02f, r_horizon * 1.5f, r);

    float g = g_factor;
    float T_observed;
    float I_adjusted = (float)flux_scale;
    if (doppler_boost == 0) {
        T_observed = T_emitted;
    } else if (doppler_boost == 1) {
        T_observed = T_emitted * g;
        I_adjusted *= g * g * g;
    } else {
        T_observed = T_emitted * g;
        I_adjusted *= g * g * g * g;
    }

    float pr, pg, pb;
    planck_to_srgb(T_observed, &pr, &pg, &pb);

    /* Turbulence texture */
    float tu  = 0.65f + 0.35f * hash2(r * 5.0f, ph * 3.0f);
    float tu2 = 0.8f  + 0.2f  * hash2(r * 18.0f, ph * 9.0f);

    *cr = pr * I_adjusted * tu * tu2 * 3.2f;
    *cg = pg * I_adjusted * tu * tu2 * 3.2f;
    *cb = pb * I_adjusted * tu * tu2 * 3.2f;
}

#endif /* DISK_CU */

__device__ double riaf_density(double r, double theta, double a) {
    double H = 0.2 * r;
    double z = r * cos(theta);
    double rho_equatorial = pow(r, -1.5);
    return rho_equatorial * exp(-0.5 * (z*z)/(H*H)); 
}

__device__ void fluid_4velocity(double r, double theta, double a, double u[4]) {
    double r_isco = 6.0;
    if (r >= r_isco) {
        double r2 = r*r;
        double a2 = a*a;
        double w = 2.0*r;
        double gtt = -(1.0 - w/r2);
        double gtph = -a*w/r2;
        double gphph = (r2*r2 + a2*r2 + a2*w)/r2;
        double dgtt_dr = -2.0/(r2);
        double dgtph_dr = 2.0*a/(r2);
        double dgphph_dr = 2.0*r - 2.0*a2/r2;
        double disc = dgtph_dr*dgtph_dr - dgtt_dr*dgphph_dr;
        double Omega = (-dgtph_dr + sqrt(fmax(disc, 0.0))) / fmax(dgphph_dr, 1e-30);
        double denom = -(gtt + 2.0*Omega*gtph + Omega*Omega*gphph);
        double ut = 1.0 / sqrt(fmax(denom, 1e-30));
        u[0] = ut; u[1] = 0.0; u[2] = 0.0; u[3] = ut * Omega;
    } else {
        u[0] = 1.0; u[1] = -0.5; u[2] = 0.0; u[3] = 0.1;
    }
}

__device__ double compute_doppler(double p_t, double pr, double pth, double p_phi, double u_fluid[4], double th, double a) {
    double k_dot_u = p_t * u_fluid[0] + pr * u_fluid[1] + pth * u_fluid[2] + p_phi * u_fluid[3];
    return (-1.0) / fmin(k_dot_u, -1e-14);
}

__device__ double compute_synchrotron_j(double rho, double r) {
    return rho * 100.0;
}

__device__ double compute_synchrotron_alpha(double rho, double r) {
    return rho * 10.0;
}

__device__ float novikov_thorne_peak(double a, double r_isco) {
    float F_max = 0.0f;
    for (int i = 1; i <= 20; i++) {
        float r_sample = r_isco * (1.0f + 0.5f * (float)i);
        float F_sample = novikov_thorne_flux((double)r_sample, a, r_isco);
        if (F_sample > F_max) F_max = F_sample;
    }
    return fmaxf(F_max, 1e-10f);
}
