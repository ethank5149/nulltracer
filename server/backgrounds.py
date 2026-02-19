"""
Background GLSL functions for the Nulltracer shader.

Contains bgStars, bgChecker, bgColorMap, background selector,
sphereDir, and the accretion disk function.
"""


def background_functions():
    """Return the GLSL background functions and related utilities.

    Includes: bgStars, bgChecker, bgColorMap, background (selector),
    sphereDir, and the accretion disk function.

    These all operate on cube-map-projected coordinates to avoid
    pole artifacts.
    """
    return """\

// ===========================================================
//  BACKGROUND FUNCTIONS
//  All take vec3 d (unit direction). Never use theta or phi.
// ===========================================================

vec3 bgStars(vec3 d) {
    float face; vec2 uv;
    cubeMap(d, face, uv);
    vec2 fuv = uv * 0.5 + 0.5;  // [0,1] on each face

    // Milky Way band: bright near equator (d.z ≈ 0)
    float mw = exp(-8.0 * d.z * d.z);

    // Nebula noise — all cube-face-local, no (θ,φ) anywhere
    vec3 c = vec3(0.007, 0.008, 0.018);
    c += vec3(0.018, 0.014, 0.028) * mw;
    vec2 ns1 = floor(fuv * 5.0 + face * 7.0);
    c += vec3(0.018, 0.006, 0.022) * hash(ns1) * mw * 0.5;
    vec2 ns2 = floor(fuv * 3.0 + face * 13.0 + 40.0);
    c += vec3(0.004, 0.012, 0.022) * hash(ns2) * 0.25;

    // Stars — cube-face-local cells
    for (int L = 0; L < STAR_LAYERS; L++) {
        float sc = 10.0 + float(L) * 14.0;
        vec2 cell = floor(fuv * sc);
        vec2 seed = cell + face * 100.0 + float(L) * 47.0;
        float h = hash(seed);
        if (h > 0.88) {
            vec2 sp = (cell + 0.3 + 0.4 * vec2(hash(seed+0.5), hash(seed+1.5))) / sc;
            float dist = length(fuv - sp) * sc;
            float s = exp(-dist * dist * 5.0);
            float t = hash(seed + 77.0);
            vec3 sc2 = t < 0.2 ? vec3(1,0.7,0.4) : t < 0.55 ? vec3(1,0.95,0.8) :
                       t < 0.8 ? vec3(0.8,0.9,1) : vec3(0.6,0.75,1);
            c += sc2 * s * (0.4 + 2.0 * hash(seed + 33.0));
        }
    }
    return c;
}

vec3 bgChecker(vec3 d) {
    float face; vec2 uv;
    cubeMap(d, face, uv);
    float check = cubeChecker(uv, 6.0);
    vec3 base = mix(vec3(0.05,0.045,0.065), vec3(0.09,0.07,0.045), check);
    base += faceColor(face) * (0.4 + 0.15 * check);
    base += vec3(0.18,0.15,0.10) * cubeGrid(uv, 6.0);
    // Face-edge seams (where cube faces meet)
    float edgeDist = 1.0 - max(abs(uv.x), abs(uv.y));
    base += vec3(0.06,0.05,0.04) * (1.0 - smoothstep(0.0, 0.04, edgeDist));
    // Equator highlight (z ≈ 0) — smooth in Cartesian
    base += vec3(0.22,0.14,0.05) * (1.0 - smoothstep(0.0, 0.04, abs(d.z)));
    return base;
}

vec3 bgColorMap(vec3 d) {
    // Direct axis → channel mapping. Inherently pole-safe.
    vec3 col;
    col.r = 0.08 + 0.35 * (d.x * 0.5 + 0.5);
    col.g = 0.08 + 0.35 * (d.y * 0.5 + 0.5);
    col.b = 0.08 + 0.35 * max(-d.z, 0.0);
    col += vec3(0.08,0.04,0.02) * max(d.z, 0.0);
    col = pow(col, vec3(0.8));
    // Grid overlay
    float face; vec2 uv;
    cubeMap(d, face, uv);
    col += vec3(0.12,0.10,0.08) * cubeGrid(uv, 6.0);
    // Equator
    col += vec3(0.15,0.12,0.05) * (1.0 - smoothstep(0.0, 0.04, abs(d.z)));
    return col;
}

vec3 background(vec3 d) {
    vec3 c;
#if BG_MODE == 0
    c = bgStars(d);
#elif BG_MODE == 1
    c = bgChecker(d);
#else
    c = bgColorMap(d);
#endif
    // Extra grid for stars mode
    if (u_grid > 0.5 && BG_MODE == 0) {
        float face; vec2 uv;
        cubeMap(d, face, uv);
        c += vec3(0.055,0.04,0.028) * cubeGrid(uv, 6.0);
        c += vec3(0.07,0.035,0.02) * (1.0 - smoothstep(0.0, 0.03, abs(d.z)));
    }
    return c;
}

// Convert (theta, phi) to unit direction vector.
// At the poles, sinθ→0, so φ-error vanishes from the result.
vec3 sphereDir(float th, float ph) {
    float sth = sin(th);
    return vec3(sth * cos(ph), sth * sin(ph), cos(th));
}

// ===========================================================
//  BLACKBODY SPECTRUM
// ===========================================================

vec3 blackbody(float T) {
    // Attempt Helland-style approximation for blackbody → linear sRGB
    // Input: T in Kelvin (1000K to 40000K range)
    // Output: linear sRGB color (NOT gamma corrected), normalized so white ≈ (1,1,1) at ~6500K
    float t = T / 100.0;
    float r, g, b;
    
    // Red
    if (t <= 66.0) {
        r = 1.0;
    } else {
        r = 1.292936186 * pow(t - 60.0, -0.1332047592);
        r = clamp(r, 0.0, 1.0);
    }
    
    // Green
    if (t <= 66.0) {
        g = 0.3900815788 * log(t) - 0.6318414438;
        g = clamp(g, 0.0, 1.0);
    } else {
        g = 1.129890861 * pow(t - 60.0, -0.0755148492);
        g = clamp(g, 0.0, 1.0);
    }
    
    // Blue
    if (t >= 66.0) {
        b = 1.0;
    } else if (t <= 19.0) {
        b = 0.0;
    } else {
        b = 0.5432067891 * log(t - 10.0) - 1.19625409;
        b = clamp(b, 0.0, 1.0);
    }
    
    return vec3(r, g, b);
}

// ===========================================================
//  ACCRETION DISK
// ===========================================================

vec3 disk(float r, float ph, float a, float b_impact) {
    float ri = u_isco;
    if (r < ri * 0.85 || r > RDISK) return vec3(0);
    
    // --- Novikov-Thorne temperature profile ---
    float x = r / ri;
    float T_base = 8000.0 * u_temp;  // Base temperature at ISCO, scaled by user control
    float T_emit = T_base * pow(x, -0.75);
    
    // --- Intensity: Stefan-Boltzmann I ∝ T^4 / r ---
    float I = pow(T_emit / T_base, 4.0) / (r * 0.3);
    
    // --- Edge smoothing ---
    I *= smoothstep(ri * 0.85, ri * 1.3, r);
    I *= smoothstep(RDISK, RDISK * 0.55, r);
    
    // --- Full GR redshift factor for Kerr-Newman ---
    // Metric components at equator (theta = pi/2, so Sigma = r^2)
    float r2 = r * r;
    float a2 = a * a;
    float Q2 = u_Q * u_Q;
    float Delta = r2 - 2.0 * r + a2 + Q2;
    float w = 2.0 * r - Q2;  // = 2Mr - Q^2 with M=1
    
    // Covariant metric at equator (Sigma = r^2):
    // g_tt = -(1 - w/r^2)
    // g_tphi = -a * w / r^2
    // g_phiphi = (r^4 + a^2*r^2 + a^2*w) / r^2
    float gtt = -(1.0 - w / r2);
    float gtph = -a * w / r2;
    float gphph = (r2 * r2 + a2 * r2 + a2 * w) / r2;
    
    // Angular velocity of prograde circular orbit
    // From circular orbit condition: dg_tt/dr + 2*Omega*dg_tphi/dr + Omega^2*dg_phiphi/dr = 0
    float dgtt_dr = 2.0 * (Q2 - r) / (r2 * r);
    float dgtph_dr = 2.0 * a * (r - Q2) / (r2 * r);
    float dgphph_dr = 2.0 * r + a2 * (-2.0 / r2 + 2.0 * Q2 / (r2 * r));
    
    // Solve quadratic: dgphph*Omega^2 + 2*dgtph*Omega + dgtt = 0
    float disc = dgtph_dr * dgtph_dr - dgtt_dr * dgphph_dr;
    float Omega = (-dgtph_dr + sqrt(max(disc, 0.0))) / max(dgphph_dr, 1e-10);  // prograde
    
    // Four-velocity normalization: u^t = 1/sqrt(-(g_tt + 2*Omega*g_tphi + Omega^2*g_phiphi))
    float denom = -(gtt + 2.0 * Omega * gtph + Omega * Omega * gphph);
    float ut = 1.0 / sqrt(max(denom, 1e-10));
    
    // Redshift factor: g = 1 / (u^t * (1 - b * Omega))
    // where b is the photon impact parameter = -alpha * sin(theta_obs)
    float g = 1.0 / (ut * max(abs(1.0 - b_impact * Omega), 1e-6));
    // Preserve sign: if (1 - b*Omega) < 0, the photon is counter-rotating
    // but g should always be positive for physical photons reaching the observer
    g = abs(g);
    
    // Apply relativistic beaming: T_obs = g * T_emit, I_obs = g^4 * I_emit
    float T_obs = g * T_emit;
    float g4 = g * g * g * g;
    I *= g4;
    
    // --- Blackbody color from observed temperature ---
    vec3 col = blackbody(T_obs);
    
    // --- Turbulence texture (preserved from original) ---
    float tu = 0.65 + 0.35 * hash(vec2(r * 5.0, ph * 3.0));
    float tu2 = 0.8 + 0.2 * hash(vec2(r * 18.0, ph * 9.0));
    
    return col * I * tu * tu2 * 3.2;
}
"""
