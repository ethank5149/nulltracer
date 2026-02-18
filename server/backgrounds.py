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
//  ACCRETION DISK
// ===========================================================

vec3 disk(float r, float ph, float a) {
    float ri = u_isco;
    if (r < ri * 0.85 || r > RDISK) return vec3(0);
    float x = r / ri;
    float tp = pow(x, -0.75) * u_temp;
    float I = pow(tp, 4.0) / (r * 0.3);
    I *= smoothstep(ri*0.85, ri*1.3, r);
    I *= smoothstep(RDISK, RDISK*0.55, r);
    float vo = 1.0 / sqrt(r);
    float dop = 1.0 + 0.65 * vo * sin(ph);
    float boost = pow(max(dop, 0.1), 3.0);
    I *= boost;
    float t = clamp(tp * boost * 0.45, 0.0, 3.5);
    vec3 col;
    if      (t < 0.4) col = mix(vec3(0.25,0.03,0), vec3(0.85,0.15,0.01), t*2.5);
    else if (t < 0.9) col = mix(vec3(0.85,0.15,0.01), vec3(1,0.55,0.08), (t-0.4)*2.0);
    else if (t < 1.7) col = mix(vec3(1,0.55,0.08), vec3(1,0.92,0.6), (t-0.9)/0.8);
    else if (t < 2.5) col = mix(vec3(1,0.92,0.6), vec3(1,1,0.95), (t-1.7)/0.8);
    else col = vec3(1);
    float tu = 0.65 + 0.35 * hash(vec2(r*5.0, ph*3.0));
    float tu2 = 0.8 + 0.2 * hash(vec2(r*18.0, ph*9.0));
    return col * I * tu * tu2 * 3.2;
}
"""
