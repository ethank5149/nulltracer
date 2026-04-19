/* ============================================================
 *  BACKGROUNDS — Procedural background rendering (float32)
 *
 *  Originally ported from GLSL background rendering functions.
 *  All functions take a Cartesian direction vector (dx, dy, dz)
 *  and never use θ or φ directly — pole-safe by construction.
 *
 *  Requires geodesic_base.cu to be included first (for hash,
 *  cubeMap, cubeChecker, cubeGrid, faceColor, smoothstepf).
 * ============================================================ */

#ifndef BACKGROUNDS_CU
#define BACKGROUNDS_CU


/* ── Star field background ────────────────────────────────── */

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


/* ── Checker background ───────────────────────────────────── */

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


/* ── Color-map background ─────────────────────────────────── */

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


/* ── Skymap texture background (equirectangular) ──────────── */

/* Samples from an equirectangular float32 RGB texture using the
 * escaped ray's spherical direction. The texture is stored as
 * packed float RGB (3 floats per pixel, linear light), row-major,
 * with the first row at the top (θ=0, north pole).
 *
 * The Python side normalizes all input formats (JPEG, PNG, EXR)
 * to float32 linear light before uploading to GPU, so this
 * function needs no format-specific logic.
 *
 * Uses bilinear interpolation for smooth sampling.
 */
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


/* ── Background dispatcher ────────────────────────────────── */

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


#endif /* BACKGROUNDS_CU */
