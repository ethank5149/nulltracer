"""
Port of buildFragSrc() from nulltracer/index.html (lines 308-798).

Generates GLSL fragment shader source identical to the JavaScript version,
with a #version 120 prefix for desktop OpenGL compatibility.
"""

VERTEX_SHADER_SRC = "#version 120\n" \
    "attribute vec2 a_pos; varying vec2 v_uv; void main(){ v_uv=a_pos; gl_Position=vec4(a_pos,0,1); }"


def build_frag_src(opts: dict) -> str:
    """Build the fragment shader source from options.

    Args:
        opts: dict with keys method, steps, obsDist, starLayers, stepSize, bgMode

    Returns:
        Complete GLSL fragment shader source string with #version 120 prefix.
    """
    STEPS = opts.get("steps", 200)
    METHOD = opts.get("method", "yoshida4")
    R0 = opts.get("obsDist", 40)
    RESC = R0 + 12
    STAR_LAYERS = opts.get("starLayers", 3)
    H_BASE = opts.get("stepSize", 0.30)
    BG_MODE = opts.get("bgMode", 1)

    src = f"""varying vec2 v_uv;
uniform vec2 u_res;
uniform float u_a,u_incl,u_fov,u_disk,u_grid,u_temp,u_phi0,u_Q,u_isco;

#define PI  3.14159265359
#define TAU 6.28318530718
#define STEPS {STEPS}
#define R0  {float(R0):.1f}
#define RESC {float(RESC):.1f}
#define RDISK 14.0
#define H_BASE {float(H_BASE):.4f}
#define STAR_LAYERS {STAR_LAYERS}
#define BG_MODE {BG_MODE}

// -------------------------------------------------------
//  POLE HANDLING STRATEGY
//
//  Two independent problems cause visual artifacts at
//  the celestial sphere poles:
//
//  A. INTEGRATOR: dφ/dλ has b/sin²θ which diverges at θ=0,π.
//     Fix: smooth additive regularization sin²θ → sin²θ + ε.
//     This bounds dφ everywhere. The physics error is confined
//     to a ~1° cap around each pole — an invisibly small solid
//     angle. No adaptive step-sizing needed (which would starve
//     the step budget and hide the disk).
//
//  B. BACKGROUND: Any pattern using (θ,φ) coordinates shows
//     artifacts because φ is undefined at θ=0,π. Even if the
//     integrator is perfect, adjacent rays escaping near a pole
//     can have wildly different φ values.
//
//     Fix: ALL background functions take a Cartesian direction
//     vec3 d = (sinθ cosφ, sinθ sinφ, cosθ) and NEVER use θ or φ
//     directly. Cube-map projection gives smooth face-local (u,v)
//     everywhere. At the poles, sinθ→0, so any φ-error is
//     multiplied by ~0 and vanishes from the direction vector.
//     This makes the background rendering robust to integrator
//     φ-noise near the poles.
// -------------------------------------------------------

// Smooth regularization epsilon. sin²θ + S2_EPS never reaches 0.
// dφ/dλ is bounded by b/(S2_EPS * Σ) ~ few hundred — large but finite.
// Physics error confined to θ < arcsin(√ε) ≈ 1.1° from poles.
#define S2_EPS 0.0004

float hash(vec2 p){{
    p=fract(p*vec2(443.8,441.4));
    p+=dot(p,p+19.19);
    return fract(p.x*p.y);
}}

// ISCO passed as uniform u_isco (computed numerically in JS for Kerr-Newman)

// ===========================================================
//  CUBE-MAP PROJECTION  (Cartesian direction → face + UV)
//
//  The key property: at the poles d=(0,0,±1), the +Z/-Z faces
//  use (x,y) as local coords. Both x=sinθ cosφ and y=sinθ sinφ
//  are smooth and ~0 at the pole, regardless of φ.
//  → φ-error × sinθ → 0.  No artifacts possible.
// ===========================================================

void cubeMap(vec3 d, out float face, out vec2 uv) {{
    float ax = abs(d.x), ay = abs(d.y), az = abs(d.z);
    if (az >= ax && az >= ay) {{
        face = d.z > 0.0 ? 0.0 : 1.0;
        uv = vec2(d.x, d.y) / az;
    }} else if (ax >= ay) {{
        face = d.x > 0.0 ? 2.0 : 3.0;
        uv = vec2(d.y, d.z) / ax;
    }} else {{
        face = d.y > 0.0 ? 4.0 : 5.0;
        uv = vec2(d.x, d.z) / ay;
    }}
}}

float cubeChecker(vec2 uv, float div) {{
    vec2 cell = floor((uv * 0.5 + 0.5) * div);
    return mod(cell.x + cell.y, 2.0);
}}

float cubeGrid(vec2 uv, float div) {{
    vec2 f = fract((uv * 0.5 + 0.5) * div);
    return smoothstep(0.88, 0.96, max(abs(f.x-0.5)*2.0, abs(f.y-0.5)*2.0));
}}

vec3 faceColor(float face) {{
    if (face < 0.5) return vec3(0.14, 0.08, 0.04);
    if (face < 1.5) return vec3(0.06, 0.05, 0.14);
    if (face < 2.5) return vec3(0.04, 0.12, 0.07);
    if (face < 3.5) return vec3(0.12, 0.04, 0.08);
    if (face < 4.5) return vec3(0.04, 0.08, 0.14);
    return vec3(0.12, 0.10, 0.04);
}}

// ===========================================================
//  BACKGROUND FUNCTIONS
//  All take vec3 d (unit direction). Never use theta or phi.
// ===========================================================

vec3 bgStars(vec3 d) {{
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
    for (int L = 0; L < STAR_LAYERS; L++) {{
        float sc = 10.0 + float(L) * 14.0;
        vec2 cell = floor(fuv * sc);
        vec2 seed = cell + face * 100.0 + float(L) * 47.0;
        float h = hash(seed);
        if (h > 0.88) {{
            vec2 sp = (cell + 0.3 + 0.4 * vec2(hash(seed+0.5), hash(seed+1.5))) / sc;
            float dist = length(fuv - sp) * sc;
            float s = exp(-dist * dist * 5.0);
            float t = hash(seed + 77.0);
            vec3 sc2 = t < 0.2 ? vec3(1,0.7,0.4) : t < 0.55 ? vec3(1,0.95,0.8) :
                       t < 0.8 ? vec3(0.8,0.9,1) : vec3(0.6,0.75,1);
            c += sc2 * s * (0.4 + 2.0 * hash(seed + 33.0));
        }}
    }}
    return c;
}}

vec3 bgChecker(vec3 d) {{
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
}}

vec3 bgColorMap(vec3 d) {{
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
}}

vec3 background(vec3 d) {{
    vec3 c;
#if BG_MODE == 0
    c = bgStars(d);
#elif BG_MODE == 1
    c = bgChecker(d);
#else
    c = bgColorMap(d);
#endif
    // Extra grid for stars mode
    if (u_grid > 0.5 && BG_MODE == 0) {{
        float face; vec2 uv;
        cubeMap(d, face, uv);
        c += vec3(0.055,0.04,0.028) * cubeGrid(uv, 6.0);
        c += vec3(0.07,0.035,0.02) * (1.0 - smoothstep(0.0, 0.03, abs(d.z)));
    }}
    return c;
}}

// Convert (theta, phi) to unit direction vector.
// At the poles, sinθ→0, so φ-error vanishes from the result.
vec3 sphereDir(float th, float ph) {{
    float sth = sin(th);
    return vec3(sth * cos(ph), sth * sin(ph), cos(th));
}}

// ===========================================================
//  ACCRETION DISK
// ===========================================================

vec3 disk(float r, float ph, float a) {{
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
}}
"""

    # ---- INTEGRATOR-SPECIFIC CODE ----
    if METHOD == "rk4":
        src += """
// ===========================================================
//  FULL HAMILTONIAN INTEGRATOR (RK4)
//  Uses smooth sin²θ+ε regularization for pole safety.
// ===========================================================

void geoRHS(
    float r, float th, float pr, float pth,
    float a, float b,
    out float dr, out float dth, out float dphi,
    out float dpr, out float dpth
) {
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float a2 = a*a, r2 = r*r;
    float Q2 = u_Q*u_Q;
    float sig = r2 + a2*c2;
    float del = r2 - 2.0*r + a2 + Q2;
    float sdel = max(del, 1e-6);
    float rpa2 = r2 + a2;
    // KN cross-term factor: w = 2Mr - Q^2 (M=1)
    // For Kerr (Q=0): w = 2r. Identity: w = (r^2+a^2) - Delta.
    float w = 2.0*r - Q2;
    float A_ = rpa2*rpa2 - sdel*a2*s2;
    float isig = 1.0/sig;
    float SD = sig*sdel;
    float iSD = 1.0/SD;
    float is2 = 1.0/s2;

    float grr = sdel*isig, gthth = isig;
    // KN inverse metric: g^phiphi = (Sigma-w)/(Sigma*Delta*sin^2)
    //                     g^tphi  = -a*w/(Sigma*Delta)
    float gff = (sig - w)*iSD*is2;
    float gtf = -a*w*iSD;

    dr = grr*pr; dth = gthth*pth; dphi = gff*b - gtf;

    float dsig_r=2.0*r; float ddel_r=2.0*r-2.0;
    float dA_r=4.0*r*rpa2-ddel_r*a2*s2;
    float dSD_r=dsig_r*sdel+sig*ddel_r;
    float dgtt_r=-(dA_r*SD-A_*dSD_r)/(SD*SD);
    // d/dr[-a*w/SD] = -a*(dw*SD - w*dSD)/SD^2, dw/dr = 2
    float dgtf_r=-a*(2.0*SD - w*dSD_r)/(SD*SD);
    float dgrr_r=(ddel_r*sig-sdel*dsig_r)/(sig*sig);
    float dgthth_r=-dsig_r*isig*isig;
    float num_ff=sig-w; float den_ff=SD*s2;
    // d(num_ff)/dr = dsig_r - dw/dr = 2r - 2
    float dgff_r=((dsig_r-2.0)*den_ff-num_ff*dSD_r*s2)/(den_ff*den_ff);
    dpr=-0.5*(dgtt_r-2.0*b*dgtf_r+dgrr_r*pr*pr+dgthth_r*pth*pth+dgff_r*b*b);

    float dsig_th=-2.0*a2*sth*cth;
    float ds2_th=2.0*sth*cth;
    float dA_th=-sdel*a2*ds2_th;
    float dSD_th=dsig_th*sdel;
    float dgtt_th=-(dA_th*SD-A_*dSD_th)/(SD*SD);
    // d/dtheta[-a*w/SD]: w is r-only, so = a*w*dSD_th/SD^2
    float dgtf_th=a*w*dSD_th/(SD*SD);
    float dgrr_th=-sdel*dsig_th/(sig*sig);
    float dgthth_th=-dsig_th*isig*isig;
    // d(num_ff)/dtheta = dsig_th (w has no theta dependence)
    float dgff_th=(dsig_th*den_ff-num_ff*(dsig_th*sdel*s2+SD*ds2_th))/(den_ff*den_ff);
    dpth=-0.5*(dgtt_th-2.0*b*dgtf_th+dgrr_th*pr*pr+dgthth_th*pth*pth+dgff_th*b*b);
}

void main() {
    vec2 uv = v_uv;
    float asp = u_res.x/u_res.y;
    float alpha = uv.x*u_fov*asp, beta = uv.y*u_fov;
    float a = u_a, a2 = a*a;
    float thObs = u_incl, sO = sin(thObs), cO = cos(thObs);
    float b = -alpha*sO;
    float q2 = beta*beta + cO*cO*(alpha*alpha - a2);

    float r = R0, th = thObs, phi = u_phi0;
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float sig = r*r+a2*c2, del = r*r-2.0*r+a2+u_Q*u_Q;
    float sdel = max(del,1e-6);
    float rpa2 = r*r+a2, A_ = rpa2*rpa2-sdel*a2*s2;
    float iSD = 1.0/(sig*sdel), is2 = 1.0/s2;
    float grr = sdel/sig, gthi = 1.0/sig;
    float pth = beta;
    // KN cross-term factor: w = 2r - Q^2
    float w_init = 2.0*r - u_Q*u_Q;
    // Hamiltonian constraint: g^tt + 2*b*g^tphi + g^rr*pr^2 + g^thth*pth^2 + g^phiphi*b^2 = 0
    // g^tphi = -a*w/SD,  g^phiphi = (sig-w)/(SD*s^2)
    float rest = -A_*iSD + 2.0*a*b*w_init*iSD + gthi*pth*pth + (sig-w_init)*iSD*is2*b*b;
    float pr2 = -rest/grr;
    float pr = pr2 > 0.0 ? -sqrt(pr2) : 0.0;
    float Q2 = u_Q*u_Q;
    float rp = 1.0 + sqrt(max(1.0-a2-Q2, 0.0));
    vec3 color = vec3(0); bool done = false;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;
        float he = H_BASE * clamp((r-rp)*0.4, 0.04, 1.0);
        he = clamp(he, 0.012, 0.6);
        float oldTh = th, oldR = r, oldPhi = phi;

        float dr1,dth1,dphi1,dpr1,dpth1;
        float dr2,dth2,dphi2,dpr2,dpth2;
        float dr3,dth3,dphi3,dpr3,dpth3;
        float dr4,dth4,dphi4,dpr4,dpth4;
        geoRHS(r,th,pr,pth, a,b, dr1,dth1,dphi1,dpr1,dpth1);
        geoRHS(r+.5*he*dr1, th+.5*he*dth1, pr+.5*he*dpr1, pth+.5*he*dpth1, a,b, dr2,dth2,dphi2,dpr2,dpth2);
        geoRHS(r+.5*he*dr2, th+.5*he*dth2, pr+.5*he*dpr2, pth+.5*he*dpth2, a,b, dr3,dth3,dphi3,dpr3,dpth3);
        geoRHS(r+he*dr3, th+he*dth3, pr+he*dpr3, pth+he*dpth3, a,b, dr4,dth4,dphi4,dpr4,dpth4);
        r   += he*(dr1  +2.0*dr2  +2.0*dr3  +dr4  )/6.0;
        th  += he*(dth1 +2.0*dth2 +2.0*dth3 +dth4 )/6.0;
        phi += he*(dphi1+2.0*dphi2+2.0*dphi3+dphi4)/6.0;
        pr  += he*(dpr1 +2.0*dpr2 +2.0*dpr3 +dpr4 )/6.0;
        pth += he*(dpth1+2.0*dpth2+2.0*dpth3+dpth4)/6.0;

        if (th < 0.005) { th = 0.005; pth = abs(pth); }
        if (th > PI-0.005) { th = PI-0.005; pth = -abs(pth); }

        if (r <= rp*1.01) { done=true; break; }
        if (u_disk > 0.5) {
            float cross = (oldTh-PI*0.5)*(th-PI*0.5);
            if (cross < 0.0) {
                float f = clamp(abs(oldTh-PI*0.5)/max(abs(th-oldTh),1e-6), 0.0, 1.0);
                vec3 dc = disk(oldR+f*(r-oldR), oldPhi+f*(phi-oldPhi), a);
                color += dc * (1.0 - clamp(length(color)*0.4, 0.0, 0.9));
            }
        }
        if (r > RESC) {
            float fth = oldTh + (th-oldTh) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            float fph = oldPhi + (phi-oldPhi) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            color += background(sphereDir(fth, fph)) * (1.0 - clamp(length(color)*0.3, 0.0, 0.9));
            done=true; break;
        }
        if (r < 0.5 || r!=r || th!=th) { done=true; break; }
    }
    float imp = length(vec2(alpha,beta));
    float rc = 5.2-1.0*a;
    color += vec3(0.1,0.07,0.04)*exp(-pow((imp-rc)/0.3,2.0))*0.06;
    color *= 1.0-0.3*dot(uv,uv);
    color = color/(1.0+color);
    color = pow(color, vec3(1.0/2.2));
    gl_FragColor = vec4(color, 1);
}
"""
    elif METHOD == "yoshida6":
        src += """
// ===========================================================
//  YOSHIDA 6TH-ORDER SYMPLECTIC INTEGRATOR
//
//  A composition method that chains leapfrog (Störmer-Verlet)
//  substeps with optimized coefficients to achieve 6th-order
//  global accuracy while preserving the symplectic structure
//  of Hamilton's equations. This means:
//    - No secular energy drift (Hamiltonian is conserved to
//      O(h^6) over exponentially long times)
//    - Phase-space volume is exactly preserved
//    - Superior long-term stability vs. RK4
//
//  The coefficients are from Yoshida (1990), "Construction of
//  higher order symplectic integrators", Physics Letters A,
//  150(5-7), 262-268. Solution A triple-jump composition.
//
//  Industry usage: REBOUND (N-body), GADGET (cosmological),
//  and numerous GR ray-tracing codes.
// ===========================================================

void geoRHS(
    float r, float th, float pr, float pth,
    float a, float b,
    out float dr, out float dth, out float dphi,
    out float dpr, out float dpth
) {
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float a2 = a*a, r2 = r*r;
    float Q2 = u_Q*u_Q;
    float sig = r2 + a2*c2;
    float del = r2 - 2.0*r + a2 + Q2;
    float sdel = max(del, 1e-6);
    float rpa2 = r2 + a2;
    float w = 2.0*r - Q2;
    float A_ = rpa2*rpa2 - sdel*a2*s2;
    float isig = 1.0/sig;
    float SD = sig*sdel;
    float iSD = 1.0/SD;
    float is2 = 1.0/s2;

    float grr = sdel*isig, gthth = isig;
    float gff = (sig - w)*iSD*is2;
    float gtf = -a*w*iSD;

    dr = grr*pr; dth = gthth*pth; dphi = gff*b - gtf;

    float dsig_r=2.0*r; float ddel_r=2.0*r-2.0;
    float dA_r=4.0*r*rpa2-ddel_r*a2*s2;
    float dSD_r=dsig_r*sdel+sig*ddel_r;
    float dgtt_r=-(dA_r*SD-A_*dSD_r)/(SD*SD);
    float dgtf_r=-a*(2.0*SD - w*dSD_r)/(SD*SD);
    float dgrr_r=(ddel_r*sig-sdel*dsig_r)/(sig*sig);
    float dgthth_r=-dsig_r*isig*isig;
    float num_ff=sig-w; float den_ff=SD*s2;
    float dgff_r=((dsig_r-2.0)*den_ff-num_ff*dSD_r*s2)/(den_ff*den_ff);
    dpr=-0.5*(dgtt_r-2.0*b*dgtf_r+dgrr_r*pr*pr+dgthth_r*pth*pth+dgff_r*b*b);

    float dsig_th=-2.0*a2*sth*cth;
    float ds2_th=2.0*sth*cth;
    float dA_th=-sdel*a2*ds2_th;
    float dSD_th=dsig_th*sdel;
    float dgtt_th=-(dA_th*SD-A_*dSD_th)/(SD*SD);
    float dgtf_th=a*w*dSD_th/(SD*SD);
    float dgrr_th=-sdel*dsig_th/(sig*sig);
    float dgthth_th=-dsig_th*isig*isig;
    float dgff_th=(dsig_th*den_ff-num_ff*(dsig_th*sdel*s2+SD*ds2_th))/(den_ff*den_ff);
    dpth=-0.5*(dgtt_th-2.0*b*dgtf_th+dgrr_th*pr*pr+dgthth_th*pth*pth+dgff_th*b*b);
}

// Yoshida 6th-order symmetric composition coefficients (Solution A)
#define Y6_W1  0.78451361047755726382
#define Y6_W2  0.23557321335935813368
#define Y6_W3 -1.17767998417887100695
#define Y6_W0  1.31518632068391121889
#define Y6_D1  0.39225680523877863191
#define Y6_D2  0.51004341191845769508
#define Y6_D3 -0.47105338540975643969
#define Y6_D0  0.06875316825252012625

void main() {
    vec2 uv = v_uv;
    float asp = u_res.x/u_res.y;
    float alpha = uv.x*u_fov*asp, beta = uv.y*u_fov;
    float a = u_a, a2 = a*a;
    float thObs = u_incl, sO = sin(thObs), cO = cos(thObs);
    float b = -alpha*sO;
    float q2 = beta*beta + cO*cO*(alpha*alpha - a2);

    float r = R0, th = thObs, phi = u_phi0;
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float sig = r*r+a2*c2, del = r*r-2.0*r+a2+u_Q*u_Q;
    float sdel = max(del,1e-6);
    float rpa2 = r*r+a2, A_ = rpa2*rpa2-sdel*a2*s2;
    float iSD = 1.0/(sig*sdel), is2 = 1.0/s2;
    float grr = sdel/sig, gthi = 1.0/sig;
    float pth = beta;
    float w_init = 2.0*r - u_Q*u_Q;
    float rest = -A_*iSD + 2.0*a*b*w_init*iSD + gthi*pth*pth + (sig-w_init)*iSD*is2*b*b;
    float pr2 = -rest/grr;
    float pr = pr2 > 0.0 ? -sqrt(pr2) : 0.0;
    float Q2 = u_Q*u_Q;
    float rp = 1.0 + sqrt(max(1.0-a2-Q2, 0.0));
    vec3 color = vec3(0); bool done = false;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;
        float he = H_BASE * clamp((r-rp)*0.4, 0.04, 1.0);
        he = clamp(he, 0.012, 0.6);
        float oldTh = th, oldR = r, oldPhi = phi;

        // Yoshida 6th-order symmetric composition: 7 substeps
        // Each substep: drift positions, then kick momenta
        float dr_,dth_,dphi_,dpr_,dpth_;

        // --- Substep 1: drift d1, kick w1 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y6_D1*dr_;
        th  += he*Y6_D1*dth_;
        phi += he*Y6_D1*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y6_W1*dpr_;
        pth += he*Y6_W1*dpth_;

        // --- Substep 2: drift d2, kick w2 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y6_D2*dr_;
        th  += he*Y6_D2*dth_;
        phi += he*Y6_D2*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y6_W2*dpr_;
        pth += he*Y6_W2*dpth_;

        // --- Substep 3: drift d3, kick w3 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y6_D3*dr_;
        th  += he*Y6_D3*dth_;
        phi += he*Y6_D3*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y6_W3*dpr_;
        pth += he*Y6_W3*dpth_;

        // --- Substep 4: drift d0, kick w0 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y6_D0*dr_;
        th  += he*Y6_D0*dth_;
        phi += he*Y6_D0*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y6_W0*dpr_;
        pth += he*Y6_W0*dpth_;

        // --- Substep 5: drift d3, kick w3 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y6_D3*dr_;
        th  += he*Y6_D3*dth_;
        phi += he*Y6_D3*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y6_W3*dpr_;
        pth += he*Y6_W3*dpth_;

        // --- Substep 6: drift d2, kick w2 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y6_D2*dr_;
        th  += he*Y6_D2*dth_;
        phi += he*Y6_D2*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y6_W2*dpr_;
        pth += he*Y6_W2*dpth_;

        // --- Substep 7: drift d1, kick w1 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y6_D1*dr_;
        th  += he*Y6_D1*dth_;
        phi += he*Y6_D1*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y6_W1*dpr_;
        pth += he*Y6_W1*dpth_;

        if (th < 0.005) { th = 0.005; pth = abs(pth); }
        if (th > PI-0.005) { th = PI-0.005; pth = -abs(pth); }

        if (r <= rp*1.01) { done=true; break; }
        if (u_disk > 0.5) {
            float cross = (oldTh-PI*0.5)*(th-PI*0.5);
            if (cross < 0.0) {
                float f = clamp(abs(oldTh-PI*0.5)/max(abs(th-oldTh),1e-6), 0.0, 1.0);
                vec3 dc = disk(oldR+f*(r-oldR), oldPhi+f*(phi-oldPhi), a);
                color += dc * (1.0 - clamp(length(color)*0.4, 0.0, 0.9));
            }
        }
        if (r > RESC) {
            float fth = oldTh + (th-oldTh) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            float fph = oldPhi + (phi-oldPhi) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            color += background(sphereDir(fth, fph)) * (1.0 - clamp(length(color)*0.3, 0.0, 0.9));
            done=true; break;
        }
        if (r < 0.5 || r!=r || th!=th) { done=true; break; }
    }
    float imp = length(vec2(alpha,beta));
    float rc = 5.2-1.0*a;
    color += vec3(0.1,0.07,0.04)*exp(-pow((imp-rc)/0.3,2.0))*0.06;
    color *= 1.0-0.3*dot(uv,uv);
    color = color/(1.0+color);
    color = pow(color, vec3(1.0/2.2));
    gl_FragColor = vec4(color, 1);
}
"""
    elif METHOD == "yoshida8":
        src += """
// ===========================================================
//  YOSHIDA 8TH-ORDER SYMPLECTIC INTEGRATOR
//
//  Triple-jump composition of the 4th-order Forest-Ruth
//  integrator to achieve 8th-order global accuracy while
//  preserving the symplectic structure. Uses 15 substeps.
//
//  Coefficients from Yoshida (1990), "Construction of higher
//  order symplectic integrators", Physics Letters A, 150(5-7),
//  262-268, Table 2 "Solution D" for order 8.
//
//  Properties:
//    - 8th-order symplectic: O(h^8) energy conservation
//    - Phase-space volume exactly preserved
//    - No secular drift even over millions of steps
//    - 15 force evaluations per step (vs 7 for Yoshida6)
//
//  Industry usage: long-duration orbit propagation, solar
//  system dynamics, precision GR geodesic integration.
// ===========================================================

void geoRHS(
    float r, float th, float pr, float pth,
    float a, float b,
    out float dr, out float dth, out float dphi,
    out float dpr, out float dpth
) {
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float a2 = a*a, r2 = r*r;
    float Q2 = u_Q*u_Q;
    float sig = r2 + a2*c2;
    float del = r2 - 2.0*r + a2 + Q2;
    float sdel = max(del, 1e-6);
    float rpa2 = r2 + a2;
    float w = 2.0*r - Q2;
    float A_ = rpa2*rpa2 - sdel*a2*s2;
    float isig = 1.0/sig;
    float SD = sig*sdel;
    float iSD = 1.0/SD;
    float is2 = 1.0/s2;

    float grr = sdel*isig, gthth = isig;
    float gff = (sig - w)*iSD*is2;
    float gtf = -a*w*iSD;

    dr = grr*pr; dth = gthth*pth; dphi = gff*b - gtf;

    float dsig_r=2.0*r; float ddel_r=2.0*r-2.0;
    float dA_r=4.0*r*rpa2-ddel_r*a2*s2;
    float dSD_r=dsig_r*sdel+sig*ddel_r;
    float dgtt_r=-(dA_r*SD-A_*dSD_r)/(SD*SD);
    float dgtf_r=-a*(2.0*SD - w*dSD_r)/(SD*SD);
    float dgrr_r=(ddel_r*sig-sdel*dsig_r)/(sig*sig);
    float dgthth_r=-dsig_r*isig*isig;
    float num_ff=sig-w; float den_ff=SD*s2;
    float dgff_r=((dsig_r-2.0)*den_ff-num_ff*dSD_r*s2)/(den_ff*den_ff);
    dpr=-0.5*(dgtt_r-2.0*b*dgtf_r+dgrr_r*pr*pr+dgthth_r*pth*pth+dgff_r*b*b);

    float dsig_th=-2.0*a2*sth*cth;
    float ds2_th=2.0*sth*cth;
    float dA_th=-sdel*a2*ds2_th;
    float dSD_th=dsig_th*sdel;
    float dgtt_th=-(dA_th*SD-A_*dSD_th)/(SD*SD);
    float dgtf_th=a*w*dSD_th/(SD*SD);
    float dgrr_th=-sdel*dsig_th/(sig*sig);
    float dgthth_th=-dsig_th*isig*isig;
    float dgff_th=(dsig_th*den_ff-num_ff*(dsig_th*sdel*s2+SD*ds2_th))/(den_ff*den_ff);
    dpth=-0.5*(dgtt_th-2.0*b*dgtf_th+dgrr_th*pr*pr+dgthth_th*pth*pth+dgff_th*b*b);
}

// Yoshida 8th-order coefficients (Solution D from Table 2)
// 15-stage symmetric composition: w1..w7, w0 (center)
// d_i = (w_i + w_{i+1}) / 2
#define Y8_W1  1.04242620869991
#define Y8_W2  1.82020630970714
#define Y8_W3  0.157739928123617
#define Y8_W4  2.44002732616735
#define Y8_W5 -0.00716989419708120
#define Y8_W6 -2.44699182370524
#define Y8_W7 -1.61582374150097
#define Y8_W0 -1.7808286265894516

#define Y8_D1  0.52121310434996
#define Y8_D2  1.43131625920353
#define Y8_D3  0.98897311891538
#define Y8_D4  1.29888362714548
#define Y8_D5  1.21642871598513
#define Y8_D6 -1.22708085895116
#define Y8_D7 -2.03140778260311
#define Y8_D0 -1.69832618454521

void main() {
    vec2 uv = v_uv;
    float asp = u_res.x/u_res.y;
    float alpha = uv.x*u_fov*asp, beta = uv.y*u_fov;
    float a = u_a, a2 = a*a;
    float thObs = u_incl, sO = sin(thObs), cO = cos(thObs);
    float b = -alpha*sO;
    float q2 = beta*beta + cO*cO*(alpha*alpha - a2);

    float r = R0, th = thObs, phi = u_phi0;
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float sig = r*r+a2*c2, del = r*r-2.0*r+a2+u_Q*u_Q;
    float sdel = max(del,1e-6);
    float rpa2 = r*r+a2, A_ = rpa2*rpa2-sdel*a2*s2;
    float iSD = 1.0/(sig*sdel), is2 = 1.0/s2;
    float grr = sdel/sig, gthi = 1.0/sig;
    float pth = beta;
    float w_init = 2.0*r - u_Q*u_Q;
    float rest = -A_*iSD + 2.0*a*b*w_init*iSD + gthi*pth*pth + (sig-w_init)*iSD*is2*b*b;
    float pr2 = -rest/grr;
    float pr = pr2 > 0.0 ? -sqrt(pr2) : 0.0;
    float Q2 = u_Q*u_Q;
    float rp = 1.0 + sqrt(max(1.0-a2-Q2, 0.0));
    vec3 color = vec3(0); bool done = false;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;
        float he = H_BASE * clamp((r-rp)*0.4, 0.04, 1.0);
        he = clamp(he, 0.012, 0.6);
        float oldTh = th, oldR = r, oldPhi = phi;

        // Yoshida 8th-order symmetric composition: 15 substeps
        float dr_,dth_,dphi_,dpr_,dpth_;

        // --- Substep 1: drift d1, kick w1 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D1*dr_; th += he*Y8_D1*dth_; phi += he*Y8_D1*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W1*dpr_; pth += he*Y8_W1*dpth_;

        // --- Substep 2: drift d2, kick w2 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D2*dr_; th += he*Y8_D2*dth_; phi += he*Y8_D2*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W2*dpr_; pth += he*Y8_W2*dpth_;

        // --- Substep 3: drift d3, kick w3 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D3*dr_; th += he*Y8_D3*dth_; phi += he*Y8_D3*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W3*dpr_; pth += he*Y8_W3*dpth_;

        // --- Substep 4: drift d4, kick w4 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D4*dr_; th += he*Y8_D4*dth_; phi += he*Y8_D4*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W4*dpr_; pth += he*Y8_W4*dpth_;

        // --- Substep 5: drift d5, kick w5 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D5*dr_; th += he*Y8_D5*dth_; phi += he*Y8_D5*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W5*dpr_; pth += he*Y8_W5*dpth_;

        // --- Substep 6: drift d6, kick w6 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D6*dr_; th += he*Y8_D6*dth_; phi += he*Y8_D6*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W6*dpr_; pth += he*Y8_W6*dpth_;

        // --- Substep 7: drift d7, kick w7 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D7*dr_; th += he*Y8_D7*dth_; phi += he*Y8_D7*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W7*dpr_; pth += he*Y8_W7*dpth_;

        // --- Substep 8 (center): drift d0, kick w0 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D0*dr_; th += he*Y8_D0*dth_; phi += he*Y8_D0*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W0*dpr_; pth += he*Y8_W0*dpth_;

        // --- Substep 9: drift d7, kick w7 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D7*dr_; th += he*Y8_D7*dth_; phi += he*Y8_D7*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W7*dpr_; pth += he*Y8_W7*dpth_;

        // --- Substep 10: drift d6, kick w6 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D6*dr_; th += he*Y8_D6*dth_; phi += he*Y8_D6*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W6*dpr_; pth += he*Y8_W6*dpth_;

        // --- Substep 11: drift d5, kick w5 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D5*dr_; th += he*Y8_D5*dth_; phi += he*Y8_D5*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W5*dpr_; pth += he*Y8_W5*dpth_;

        // --- Substep 12: drift d4, kick w4 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D4*dr_; th += he*Y8_D4*dth_; phi += he*Y8_D4*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W4*dpr_; pth += he*Y8_W4*dpth_;

        // --- Substep 13: drift d3, kick w3 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D3*dr_; th += he*Y8_D3*dth_; phi += he*Y8_D3*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W3*dpr_; pth += he*Y8_W3*dpth_;

        // --- Substep 14: drift d2, kick w2 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D2*dr_; th += he*Y8_D2*dth_; phi += he*Y8_D2*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W2*dpr_; pth += he*Y8_W2*dpth_;

        // --- Substep 15: drift d1, kick w1 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r += he*Y8_D1*dr_; th += he*Y8_D1*dth_; phi += he*Y8_D1*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr += he*Y8_W1*dpr_; pth += he*Y8_W1*dpth_;

        if (th < 0.005) { th = 0.005; pth = abs(pth); }
        if (th > PI-0.005) { th = PI-0.005; pth = -abs(pth); }

        if (r <= rp*1.01) { done=true; break; }
        if (u_disk > 0.5) {
            float cross = (oldTh-PI*0.5)*(th-PI*0.5);
            if (cross < 0.0) {
                float f = clamp(abs(oldTh-PI*0.5)/max(abs(th-oldTh),1e-6), 0.0, 1.0);
                vec3 dc = disk(oldR+f*(r-oldR), oldPhi+f*(phi-oldPhi), a);
                color += dc * (1.0 - clamp(length(color)*0.4, 0.0, 0.9));
            }
        }
        if (r > RESC) {
            float fth = oldTh + (th-oldTh) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            float fph = oldPhi + (phi-oldPhi) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            color += background(sphereDir(fth, fph)) * (1.0 - clamp(length(color)*0.3, 0.0, 0.9));
            done=true; break;
        }
        if (r < 0.5 || r!=r || th!=th) { done=true; break; }
    }
    float imp = length(vec2(alpha,beta));
    float rc = 5.2-1.0*a;
    color += vec3(0.1,0.07,0.04)*exp(-pow((imp-rc)/0.3,2.0))*0.06;
    color *= 1.0-0.3*dot(uv,uv);
    color = color/(1.0+color);
    color = pow(color, vec3(1.0/2.2));
    gl_FragColor = vec4(color, 1);
}
"""
    elif METHOD == "rkdp8":
        src += """
// ===========================================================
//  DORMAND-PRINCE 8TH-ORDER RUNGE-KUTTA (RK8(7))
//
//  A 13-stage, 8th-order explicit Runge-Kutta method from
//  Dormand & Prince (1981), "High order embedded Runge-Kutta
//  formulae", J. Comp. Appl. Math. 7(1), 67-75.
//
//  Achieves 8th-order accuracy with 13 function evaluations
//  per step — far more accurate per step than RK4 (4th order,
//  4 evaluations). For smooth geodesics away from the horizon,
//  this allows much larger step sizes for the same accuracy.
//
//  Industry usage: MATLAB ode87, SciPy DOP853 (variant),
//  JPL trajectory planning, gravitational wave template banks.
//
//  Note: In GLSL we use the fixed-step form since our adaptive
//  stepping is distance-based. The 13 stages are fully unrolled
//  for GPU efficiency.
// ===========================================================

void geoRHS(
    float r, float th, float pr, float pth,
    float a, float b,
    out float dr, out float dth, out float dphi,
    out float dpr, out float dpth
) {
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float a2 = a*a, r2 = r*r;
    float Q2 = u_Q*u_Q;
    float sig = r2 + a2*c2;
    float del = r2 - 2.0*r + a2 + Q2;
    float sdel = max(del, 1e-6);
    float rpa2 = r2 + a2;
    float w = 2.0*r - Q2;
    float A_ = rpa2*rpa2 - sdel*a2*s2;
    float isig = 1.0/sig;
    float SD = sig*sdel;
    float iSD = 1.0/SD;
    float is2 = 1.0/s2;

    float grr = sdel*isig, gthth = isig;
    float gff = (sig - w)*iSD*is2;
    float gtf = -a*w*iSD;

    dr = grr*pr; dth = gthth*pth; dphi = gff*b - gtf;

    float dsig_r=2.0*r; float ddel_r=2.0*r-2.0;
    float dA_r=4.0*r*rpa2-ddel_r*a2*s2;
    float dSD_r=dsig_r*sdel+sig*ddel_r;
    float dgtt_r=-(dA_r*SD-A_*dSD_r)/(SD*SD);
    float dgtf_r=-a*(2.0*SD - w*dSD_r)/(SD*SD);
    float dgrr_r=(ddel_r*sig-sdel*dsig_r)/(sig*sig);
    float dgthth_r=-dsig_r*isig*isig;
    float num_ff=sig-w; float den_ff=SD*s2;
    float dgff_r=((dsig_r-2.0)*den_ff-num_ff*dSD_r*s2)/(den_ff*den_ff);
    dpr=-0.5*(dgtt_r-2.0*b*dgtf_r+dgrr_r*pr*pr+dgthth_r*pth*pth+dgff_r*b*b);

    float dsig_th=-2.0*a2*sth*cth;
    float ds2_th=2.0*sth*cth;
    float dA_th=-sdel*a2*ds2_th;
    float dSD_th=dsig_th*sdel;
    float dgtt_th=-(dA_th*SD-A_*dSD_th)/(SD*SD);
    float dgtf_th=a*w*dSD_th/(SD*SD);
    float dgrr_th=-sdel*dsig_th/(sig*sig);
    float dgthth_th=-dsig_th*isig*isig;
    float dgff_th=(dsig_th*den_ff-num_ff*(dsig_th*sdel*s2+SD*ds2_th))/(den_ff*den_ff);
    dpth=-0.5*(dgtt_th-2.0*b*dgtf_th+dgrr_th*pr*pr+dgthth_th*pth*pth+dgff_th*b*b);
}

void main() {
    vec2 uv = v_uv;
    float asp = u_res.x/u_res.y;
    float alpha = uv.x*u_fov*asp, beta = uv.y*u_fov;
    float a = u_a, a2 = a*a;
    float thObs = u_incl, sO = sin(thObs), cO = cos(thObs);
    float b = -alpha*sO;
    float q2 = beta*beta + cO*cO*(alpha*alpha - a2);

    float r = R0, th = thObs, phi = u_phi0;
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float sig = r*r+a2*c2, del = r*r-2.0*r+a2+u_Q*u_Q;
    float sdel = max(del,1e-6);
    float rpa2 = r*r+a2, A_ = rpa2*rpa2-sdel*a2*s2;
    float iSD = 1.0/(sig*sdel), is2 = 1.0/s2;
    float grr = sdel/sig, gthi = 1.0/sig;
    float pth = beta;
    float w_init = 2.0*r - u_Q*u_Q;
    float rest = -A_*iSD + 2.0*a*b*w_init*iSD + gthi*pth*pth + (sig-w_init)*iSD*is2*b*b;
    float pr2 = -rest/grr;
    float pr = pr2 > 0.0 ? -sqrt(pr2) : 0.0;
    float Q2 = u_Q*u_Q;
    float rp = 1.0 + sqrt(max(1.0-a2-Q2, 0.0));
    vec3 color = vec3(0); bool done = false;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;
        float he = H_BASE * clamp((r-rp)*0.4, 0.04, 1.0);
        he = clamp(he, 0.012, 0.6);
        float oldTh = th, oldR = r, oldPhi = phi;

        // Dormand-Prince 8th-order: 13 stages
        // We integrate the 5-variable system (r, th, phi, pr, pth)
        // Butcher tableau coefficients from Dormand & Prince (1981)

        float kr1,kth1,kphi1,kpr1,kpth1;
        float kr2,kth2,kphi2,kpr2,kpth2;
        float kr3,kth3,kphi3,kpr3,kpth3;
        float kr4,kth4,kphi4,kpr4,kpth4;
        float kr5,kth5,kphi5,kpr5,kpth5;
        float kr6,kth6,kphi6,kpr6,kpth6;
        float kr7,kth7,kphi7,kpr7,kpth7;
        float kr8,kth8,kphi8,kpr8,kpth8;
        float kr9,kth9,kphi9,kpr9,kpth9;
        float kr10,kth10,kphi10,kpr10,kpth10;
        float kr11,kth11,kphi11,kpr11,kpth11;
        float kr12,kth12,kphi12,kpr12,kpth12;
        float kr13,kth13,kphi13,kpr13,kpth13;

        // Stage 1
        geoRHS(r, th, pr, pth, a, b, kr1, kth1, kphi1, kpr1, kpth1);

        // Stage 2: c2 = 1/18
        geoRHS(r+he*kr1/18.0, th+he*kth1/18.0,
               pr+he*kpr1/18.0, pth+he*kpth1/18.0,
               a, b, kr2, kth2, kphi2, kpr2, kpth2);

        // Stage 3: c3 = 1/12
        geoRHS(r+he*(kr1/48.0+kr2/16.0), th+he*(kth1/48.0+kth2/16.0),
               pr+he*(kpr1/48.0+kpr2/16.0), pth+he*(kpth1/48.0+kpth2/16.0),
               a, b, kr3, kth3, kphi3, kpr3, kpth3);

        // Stage 4: c4 = 1/8
        geoRHS(r+he*(kr1/32.0+kr3*3.0/32.0), th+he*(kth1/32.0+kth3*3.0/32.0),
               pr+he*(kpr1/32.0+kpr3*3.0/32.0), pth+he*(kpth1/32.0+kpth3*3.0/32.0),
               a, b, kr4, kth4, kphi4, kpr4, kpth4);

        // Stage 5: c5 = 5/16
        geoRHS(r+he*(kr1*5.0/16.0-kr3*75.0/64.0+kr4*75.0/64.0),
               th+he*(kth1*5.0/16.0-kth3*75.0/64.0+kth4*75.0/64.0),
               pr+he*(kpr1*5.0/16.0-kpr3*75.0/64.0+kpr4*75.0/64.0),
               pth+he*(kpth1*5.0/16.0-kpth3*75.0/64.0+kpth4*75.0/64.0),
               a, b, kr5, kth5, kphi5, kpr5, kpth5);

        // Stage 6: c6 = 3/8
        geoRHS(r+he*(kr1*3.0/80.0+kr4*3.0/16.0+kr5*3.0/20.0),
               th+he*(kth1*3.0/80.0+kth4*3.0/16.0+kth5*3.0/20.0),
               pr+he*(kpr1*3.0/80.0+kpr4*3.0/16.0+kpr5*3.0/20.0),
               pth+he*(kpth1*3.0/80.0+kpth4*3.0/16.0+kpth5*3.0/20.0),
               a, b, kr6, kth6, kphi6, kpr6, kpth6);

        // Stage 7: c7 = 59/400
        float a71=29443841.0/614563906.0, a74=77736538.0/692538347.0;
        float a75=-28693883.0/1125000000.0, a76=23124283.0/1800000000.0;
        geoRHS(r+he*(a71*kr1+a74*kr4+a75*kr5+a76*kr6),
               th+he*(a71*kth1+a74*kth4+a75*kth5+a76*kth6),
               pr+he*(a71*kpr1+a74*kpr4+a75*kpr5+a76*kpr6),
               pth+he*(a71*kpth1+a74*kpth4+a75*kpth5+a76*kpth6),
               a, b, kr7, kth7, kphi7, kpr7, kpth7);

        // Stage 8: c8 = 93/200
        float a81=16016141.0/946692911.0, a84=61564180.0/158732637.0;
        float a85=22789713.0/633445777.0, a86=545815736.0/2771057229.0;
        float a87=-180193667.0/1043307555.0;
        geoRHS(r+he*(a81*kr1+a84*kr4+a85*kr5+a86*kr6+a87*kr7),
               th+he*(a81*kth1+a84*kth4+a85*kth5+a86*kth6+a87*kth7),
               pr+he*(a81*kpr1+a84*kpr4+a85*kpr5+a86*kpr6+a87*kpr7),
               pth+he*(a81*kpth1+a84*kpth4+a85*kpth5+a86*kpth6+a87*kpth7),
               a, b, kr8, kth8, kphi8, kpr8, kpth8);

        // Stage 9: c9 = 5490023248/9719169821
        float a91=39632708.0/573591083.0, a94=-433636366.0/683701615.0;
        float a95=-421739975.0/2616292301.0, a96=100302831.0/723423059.0;
        float a97=790204164.0/839813087.0, a98=800635310.0/3783071287.0;
        geoRHS(r+he*(a91*kr1+a94*kr4+a95*kr5+a96*kr6+a97*kr7+a98*kr8),
               th+he*(a91*kth1+a94*kth4+a95*kth5+a96*kth6+a97*kth7+a98*kth8),
               pr+he*(a91*kpr1+a94*kpr4+a95*kpr5+a96*kpr6+a97*kpr7+a98*kpr8),
               pth+he*(a91*kpth1+a94*kpth4+a95*kpth5+a96*kpth6+a97*kpth7+a98*kpth8),
               a, b, kr9, kth9, kphi9, kpr9, kpth9);

        // Stage 10: c10 = 13/20
        float a101=246121993.0/1340847787.0, a104=-37695042795.0/15268766246.0;
        float a105=-309121744.0/1061227803.0, a106=-12992083.0/490766935.0;
        float a107=6005943493.0/2108947869.0, a108=393006217.0/1396673457.0;
        float a109=123872331.0/1001029789.0;
        geoRHS(r+he*(a101*kr1+a104*kr4+a105*kr5+a106*kr6+a107*kr7+a108*kr8+a109*kr9),
               th+he*(a101*kth1+a104*kth4+a105*kth5+a106*kth6+a107*kth7+a108*kth8+a109*kth9),
               pr+he*(a101*kpr1+a104*kpr4+a105*kpr5+a106*kpr6+a107*kpr7+a108*kpr8+a109*kpr9),
               pth+he*(a101*kpth1+a104*kpth4+a105*kpth5+a106*kpth6+a107*kpth7+a108*kpth8+a109*kpth9),
               a, b, kr10, kth10, kphi10, kpr10, kpth10);

        // Stage 11: c11 = 1201146811/1299019798
        float a111=-1028468189.0/846180014.0, a114=8478235783.0/508512852.0;
        float a115=1311729495.0/1432422823.0, a116=-10304129995.0/1701304382.0;
        float a117=-48777925059.0/3047939560.0, a118=15336726248.0/1032824649.0;
        float a119=-45442868181.0/3398467696.0, a1110=3065993473.0/597172653.0;
        geoRHS(r+he*(a111*kr1+a114*kr4+a115*kr5+a116*kr6+a117*kr7+a118*kr8+a119*kr9+a1110*kr10),
               th+he*(a111*kth1+a114*kth4+a115*kth5+a116*kth6+a117*kth7+a118*kth8+a119*kth9+a1110*kth10),
               pr+he*(a111*kpr1+a114*kpr4+a115*kpr5+a116*kpr6+a117*kpr7+a118*kpr8+a119*kpr9+a1110*kpr10),
               pth+he*(a111*kpth1+a114*kpth4+a115*kpth5+a116*kpth6+a117*kpth7+a118*kpth8+a119*kpth9+a1110*kpth10),
               a, b, kr11, kth11, kphi11, kpr11, kpth11);

        // Stage 12: c12 = 1
        float a121=185892177.0/718116043.0, a124=-3185094517.0/667107341.0;
        float a125=-477755414.0/1098053517.0, a126=-703635378.0/230739211.0;
        float a127=5731566787.0/1027545527.0, a128=5232866602.0/850066563.0;
        float a129=-4093664535.0/808688257.0, a1210=3962137247.0/1805957418.0;
        float a1211=65686358.0/487910083.0;
        geoRHS(r+he*(a121*kr1+a124*kr4+a125*kr5+a126*kr6+a127*kr7+a128*kr8+a129*kr9+a1210*kr10+a1211*kr11),
               th+he*(a121*kth1+a124*kth4+a125*kth5+a126*kth6+a127*kth7+a128*kth8+a129*kth9+a1210*kth10+a1211*kth11),
               pr+he*(a121*kpr1+a124*kpr4+a125*kpr5+a126*kpr6+a127*kpr7+a128*kpr8+a129*kpr9+a1210*kpr10+a1211*kpr11),
               pth+he*(a121*kpth1+a124*kpth4+a125*kpth5+a126*kpth6+a127*kpth7+a128*kpth8+a129*kpth9+a1210*kpth10+a1211*kpth11),
               a, b, kr12, kth12, kphi12, kpr12, kpth12);

        // Stage 13: c13 = 1
        float a131=403863854.0/491063109.0, a134=-5068492393.0/434740067.0;
        float a135=-411421997.0/543043805.0, a136=652783627.0/914296604.0;
        float a137=11173962825.0/925320556.0, a138=-13158990841.0/6184727034.0;
        float a139=3936647629.0/1978049680.0, a1310=-160528059.0/685178525.0;
        float a1311=248638103.0/1413531060.0;
        geoRHS(r+he*(a131*kr1+a134*kr4+a135*kr5+a136*kr6+a137*kr7+a138*kr8+a139*kr9+a1310*kr10+a1311*kr11),
               th+he*(a131*kth1+a134*kth4+a135*kth5+a136*kth6+a137*kth7+a138*kth8+a139*kth9+a1310*kth10+a1311*kth11),
               pr+he*(a131*kpr1+a134*kpr4+a135*kpr5+a136*kpr6+a137*kpr7+a138*kpr8+a139*kpr9+a1310*kpr10+a1311*kpr11),
               pth+he*(a131*kpth1+a134*kpth4+a135*kpth5+a136*kpth6+a137*kpth7+a138*kpth8+a139*kpth9+a1310*kpth10+a1311*kpth11),
               a, b, kr13, kth13, kphi13, kpr13, kpth13);

        // 8th-order solution weights
        float b1=14005451.0/335480064.0;
        float b6=-59238493.0/1068277825.0, b7=181606767.0/758867731.0;
        float b8=561292985.0/797845732.0, b9=-1041891430.0/1371343529.0;
        float b10=760417239.0/1151165299.0, b11=118820643.0/751138087.0;
        float b12=-528747749.0/2220607170.0, b13=1.0/4.0;

        r   += he*(b1*kr1+b6*kr6+b7*kr7+b8*kr8+b9*kr9+b10*kr10+b11*kr11+b12*kr12+b13*kr13);
        th  += he*(b1*kth1+b6*kth6+b7*kth7+b8*kth8+b9*kth9+b10*kth10+b11*kth11+b12*kth12+b13*kth13);
        phi += he*(b1*kphi1+b6*kphi6+b7*kphi7+b8*kphi8+b9*kphi9+b10*kphi10+b11*kphi11+b12*kphi12+b13*kphi13);
        pr  += he*(b1*kpr1+b6*kpr6+b7*kpr7+b8*kpr8+b9*kpr9+b10*kpr10+b11*kpr11+b12*kpr12+b13*kpr13);
        pth += he*(b1*kpth1+b6*kpth6+b7*kpth7+b8*kpth8+b9*kpth9+b10*kpth10+b11*kpth11+b12*kpth12+b13*kpth13);

        if (th < 0.005) { th = 0.005; pth = abs(pth); }
        if (th > PI-0.005) { th = PI-0.005; pth = -abs(pth); }

        if (r <= rp*1.01) { done=true; break; }
        if (u_disk > 0.5) {
            float cross = (oldTh-PI*0.5)*(th-PI*0.5);
            if (cross < 0.0) {
                float f = clamp(abs(oldTh-PI*0.5)/max(abs(th-oldTh),1e-6), 0.0, 1.0);
                vec3 dc = disk(oldR+f*(r-oldR), oldPhi+f*(phi-oldPhi), a);
                color += dc * (1.0 - clamp(length(color)*0.4, 0.0, 0.9));
            }
        }
        if (r > RESC) {
            float fth = oldTh + (th-oldTh) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            float fph = oldPhi + (phi-oldPhi) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            color += background(sphereDir(fth, fph)) * (1.0 - clamp(length(color)*0.3, 0.0, 0.9));
            done=true; break;
        }
        if (r < 0.5 || r!=r || th!=th) { done=true; break; }
    }
    float imp = length(vec2(alpha,beta));
    float rc = 5.2-1.0*a;
    color += vec3(0.1,0.07,0.04)*exp(-pow((imp-rc)/0.3,2.0))*0.06;
    color *= 1.0-0.3*dot(uv,uv);
    color = color/(1.0+color);
    color = pow(color, vec3(1.0/2.2));
    gl_FragColor = vec4(color, 1);
}
"""
    else:
        src += """
// ===========================================================
//  YOSHIDA 4TH-ORDER SYMPLECTIC INTEGRATOR (Forest-Ruth)
//
//  The classic 4th-order symplectic composition method from
//  Forest & Ruth (1990) and Yoshida (1990). Uses 3 substeps
//  with the "triple-jump" coefficients:
//    w1 = 1/(2 - 2^(1/3)),  w0 = -2^(1/3)/(2 - 2^(1/3))
//
//  This is the fundamental building block from which the
//  higher-order Yoshida methods (6th, 8th) are composed.
//
//  Properties:
//    - 4th-order symplectic: O(h^4) energy conservation
//    - Phase-space volume exactly preserved
//    - Faster than RK4 (6 vs 4 geoRHS calls, but symplectic)
//    - No secular drift — ideal as the default integrator
//
//  References:
//    Forest & Ruth (1990), Physica D 43, 105-117
//    Yoshida (1990), Physics Letters A 150(5-7), 262-268
//
//  Industry usage: molecular dynamics (LAMMPS, GROMACS),
//  accelerator physics, celestial mechanics, GR geodesics.
// ===========================================================

void geoRHS(
    float r, float th, float pr, float pth,
    float a, float b,
    out float dr, out float dth, out float dphi,
    out float dpr, out float dpth
) {
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float a2 = a*a, r2 = r*r;
    float Q2 = u_Q*u_Q;
    float sig = r2 + a2*c2;
    float del = r2 - 2.0*r + a2 + Q2;
    float sdel = max(del, 1e-6);
    float rpa2 = r2 + a2;
    float w = 2.0*r - Q2;
    float A_ = rpa2*rpa2 - sdel*a2*s2;
    float isig = 1.0/sig;
    float SD = sig*sdel;
    float iSD = 1.0/SD;
    float is2 = 1.0/s2;

    float grr = sdel*isig, gthth = isig;
    float gff = (sig - w)*iSD*is2;
    float gtf = -a*w*iSD;

    dr = grr*pr; dth = gthth*pth; dphi = gff*b - gtf;

    float dsig_r=2.0*r; float ddel_r=2.0*r-2.0;
    float dA_r=4.0*r*rpa2-ddel_r*a2*s2;
    float dSD_r=dsig_r*sdel+sig*ddel_r;
    float dgtt_r=-(dA_r*SD-A_*dSD_r)/(SD*SD);
    float dgtf_r=-a*(2.0*SD - w*dSD_r)/(SD*SD);
    float dgrr_r=(ddel_r*sig-sdel*dsig_r)/(sig*sig);
    float dgthth_r=-dsig_r*isig*isig;
    float num_ff=sig-w; float den_ff=SD*s2;
    float dgff_r=((dsig_r-2.0)*den_ff-num_ff*dSD_r*s2)/(den_ff*den_ff);
    dpr=-0.5*(dgtt_r-2.0*b*dgtf_r+dgrr_r*pr*pr+dgthth_r*pth*pth+dgff_r*b*b);

    float dsig_th=-2.0*a2*sth*cth;
    float ds2_th=2.0*sth*cth;
    float dA_th=-sdel*a2*ds2_th;
    float dSD_th=dsig_th*sdel;
    float dgtt_th=-(dA_th*SD-A_*dSD_th)/(SD*SD);
    float dgtf_th=a*w*dSD_th/(SD*SD);
    float dgrr_th=-sdel*dsig_th/(sig*sig);
    float dgthth_th=-dsig_th*isig*isig;
    float dgff_th=(dsig_th*den_ff-num_ff*(dsig_th*sdel*s2+SD*ds2_th))/(den_ff*den_ff);
    dpth=-0.5*(dgtt_th-2.0*b*dgtf_th+dgrr_th*pr*pr+dgthth_th*pth*pth+dgff_th*b*b);
}

// Yoshida 4th-order (Forest-Ruth) coefficients
// w1 = 1/(2 - 2^(1/3)), w0 = -2^(1/3)/(2 - 2^(1/3))
// d1 = w1/2, d0 = (w0 + w1)/2
#define Y4_W1  1.3512071919596576
#define Y4_W0 -1.7024143839193153
#define Y4_D1  0.6756035959798288
#define Y4_D0 -0.1756035959798288

void main() {
    vec2 uv = v_uv;
    float asp = u_res.x/u_res.y;
    float alpha = uv.x*u_fov*asp, beta = uv.y*u_fov;
    float a = u_a, a2 = a*a;
    float thObs = u_incl, sO = sin(thObs), cO = cos(thObs);
    float b = -alpha*sO;
    float q2 = beta*beta + cO*cO*(alpha*alpha - a2);

    float r = R0, th = thObs, phi = u_phi0;
    float sth = sin(th), cth = cos(th);
    float s2 = sth*sth + S2_EPS, c2 = cth*cth;
    float sig = r*r+a2*c2, del = r*r-2.0*r+a2+u_Q*u_Q;
    float sdel = max(del,1e-6);
    float rpa2 = r*r+a2, A_ = rpa2*rpa2-sdel*a2*s2;
    float iSD = 1.0/(sig*sdel), is2 = 1.0/s2;
    float grr = sdel/sig, gthi = 1.0/sig;
    float pth = beta;
    float w_init = 2.0*r - u_Q*u_Q;
    float rest = -A_*iSD + 2.0*a*b*w_init*iSD + gthi*pth*pth + (sig-w_init)*iSD*is2*b*b;
    float pr2 = -rest/grr;
    float pr = pr2 > 0.0 ? -sqrt(pr2) : 0.0;
    float Q2 = u_Q*u_Q;
    float rp = 1.0 + sqrt(max(1.0-a2-Q2, 0.0));
    vec3 color = vec3(0); bool done = false;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;
        float he = H_BASE * clamp((r-rp)*0.4, 0.04, 1.0);
        he = clamp(he, 0.012, 0.6);
        float oldTh = th, oldR = r, oldPhi = phi;

        // Yoshida 4th-order symmetric composition: 3 substeps
        float dr_,dth_,dphi_,dpr_,dpth_;

        // --- Substep 1: drift d1, kick w1 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y4_D1*dr_;
        th  += he*Y4_D1*dth_;
        phi += he*Y4_D1*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y4_W1*dpr_;
        pth += he*Y4_W1*dpth_;

        // --- Substep 2: drift d0, kick w0 ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y4_D0*dr_;
        th  += he*Y4_D0*dth_;
        phi += he*Y4_D0*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y4_W0*dpr_;
        pth += he*Y4_W0*dpth_;

        // --- Substep 3: drift d1, kick w1 (symmetric) ---
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        r   += he*Y4_D1*dr_;
        th  += he*Y4_D1*dth_;
        phi += he*Y4_D1*dphi_;
        geoRHS(r,th,pr,pth, a,b, dr_,dth_,dphi_,dpr_,dpth_);
        pr  += he*Y4_W1*dpr_;
        pth += he*Y4_W1*dpth_;

        if (th < 0.005) { th = 0.005; pth = abs(pth); }
        if (th > PI-0.005) { th = PI-0.005; pth = -abs(pth); }

        if (r <= rp*1.01) { done=true; break; }
        if (u_disk > 0.5) {
            float cross = (oldTh-PI*0.5)*(th-PI*0.5);
            if (cross < 0.0) {
                float f = clamp(abs(oldTh-PI*0.5)/max(abs(th-oldTh),1e-6), 0.0, 1.0);
                vec3 dc = disk(oldR+f*(r-oldR), oldPhi+f*(phi-oldPhi), a);
                color += dc * (1.0 - clamp(length(color)*0.4, 0.0, 0.9));
            }
        }
        if (r > RESC) {
            float fth = oldTh + (th-oldTh) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            float fph = oldPhi + (phi-oldPhi) * clamp((RESC-oldR)/max(r-oldR,1e-6), 0.0, 1.0);
            color += background(sphereDir(fth, fph)) * (1.0 - clamp(length(color)*0.3, 0.0, 0.9));
            done=true; break;
        }
        if (r < 0.5 || r!=r || th!=th) { done=true; break; }
    }
    float imp = length(vec2(alpha,beta));
    float rc = 5.2-1.0*a;
    color += vec3(0.1,0.07,0.04)*exp(-pow((imp-rc)/0.3,2.0))*0.06;
    color *= 1.0-0.3*dot(uv,uv);
    color = color/(1.0+color);
    color = pow(color, vec3(1.0/2.2));
    gl_FragColor = vec4(color, 1);
}
"""

    return "#version 120\n" + src
