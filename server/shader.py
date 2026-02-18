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
    METHOD = opts.get("method", "separated")
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
    if METHOD == "hamiltonian":
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
    else:
        src += """
// ===========================================================
//  SEPARATED GEODESIC EQUATIONS (Verlet midpoint)
//  θ-form with smooth sin²θ+ε regularization.
//  At escape, convert to Cartesian direction for background
//  lookup — sinθ multiplier naturally suppresses φ-error.
// ===========================================================

void main() {
    vec2 uv = v_uv;
    float asp = u_res.x / u_res.y;
    float alpha = uv.x * u_fov * asp, beta = uv.y * u_fov;
    float a = u_a, a2 = a*a;
    float thObs = u_incl, sO = sin(thObs), cO = cos(thObs);
    float b = -alpha * sO;
    float q2 = beta*beta + cO*cO*(alpha*alpha - a2);
    float r = R0, th = thObs, phi = u_phi0;
    float sr = -1.0, st = beta >= 0.0 ? -1.0 : 1.0;
    float Q2 = u_Q*u_Q;
    float rp = 1.0 + sqrt(max(1.0-a2-Q2, 0.0));
    float bma = b-a, bma2 = bma*bma;
    vec3 color = vec3(0); bool done = false;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;

        // Regularized quantities — sin²θ + ε is smooth, never zero
        float sth = sin(th), cth = cos(th);
        float s2 = sth*sth + S2_EPS, c2 = cth*cth;
        float sig = r*r + a2*c2, isig = 1.0/sig;
        float del = r*r - 2.0*r + a2 + u_Q*u_Q;
        float sdel = max(del, 1e-6);

        float T = r*r + a2 - a*b;
        float Rp = T*T - del*(q2 + bma2);
        float Tp = q2 + a2*c2 - c2*b*b/s2;
        if (Rp < 0.0) { Rp = 0.0; sr = -sr; }
        if (Tp < 0.0) { Tp = 0.0; st = -st; }

        float dr   = sr * sqrt(Rp) * isig;
        float dth  = st * sqrt(Tp) * isig;
        float dphi = (b/s2 - a + a*T/sdel) * isig;

        // Adaptive step: smaller near horizon only
        float he = H_BASE * clamp((r-rp)*0.35, 0.04, 1.0);
        he = clamp(he, 0.012, 0.7);

        float oldTh = th, oldR = r, oldPhi = phi;

        // Verlet midpoint
        float rH  = r  + 0.5*he*dr;
        float thH = th + 0.5*he*dth;
        thH = clamp(thH, 0.005, PI-0.005);
        float sthH = sin(thH), cthH = cos(thH);
        float s2H = sthH*sthH + S2_EPS, c2H = cthH*cthH;
        float sigH = rH*rH + a2*c2H, isigH = 1.0/sigH;
        float delH = rH*rH - 2.0*rH + a2 + u_Q*u_Q;
        float sdelH = max(delH, 1e-6);
        float TH = rH*rH + a2 - a*b;
        float RpH = TH*TH - delH*(q2 + bma2);
        float TpH = q2 + a2*c2H - c2H*b*b/s2H;
        if (RpH < 0.0) { RpH = 0.0; sr = -sr; }
        if (TpH < 0.0) { TpH = 0.0; st = -st; }

        float drH   = sr * sqrt(RpH) * isigH;
        float dthH  = st * sqrt(TpH) * isigH;
        float dphiH = (b/s2H - a + a*TH/sdelH) * isigH;

        r   += he * drH;
        th  += he * dthH;
        phi += he * dphiH;

        // Soft boundary reflection
        if (th < 0.005) { th = 0.005; st = 1.0; }
        if (th > PI-0.005) { th = PI-0.005; st = -1.0; }

        if (r <= rp*1.005) { done = true; break; }

        // Disk crossing
        if (u_disk > 0.5) {
            float cross = (oldTh - PI*0.5) * (th - PI*0.5);
            if (cross < 0.0) {
                float f = clamp(abs(oldTh-PI*0.5)/max(abs(th-oldTh),1e-6), 0.0, 1.0);
                vec3 dc = disk(oldR + f*(r-oldR), oldPhi + f*(phi-oldPhi), a);
                color += dc * (1.0 - clamp(length(color)*0.4, 0.0, 0.9));
            }
        }

        // Escape — convert to Cartesian direction for background
        if (r > RESC) {
            // Interpolate to escape radius for smoother background
            float frac = clamp((RESC - oldR) / max(r - oldR, 1e-6), 0.0, 1.0);
            float escTh  = oldTh  + (th  - oldTh)  * frac;
            float escPhi = oldPhi + (phi - oldPhi) * frac;
            vec3 d = sphereDir(escTh, escPhi);
            color += background(d) * (1.0 - clamp(length(color)*0.3, 0.0, 0.9));
            done = true; break;
        }
        if (r!=r || th!=th) { done = true; break; }
    }

    float imp = length(vec2(alpha, beta));
    float rc = 5.2 - 1.0*a;
    color += vec3(0.1,0.07,0.04) * exp(-pow((imp-rc)/0.3, 2.0)) * 0.06;
    color *= 1.0 - 0.3 * dot(uv, uv);
    color = color / (1.0 + color);
    color = pow(color, vec3(1.0/2.2));
    gl_FragColor = vec4(color, 1);
}
"""

    return "#version 120\n" + src
