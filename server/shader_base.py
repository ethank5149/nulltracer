"""
Shared GLSL shader components: header, geoRHS function, and common utilities.

These are used by all integrators and assembled by shader.py's build_frag_src().
"""


def shader_header(STEPS, R0, RESC, H_BASE, STAR_LAYERS, BG_MODE):
    """Generate the GLSL header: varying/uniform declarations and #defines.

    This is the only part that needs Python string interpolation.
    """
    return (
        f"varying vec2 v_uv;\n"
        f"uniform vec2 u_res;\n"
        f"uniform float u_a,u_incl,u_fov,u_disk,u_grid,u_temp,u_phi0,u_Q,u_isco;\n"
        f"\n"
        f"#define PI  3.14159265359\n"
        f"#define TAU 6.28318530718\n"
        f"#define STEPS {STEPS}\n"
        f"#define R0  {float(R0):.1f}\n"
        f"#define RESC {float(RESC):.1f}\n"
        f"#define RDISK 14.0\n"
        f"#define H_BASE {float(H_BASE):.4f}\n"
        f"#define STAR_LAYERS {STAR_LAYERS}\n"
        f"#define BG_MODE {BG_MODE}\n"
    )


# ── Static GLSL blocks (no Python interpolation needed) ─────

_COMMENTS_AND_DEFINES = """\

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

float hash(vec2 p){
    p=fract(p*vec2(443.8,441.4));
    p+=dot(p,p+19.19);
    return fract(p.x*p.y);
}

// ISCO passed as uniform u_isco (computed numerically in JS for Kerr-Newman)

// ===========================================================
//  CUBE-MAP PROJECTION  (Cartesian direction → face + UV)
//
//  The key property: at the poles d=(0,0,±1), the +Z/-Z faces
//  use (x,y) as local coords. Both x=sinθ cosφ and y=sinθ sinφ
//  are smooth and ~0 at the pole, regardless of φ.
//  → φ-error × sinθ → 0.  No artifacts possible.
// ===========================================================

void cubeMap(vec3 d, out float face, out vec2 uv) {
    float ax = abs(d.x), ay = abs(d.y), az = abs(d.z);
    if (az >= ax && az >= ay) {
        face = d.z > 0.0 ? 0.0 : 1.0;
        uv = vec2(d.x, d.y) / az;
    } else if (ax >= ay) {
        face = d.x > 0.0 ? 2.0 : 3.0;
        uv = vec2(d.y, d.z) / ax;
    } else {
        face = d.y > 0.0 ? 4.0 : 5.0;
        uv = vec2(d.x, d.z) / ay;
    }
}

float cubeChecker(vec2 uv, float div) {
    vec2 cell = floor((uv * 0.5 + 0.5) * div);
    return mod(cell.x + cell.y, 2.0);
}

float cubeGrid(vec2 uv, float div) {
    vec2 f = fract((uv * 0.5 + 0.5) * div);
    return smoothstep(0.88, 0.96, max(abs(f.x-0.5)*2.0, abs(f.y-0.5)*2.0));
}

vec3 faceColor(float face) {
    if (face < 0.5) return vec3(0.14, 0.08, 0.04);
    if (face < 1.5) return vec3(0.06, 0.05, 0.14);
    if (face < 2.5) return vec3(0.04, 0.12, 0.07);
    if (face < 3.5) return vec3(0.12, 0.04, 0.08);
    if (face < 4.5) return vec3(0.04, 0.08, 0.14);
    return vec3(0.12, 0.10, 0.04);
}
"""

_GEO_RHS = """\

// ===========================================================
//  KERR-NEWMAN GEODESIC RHS (shared by all integrators)
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
"""


def common_glsl_prefix():
    """Return the static GLSL code that follows the header defines:
    comments, S2_EPS, hash, cube-map utilities, and faceColor.

    This comes BEFORE the background functions.
    """
    return _COMMENTS_AND_DEFINES


def geo_rhs_function():
    """Return the geoRHS() GLSL function — the Kerr-Newman geodesic
    right-hand side. Shared by ALL integrators.

    Previously duplicated 5 times (once per integrator).
    """
    return _GEO_RHS
