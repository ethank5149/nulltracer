// ============================================================
//  ISCO CALCULATOR
//  Innermost Stable Circular Orbit calculations for
//  Kerr and Kerr-Newman black holes.
// ============================================================

// Numerical ISCO for Kerr-Newman (uncharged test particle, M=1).
// Uses dE/dr = 0 condition via bisection with finite-difference derivative.
function iscoKN(a, Q) {
    const a2 = a*a, Q2 = Q*Q;
    const rh = 1 + Math.sqrt(Math.max(1 - a2 - Q2, 1e-12));
    // Angular velocity of prograde circular orbit at radius r
    function omega(r) {
        const f = r - Q2/r;  // = (r^2 - Q^2)/r
        const D = r*r - 2*r + a2 + Q2;
        // From geodesic equation: Omega = (sqrt(M/r - Q^2/r^2) - a*(...)) 
        // More robust: use the condition dg_tt/dr + 2 Omega dg_tphi/dr + Omega^2 dg_phiphi/dr = 0
        // At equator (Sigma = r^2):
        // A = 2*(Q2 - r)/r^3, B = 2*a*(r - Q2/r)/r^2... let me use the direct formula
        const rm = r - Q2/r;  // = (r^2 - Q^2)/r
        if (rm <= 0) return NaN;
        const sqrm = Math.sqrt(rm/r);  // sqrt((r - Q^2/r)/r) = sqrt(1/r - Q^2/r^2)
        return 1 / (r*r/sqrm/r + a);  // simplified: 1/(r^(3/2)/sqrt(1-Q^2/r^2) + a)
    }
    // Energy of circular orbit via metric components at equator
    function energy(r) {
        const r2 = r*r;
        const del = r2 - 2*r + a2 + Q2;
        if (del <= 0) return NaN;
        // Metric at equator (Sigma = r^2):
        const gtt = -(1 - (2*r - Q2)/r2);
        const gtph = -a*(2*r - Q2)/r2;
        const gphph = (r2*r2 + a2*r2 + a2*(2*r - Q2))/r2;
        // Solve for Omega from circular orbit condition
        const A = gphph, B = 2*gtph, C = gtt;
        // d/dr components:
        const dgtt = 2*(Q2 - r)/(r2*r);
        const dgtph = 2*a*(r2 - 2*Q2*r + Q2*r)/(r2*r2);  
        // Use cleaner formula: g_tphi = -a(2r-Q^2)/r^2
        // dg_tphi/dr = -a[(2r^2 - (2r-Q^2)*2r)/r^4] = -a[(2r^2-4r^2+2Q^2r)/r^4] = -a[(-2r^2+2Q^2r)/r^4]
        // = 2a(r-Q^2)/r^3  ... wait: 2a(r^2 - Q^2*r) hmm
        // Let me just use g_tphi = -a(2r-Q^2)/r^2
        // dg_tphi/dr = -a * [(2*r^2 - (2r-Q^2)*2r) / r^4]
        //            = -a * [(2r^2 - 4r^2 + 2Q^2*r) / r^4]
        //            = -a * [(-2r^2 + 2Q^2*r) / r^4]
        //            = 2*a*(r - Q2) / (r*r*r)
        const dgtph_clean = 2*a*(r - Q2)/(r*r2);
        // g_phiphi = (r^4 + a^2*r^2 + a^2*(2r-Q^2))/r^2
        //          = r^2 + a^2 + a^2*(2r-Q^2)/r^2
        const A2 = r2 + a2 + a2*(2*r-Q2)/r2;
        // dg_phiphi/dr = 2r + a^2 * d[(2r-Q^2)/r^2]/dr
        //              = 2r + a^2 * [(2r^2 - (2r-Q^2)*2r)/r^4]
        //              = 2r + a^2 * [(-2r^2 + 2Q^2*r)/r^4]  ... same as before
        //              = 2r - 2*a^2*(r-Q2)/r^3  hmm
        // Actually: (2r-Q^2)/r^2 = 2/r - Q^2/r^2
        // d/dr = -2/r^2 + 2Q^2/r^3
        const dgphph = 2*r + a2*(-2/r2 + 2*Q2/(r2*r));

        // Circular orbit: dgtt + 2*Om*dgtph + Om^2*dgphph = 0
        const disc = dgtph_clean*dgtph_clean - dgtt*dgphph;
        if (disc < 0) return NaN;
        const Om = (-dgtph_clean + Math.sqrt(disc)) / dgphph;  // prograde
        
        // E = -(g_tt + g_tphi * Om) * u^t
        // u^t = 1/sqrt(-(g_tt + 2*g_tphi*Om + g_phiphi*Om^2))
        const denom = -(gtt + 2*gtph*Om + gphph*Om*Om);
        if (denom <= 0) return NaN;
        const ut = 1/Math.sqrt(denom);
        return -(gtt + gtph*Om) * ut;
    }
    // Bisect on dE/dr = 0
    const dr = 1e-5;
    function dEdr(r) {
        const Ep = energy(r + dr), Em = energy(r - dr);
        if (isNaN(Ep) || isNaN(Em)) return NaN;
        return (Ep - Em) / (2*dr);
    }
    let lo = rh + 0.01, hi = 9.0;
    // ISCO is where dE/dr changes from negative to positive
    for (let i = 0; i < 80; i++) {
        const mid = (lo + hi) / 2;
        const d = dEdr(mid);
        if (isNaN(d) || d < 0) lo = mid;
        else hi = mid;
    }
    return (lo + hi) / 2;
}

export function rPlus(a, Q) { return 1+Math.sqrt(Math.max(1-a*a-Q*Q,0)); }

export function iscoJS(a, Q) {
    if (!Q || Q === 0) {
        // Analytic Kerr formula (Bardeen, Press & Teukolsky 1972)
        const z1=1+Math.pow(1-a*a,1/3)*(Math.pow(1+a,1/3)+Math.pow(Math.max(1-a,0),1/3));
        const z2=Math.sqrt(3*a*a+z1*z1);
        return 3+z2-Math.sqrt((3-z1)*(3+z1+2*z2));
    }
    return iscoKN(a, Q);
}
