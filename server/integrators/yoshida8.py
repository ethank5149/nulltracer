"""
Yoshida 8th-order symplectic integrator GLSL main loop.
"""


def yoshida8_integrator():
    """Return the GLSL constants and main() function for the Yoshida8 integrator.

    Uses 15 symmetric substeps with Solution D coefficients from Table 2.
    """
    return """
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
    float pth = -beta;
    float w_init = 2.0*r - u_Q*u_Q;
    float rest = -A_*iSD + 2.0*a*b*w_init*iSD + gthi*pth*pth + (sig-w_init)*iSD*is2*b*b;
    float pr2 = -rest/grr;
    float pr = pr2 > 0.0 ? -sqrt(pr2) : 0.0;
    float Q2 = u_Q*u_Q;
    float rp = 1.0 + sqrt(max(1.0-a2-Q2, 0.0));
    vec3 color = vec3(0); bool done = false;

    for (int i = 0; i < STEPS; i++) {
        if (done) break;
        float he = H_BASE * (R0 / 30.0) * clamp((r-rp)*0.4, 0.04, 1.0);
        he = clamp(he, 0.012, 1.0);
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
                vec3 dc = disk(oldR+f*(r-oldR), oldPhi+f*(phi-oldPhi), a, b);
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
