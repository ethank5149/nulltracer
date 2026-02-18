"""
RK4 (classical 4th-order Runge-Kutta) integrator GLSL main loop.
"""


def rk4_integrator():
    """Return the GLSL main() function for the RK4 integrator.

    Uses 4 stages per step with the classic 1/6, 1/3, 1/3, 1/6 weights.
    """
    return """
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
