"""
Dormand-Prince 8th-order Runge-Kutta (RK8(7)) integrator GLSL main loop.
"""


def rkdp8_integrator():
    """Return the GLSL main() function for the RKDP8 integrator.

    Uses 13 stages per step with the Dormand-Prince (1981) coefficients.
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
        float he = H_BASE * (R0 / 30.0) * clamp((r-rp)*0.4, 0.04, 1.0);
        he = clamp(he, 0.012, 1.0);
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
