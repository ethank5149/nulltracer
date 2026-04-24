#ifndef RK4_CKS_CU
#define RK4_CKS_CU

#include "../geodesic_base.cu"
#include "../cks_metric.cu"
#include "../disk.cu"

extern "C" __global__
void trace_rk4_cks(const RenderParams *pp, unsigned char *output, const float *skymap, unsigned int *progress_counter) {
    const RenderParams &p = *pp;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int W = (int)p.width, H = (int)p.height;
    if (ix >= W || iy >= H) return;

    double r, th, phi, pr, pth, b, rp;
    float alpha, beta;
    if (!initRay(ix, iy, p, &r, &th, &phi, &pr, &pth, &b, &rp, &alpha, &beta)) return;

    double a = p.spin;
    double Q2 = p.charge * p.charge;
    double E_init = 1.0;
    double Lz_init = b;
    double Q_init = computeCarter(th, pth, a, b, Q2);

    double state[8];
    double rpa2 = sqrt(r*r + a*a);
    double sth = sin(th), cth = cos(th);
    double sphi = sin(phi), cphi = cos(phi);
    
    state[0] = 0.0;
    state[1] = rpa2 * sth * cphi;
    state[2] = rpa2 * sth * sphi;
    state[3] = r * cth;
    state[4] = -1.0; // p_t
    state[5] = pr; // Approx
    state[6] = 0.0; // Approx
    state[7] = pth; // Approx
    // Just a placeholder, exact CKS momentum transform from BL is needed but complex.
    
    int STEPS = (int)p.steps;
    double he = p.step_size;
    double pixel_intensity = 0.0;

    for (int i = 0; i < STEPS; i++) {
        double d1[8], d2[8], d3[8], d4[8], temp[8];
        cks_hamiltonian_derivatives(state, a, sqrt(Q2), d1);
        for(int j=0; j<8; j++) temp[j] = state[j] + 0.5 * he * d1[j];
        
        cks_hamiltonian_derivatives(temp, a, sqrt(Q2), d2);
        for(int j=0; j<8; j++) temp[j] = state[j] + 0.5 * he * d2[j];
        
        cks_hamiltonian_derivatives(temp, a, sqrt(Q2), d3);
        for(int j=0; j<8; j++) temp[j] = state[j] + he * d3[j];
        
        cks_hamiltonian_derivatives(temp, a, sqrt(Q2), d4);
        
        for(int j=0; j<8; j++) {
            state[j] += (he / 6.0) * (d1[j] + 2.0*d2[j] + 2.0*d3[j] + d4[j]);
        }
        
        stabilize_constants(state, a, E_init, Lz_init, Q_init);
        
        double current_r = compute_r_cks(state[1], state[2], state[3], a);
        if (current_r <= rp * 1.01) break;
        if (current_r > p.esc_radius) break;
    }

    int idx = (iy * W + ix) * 3;
    output[idx + 0] = (unsigned char)255;
    output[idx + 1] = (unsigned char)0;
    output[idx + 2] = (unsigned char)0;
    atomicAdd(progress_counter, 1);
}

extern "C" __global__
void ray_trace_rk4_cks(const RenderParams *pp, double *output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
}

#endif
