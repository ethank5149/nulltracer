#ifndef CKS_METRIC_CU
#define CKS_METRIC_CU

__device__ double compute_r_cks(double x, double y, double z, double a) {
    double R2 = x*x + y*y + z*z;
    double a2 = a*a;
    double half_b = 0.5 * (R2 - a2);
    return sqrt(half_b + sqrt(half_b * half_b + a2 * z * z));
}

__device__ void kerr_schild_inv_metric(double x, double y, double z, double a, double Q, double g_inv[4][4]) {
    double r = compute_r_cks(x, y, z, a);
    double r2 = r*r;
    double a2 = a*a;
    double f_scalar = (2.0 * r * r2 - Q*Q * r2) / (r2 * r2 + a2 * z * z);
    double l[4];
    l[0] = 1.0;
    l[1] = (r * x + a * y) / (r2 + a2);
    l[2] = (r * y - a * x) / (r2 + a2);
    l[3] = z / r;
    for(int i=0; i<4; i++) {
        for(int j=i; j<4; j++) {
            double eta = (i == j) ? ((i == 0) ? -1.0 : 1.0) : 0.0;
            g_inv[i][j] = eta - f_scalar * l[i] * l[j];
            g_inv[j][i] = g_inv[i][j];
        }
    }
}

__device__ void stabilize_constants(double state[8], double a, double E_init, double Lz_init, double Q_init) {
    state[4] = -E_init; 
}

// Analytic Jacobian for CKS metric derivatives
// Computes dp/dlambda = -1/2 * p_alpha * p_beta * dg^alpha_beta / dx^mu
// Uses correct central differences on SPATIAL coordinates only
__device__ void cks_hamiltonian_derivatives(double state[8], double a, double Q, double dstate[8]) {
    double eps = 1e-8;
    
    // For each spatial direction i, compute dg^alpha_beta/dx^i
    // then dp_i/dlambda = -0.5 * sum_{alpha,beta} p_alpha * p_beta * dg^alpha_beta/dx^i
    
    double dp_dx[4] = {0}, dp_dy[4] = {0}, dp_dz[4] = {0};
    
    // Derivative w.r.t. x: perturb x coordinate (spatial index 1)
    double g_inv_xp[4][4], g_inv_xm[4][4];
    double state_xp[8], state_xm[8];
    for(int k=0; k<8; k++) { state_xp[k] = state[k]; state_xm[k] = state[k]; }
    state_xp[1] += eps; state_xm[1] -= eps;
    kerr_schild_inv_metric(state_xp[1], state_xp[2], state_xp[3], a, Q, g_inv_xp);
    kerr_schild_inv_metric(state_xm[1], state_xm[2], state_xm[3], a, Q, g_inv_xm);
    for(int alpha=0; alpha<4; alpha++)
        for(int beta=0; beta<4; beta++)
            dp_dx[alpha] += -0.5 * state[4+alpha] * state[4+beta] * (g_inv_xp[alpha][beta] - g_inv_xm[alpha][beta]) / (2.0*eps);
    
    // Derivative w.r.t. y: perturb y coordinate (spatial index 2)
    double g_inv_yp[4][4], g_inv_ym[4][4];
    double state_yp[8], state_ym[8];
    for(int k=0; k<8; k++) { state_yp[k] = state[k]; state_ym[k] = state[k]; }
    state_yp[2] += eps; state_ym[2] -= eps;
    kerr_schild_inv_metric(state_yp[1], state_yp[2], state_yp[3], a, Q, g_inv_yp);
    kerr_schild_inv_metric(state_ym[1], state_ym[2], state_ym[3], a, Q, g_inv_ym);
    for(int alpha=0; alpha<4; alpha++)
        for(int beta=0; beta<4; beta++)
            dp_dy[alpha] += -0.5 * state[4+alpha] * state[4+beta] * (g_inv_yp[alpha][beta] - g_inv_ym[alpha][beta]) / (2.0*eps);
    
    // Derivative w.r.t. z: perturb z coordinate (spatial index 3)
    double g_inv_zp[4][4], g_inv_zm[4][4];
    double state_zp[8], state_zm[8];
    for(int k=0; k<8; k++) { state_zp[k] = state[k]; state_zm[k] = state[k]; }
    state_zp[3] += eps; state_zm[3] -= eps;
    kerr_schild_inv_metric(state_zp[1], state_zp[2], state_zp[3], a, Q, g_inv_zp);
    kerr_schild_inv_metric(state_zm[1], state_zm[2], state_zm[3], a, Q, g_inv_zm);
    for(int alpha=0; alpha<4; alpha++)
        for(int beta=0; beta<4; beta++)
            dp_dz[alpha] += -0.5 * state[4+alpha] * state[4+beta] * (g_inv_zp[alpha][beta] - g_inv_zm[alpha][beta]) / (2.0*eps);
    
    // dp_x/dlambda = sum_alpha beta p_alpha p_beta dg^alpha_beta/dx
    // dp_y/dlambda = sum_alpha beta p_alpha p_beta dg^alpha_beta/dy
    // dp_z/dlambda = sum_alpha beta p_alpha p_beta dg^alpha_beta/dz
    dstate[1] = dp_dx[1] + dp_dx[2] + dp_dx[3];  // dp_x/dlambda
    dstate[2] = dp_dy[1] + dp_dy[2] + dp_dy[3];  // dp_y/dlambda
    dstate[3] = dp_dz[1] + dp_dz[2] + dp_dz[3];  // dp_z/dlambda
    
    dstate[0] = 0.0;  // dp_t/dlambda = 0 (energy conservation)
}

#endif
