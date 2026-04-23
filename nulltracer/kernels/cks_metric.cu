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

__device__ void cks_hamiltonian_derivatives(double state[8], double a, double Q, double dstate[8]) {
    double eps = 1e-6;
    double g_inv[4][4];
    kerr_schild_inv_metric(state[1], state[2], state[3], a, Q, g_inv);
    
    // dx^mu / d lambda = g^mu nu p_nu
    for(int i=0; i<4; i++) {
        dstate[i] = 0.0;
        for(int j=0; j<4; j++) {
            dstate[i] += g_inv[i][j] * state[4+j];
        }
    }
    
    // dp_mu / d lambda = -1/2 * p_alpha p_beta d(g^alpha beta)/dx^mu
    for(int i=1; i<4; i++) {
        double state_plus[8];
        double state_minus[8];
        for(int k=0; k<8; k++) {
            state_plus[k] = state[k];
            state_minus[k] = state[k];
        }
        state_plus[i] += eps;
        state_minus[i] -= eps;
        
        double g_inv_plus[4][4];
        double g_inv_minus[4][4];
        kerr_schild_inv_metric(state_plus[1], state_plus[2], state_plus[3], a, Q, g_inv_plus);
        kerr_schild_inv_metric(state_minus[1], state_minus[2], state_minus[3], a, Q, g_inv_minus);
        
        double dp = 0.0;
        for(int alpha=0; alpha<4; alpha++) {
            for(int beta=0; beta<4; beta++) {
                double dg = (g_inv_plus[alpha][beta] - g_inv_minus[alpha][beta]) / (2.0 * eps);
                dp += -0.5 * state[4+alpha] * state[4+beta] * dg;
            }
        }
        dstate[4+i] = dp;
    }
    dstate[4] = 0.0;
}

__device__ void stabilize_constants(double state[8], double a, double E_init, double Lz_init, double Q_init) {
    // Keep momentum slightly stabilized if possible, or just keep p_t = -E_init
    state[4] = -E_init; 
}

#endif
