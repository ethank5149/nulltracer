with open("nulltracer/kernels/disk.cu", "r") as f:
    content = f.read()

new_g = """
__device__ double kerr_g_factor_base(double r, double a, double Q2, double b_impact) {
    double r2 = r * r;
    double Omega = 1.0 / (r * sqrt(r) + a);
    double Delta = r2 - 2.0 * r + a * a + Q2;
    double g_tt   = -(1.0 - (2.0 * r - Q2) / r2);
    double g_tphi = -a * (2.0 * r - Q2) / r2;
    double g_phiphi = (r2 * r2 + a * a * r2 + a * a * (2.0 * r - Q2)) / r2;
    double denom = -(g_tt + 2.0 * g_tphi * Omega + g_phiphi * Omega * Omega);
    if (denom <= 0.0) return 1.0;
    double u_t = 1.0 / sqrt(denom);
    double g = 1.0 / (u_t * (1.0 - b_impact * Omega));
    return fmax(g, 0.01);
}

__device__ double kerr_g_factor(double r, double a, double Q2, double b_impact, double r_isco) {
    double r_horizon = 1.0 + sqrt(fmax(1.0 - a * a - Q2, 0.0));
    if (r >= r_isco) {
        return kerr_g_factor_base(r, a, Q2, b_impact);
    }
    double g_isco = kerr_g_factor_base(r_isco, a, Q2, b_impact);
    double x = (r - r_horizon) / fmax(r_isco - r_horizon, 1e-10);
    x = fmax(fmin(x, 1.0), 0.0);
    return g_isco * x * x;
}
"""

import re
content = re.sub(r'__device__ double kerr_g_factor\(double r, double a, double Q2,\s*double b_impact\) \{.*?\n\}', new_g, content, flags=re.DOTALL)

with open("nulltracer/kernels/disk.cu", "w") as f:
    f.write(content)

# Now update the integrators to pass `p.isco`
import glob
for filepath in glob.glob("nulltracer/kernels/integrators/*.cu"):
    with open(filepath, "r") as f:
        content = f.read()
    content = content.replace("kerr_g_factor(r_hit, a, Q2, b_impact)", "kerr_g_factor(r_hit, a, Q2, b_impact, (double)p.isco)")
    with open(filepath, "w") as f:
        f.write(content)

with open("nulltracer/kernels/ray_trace.cu", "r") as f:
    content = f.read()
content = content.replace("kerr_g_factor(r_hit, a, Q2, b)", "kerr_g_factor(r_hit, a, Q2, b, (double)p.isco)")
with open("nulltracer/kernels/ray_trace.cu", "w") as f:
    f.write(content)
print("Done")
