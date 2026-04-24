with open("nulltracer/kernels/disk.cu", "r") as f:
    content = f.read()

plunging_logic = """
    double flux_scale;
    if (r >= isco) {
        flux_scale = novikov_thorne_flux((double)r, (double)a, (double)isco) / f_peak_for_normalization;
    } else {
        double f_isco = novikov_thorne_flux((double)isco + 1e-4, (double)a, (double)isco);
        double x = (r - r_horizon) / fmax(isco - r_horizon, 1e-6);
        x = fmax(x, 0.0);
        flux_scale = (f_isco / f_peak_for_normalization) * x * x;
    }
"""

content = content.replace("    double flux_scale = novikov_thorne_flux((double)r, (double)a, (double)isco) / f_peak_for_normalization;", plunging_logic)

with open("nulltracer/kernels/disk.cu", "w") as f:
    f.write(content)
print("Done")
