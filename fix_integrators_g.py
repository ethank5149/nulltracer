import glob
for filepath in glob.glob("nulltracer/kernels/integrators/*.cu"):
    with open(filepath, "r") as f:
        content = f.read()
    content = content.replace("kerr_g_factor(r_hit, a, Q2, b)", "kerr_g_factor(r_hit, a, Q2, b, (double)p.isco)")
    with open(filepath, "w") as f:
        f.write(content)
