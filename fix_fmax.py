with open("nulltracer/kernels/disk.cu", "r") as f:
    content = f.read()

content = content.replace("fmax(isco - r_horizon, 1e-6)", "fmax((double)(isco - r_horizon), 1e-6)")
content = content.replace("x = fmax(x, 0.0);", "x = fmax(x, 0.0);")

with open("nulltracer/kernels/disk.cu", "w") as f:
    f.write(content)
print("Done")
