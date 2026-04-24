import glob
import re

for filepath in glob.glob("nulltracer/kernels/integrators/*.cu"):
    with open(filepath, 'r') as f:
        content = f.read()

    adaptive_logic = """
            double r_photon_sphere = 3.0;
            if (a != 0.0) {
                r_photon_sphere = 2.0 * (1.0 + cos(2.0/3.0 * acos(-a)));
            }
            double dist_to_photon_sphere = fabs(r - r_photon_sphere);
            double adaptive_factor = 1.0;
            if (dist_to_photon_sphere < 1.0) {
                adaptive_factor = 0.1 + 0.9 * dist_to_photon_sphere;
            }
            double effective_step = p.step_size * adaptive_factor;
            // Cap he if needed
            if (he > effective_step) he = effective_step;
    """
    
    # Let's insert it at the start of the while loop or after `for (int i = 0; i < STEPS; i++) {`
    # Wait, some integrators use `double he = ...` and some don't have adaptive step.
    # We can just insert it at the start of the loop.
    # In `rkdp8.cu` it's `for (int i = 0; i < STEPS; i++) {` and then `if (done) break;`
    content = re.sub(r'(for \(int i = 0; i < STEPS; i\+\+\) \{\s*if \(done\) break;)', r'\1' + adaptive_logic, content)

    with open(filepath, 'w') as f:
        f.write(content)

print("Done")
