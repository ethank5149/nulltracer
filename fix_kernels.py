import os
import glob

for filepath in glob.glob("nulltracer/kernels/integrators/*.cu"):
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'diskColor' not in content:
        continue
        
    if 'F_peak' not in content:
        # Add F_peak computation where Q2 is defined
        content = content.replace('double Q2 = p.charge * p.charge;', 'double Q2 = p.charge * p.charge;\n        float F_peak = novikov_thorne_peak(a, (double)p.isco);')
    
    # Replace diskColor call
    # Note: there might be multiple spaces, so we'll do a simple replace
    # We will search for 'g, (int)p.doppler_boost,\n                             &dcr'
    # Actually, let's use regex
    import re
    content = re.sub(r'g,\s*\(int\)p\.doppler_boost,\s*&dcr,\s*&dcg,\s*&dcb\);', r'g, (int)p.doppler_boost, F_peak,\n                             &dcr, &dcg, &dcb);', content)
    
    with open(filepath, 'w') as f:
        f.write(content)
