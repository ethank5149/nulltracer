import glob
import re

for filepath in glob.glob("nulltracer/kernels/integrators/*.cu"):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Update diskColor and g-factor
    # Find the disk crossing block
    pattern = r'(float\s+g\s*=\s*compute_g_factor_extended\([^;]+;\s*float\s+dcr,\s*dcg,\s*dcb;\s*diskColor\([^;]+;\s*float\s+crossing_alpha\s*=\s*base_alpha;)'
    replacement = """
                    double b_impact = b;
                    float g = (float)kerr_g_factor(r_hit, a, Q2, b_impact);
                    float dcr, dcg, dcb;
                    diskColor(dr_f, dphi_f, (float)a, (float)Q2,
                             (float)p.isco, (float)p.disk_outer, (float)p.disk_temp,
                             g, (int)p.doppler_boost, F_peak,
                             &dcr, &dcg, &dcb);
                    
                    double p_total = sqrt(pr * pr + pth * pth + b * b);
                    float cos_em = (float)(fabs(pth) / fmax(p_total, 1e-15));
                    float limb = limb_darkening(cos_em);
                    dcr *= limb;
                    dcg *= limb;
                    dcb *= limb;

                    float crossing_alpha;
                    if (disk_crossings == 0) {
                        crossing_alpha = base_alpha;
                    } else if (disk_crossings == 1) {
                        crossing_alpha = base_alpha * 0.85f;
                    } else {
                        float ring_brightness_boost = powf(2.71828f, (float)(disk_crossings - 1) * 0.5f);
                        crossing_alpha = fminf(base_alpha * ring_brightness_boost, 1.0f);
                    }
    """
    
    # Actually wait, in rkdp8.cu, there is `float crossing_alpha = base_alpha;`
    # Let's replace the block more carefully
    content = re.sub(r'float g = compute_g_factor_extended\([^;]+;', 'float g = (float)kerr_g_factor(r_hit, a, Q2, b);', content)
    
    # Add limb darkening and crossing alpha
    crossing_block = """
                    double p_total = sqrt(pr * pr + pth * pth + b * b);
                    float cos_em = (float)(fabs(pth) / fmax(p_total, 1e-15));
                    float limb = limb_darkening(cos_em);
                    dcr *= limb;
                    dcg *= limb;
                    dcb *= limb;

                    float crossing_alpha;
                    if (disk_crossings == 0) {
                        crossing_alpha = base_alpha;
                    } else if (disk_crossings == 1) {
                        crossing_alpha = base_alpha * 0.85f;
                    } else {
                        float ring_brightness_boost = powf(2.71828f, (float)(disk_crossings - 1) * 0.5f);
                        crossing_alpha = fminf(base_alpha * ring_brightness_boost, 1.0f);
                    }
"""
    content = re.sub(r'float crossing_alpha = base_alpha;', crossing_block, content)

    # Hawking radiation glow
    hawking_glow = """
            if (r <= rp * 1.01) {
                float hr, hg, hb;
                hawking_glow_color(r, a, Q2, p.hawking_boost, &hr, &hg, &hb);
                if (hr > 0 || hg > 0 || hb > 0) {
                    blendColor(hr, hg, hb, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
                }
                blendColor(0.0f, 0.0f, 0.0f, 1.0f, &acc_r, &acc_g, &acc_b, &acc_a);
                done = true; break;
            }
"""
    content = re.sub(r'if \(r <= rp \* 1\.01\) \{\s*blendColor\(0\.0f, 0\.0f, 0\.0f, 1\.0f, &acc_r, &acc_g, &acc_b, &acc_a\);\s*done = true; break;\s*\}', hawking_glow, content)

    with open(filepath, 'w') as f:
        f.write(content)

print("Done")
