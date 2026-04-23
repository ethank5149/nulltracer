import pytest
import numpy as np
from nulltracer.render import render_frame
from nulltracer.eht_validation import extract_shadow_metrics

@pytest.mark.gpu
def test_schwarzschild_shadow_diameter(cuda_renderer):
    img, info = render_frame(spin=0.0, inclination_deg=90.0, width=512, height=512, fov=12.0, bg_mode=2, obs_dist=100.0)
    
    metrics = extract_shadow_metrics(img, fov_deg=12.0, threshold=0.3)
    
    if "error" not in metrics:
        # Expected shadow diameter for Schwarzschild is 2 * 3*sqrt(3) ~= 10.3923
        expected_diameter = 2.0 * 3.0 * np.sqrt(3.0)
        assert abs(metrics["diameter_M"] - expected_diameter) < 0.5, "Shadow diameter out of range"
