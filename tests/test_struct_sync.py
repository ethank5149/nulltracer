import ctypes
from nulltracer._params import RenderParams

def test_struct_size():
    assert ctypes.sizeof(RenderParams) == 208, f"RenderParams size {ctypes.sizeof(RenderParams)} != 208 bytes"

def test_struct_fields():
    expected_fields = [
        "width", "height", "spin", "charge", "incl", "fov", "phi0", "isco",
        "steps", "obs_dist", "esc_radius", "disk_outer", "step_size",
        "bg_mode", "star_layers", "show_disk", "show_grid", "disk_temp",
        "doppler_boost", "srgb_output", "disk_alpha", "disk_max_crossings",
        "disk_mode", "bloom_enabled", "debug_trace", "aa_samples", "sky_width", "sky_height"
    ]
    
    actual_fields = [f[0] for f in RenderParams._fields_]
    assert actual_fields == expected_fields, f"Fields do not match expected canonical list: {actual_fields} vs {expected_fields}"
