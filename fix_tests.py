with open("tests/test_shadow_render.py", "r") as f:
    content = f.read()

content = content.replace('fov=12.0, bg_mode=2, obs_dist=100.0,\n        )', 'fov=12.0, bg_mode=2, obs_dist=100.0, show_disk=False,\n        )')

with open("tests/test_shadow_render.py", "w") as f:
    f.write(content)

with open("tests/test_eht_validation.py", "r") as f:
    content = f.read()

content = content.replace('aa_samples=1,\n            bg_mode=2, star_layers=0,\n        )', 'aa_samples=1,\n            bg_mode=2, star_layers=0, show_disk=False,\n        )')

with open("tests/test_eht_validation.py", "w") as f:
    f.write(content)
print("Done")
