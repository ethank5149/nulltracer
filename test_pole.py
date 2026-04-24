import numpy as np
import nulltracer as nt
import matplotlib.pyplot as plt

renderer = nt.CudaRenderer()
renderer.initialize()

params = {
    'spin': 0.998, 'inclination': 0.1,
    'width': 512, 'height': 512, 'fov': 5.0,
    'show_disk': False, 'bg_mode': 1,
    'obs_dist': 500, 'step_size': 0.1, 'method': 'symplectic8',
}

raw_bytes = renderer.render_frame(params)
img = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((512, 512, 3))

plt.imsave("pole_artifact.png", img)
print("Saved pole_artifact.png")
print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.2f}")
print("Center region mean:", img[120:136, 120:136].mean())
