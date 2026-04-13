# Nulltracer

**GPU-accelerated null geodesic ray tracer for Kerr–Newman black holes.**

Nulltracer traces light paths through curved spacetime using CUDA compute
kernels, rendering realistic images of black hole shadows, photon rings,
accretion disk Doppler effects, and gravitational lensing.

## Quick Start

```bash
# Clone and install
git clone https://github.com/ethank5149/nulltracer.git
cd nulltracer
pip install -e .

# Launch the notebook
jupyter lab notebooks/nulltracer.ipynb
```

### Minimal example

```python
import nulltracer as nt
import matplotlib.pyplot as plt

# Render M87*-like parameters (a ≈ 0.94, θ ≈ 17°)
img, info = nt.render_frame(spin=0.94, inclination_deg=17.0, width=1024, height=1024)
print(f"Rendered in {info.render_ms:.0f} ms ({info.method}, {info.max_steps} steps)")

plt.imshow(img)
plt.axis("off")
plt.show()
```

## Features

- **Kerr–Newman metric** — rotating, electrically charged black holes in
  Boyer–Lindquist coordinates
- **Seven integration methods** — RK4, Dormand–Prince 8(7) adaptive,
  Yoshida 4th/6th-order symplectic, Kahan–Li 8th-order symplectic (BL and
  Kerr–Schild), Tao + Kahan–Li 8th symplectic
- **Production CUDA kernels** — float64 geodesic integration, one thread per
  pixel, compiled via CuPy `RawKernel`
- **Accretion disk physics** — Doppler boosting (g³ thin / g⁴ thick), colour
  temperature, multi-crossing accumulation
- **Shadow classification** — dedicated lightweight kernel for quantitative
  shadow measurement against Bardeen (1973) analytic curves
- **Single-ray diagnostics** — full trajectory, equatorial-plane crossings,
  g-factors, Novikov–Thorne flux per crossing
- **Equirectangular skymap backgrounds** — Gaia EDR3, Hipparcos/Tycho-2, or
  custom HDR environments
- **Airy disk bloom** — physically motivated diffraction post-processing

## Requirements

- **Python ≥ 3.10**
- **NVIDIA GPU** with CUDA support
- **CuPy** (`cupy-cuda12x` for CUDA 12.x)
- NumPy, SciPy, Matplotlib

Optional: Pillow (skymap loading), OpenEXR (HDR skymaps), JupyterLab

## API Reference

### Rendering

```python
# Full visual pipeline → (H, W, 3) uint8 sRGB image
img, info = nt.render_frame(spin, inclination_deg, **kwargs)

# Shadow classification → boolean (H, W) mask
mask, info = nt.classify_shadow(spin, inclination_deg, **kwargs)

# Single-ray diagnostic → dict with trajectory, crossings, physics
data = nt.trace_ray(spin=0.6, mode="impact_parameter", alpha=5.0, beta=0.0)
```

### Utilities

```python
nt.compile_all()                         # pre-compile all CUDA kernels
nt.available_methods()                   # list integrator names
nt.isco(a, Q=0.0)                       # ISCO radius
nt.r_plus(a, Q=0.0)                     # event horizon radius
nt.shadow_boundary(a, theta_obs, N=1000) # Bardeen analytic contour
nt.load_skymap("path/to/skymap.exr")    # load background texture
results, fig = nt.compare_integrators() # side-by-side benchmark
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `spin` | — | Dimensionless spin *a*, 0 ≤ *a* < 1 |
| `inclination_deg` | — | Observer inclination (0° = pole, 90° = equator) |
| `charge` | 0.0 | Dimensionless charge *Q* (Kerr–Newman) |
| `method` | `"rkdp8"` | Integration method |
| `obs_dist` | 40.0 | Observer distance in *M* |
| `fov` | 7.0 | Field of view (degrees) |
| `step_size` | 0.3 | Base affine-parameter step |
| `max_steps` | auto | Integration budget (auto-estimated if None) |

## Project Structure

```
nulltracer/
├── nulltracer/              # Python package
│   ├── __init__.py          # public API
│   ├── render.py            # render_frame(), classify_shadow()
│   ├── ray.py               # trace_ray() diagnostics
│   ├── isco.py              # ISCO calculations
│   ├── bloom.py             # Airy disk post-processing
│   ├── skymap.py            # equirectangular skymap loader
│   ├── compare.py           # shadow_boundary(), compare_integrators()
│   ├── _params.py           # RenderParams ctypes struct
│   ├── _kernel_utils.py     # kernel compilation cache
│   └── kernels/             # CUDA source (float64 geodesics)
│       ├── geodesic_base.cu # metric, geoRHS, ray init
│       ├── backgrounds.cu   # star field, checker, skymap
│       ├── disk.cu          # accretion disk emission
│       ├── ray_trace.cu     # single-ray entry points
│       └── integrators/     # per-method trace kernels
├── notebooks/               # Jupyter notebooks
│   └── nulltracer.ipynb     # main publication notebook
├── data/                    # skymaps and reference images
├── pyproject.toml           # package metadata
└── README.md
```

## Physics References

- Bardeen, J. M. (1973). Timelike and null geodesics in the Kerr metric.
  *Les Houches Summer School*, 215–240.
- Bardeen, Press & Teukolsky (1972). Rotating black holes.
  *ApJ* **178**, 347.
- Fuerst & Wu (2004). Radiation transfer of emission lines in curved
  space-time. *A&A* **424**, 733.
- Event Horizon Telescope Collaboration (2019). First M87 Event Horizon
  Telescope results. I–VI. *ApJL* **875**.

## License

MIT

## Author

Ethan Knox — [ethank5149@gmail.com](mailto:ethank5149@gmail.com) —
[github.com/ethank5149](https://github.com/ethank5149)
