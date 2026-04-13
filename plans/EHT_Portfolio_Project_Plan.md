# Nulltracer × EHT: Portfolio Project Plan

## The Pitch (What a Hiring Manager Sees)

> "Built a GPU-accelerated general-relativistic ray tracer from scratch using CUDA, then validated it against Event Horizon Telescope observations of M87* and Sgr A*. Embedded the Kerr metric geodesic equations directly in a self-contained CuPy notebook, performed parameter sweeps over black hole spin, extracted quantitative shadow observables, and demonstrated agreement with published EHT measurements to within observational uncertainty."

That's the one-liner on your resume. Here's how to get there.

### Key Upgrade: GPU-Native Notebook

Rather than just calling the Nulltracer server API, the notebook embeds the actual physics — the Kerr metric, the Hamiltonian geodesic equations, the RK4 integrator, and the ISCO calculation — directly as a CuPy CUDA RawKernel. This means:
- The notebook is **completely self-contained** (any machine with CuPy + NVIDIA GPU can run it)
- It demonstrates **CUDA/GPU programming proficiency** alongside the physics
- The math is **visible and explained**, not hidden behind an API call
- It shows you understand the code you wrote, not just how to call it

---

## Published EHT Measurements (Your Ground Truth)

### M87* (2017 observations, published April 2019)
- **Mass:** (6.5 ± 0.7) × 10⁹ M☉
- **Distance:** 16.8 Mpc
- **Ring diameter:** 42 ± 3 μas
- **Circularity deviation:** ΔC ≲ 0.10
- **Brightness asymmetry:** southern crescent (approaching jet side)
- **Inclination:** ~17° (from jet axis observations)
- **Spin:** excluded a = 0; estimates range 0.5–0.94 depending on method
- **Flux ratio (ring to central depression):** ≳ 10:1

### Sgr A* (2017 observations, published May 2022)
- **Mass:** 4.0⁺¹·¹₋₀.₆ × 10⁶ M☉
- **Distance:** ~8.28 kpc
- **Shadow angular diameter:** 48.7 ± 7 μas
- **Emission ring diameter:** 51.8 ± 2.3 μas
- **Inclination:** ~50° (less constrained than M87*)
- **Spin:** ~0.65–0.9 (model-dependent)

---

## Deliverables

1. **Cleaned-up Nulltracer GitHub repo** (README polish, license, remove dead code)
2. **Jupyter notebook:** `eht_comparison.ipynb` — the main portfolio piece
3. **Updated resume** with a "Projects" section featuring this work

---

## Notebook Structure (6 Sections)

### Section 1: Introduction & Motivation (~0.5 page)
- What is the EHT? What did it observe?
- What is Nulltracer? (Brief, link to repo)
- Goal: validate a forward ray-tracing model against real observations

### Section 2: The Ray Tracer (~1 page)
- Brief description of Nulltracer's physics (Kerr metric, geodesic integration)
- Show a few rendered images at different spin values to demonstrate the tool
- Render a grid: spin a ∈ {0, 0.3, 0.6, 0.9, 0.998} × inclination θ ∈ {17°, 50°, 90°}
- This section establishes credibility — you built a serious simulation

### Section 3: Shadow Observable Extraction (~1.5 pages)
This is the **data science core** — you're extracting quantitative measurements from images.

For each rendered image, compute:
- **Shadow diameter** — fit a circle or ellipse to the photon ring boundary
  - Method: threshold the image, find the ring contour, fit an ellipse (OpenCV or scipy)
- **Circularity** — ratio of minor to major axis of the fitted ellipse, or ΔC deviation
- **Brightness asymmetry** — ratio of flux in the bright crescent vs. opposite side
  - Divide the ring into angular sectors, compute integrated brightness per sector
- **Central depression ratio** — flux inside shadow vs. flux on ring

Write reusable functions for each metric. Show the pipeline on a single example image first, then apply to the full grid.

### Section 4: Parameter Sweep & Comparison to M87* (~2 pages)
- Render a dense grid at M87*'s inclination (θ = 17°):
  - Spin: a ∈ [0, 0.998] in steps of ~0.05
  - Fixed inclination: 17°
- For each render, extract shadow observables
- Plot shadow diameter vs. spin — overlay the EHT measurement band (42 ± 3 μas)
- Plot circularity vs. spin — overlay ΔC ≲ 0.10
- Convert from pixel units to μas using the known M87* mass and distance:
  - θ_shadow = (d_shadow / M) × (GM/c²) / D
  - where M = 6.5 × 10⁹ M☉, D = 16.8 Mpc
- Identify the spin range consistent with ALL observables simultaneously
- Compare your constraint to the EHT's published constraint
- Side-by-side: your best-fit render vs. the published EHT image

### Section 5: Comparison to Sgr A* (~1 page)
- Repeat the parameter sweep at θ = 50° (Sgr A*'s estimated inclination)
- Convert to μas using Sgr A*'s mass and distance
- Overlay the EHT measurement band (48.7 ± 7 μas for shadow, 51.8 ± 2.3 μas for ring)
- Note: Sgr A* is more variable and harder to constrain — acknowledge this honestly

### Section 6: Discussion & Conclusions (~0.5 page)
- Summary of results
- Limitations of the model (no GRMHD, simplified disk model, no scattering)
- What would be needed to go further (visibility fitting, GRMHD, etc.)
- Link to the full Nulltracer codebase

---

## Key Python Libraries Needed

```
cupy (cupy-cuda12x), numpy, scipy, matplotlib, Pillow
```

Optional but nice: `astropy` (for unit conversions), `scikit-image`, `seaborn`.

**Hardware requirement:** NVIDIA GPU with CUDA support. Any modern GPU works — even a laptop GTX 1650 will render each frame in under a second. The full parameter sweep (~20 spin values × 2 targets) takes a few minutes total.

---

## Timeline

### Days 1–3: Repo cleanup
- Clean up README (it's already good — just make sure it's portfolio-ready)
- Add a LICENSE file
- Remove any debug/scratch files
- Make sure Docker build works cleanly
- Add a screenshot or GIF to the README showing the raytracer in action

### Days 4–7: Notebook sections 1–3
- Write intro and raytracer description
- Render the image grid (you'll need your GPU server for this)
- Build the shadow extraction pipeline — this is the most important code to get right
- Test the extraction functions thoroughly on known cases (e.g., Schwarzschild should give a known shadow radius of √27 M)

### Days 8–12: Notebook sections 4–5
- Run the full parameter sweeps
- Build the comparison plots
- Get the unit conversions right (this is where the physics really matters)
- Create the final side-by-side comparison figures

### Days 13–14: Polish
- Write section 6
- Clean up all plots (proper axis labels, consistent style, publication-quality)
- Add markdown narrative connecting each section
- Proofread everything
- Update resume to include this project

---

## Tips for Maximum Impact

1. **Matplotlib style matters.** Use a dark background theme that matches the astrophysics aesthetic. `plt.style.use('dark_background')` as a starting point, then customize.

2. **Show your code, but not all of it.** The notebook should show the analysis pipeline clearly, but the CUDA kernel code should live in the repo, not in the notebook. Import Nulltracer as a library.

3. **Narrate like a scientist.** Each section should have markdown cells explaining what you're doing and why. A hiring manager should be able to read just the markdown and understand the project.

4. **Be honest about limitations.** Acknowledging that your model is simplified (no GRMHD, no scattering) shows maturity. Every real physicist knows models have limits.

5. **The comparison plot is the hero.** The single most important figure is the one showing your shadow diameter vs. spin curve with the EHT measurement band overlaid. Make it beautiful.

---

## What This Project Demonstrates to Employers

| Skill | Evidence |
|-------|----------|
| GPU/CUDA programming | CuPy RawKernel implementing geodesic equations directly on GPU |
| Scientific computing | Null geodesic integration, Kerr metric, numerical methods (RK4) |
| Physics & mathematics | General relativity, Hamiltonian mechanics, differential equations, conserved quantities |
| Data analysis | Extracting quantitative measurements from image data (ellipse fitting, contour extraction) |
| Software engineering | Full Nulltracer codebase: Docker, REST API, caching, clean architecture |
| Data visualization | Publication-quality matplotlib plots, parameter space exploration |
| Scientific communication | Clear notebook narrative, LaTeX equations, honest discussion of limitations |
| Python proficiency | NumPy, SciPy, CuPy, Matplotlib, image processing pipeline |
| C/C++/CUDA | High-performance GPU kernels (both in notebook and Nulltracer repo) |
