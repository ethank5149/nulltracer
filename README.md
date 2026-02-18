# Nulltracer

**A GPU-accelerated Kerr-Newman black hole ray tracer** — an interactive WebGL application that visualizes the appearance of rotating and electrically charged black holes by tracing light paths (null geodesics) through curved spacetime.

## Overview

Nulltracer simulates the visual appearance of black holes as they would appear to an external observer. By tracing null geodesics (light paths) through Kerr-Newman spacetime using GPU-accelerated WebGL fragment shaders, the application renders realistic depictions of black hole phenomena including:

- The **black hole shadow** — the dark silhouette cast by the event horizon
- **Photon rings** — unstable light orbits around the black hole
- **Accretion disk emissions** — with Doppler boosting effects that make the approaching side appear hotter and brighter
- **Gravitational lensing** — the bending of light from background stars and structures
- **Frame-dragging effects** — the warping of spacetime by the black hole's rotation

The simulator supports both **Kerr black holes** (spinning) and **Kerr-Newman black holes** (spinning with electric charge), allowing exploration of how these parameters affect the visual appearance.

## Features

- **Real-time interactive rendering** — adjust black hole parameters and see results instantly
- **Kerr-Newman metric support** — model rotating, electrically charged black holes
- **Interactive controls** — modify spin parameter (a), electric charge (Q), and observer inclination (θ)
- **Multiple background modes** — Stars (cube-mapped), Checker pattern, or Color-mapped sphere
- **Accretion disk rendering** — with Doppler temperature boosting
- **Quality presets** — Low, Medium, High, and Ultra quality settings with performance tuning
- **Advanced controls** — configure integration steps, resolution scaling, step size, and observer distance
- **Integrator options** — switch between separated first-order equations or Hamiltonian integration
- **Full-screen capable** — for immersive visualization

## Usage

Simply open `index.html` in a modern web browser with WebGL 2.0 support. No installation or server required — the entire application runs locally in your browser.

```
Open nulltracer/index.html in your browser
```

## Controls

- **Spin (a):** Adjust black hole rotation from 0 (non-rotating Schwarzschild) to near-maximal values
- **Charge (Q):** Set electric charge parameter for Kerr-Newman black holes
- **Inclination (θ):** Change observer viewing angle relative to the black hole's rotation axis
- **Disk Temperature:** Adjust the color temperature of the accretion disk
- **Quality Preset:** Choose from Low/Medium/High/Ultra to balance visual fidelity and performance
- **Integration Method:** Select between Separated equations (faster) or Hamiltonian (more stable)
- **Integration Steps:** Control ray-tracing precision
- **Resolution Scaling:** Adjust internal rendering resolution for performance
- **Background Mode:** Switch between different background textures and patterns

## Technical Details

### Ray Tracing Approach

Nulltracer uses **WebGL 2.0 fragment shaders** to perform real-time ray tracing. Each pixel on screen corresponds to a light ray traced backward from the observer's eye through spacetime. The integration follows the equations of motion for null geodesics in the Kerr-Newman metric.

### Kerr-Newman Metric

The application solves the geodesic equations in Boyer-Lindquist coordinates, supporting both:
- **Kerr metric** — spinning (uncharged) black holes
- **Kerr-Newman metric** — spinning black holes with electric charge

### Integration Methods

1. **Separated First-Order Equations** (~40% faster) — optimized for performance
2. **Hamiltonian Integration** — uses conserved quantities for improved numerical stability

### Optimizations

- **μ = cos(θ) coordinate substitution** for robust pole handling
- **Adaptive stepping refinements** to balance accuracy and performance
- **Smooth regularization** techniques for numerical stability
- **Equal-area sphere tiling** to eliminate polar distortion in background rendering

## Version History

All versions are preserved as git tags. To view a previous release, use `git checkout v0.X`:

| Version | Tag | Date | Key Changes |
|---------|-----|------|-------------|
| v0.0.1 | `v0.0.1` | Initial | Kerr black hole with Hamiltonian RK4 integration |
| v0.1 | `v0.1` | | Refactored to separated first-order equations (~40% faster) |
| v0.2 | `v0.2` | | UX overhaul: legend, settings panel, multiple backgrounds |
| v0.3 | `v0.3` | | Equal-area sphere tiling (fixes polar pinching) |
| v0.4 | `v0.4` | | Numerical stability improvements |
| v0.5 | `v0.5` | | Smooth regularization and cube-map projection |
| v0.6 | `v0.6` | | μ=cos(θ) coordinate substitution for pole handling |
| v0.7 | `v0.7` | | Adaptive stepping refinements |
| v0.8 | `v0.8` | | Kerr-Newman extension (electric charge parameter) |
| v0.9 | `v0.9` | Current | Polished Kerr-Newman release |

**Accessing previous versions:**
```bash
git checkout v0.8    # View the Kerr-Newman release before current
git checkout v0.1    # View the first performance-optimized version
git checkout main    # Return to the latest version
```

## Requirements

- **Modern web browser** with WebGL 2.0 support
- **GPU acceleration** strongly recommended for real-time performance
- No external dependencies or server required — runs entirely in the browser

### Browser Compatibility

- Chrome/Chromium 56+
- Firefox 51+
- Safari 15+ (on macOS/iOS)
- Edge 79+

---

**License:** Check the repository for licensing information.

**Author:** Nulltracer project contributors
