# Nulltracer: Unified Roadmap

## Executive Summary

This document merges the **CUDA Migration & Advanced Techniques** plan and the **Composable Integrator Architecture** plan into a single unified roadmap, accounting for all work already completed. The goal: a composable, flag-driven CUDA kernel architecture supporting multiple spacetime metrics, coordinate systems, time parameterizations, and advanced rendering techniques — all exposed through the existing WebSocket-based browser UI.

---

## Completed Work (for reference)

The following items from the original plans are **already implemented** and do not appear in the roadmap below:

| Original Plan Item | Status | Evidence |
|---|---|---|
| **Phase 0B**: CUDA compute backend | ✅ Done | `server/renderer.py` uses CuPy `RawKernel`; Dockerfile is `nvidia/cuda:12.2.0-devel`; `cupy-cuda12x` in requirements |
| **Phase 0A**: WebGL client removal | ✅ Done | No `webgl-renderer.js`, `shader-generator.js`, or `isco-calculator.js`; `index.html` uses `<img id="server-frame">` |
| **Phase 0A server**: OpenGL removal | ✅ Done | No `shader.py`, `shader_base.py`, `backgrounds.py` Python files; no PyOpenGL in requirements |
| **Phase 0C**: WebSocket streaming | ✅ Done | `js/ws-client.js` + `/stream` endpoint in `server/app.py` |
| **Phase 1A**: Redshift & Beaming g-factor | ✅ Done | `compute_g_factor()` in `disk.cu`; `test_gfactor.py` validation against Schwarzschild analytic |
| **Phase 2B partial**: Extended phase space integrators | ✅ Done | `tao_yoshida4.cu`, `tao_yoshida6.cu`, `tao_kahan_li8.cu` — Tao splitting with Yoshida/KL8 composition |
| Novikov-Thorne disk model | ✅ Done | `normalized_nt_flux()` in `disk.cu` with Planck spectrum LUT |
| Single-ray tracing API | ✅ Done | `ray_trace.cu` with per-integrator entry points; `/ray` endpoint |
| Bench comparison endpoint | ✅ Done | `/bench` endpoint for side-by-side integrator comparison |

### Current Integrator Inventory

| Kernel | Coords | Time Param | Kahan | Corrector | Projection | Extended Phase Space |
|---|---|---|---|---|---|---|
| `rk4.cu` | BL | Affine | ❌ | ❌ | ✅ H-proj | ❌ |
| `yoshida4.cu` | BL | Affine | ❌ | ❌ | ✅ H-proj | ❌ |
| `rkdp8.cu` | BL | Affine/Adaptive | ❌ | ❌ | ✅ H-proj | ❌ |
| `kahanli8s.cu` | BL | Sundman/Mino | ✅ | ✅ Wisdom | ✅ H-proj | ❌ |
| `kahanli8s_ks.cu` | KS | Sundman/Mino | ✅ | ✅ Wisdom | ✅ H-proj KS | ❌ |
| `tao_yoshida4.cu` | BL | Affine | ❌ | ❌ | ✅ H-proj | ✅ Tao |
| `tao_yoshida6.cu` | BL | Affine | ❌ | ❌ | ✅ H-proj | ✅ Tao |
| `tao_kahan_li8.cu` | BL | Affine | ❌ | ❌ | ✅ H-proj | ✅ Tao |

**Problem**: 8 monolithic kernel files with ~80 lines of duplicated boilerplate each. Features like KS coordinates, Sundman time, and Kahan summation are locked to specific files.

---

## Phase 1: Composable Kernel Architecture (Foundation)

**Goal**: Refactor all 8 monolithic integrator kernels into a composable, flag-driven architecture. One generic integration loop + thin per-kernel flag files. Zero behavior change — all existing renders must be bit-identical.

### 1.1 Create Dispatch Layer Files

Create `server/kernels/dispatch/` directory with 8 dispatch files:

| File | Purpose | Initial Scope |
|---|---|---|
| `metric_dispatch.cu` | Select spacetime geometry | Kerr-Newman only — default, no branching needed yet |
| `coord_dispatch.cu` | Map generic names to BL or KS implementations | BL + KS — extract from existing `geodesic_base.cu` |
| `time_dispatch.cu` | Select time parameterization | Affine + Sundman Σ — extract from existing `adaptive_step.cu` |
| `accum_dispatch.cu` | Select accumulation strategy | Naive + Kahan — extract from existing `kahanli8s.cu` |
| `hamiltonian_dispatch.cu` | Select constraint formulation | Raw only initially |
| `correction_dispatch.cu` | Select corrector + projection | None + Wisdom corrector + H-projection |
| `shadow_dispatch.cu` | Shadow Hamiltonian analysis | None only initially |
| `ray_dispatch.cu` | Select geodesic type | Null only initially |

### 1.2 Create Generic Integration Loop

Create `server/kernels/integrators/trace_generic.cu` — the ONE shared integration loop. Before including, the caller defines `KERNEL_NAME`, `STEP_BODY(he)`, and `STEP_MULTIPLIER`.

The loop handles: ray init → coordinate transform → time init → integration loop → disk crossing → escape → post-processing → output.

### 1.3 Refactor steps.cu for Dispatch Compatibility

Update `server/kernels/integrators/steps.cu` to use dispatch macros (`geo_rhs`, `geo_velocity`, `geo_force`) instead of hardcoded `geoRHS`, `geoVelocity`, `geoForce`.

### 1.4 Convert Existing Kernels to Thin Flag Files

Convert all 8 existing kernel files to ~15-20 line flag files that set preprocessor defines and include `trace_generic.cu`:

- `rk4.cu` → BL + Affine + Naive + no corrector
- `yoshida4.cu` → BL + Affine + Naive + no corrector
- `rkdp8.cu` → BL + Affine/Adaptive + Naive + no corrector
- `kahanli8s.cu` → BL + Sundman Σ + Kahan + Wisdom + H-proj
- `kahanli8s_ks.cu` → KS + Sundman Σ + Kahan + Wisdom + H-proj KS
- `tao_yoshida4.cu` → BL + Affine + Naive + H-proj + Tao extended
- `tao_yoshida6.cu` → BL + Affine + Naive + H-proj + Tao extended
- `tao_kahan_li8.cu` → BL + Affine + Naive + H-proj + Tao extended

### 1.5 Refactor ray_trace.cu for Dispatch Compatibility

Update `server/kernels/ray_trace.cu` to use the same dispatch macros, or refactor it to share `trace_generic.cu` with per-ray output instead of per-pixel output.

### 1.6 Validation

- All existing tests pass unchanged
- Rendered images are bit-identical before/after refactor for all 8 integrators
- Bench endpoint produces identical results
- Single-ray tracing produces identical trajectories

---

## Phase 2: Unlock Cross-Cutting Feature Combinations

**Goal**: With the dispatch architecture in place, enable feature combinations that were previously impossible — any stepper can now use any coordinate system, time parameterization, and accumulation strategy.

### 2.1 New Sundman Time Variants

Add to `time_dispatch.cu`:
- `TIME_SUNDMAN_DELTA` — dτ = dλ/Δ, regularizes BL horizon
- `TIME_SUNDMAN_SIGMA_DELTA` — dτ = dλ/(Σ·Δ), double regularization
- `TIME_SUNDMAN_R2` — dτ = dλ/r², simplified far-field

### 2.2 Kahan Summation for All Integrators

With dispatch, `ACCUM_KAHAN` becomes available to `rk4`, `yoshida4`, `rkdp8`, and all Tao integrators — previously locked to `kahanli8s*` only.

### 2.3 KS Coordinates for All Integrators

With dispatch, `COORD_KS` becomes available to all integrators — previously locked to `kahanli8s_ks` only.

### 2.4 Wisdom Corrector for Yoshida Integrators

With dispatch, `CORRECT_WISDOM` becomes available to `yoshida4` and all Tao integrators — previously locked to `kahanli8s*` only.

### 2.5 Optimized Metric Fast Paths

Add compile-time fast paths in `metric_dispatch.cu`:
- `METRIC_SCHWARZSCHILD` — skip all spin/charge terms
- `METRIC_KERR` — skip Q² terms
- `METRIC_REISSNER_NORDSTROM` — skip frame-dragging

### 2.6 Python Dynamic Kernel Builder

Update `server/renderer.py`:
- Add `_FEATURE_FLAGS` registry mapping feature names to `#define` strings
- Add `_build_kernel_source()` that composes kernel source from stepper + feature flags
- Add `_validate_combination()` for incompatibility rules
- Update `_get_kernel()` to accept feature kwargs and cache by full feature tuple
- CuPy handles kernel caching automatically

### 2.7 API and UI Updates

- Update `RenderRequest` in `app.py` to accept optional feature flags
- Update frontend UI with dropdowns for coordinate system, time parameterization, accumulation
- Add presets: "Fast Preview", "Maximum Accuracy", "Near-Horizon"

### 2.8 Validation

- New combinations produce physically correct results
- Kahan summation improves Hamiltonian conservation for all integrators
- KS coordinates allow closer horizon approach for all integrators

---

## Phase 3: Core Accuracy Improvements

**Goal**: Implement the remaining advanced numerical techniques from the original CUDA plan.

### 3.1 Time Transformation — Mikkola/Preto-Tremaine (Technique #1)

- New integrator variant using transformed independent variable ds = f(r) · dλ
- Transformation function f(r) = 1/Δ makes RHS bounded everywhere
- Remove ad-hoc step-size clamp `clamp((r-rp)*0.4, 0.04, 1.0)` from all integrators
- Compatible with symplectic integrators via extended Hamiltonian
- **Validation**: Compare near-horizon geodesics against Weierstrass exact solutions (Phase 5)

### 3.2 Automatic Differentiation — Geodesic Deviation (Technique #9)

- Define `DualFloat64` struct in CUDA: `struct Dual { double val; double d_alpha; double d_beta; }`
- All metric functions get dual-number overloads
- Each integration step propagates geodesic AND its derivatives w.r.t. initial ray direction
- Output per ray: position + 2×2 Jacobian matrix d(r_disk, phi_disk) / d(alpha, beta)
- Jacobian determinant = magnification = 1/|det(J)|
- **Validation**: AD Jacobian vs finite-difference Jacobian — agree to ~12 digits

### 3.3 Regularized Hamiltonian

Implement `HAMILTONIAN_REGULARIZED` in `hamiltonian_dispatch.cu`:
- H_reg = Σ · H — eliminates 1/Σ singularity near ring singularity
- Equations become polynomial in (r, cosθ, sinθ)

### 3.4 Carter-Separated Hamiltonian

Implement `HAMILTONIAN_CARTER_SEPARATED` in `hamiltonian_dispatch.cu`:
- Decoupled r(τ) and θ(τ) ODEs
- Eliminates r-θ coupling errors
- Kerr family only

### 3.5 Carter Constant Projection

Implement `PROJECT_CARTER` in `correction_dispatch.cu`:
- Algebraic solve for p_θ given Q₀
- Preserves Carter constant to machine precision

### 3.6 Shadow Hamiltonian Analysis

Implement all `SHADOW_*` variants in `shadow_dispatch.cu`:
- `SHADOW_COMPUTE` — monitor shadow H̃ for diagnostics
- `SHADOW_PROJECT` — project onto H̃ = H̃₀ each step
- `SHADOW_BACKWARD_CORRECT` — single post-integration correction
- `SHADOW_FULL` — combined shadow projection + backward correction
- **Validation**: Hamiltonian conservation to machine precision over 10⁴ steps

### 3.7 Advanced Accumulation Strategies

Implement in `accum_dispatch.cu`:
- `ACCUM_DOUBLE_DOUBLE` — (hi, lo) pair arithmetic, ~32 digits for all ops
- `ACCUM_DEKKER_PRODUCT` — Dekker splitting for exact products + Kahan for sums

---

## Phase 4: New Coordinate Systems

**Goal**: Implement alternative coordinate charts for Kerr-Newman, each providing horizon-penetrating or singularity-free properties.

For each coordinate system, implement: `geoVelocity_*`, `geoForce_*`, `computeHamiltonian_*`, `projectHamiltonian_*`, and `transformBLto_*`.

### 4.1 Outgoing Kerr-Schild

`COORD_KS_OUTGOING` — mirror of ingoing KS with sign flips. Regular at past horizon; retarded time u.

### 4.2 Painlevé-Gullstrand

`COORD_PAINLEVE` — flat spatial slices; "rain" observers. Regular at future horizon.

### 4.3 Doran Coordinates

`COORD_DORAN` — rotating Painlevé-Gullstrand for Kerr. Flat spatial slices.

### 4.4 Natário River Model

`COORD_NATARIO` — explicit flow velocity field. Intuitive visualization of frame-dragging.

### 4.5 Tortoise Coordinates

`COORD_TORTOISE` — compactified radial; horizon at r* → −∞. Requires event horizon.

### 4.6 Kerr-Schild Cartesian

`COORD_KS_CARTESIAN` — 8D state (x,y,z,px,py,pz + t,pt). No coordinate singularities at all; no pole reflection needed.

### 4.7 Cook-Scheel Harmonic

`COORD_HARMONIC` — horizon-penetrating + asymptotically Minkowski.

---

## Phase 5: Exact Solutions & Validation Oracle

**Goal**: Implement Weierstrass elliptic function solutions for Kerr (Q=0) geodesics as the ultimate validation oracle.

### 5.1 Carlson Symmetric Elliptic Integrals

New kernel module `server/kernels/analytic/carlson.cu`:
- Implements RF, RJ, RD, RC in float64
- These are the building blocks for all elliptic function evaluations

### 5.2 Weierstrass Elliptic Functions

New kernel module `server/kernels/analytic/weierstrass.cu`:
- Weierstrass P, zeta, sigma functions
- Built on Carlson integrals

### 5.3 Geodesic Classification

New kernel module `server/kernels/analytic/geodesic_classify.cu`:
- Classify geodesic type from conserved quantities (b, q)
- Determine root structure of radial/polar potentials

### 5.4 Analytic Kerr Geodesic Solver

New kernel module `server/kernels/analytic/kerr_analytic.cu`:
- Compute r(λ), θ(λ), φ(λ) from elliptic functions
- Falls back to numerical integration for Q ≠ 0

### 5.5 Validation

- Compare analytic solutions against all numerical integrators at various (a, θ_obs, α, β)
- Quantify numerical error per integrator per feature combination
- This IS the validation oracle for all other phases

---

## Phase 6: New Spacetime Metrics

**Goal**: Implement alternative spacetimes beyond Kerr-Newman.

For each metric, implement the full set of geodesic functions + update `RenderParams` with new parameters.

### 6.1 Kerr-Newman-de Sitter

`METRIC_KERR_NEWMAN_DS` — cosmological constant Λ. Δ_r = r² − 2r + a² + Q² − (Λ/3)r⁴. Creates cosmological horizon.

### 6.2 Bardeen Regular Black Hole

`METRIC_BARDEEN` — M(r) = Mr³/(r² + ℓ²)^(3/2). No singularity.

### 6.3 Hayward Regular Black Hole

`METRIC_HAYWARD` — M(r) = Mr³/(r³ + 2Mℓ²). de Sitter core.

### 6.4 Morris-Thorne Wormhole

`METRIC_WORMHOLE_MT` — no horizon. Rays pass through throat.

### 6.5 Teo Rotating Wormhole

`METRIC_WORMHOLE_TEO` — rotating Morris-Thorne with frame-dragging.

### 6.6 Kerr-Newman-NUT

`METRIC_KERR_NEWMAN_NUT` — gravitomagnetic monopole. Σ = r² + (n + a·cosθ)².

### 6.7 Kerr-Sen (String Theory)

`METRIC_KERR_SEN` — heterotic string BH. Σ = r² + 2br + a²cos²θ.

### 6.8 Johannsen-Psaltis Parametric

`METRIC_JOHANNSEN_PSALTIS` — standard EHT framework for testing GR deviations.

### 6.9 Einstein-Dilaton-Gauss-Bonnet

`METRIC_GAUSS_BONNET` — leading quantum gravity correction.

### 6.10 Boson Star (Tabulated)

`METRIC_BOSON_STAR` — requires texture memory for metric interpolation. No horizon.

### 6.11 Extended RenderParams

Update the CUDA `RenderParams` struct and Python ctypes mirror with new metric parameters:
- `lambda_cosmo`, `nut_charge`, `dilaton_param`
- `jp_alpha13`, `jp_alpha22`, `jp_alpha52`, `jp_epsilon3`
- `gb_alpha`, `reg_length`, `throat_radius`, `wh_shape_param`
- `particle_mass`, `particle_charge`

---

## Phase 7: Advanced Rendering Techniques

**Goal**: Implement sophisticated rendering features that build on the AD infrastructure from Phase 3.

### 7.1 Transfer Functions (Technique #4)

- New CUDA kernel `kernels/transfer/precompute.cu` — traces dense ray grid, stores (r_disk, φ_disk, g_factor, magnification, n_crossings) per pixel
- New Python module `server/transfer.py` — manages computation, storage (HDF5/npz), interpolation
- Cached on disk keyed on (a, Q, θ_obs, fov, resolution)
- Rendering with new emission model becomes sub-millisecond 2D lookup
- New API endpoint: `POST /render_transfer`
- **Validation**: Transfer function renders vs direct ray-traced renders — pixel-identical

### 7.2 Importance Sampling (Technique #5)

- Two-pass rendering pipeline:
  1. Coarse pass: trace rays on low-res grid, compute magnification from AD Jacobian
  2. Adaptive pass: redistribute threads proportional to magnification
- Requires parallel prefix sum (CUB library) for thread redistribution
- Output: non-uniform sample set splatted into final image
- **Validation**: Convergence rate comparison vs uniform sampling

### 7.3 Spectral Ray Bundling (Technique #6)

- Each ray carries 2 deviation vectors (bundle principal axes) — from AD
- Bundle solid angle = |cross product of deviation vectors| at each step
- Proper flux integration: F = I · dΩ
- Physically correct anti-aliasing without supersampling
- **Validation**: Flux conservation — total flux through image plane = total disk luminosity

### 7.4 Polarization Transport (Technique #7)

- New CUDA function `compute_walker_penrose()` in `kernels/polarization.cu`
- For Kerr (Q=0): Walker-Penrose constant exactly conserved
- For Kerr-Newman (Q≠0): modified Walker-Penrose with charge correction
- Output per pixel: Stokes vector (I, Q, U, V) — 4 channels
- New API parameter: `polarization: bool`
- **Validation**: Walker-Penrose constant conservation along geodesic

---

## Phase 8: Ray Type Extensions

**Goal**: Support non-null geodesics.

### 8.1 Timelike Geodesics

`RAY_TIMELIKE` — massive particle with H = −½μ². Update constraint from H=0 to H=−½μ². Add `particle_mass` to RenderParams.

### 8.2 Charged Particle Geodesics

`RAY_CHARGED` — Lorentz force in KN electromagnetic field. Requires `particle_charge` parameter. Only valid for Kerr-Newman family metrics.

### 8.3 Spacelike Geodesics

`RAY_SPACELIKE` — tachyonic with H = +½. Primarily for visualization/exploration.

### 8.4 Proper Time Parameterization

`TIME_PROPER` — proper time dτ for massive particles. Requires `RAY_TIMELIKE`.

---

## Target Architecture

```mermaid
graph TD
    subgraph Browser Client
        UI[Parameter Controls<br/>Sliders + Dropdowns]
        DISPLAY[img element<br/>Full-viewport display]
        WSC[WebSocket Client<br/>Binary frame streaming]
    end

    subgraph FastAPI Server
        API[/render + /stream + /ray + /bench]
        CACHE[LRU Image Cache]
        BUILDER[Dynamic Kernel Builder<br/>Feature flag composition]
        TRANSFER[Transfer Function Cache]
    end

    subgraph CUDA Compute - RTX 3090
        subgraph Dispatch Layer
            METRIC[Metric Dispatch<br/>14 spacetimes]
            COORD[Coordinate Dispatch<br/>9 charts]
            TIME[Time Dispatch<br/>8 parameterizations]
            ACCUM[Accumulation Dispatch<br/>4 strategies]
            HAM[Hamiltonian Dispatch<br/>3 formulations]
            CORR[Correction Dispatch<br/>Wisdom + H-proj + Carter-proj]
            SHADOW[Shadow H Dispatch<br/>5 modes]
            RAY[Ray Type Dispatch<br/>4 types]
        end

        GENERIC[trace_generic.cu<br/>Shared integration loop]
        STEPS[Step functions<br/>RK4, Yoshida4/6/8, KL8, RKDP8, Tao variants]

        subgraph Advanced Features
            AD[Auto-Diff Layer<br/>Dual numbers - geodesic deviation]
            ANALYTIC[Analytic Solver<br/>Weierstrass elliptic - Kerr exact]
            POLAR[Polarization Transport<br/>Walker-Penrose]
            IMPORTANCE[Importance Sampler<br/>Adaptive thread distribution]
            BUNDLE[Ray Bundling<br/>Solid angle tracking]
        end
    end

    UI -->|onChange| WSC
    WSC -->|JSON params| API
    API --> CACHE
    API --> BUILDER
    BUILDER -->|compile flags| GENERIC
    GENERIC --> STEPS
    GENERIC --> METRIC
    GENERIC --> COORD
    GENERIC --> TIME
    GENERIC --> ACCUM
    GENERIC --> HAM
    GENERIC --> CORR
    GENERIC --> SHADOW
    GENERIC --> RAY
    STEPS --> AD
    ANALYTIC --> API
    AD --> IMPORTANCE
    AD --> BUNDLE
    AD --> POLAR
    API -->|Binary JPEG/WebP| WSC
    WSC --> DISPLAY
    API --> TRANSFER
```

---

## Validation Strategy

| Phase | Validation Method |
|---|---|
| 1 — Composable refactor | Bit-identical renders before/after for all 8 integrators |
| 2 — Cross-cutting combos | Hamiltonian conservation improves with Kahan; KS allows closer horizon |
| 3.1 — Time transform | Near-horizon geodesics match Weierstrass exact (Phase 5) |
| 3.2 — Auto-diff | AD Jacobian vs finite-difference Jacobian — agree to ~12 digits |
| 3.6 — Shadow H | Hamiltonian conservation to machine precision over 10⁴ steps |
| 5 — Weierstrass | Self-validating oracle — compare against all numerical integrators |
| 6 — New metrics | Shadow shape matches published results for each spacetime |
| 7.1 — Transfer functions | Transfer renders vs direct ray-traced — pixel-identical |
| 7.3 — Ray bundling | Flux conservation: total image flux = total disk luminosity |
| 7.4 — Polarization | Walker-Penrose constant conservation along geodesic |

---

## Incompatibility Rules

These combinations are physically or numerically invalid and must be rejected by the Python validation layer:

| Requires | Forbids | Reason |
|---|---|---|
| `HAMILTONIAN_CARTER_SEPARATED` | Non-Kerr-family metrics | Carter constant only exists for Kerr-Newman family |
| `TIME_PROPER` | `RAY_NULL` | Proper time is zero for null geodesics |
| `CORRECT_WISDOM` | `rk4`, `rkdp8` | Symplectic corrector only for symplectic integrators |
| `SHADOW_*` (non-none) | `rk4`, `rkdp8` | Shadow Hamiltonian requires symplectic integrator |
| `SHADOW_PROJECT` or `SHADOW_FULL` | `PROJECT_HAMILTONIAN` | Shadow projection supersedes standard projection |
| `METRIC_WORMHOLE_*` | `COORD_TORTOISE` | Tortoise coordinates require an event horizon |
| `METRIC_BOSON_STAR` | All non-BL coords | Tabulated metric only in BL coordinates |
| `RAY_CHARGED` | Wormhole/boson star/Bardeen/Hayward | Lorentz force requires known EM field |
| `COORD_KS_CARTESIAN` | `HAMILTONIAN_CARTER_SEPARATED` | Carter separation uses (r, θ), not Cartesian |

---

## References

### Coordinate Systems
- Kerr 1963, Phys. Rev. Lett. 11:237
- MTW 1973, "Gravitation," Box 33.2
- Visser 2007, arXiv:0706.0622
- Doran 2000, Phys. Rev. D 61:067503
- Natário 2009, arXiv:0903.3779
- Cook & Scheel 1997, Phys. Rev. D 56:4775

### Spacetime Metrics
- Carter 1968, Phys. Rev. 174:1559
- Bardeen 1968, GR5 Conference
- Hayward 2006, Phys. Rev. Lett. 96:031103
- Morris & Thorne 1988, Am. J. Phys. 56:395
- Teo 1998, Phys. Rev. D 58:024014
- Johannsen & Psaltis 2011, Phys. Rev. D 83:124015
- Sen 1992, Phys. Rev. Lett. 69:1006

### Time Parameterizations
- Mino 2003, Phys. Rev. D 67:084027
- Sundman 1913, Acta Math. 36:105
- Mikkola 1999, Celest. Mech. Dyn. Astron. 74:287
- Preto & Tremaine 1999, AJ 118:2532

### Numerical Methods
- Kahan & Li 1997, Math. Comp. 66:1089
- Kahan 1965, Comm. ACM 8(1):40
- Wisdom 2006, Astron. J. 131:2294
- Hairer, Lubich & Wanner 2006, "Geometric Numerical Integration"
- Reich 1999, SIAM J. Numer. Anal. 36:1549
- Tao 2016, Phys. Rev. E 94:043303
- Wang, Huang & Wu 2021, ApJ 907:66
- Dekker 1971, Numer. Math. 18:224

### Advanced Techniques
- Cunningham 1975, ApJ 202:788 (g-factor validation)
- Page & Thorne 1974, ApJ 191:499 (thin disk model)
- Novikov & Thorne 1973, "Black Holes" (disk flux)
- Chandrasekhar 1983, "The Mathematical Theory of Black Holes"
- Bardeen, Press & Teukolsky 1972, ApJ 178:347
