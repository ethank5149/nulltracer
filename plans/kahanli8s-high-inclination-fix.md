# Plan: Fix kahanli8s High-Inclination Rendering Artifacts

## Problem Statement

Both `kahanli8s` (Boyer-Lindquist) and `kahanli8s_ks` (Kerr-Schild) integrators produce a distorted black hole shadow with a black elliptical patch at high observer inclinations (θ > ~70°). The artifact is worst at a=0 (Schwarzschild) and θ=89° (near edge-on). At θ ≤ 45°, the integrators produce correct results matching rkdp8.

## Root Cause Analysis

### The Sundman Step Budget Problem

The kahanli8s integrators use a **Sundman/Mino time transformation** (dτ = dλ/Σ) with a **fixed Mino-time step** Δτ computed from a radial-only budget estimate:

```
τ_needed = 2 × (1/r_ph − 1/r_esc)    // radial round-trip budget
Δτ = (1 + step_size) × τ_needed / N   // fixed Mino step
Δλ = Δτ × Σ                           // physical step (varies with position)
```

This budget assumes **radial-dominated motion**. At high inclination, rays that pass near the coordinate pole (θ ≈ 0 or θ ≈ π) have significant angular motion that consumes Mino time without making radial progress. These rays exhaust their integration budget before escaping, producing black pixels.

### Why Other Integrators Work

- **rkdp8**: Uses adaptive error control — automatically reduces step size when angular rates are large
- **tao_yoshida4/6**: Uses `adaptive_step_tao()` with a tighter upper clamp (max he = 1.0) and geometric heuristic
- **rk4**: Uses `adaptive_step_rk4()` with similar geometric heuristic

### Evidence from Reference Codes

**ipole** (Mościbrodzka et al.) and **raptor** (Bronzwaer et al.) both use the **Dolence & Mościbrodzka (2009)** step sizing method with three independent constraints:

```c
// raptor/integrator.c:109
dlx1 = STEPSIZE / (|U^r| + ε)                        // radial constraint
dlx2 = STEPSIZE × min(X^θ, 1−X^θ) / (|U^θ| + ε)    // θ constraint with pole factor
dlx3 = STEPSIZE / (|U^φ| + ε)                        // φ constraint
dl = harmonic_mean(dlx1, dlx2, dlx3)                 // combined step
```

The critical insight is `dlx2`: the step is limited by **both** the angular velocity AND the distance to the nearest pole.

### Evidence from Literature

**Wu et al. (2024)** — "Explicit symplectic integrators with adaptive time steps in curved spacetimes" describes exactly our situation. Their key findings:

1. The Sundman transformation g = Σ/r² (our current approach) gives g → 1 for large r, meaning **no adaptive step sizing** at large distances — the old time step Δτ ≈ Δw (constant)
2. They propose using g₂ = Σ (i.e., g = r²·(Σ/r²) = Σ) which gives Δτ ≈ r²·Δw — adaptive steps that grow with r²
3. However, they note this is "cumbersome to implement" because the smaller step selection requires a much shorter new time step h
4. Their solution: the **Preto & Saha (2009) Φ-variable technique** — introduce a conjugate momentum Φ that acts as a rescaled time variable, adjusting steps without breaking symplecticity
5. The parameter j ≈ observer distance is recommended for ray-tracing

**Hairer (1997)** — "Variable time step integration with symplectic methods" provides the theoretical foundation: the Sundman transformation dt/dT = s(p,q) makes the system Hamiltonian only if s is constant along solutions. The "meta-algorithm" adds a perturbation term K = s(p,q)(H − H₀) that vanishes on the true solution but makes the system Hamiltonian.

## Proposed Solution

### Approach: Adopt the Wu et al. / Preto & Saha Adaptive Φ-Variable

This is the theoretically correct approach for symplectic integrators. It:
- Preserves symplecticity (unlike ad-hoc step clamping)
- Provides genuine adaptive time stepping (smaller steps near horizon AND near poles)
- Has negligible computational overhead (only 2 additional steps per integration step)
- Is proven in the literature for exactly our use case (Kerr null geodesics)

### Implementation Steps

The adaptive method AS₂ wraps the existing S₂ integrator with two additional steps:

```
Step 1: Advance Φ by −g·h·g^rr·p_r / (2r)     // half-step Φ update
Step 2: Advance τ by g·h / (2Φ)                 // half-step time update
Step 3: Evolve r, θ, p_r, p_θ using S₂(h/Φ)    // existing integrator with scaled step
Step 4: Reiterate Step 2                         // second half-step time
Step 5: Reiterate Step 1                         // second half-step Φ
```

Where:
- Φ is initialized to j/r₀ (with j ≈ obs_dist)
- g is the existing Sundman transformation function (Σ/r² for BL, similar for KS)
- h is the fixed new-time step
- The physical step becomes h/Φ instead of h, providing adaptive sizing

## Files to Modify

### CUDA Kernels
- `server/kernels/integrators/adaptive_step.cu` — Add Φ-variable helper functions
- `server/kernels/integrators/kahanli8s.cu` — Wrap integration loop with Φ updates
- `server/kernels/integrators/kahanli8s_ks.cu` — Same for Kerr-Schild variant

### No Other Files Need Changes
- The RenderParams struct does NOT need new fields (j can be derived from obs_dist)
- The API does NOT need new parameters
- The client UI does NOT need changes
- Other integrators are NOT affected

## References

1. Wu, X. et al. (2024) — "Explicit symplectic integrators with adaptive time steps in curved spacetimes" — Primary reference for the AS₂ algorithm
2. Preto, M. & Saha, P. (2009) — "Symplectic integrator with adaptive time steps" — The Φ-variable technique
3. Hairer, E. (1997) — "Variable time step integration with symplectic methods" — Theoretical foundation
4. Mikkola, S. (1997) — "Time transformations for constructing efficient symplectic algorithms" — Original Sundman approach
5. ipole (Mościbrodzka et al.) — Reference implementation of pole-cautious step sizing
6. raptor (Bronzwaer et al.) — Reference implementation of Dolence & Mościbrodzka step sizing
