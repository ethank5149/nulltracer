# Tao Extended Phase Space + Yoshida Integrators Design

## Problem

The Kerr-Newman Hamiltonian is non-separable: H = T(q,p) + V(q), not H = T(p) + V(q).
Standard Yoshida drift-kick splitting degrades to ~2nd order on non-separable systems.
Tao's method (2016) embeds the system in a doubled phase space where the extended Hamiltonian IS separable.

## Tao's Method Applied to Kerr-Newman Geodesics

### Extended Phase Space

Original state: (r, θ, φ, p_r, p_θ) — 5 variables
Extended state: (r, θ, φ, p_r, p_θ, r̃, θ̃, φ̃, p̃_r, p̃_θ) — 10 variables

The extended Hamiltonian is:

```
H_ext = H_A(q, p̃) + H_B(q̃, p) + ½ω²|q - q̃|² + ½ω²|p - p̃|²
```

where:
- H_A(q, p̃) = H(r, θ, φ, p̃_r, p̃_θ) — original H with shadow momenta
- H_B(q̃, p) = H(r̃, θ̃, φ̃, p_r, p_θ) — original H with shadow positions
- Coupling: ½ω²[(r-r̃)² + (θ-θ̃)² + (φ-φ̃)² + (p_r-p̃_r)² + (p_θ-p̃_θ)²]

### Three-Part Splitting

The extended Hamiltonian splits into three exactly-solvable pieces:

1. **H_A flow**: Advance (r, θ, φ) using geoVelocity(r, θ, p̃_r, p̃_θ) and advance (p̃_r, p̃_θ) using geoForce(r, θ, p̃_r, p̃_θ)
   - This is separable because positions (r,θ,φ) evolve via ∂H_A/∂p̃ and shadow momenta (p̃_r,p̃_θ) evolve via -∂H_A/∂q
   - Wait — this is still non-separable within H_A itself! H_A(q, p̃) has the same coupling structure.

### Corrected Understanding

Actually, Tao's method works differently. The key insight is:

**H_A(q, p̃)** — when we flow under H_A, we update q using ∂H_A/∂p̃ (which depends on q and p̃) and update p̃ using -∂H_A/∂q (which depends on q and p̃). This is STILL non-separable.

The actual Tao splitting is:

```
H_ext = [T_A(q, p̃) + V_B(q̃)] + [T_B(q̃, p) + V_A(q)] + coupling
```

where T and V are the kinetic and potential parts of H. But this doesn't help either because T depends on both q and p.

### Correct Tao Splitting for General Non-Separable H

Tao's actual method (Phys. Rev. E 94, 043303, 2016) uses a different decomposition. For a general H(q,p):

Split H into two arbitrary pieces: H = H_a(q,p) + H_b(q,p)

Then construct:
```
H_ext = H_a(q, p̃) + H_b(q̃, p) + ½ω²(|q-q̃|² + |p-p̃|²)
```

The flow of H_a(q, p̃) only moves (q, p̃) — it does NOT move (q̃, p).
The flow of H_b(q̃, p) only moves (q̃, p) — it does NOT move (q, p̃).
The coupling flow rotates (q,q̃) and (p,p̃) as harmonic oscillators.

For the Kerr-Newman Hamiltonian:
```
H = ½[g^rr(r,θ)·p_r² + g^θθ(r,θ)·p_θ²] + ½[g^tt(r,θ) + 2g^tφ(r,θ)·b + g^φφ(r,θ)·b²]
     \___________ T(q,p) ___________/     \________________ V(q) ________________/
```

Natural split: H_a = T(q,p) = ½[g^rr·p_r² + g^θθ·p_θ²], H_b = V(q) = ½[g^tt + 2g^tφ·b + g^φφ·b²]

Then:
- **H_a(q, p̃)** = ½[g^rr(r,θ)·p̃_r² + g^θθ(r,θ)·p̃_θ²] — this flow moves (r, θ, φ, p̃_r, p̃_θ)
- **H_b(q̃, p)** = ½[g^tt(r̃,θ̃) + 2g^tφ(r̃,θ̃)·b + g^φφ(r̃,θ̃)·b²] — this flow moves (r̃, θ̃, φ̃, p_r, p_θ)

Now H_a(q, p̃) is STILL non-separable (g^rr depends on q). But the key is that each piece can be integrated with a standard ODE solver (even a simple Euler step) and the overall composition with the coupling achieves the desired order.

Actually, re-reading Tao more carefully: the method requires that each H_a and H_b be **integrable** (exactly solvable), not just separable. For general non-separable Hamiltonians, Tao proposes using the **gradient splitting**:

```
H_a(q,p) = ½|p|² (free particle kinetic energy)
H_b(q,p) = H(q,p) - ½|p|² (everything else)
```

But this doesn't apply to our curved-space system where there's no flat-space kinetic energy.

### Practical Tao Implementation for Kerr-Newman

The most practical approach for our system follows Tao's Algorithm 1 (Section II.A):

Given H(q,p) non-separable, define:
- H_A(q, p̃) = H(q, p̃) — full Hamiltonian evaluated at real positions, shadow momenta
- H_B(q̃, p) = H(q̃, p) — full Hamiltonian evaluated at shadow positions, real momenta
- H_C = ½ω²(|q-q̃|² + |p-p̃|²) — harmonic coupling

The three flows are:
1. **φ_A^t**: Integrate Hamilton's equations for H_A(q, p̃) — this moves (q, p̃) while (q̃, p) are frozen
2. **φ_B^t**: Integrate Hamilton's equations for H_B(q̃, p) — this moves (q̃, p) while (q, p̃) are frozen
3. **φ_C^t**: Exact harmonic oscillator rotation of (q,q̃) and (p,p̃)

Each of φ_A and φ_B is a standard ODE integration of the SAME geodesic equations, just with different variable assignments. They don't need to be exactly solvable — they can use a simple leapfrog/Verlet step internally.

The Yoshida composition then operates on the three-way splitting:
```
Ψ(h) = φ_A(d₁h) ∘ φ_B(d₁h) ∘ φ_C(w₁h) ∘ φ_A(d₂h) ∘ φ_B(d₂h) ∘ φ_C(w₂h) ∘ ...
```

Wait — this is getting complex. Let me re-read Tao's paper more carefully.

### Tao's Actual Algorithm (Corrected)

From Tao (2016), the method for a general non-separable H(q,p) is:

**Step 1**: Choose ω (coupling strength). Tao recommends ω = O(1/h).

**Step 2**: Initialize shadow variables: q̃ = q, p̃ = p.

**Step 3**: Each step of size h consists of a symmetric composition of three maps:

For a **2nd-order base method** (Strang splitting):
```
Ψ₂(h) = φ_C(h/2) ∘ φ_A(h/2) ∘ φ_B(h) ∘ φ_A(h/2) ∘ φ_C(h/2)
```

where:
- φ_A(τ): q += τ·∂H/∂p̃(q, p̃),  p̃ += -τ·∂H/∂q(q, p̃)
- φ_B(τ): q̃ += τ·∂H/∂p(q̃, p),  p += -τ·∂H/∂q̃(q̃, p)
- φ_C(τ): exact rotation of (q,q̃) and (p,p̃) by angle ωτ

**Step 4**: Compose Ψ₂ using Yoshida coefficients to get higher order:
```
Ψ₄(h) = Ψ₂(w₁h) ∘ Ψ₂(w₀h) ∘ Ψ₂(w₁h)
Ψ₆(h) = Ψ₂(w₁h) ∘ ... ∘ Ψ₂(w₁h)  [7 compositions]
Ψ₈(h) = Ψ₂(w₁h) ∘ ... ∘ Ψ₂(w₁h)  [15 compositions]
```

**Step 5**: Read off the physical solution from q, p (ignore q̃, p̃).

### Key Implementation Details

1. **φ_A and φ_B are single Euler-like steps** — they advance the state by evaluating the geodesic RHS once. They don't need to be high-order because the Yoshida composition handles the order.

2. **φ_C is an exact rotation**:
   ```
   q_new  = q·cos(ωτ) + q̃·sin(ωτ)
   q̃_new = -q·sin(ωτ) + q̃·cos(ωτ)
   ```
   (same for p, p̃)

3. **ω = c/h** where c is a constant of order 1. Tao suggests c ≈ 1-10.

4. **Memory**: 10 doubles per ray (5 real + 5 shadow) = 80 bytes.

5. **Cost per Yoshida substep**: 2 geoRHS evaluations (one for φ_A, one for φ_B) + 1 harmonic rotation (cheap: ~20 FLOPs for sin/cos + 10 multiplies).

6. **Total cost for Y8**: 15 Yoshida substeps × (2 geoRHS + 1 rotation) = 30 geoRHS + 15 rotations per step.

## Implementation Plan

### New Files
- `server/kernels/integrators/tao_yoshida4.cu` — Tao + Y4 kernel
- `server/kernels/integrators/tao_yoshida6.cu` — Tao + Y6 kernel
- `server/kernels/integrators/tao_yoshida8.cu` — Tao + Y8 kernel

### Modified Files
- `server/kernels/integrators/steps.cu` — Add Tao step functions, Y6/Y8 coefficients
- `server/kernels/integrators/adaptive_step.cu` — Add Tao-specific step sizing
- `server/kernels/ray_trace.cu` — Add Tao ray trace kernels
- `server/renderer.py` — Register new kernels
- `server/app.py` — Add to method validation
- `index.html` — Add to integrator dropdown
- `js/ui-controller.js` — Add labels
- `bench.html` — Add to benchmark registry
- `ARCHITECTURE.md` — Document new integrators
- `README.md` — Update file tree

### Step Function Design

```c
/* Tao extended phase space step for Yoshida composition.
 *
 * State: (r, th, phi, pr, pth) — real variables
 *        (rs, ths, phis, prs, pths) — shadow variables
 *
 * Each Yoshida substep with coefficients (D_i, W_i) performs:
 *   1. φ_A(D_i·h): advance (r,th,phi) and (prs,pths) using geoRHS(r,th,prs,pths)
 *   2. φ_B(D_i·h): advance (rs,ths,phis) and (pr,pth) using geoRHS(rs,ths,pr,pth)
 *   3. φ_C(W_i·h): harmonic rotation of (q,q̃) and (p,p̃) with frequency ω
 */
```

### Coupling Parameter ω

Following Tao's recommendation: ω = c/h where h is the step size.
For our adaptive step sizing, we compute ω at each step as ω = TAO_OMEGA_C / he.
TAO_OMEGA_C ≈ 2.0 (tunable constant).

### Naming Convention

Methods will be named `tao_yoshida4`, `tao_yoshida6`, `tao_yoshida8` to distinguish from the (removed) naive Yoshida implementations.
