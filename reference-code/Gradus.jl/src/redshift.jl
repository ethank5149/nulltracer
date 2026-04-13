module RedshiftFunctions
import ..Gradus
import ..Gradus: __BoyerLindquistFO, AbstractMetric, KerrMetric, metric, dotproduct
using StaticArrays

"""
    eⱽ(M, r, a, θ)

Modified from Cunningham et al. (1975) eq. (A2a):

```math
e^\\nu = \\sqrt{\\frac{\\Delta \\Sigma}{A}}.
```

"""
eⱽ(M, r, a, θ) =
    √(__BoyerLindquistFO.Σ(r, a, θ) * __BoyerLindquistFO.Δ(M, r, a) /
      __BoyerLindquistFO.A(M, r, a, θ),)

"""
    eᶲ(M, r, a, θ)

Modified from Cunningham et al. (1975) eq. (A2b):

```math
e^\\Phi = \\sin \\theta \\sqrt{\\frac{A}{\\Sigma}}.
```
"""
eᶲ(M, r, a, θ) =
    sin(θ) * √(__BoyerLindquistFO.A(M, r, a, θ) / __BoyerLindquistFO.Σ(r, a, θ))

"""
    ω(M, r, a, θ)

From Cunningham et al. (1975) eq. (A2c):

```math
\\omega = \\frac{2 a M r}{A}.
```
"""
ω(M, r, a, θ) = 2 * a * M * r / __BoyerLindquistFO.A(M, r, a, θ)

"""
    Ωₑ(M, r, a)

Coordinate angular velocity of an accreting gas.

Taken from Cunningham et al. (1975) eq. (A7b):

```math
\\Omega_e = \\frac{\\sqrt{M}}{a \\sqrt{M} + r_e^{3/2}}.
```

# Notes

Fanton et al. (1997) use

```math
\\Omega_e = \\frac{\\sqrt{M}}{a \\sqrt{M} \\pm r_e^{3/2}},
```

where the sign is dependent on co- or contra-rotation. This function may be extended in the future to support this definition.
"""
Ωₑ(M, r, a) = √M / (r^1.5 + a * √M)

"""
    Vₑ(M, r, a, θ)

Velocity of an accreting gas in a locally non-rotating reference frame (see Bardeen et al. 1973).
Taken from Cunningham et al. (1975) eq. (A7b):

```math
V_e = (\\Omega_e - \\omega) e^{\\Phi - \\nu}.
```
"""
Vₑ(M, r, a, θ) = (Ωₑ(M, r, a) - ω(M, r, a, θ)) * eᶲ(M, r, a, θ) / eⱽ(M, r, a, θ)

"""
    Lₑ(M, rms, a)

Angular momentum of an accreting gas within ``r_ms``.

Taken from Cunningham et al. (1975) eq. (A11b):

```math
L_e = \\sqrt{M} \\frac{
        r_{\\text{ms}}^2 - 2 a \\sqrt{M r_{\\text{ms}}} + a^2
    }{
        r_{\\text{ms}}^{3/2} - 2 M \\sqrt{r_{\\text{ms}}} + a \\sqrt{M}
    }.
```
"""
Lₑ(M, rms, a) = √M * (rms^2 - 2 * a * √(M * rms) + a^2) / (rms^1.5 - 2 * M * √rms + a * √M)

"""
    H(M, rms, r, a)

Taken from Cunningham et al. (1975) eq. (A12e):

```math
H = \\frac{2 M r_e - a \\lambda_e}{\\Delta},
```

where we distinguing ``r_e`` as the position of the accreting gas.
"""
H(M, rms, r, a) = (2 * M * r - a * Lₑ(M, rms, a)) / __BoyerLindquistFO.Δ(M, r, a)

"""
    γₑ(M, rms)

Taken from Cunningham et al. (1975) eq. (A11c):

```math
\\gamma_e = \\sqrt{1 - \\frac{
        2M
    }{
        3 r_{\\text{ms}}
    }}.
```
"""
γₑ(M, rms) = √(1 - (2 * M) / (3 * rms))

"""
    uʳ(M, rms, r)

Taken from Cunningham et al. (1975) eq. (A12b):

```math
u^r = - \\sqrt{\\frac{
        2M
    }{
        3 r_{\\text{ms}}
    }} \\left(
        \\frac{ r_{\\text{ms}} }{r_e} - 1
    \\right)^{3/2}.
```
"""
uʳ(M, rms, r) = -√((2 * M) / (3 * rms)) * (rms / r - 1)^1.5

"""
    uᶲ(M, rms, r, a)

Taken from Cunningham et al. (1975) eq. (A12c):

```math
u^\\phi = \\frac{\\gamma_e}{r_e^2} \\left(
        L_e + aH
    \\right).
```
"""
uᶲ(M, rms, r, a) = γₑ(M, rms) / r^2 * (Lₑ(M, rms, a) + a * H(M, rms, r, a))

"""
    uᵗ(M, rms, r, a)

Taken from Cunningham et al. (1975) eq. (A12b):

```math
u^t = \\gamma_e \\left(
        1 + \\frac{2 M (1 + H)}{r_e}
    \\right).
```
"""
uᵗ(M, rms, r, a) = γₑ(M, rms) * (1 + 2 * M * (1 + H(M, rms, r, a)) / r)

regular_pdotu_inv(L, M, r, a, θ) =
    (eⱽ(M, r, a, θ) * √(1 - Vₑ(M, r, a, θ)^2)) / (1 - L * Ωₑ(M, r, a))

function plunging_p_dot_u(E, a, M, L, Q, rms, r, sign_r)
    inv(
        uᵗ(M, rms, r, a) - uᶲ(M, rms, r, a) * L -
        sign_r * uʳ(M, rms, r) * __BoyerLindquistFO.Σδr_δλ(E, L, M, Q, r, a) /
        __BoyerLindquistFO.Δ(M, r, a),
    )
end

function redshift_function(m::Gradus.KerrSpacetimeFirstOrder, u, p, λ, isco)
    if u[2] > isco
        regular_pdotu_inv(p.L, m.M, u[2], m.a, u[3])
    else
        # change sign if we're after the sign flip
        # TODO: this isn't robust to multiple sign changes
        # TODO: i feel like there should be a better way than checking this with two conditions
        #       used to have λ > p.changes[1] to make sure we're ahead of the time flip (but when wouldn't we be???)
        #       now p.changes[1] > 0.0 to make sure there was a time flip at all
        sign_r = (p.changes[1] > 0.0 ? 1 : -1) * p.r
        plunging_p_dot_u(m.E, m.a, m.M, p.L, p.Q, isco, u[2], sign_r)
    end
end

"""
    keplerian_orbit(m::KerrMetric, x, isco)

A `disc_velocity` for use in [`redshift_function`](@ref) which represents
standard Keplerian orbits with a Plungion region extension.
"""
function keplerian_orbit(m::KerrMetric, x, isco)
    r = Gradus._equatorial_project(x)
    if r < isco
        # plunging region
        SVector(uᵗ(m.M, isco, r, m.a), -uʳ(m.M, isco, r), zero(r), uᶲ(m.M, isco, r, m.a))
    else
        Gradus.CircularOrbits.fourvelocity(m, r)
    end
end

redshift_function(m::KerrMetric, gp, isco; disc_velocity = keplerian_orbit) =
    _redshift_dotproduct(m, gp, disc_velocity(m, gp.x, isco))

function _redshift_dotproduct(m::AbstractMetric, gp, v_disc)
    # fixed stationary observer velocity
    g_init = metric(m, gp.x_init)
    v_obs = SVector{4}(inv(√(-g_init[1])), 0, 0, 0)
    _redshift_dotproduct(metric(m, gp.x), v_disc, gp.v, g_init, v_obs, gp.v_init)
end

function _redshift_dotproduct(
    M_end::AbstractMatrix,
    u_end,
    v_end,
    M_start::AbstractMatrix,
    u_start,
    v_start,
)
    E_end = dotproduct(M_end, v_end, u_end)
    E_obs = dotproduct(M_start, v_start, u_start)
    E_obs / E_end
end

end # module

const STATIONARY_VELOCITY = Gradus.SourceVelocities.stationary

_redshift_guard(
    m::Gradus.KerrSpacetimeFirstOrder,
    gp::FirstOrderGeodesicPoint,
    isco;
    kwargs...,
) = RedshiftFunctions.redshift_function(m, gp.x, gp.p, gp.λ_max, isco; kwargs...)

_redshift_guard(m::AbstractMetric, gp, isco; kwargs...) =
    RedshiftFunctions.redshift_function(m, gp, isco; kwargs...)

"""
    redshift_function(m::AbstractMetric, x_obs::SVector{4}; kwargs...)

Construct a closure which returns a function for evaluating the redshift of a
particular medium.

For almost all spacetimes, the `velocity` keyword is supported, which allows a
different velocity function for the disc velocity to be set. The default is
[`RedshiftFunctions.keplerian_orbit`](@ref)

The closure function has the following signature, compatible with
[`PointFunction`](@ref):

    function _redshift_closure(m, gp, max_time)
        _redshift_guard(m, gp, r_isco; kwargs...)
    end

For most purposes, [`ConstPointFunctions.redshift`](@ref) should be used.
"""
function redshift_function(m::AbstractMetric, x_obs::SVector{4}; kwargs...)
    r_isco = isco(m)
    function _redshift_closure(m, gp, max_time)
        _redshift_guard(m, gp, r_isco; kwargs...)
    end
end

"""
    interpolate_redshift(plunging_interpolation, u)

`u` is the observer's position (assumed stationary). Returns a [`PointFunction`](@ref).

# Notes

For a full, annotated derivation of this method, see
[the following blog post](https://fjebaker.github.io/blog/pages/2022-05-plunging-orbits/).
"""
function interpolate_redshift(plunging_interpolation, u::SVector{4,T}; kwargs...) where {T}
    isco = Gradus.isco(plunging_interpolation.m)
    # metric matrix at observer
    m_obs = metric(plunging_interpolation.m, u)
    # fixed stationary observer velocity
    v_obs = STATIONARY_VELOCITY(plunging_interpolation.m, u)
    circ_velocity_func =
        make_circular_velocity_function(plunging_interpolation.m; kwargs...)
    function _interpolate_redshift_closure(m, gp, max_time)
        r = _equatorial_project(gp.x)
        v_disc = if r < isco
            # plunging region
            vtemp = plunging_interpolation(r)
            # we have to reverse radial velocity due to backwards tracing convention
            # see https://github.com/astro-group-bristol/Gradus.jl/issues/3
            SVector{4}(vtemp[1], -vtemp[2], vtemp[3], vtemp[4])
        else
            # regular circular orbit
            circ_velocity_func(r)
        end

        # get metric matrix at position on disc
        g = metric(plunging_interpolation.m, gp.x)
        RedshiftFunctions._redshift_dotproduct(g, v_disc, gp.v, m_obs, v_obs, gp.v_init)
    end
    _interpolate_redshift_closure
end

interpolate_redshift(m::AbstractMetric, u::SVector{4}; kwargs...) =
    interpolate_redshift(interpolate_plunging_velocities(m; kwargs...), u)

function make_circular_velocity_function(m::AbstractMetric)
    function _circular_velocity(r)
        CircularOrbits.fourvelocity(m, r)
    end
end

"""
    energyshift(m, gp::GeodesicPoint; u_init, u_end)
    energyshift(m, x_init, v_init, x_end, v_end; u_init, u_end)

Calculate the energyshift along a geodesic as measured by a medium with
velocity `u_init` of a photon originating from a medium with `u_end`. By
default, both velocities are set be locally stationary, and the redshift that
is measured is entirely determined by the relative ``g_{t t}`` metric
components.

The naming may seem reverse, but this is to reflect that the geodesic is traced
from `u_init` to `u_end`, but the energyshift is measured at `u_init`.

To get the energyshift as measured by `u_end`, simply take the inverse of the
result.
"""
energyshift(m::AbstractMetric, gp::GeodesicPoint; kwargs...) =
    energyshift(m, gp.x_init, gp.v_init, gp.x, gp.v; kwargs...)
energyshift(m::AbstractMetric, x_v_init::SVector{8}, x_v_end::SVector{8}; kwargs...) =
# TODO: make these SVectors as well
    @views energyshift(m, x_v_init[1:4], x_v_init[5:8], x_v_end[1:4], x_v_end[5:8])
function energyshift(
    m::AbstractMetric,
    x_init,
    v_init,
    x_end,
    v_end;
    u_init = STATIONARY_VELOCITY(m, x_init),
    u_end = STATIONARY_VELOCITY(m, x_end),
)
    g_init = metric(m, x_init)
    g_end = metric(m, x_end)
    RedshiftFunctions._redshift_dotproduct(g_end, u_end, v_end, g_init, u_init, v_init)
end

export RedshiftFunctions, interpolate_redshift, redshift_function, energyshift
