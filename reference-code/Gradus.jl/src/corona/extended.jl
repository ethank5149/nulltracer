module SourceVelocities

using ..Gradus:
    AbstractMetric,
    CircularOrbits,
    SVector,
    metric_components,
    propernorm,
    metric,
    constrain_all,
    isco

"""
    co_rotating(m::AbstractMetric, x::SVector{4})

Calculates the a source velocity assuming the cylinder described by the point
`x` co-rotates with the accretion disc below. This assumes Keplerian orbits, and
uses [`CircularOrbits`](@ref) to perform the calculation.
"""
function co_rotating(m::AbstractMetric, x::SVector{4})
    sinθ = sin(x[3])
    v = CircularOrbits.fourvelocity(m, max(isco(m), x[2] * sinθ)) .* sinθ
    v = v ./ sqrt(abs(propernorm(metric(m, x), v)))
    v = constrain_all(m, x, v, 1.0)
end

"""
    stationary(m::AbstractMetric, x::SVector{4})

Assumes the source is stationary, and calculates a velocity vector with
components
```math
v^\\mu = (v^t, 0, 0, 0),
```
where the time-component is determined by the metric to satisfy
```math
g_{\\mu\\nu} v^\\mu v^\\nu = -1.
```
"""
function stationary(m::AbstractMetric{T}, x) where {T}
    gcomp = metric_components(m, SVector(x[2], x[3]))
    inv(√(-gcomp[1])) * SVector{4,T}(1, 0, 0, 0)
end

end # module

"""
    DEFAULT_β_ANGLES(n = 100)

The default ``\\beta`` angles used to calculate slices of an extended corona
point approximation.
"""
DEFAULT_β_ANGLES(n = 100) = sort!(collect(range(0, π - (π / n), n)))

"""
    RingCorona

A ring-like corona, representing an infinitely thin thing at some radius and
height above the accretion disc.
"""
struct RingCorona{T,VelFunc} <: AbstractCoronaModel{T}
    "Source velocity function. May be any one of [`SourceVelocities`](@ref) or a
    custom implementation."
    vf::VelFunc
    "Radius of the ring"
    r::T
    "Height of the base of the cylinder above the disc"
    h::T
end

RingCorona(r, h) = RingCorona(SourceVelocities.co_rotating, r, h)
RingCorona(; r = 5.0, h = 5.0, vf = SourceVelocities.co_rotating) = RingCorona(vf, r, h)

function sample_position_velocity(m::AbstractMetric, model::RingCorona{T}) where {T}
    r = sqrt(model.r^2 + model.h^2)
    # (x, y) flipped because coordinates are off the Z axis
    θ = atan(model.r, model.h)
    x = SVector{4,T}(0, r, θ, 0)
    x, model.vf(m, x)
end

function emissivity_profile(
    setup::EmissivityProfileSetup{true},
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    model::RingCorona;
    βs = DEFAULT_β_ANGLES(),
    no_refine = false,
    βs_refined = no_refine ? βs : collect(range(extrema(βs)..., setup.n_samples)),
    kwargs...,
)
    slices = corona_slices(setup, m, d, model, βs; kwargs...)
    make_approximation(m, slices, setup.spectrum; βs = βs_refined)
end

function _process_ring_traces(setup::EmissivityProfileSetup, m, d, v, gps, rs, δs)
    # no need to filter intersected, as we have already done that before calling this process function
    J = sortperm(rs)
    points = gps[J]
    δs_sorted = δs[J]

    r, ε, g = _point_source_emissivity(m, d, setup.spectrum, v, rs[J], δs_sorted, points)
    t = [i.x[1] for i in points]
    (; t, r, ε = abs.(ε), θ = δs_sorted, g)
end

"""
    DiscCorona

A disk-like corona with no height but some radial extent.

Depending on the algorithm chosen, emissivity profiles for this corona are
either calculated via Monte-Carlo sampling, or by treating the extended source
as many concentric [`RingCorona`](@ref).
"""
struct DiscCorona{T,VelFunc} <: AbstractCoronaModel{T}
    "Source velocity function. May be any one of [`SourceVelocities`](@ref) or a
    custom implementation."
    vf::VelFunc
    "Radius of the disc"
    r::T
    "Height of the base of the cylinder above the disc"
    h::T
end

DiscCorona(; vf = SourceVelocities.co_rotating, r = 5.0, h = 5.0) =
    DiscCorona{typeof(vf)}(vf, r, h)

function sample_position_velocity(m::AbstractMetric, model::DiscCorona{T}) where {T}
    x = rand() * model.r
    r = sqrt(x^2 + model.h^2)
    # (x, y) flipped because coordinates are off the Z axis
    θ = atan(x, model.h)
    x = SVector{4,T}(0, r, θ, 0)
    x, model.vf(m, x)
end

function emissivity_profile(
    setup::EmissivityProfileSetup{true},
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    model::DiscCorona;
    n_rings = 10,
    kwargs...,
)
    radii = range(1e-2, model.r, n_rings)
    ring_profiles = map(radii) do r
        emissivity_profile(setup, m, d, RingCorona(model.vf, r, model.h); kwargs...)
    end
    DiscCoronaProfile(collect(radii), ring_profiles, _ -> 0)
end


export RingCorona, DiscCorona
