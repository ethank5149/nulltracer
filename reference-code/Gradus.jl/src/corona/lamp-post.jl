"""
    LampPostModel(h = 5.0)

An implementation of the lamppost coronal model. This is a point-source located
on the spin-axis of the black hole at a particular heighth `h`.
"""
Base.@kwdef struct LampPostModel{T} <: AbstractCoronaModel{T}
    h::T = 5.0
    θ::T = 0.01
    ϕ::T = 0.0
end

function sample_position_velocity(m::AbstractMetric, model::LampPostModel{T}) where {T}
    x = SVector{4,T}(0, model.h, model.θ, model.ϕ)
    gcomp = metric_components(m, SVector(x[2], x[3]))
    v = inv(√(-gcomp[1])) * SVector{4,T}(1, 0, 0, 0)
    x, v
end

"""
    BeamedPointSource(h, β)

Point source corona moving away from the black hole with specified starting
height above the disc `h` and source speed `β`.  Point sources are the most
plausible example of source that would support beaming (ref: Gonzalez et al
2017).
"""
struct BeamedPointSource{T} <: Gradus.AbstractCoronaModel{T}
    h::T
    β::T
end

# these are specific to BeamedPointSource, so we'll scope them in a module
# so that they don't pollute the namespace of Gradus
module __BeamedPointSource
import ..Gradus: AbstractMetric, metric_components, SVector

drdt(g, β) = β * √(-g[1] / g[2])
drdt(m::AbstractMetric, x, β) = drdt(metric_components(m, SVector(x[2], x[3])), β)
end # module

function sample_position_velocity(m::AbstractMetric, model::BeamedPointSource)
    x = SVector{4}(0, model.h, 1e-4, 0)
    g = metric_components(m, SVector(x[2], x[3]))
    v̄ = SVector(1, __BeamedPointSource.drdt(g, model.β), 0, 0)
    v = constrain_normalize(m, x, v̄; μ = 1)
    x, v
end

"""
    point_source_geodesics(
        m::AbstractMetric,
        d::AbstractAccretionGeometry,
        model::AbstractCoronaModel;
        kwargs...
    )
    point_source_geodesics(
        setup::EmissivityProfileSetup,
        m::AbstractMetric,
        d::AbstractAccretionGeometry,
        model::AbstractCoronaModel;
        kwargs...
    )

Calculate a series of photons in a 2-dimensional slice (i.e. with radial angle
set to zero) for a point-like source on the symmetry axis of the spacetime
[`AbstractMetric`](@ref).

This returns a named tuple `(; δs, gps, x_src, v_src)` with the initial
azimuthal angles and [`GeodesicPoint`](@ref) correponding to those geodesics.

All keyword arguments are forwarded to [`tracegeodesics`](@ref) via
[`EmissivityProfileSetup`](@ref) if called without a `setup` argument.
"""
function point_source_geodesics(
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    model::AbstractCoronaModel;
    kwargs...,
)
    # use a dummy powerlawspectrum, since it's not actually used in the call to
    # `point_source_geodesics`
    solver_kwargs, setup =
        Gradus.EmissivityProfileSetup(Float64, Gradus.PowerLawSpectrum(2.0); kwargs...)
    point_source_geodesics(setup, m, d, model; solver_kwargs...)
end

function point_source_geodesics(
    setup::EmissivityProfileSetup,
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    model::AbstractCoronaModel;
    callback = domain_upper_hemisphere(),
    kwargs...,
)
    δs = deg2rad.(range(setup.δmin, setup.δmax, setup.n_samples))
    # we assume a point source
    x_src, v_src = sample_position_velocity(m, model)
    velfunc = polar_angle_to_velfunc(m, x_src, v_src, δs)
    gps = tracegeodesics(
        m,
        x_src,
        velfunc,
        d,
        setup.λmax;
        save_on = false,
        ensemble = EnsembleEndpointThreads(),
        callback = callback,
        trajectories = length(δs),
        kwargs...,
    )
    (; δs, gps, x_src, v_src)
end

function _point_source_symmetric_emissivity_profile(
    setup::EmissivityProfileSetup,
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    model::AbstractCoronaModel;
    kwargs...,
)
    δs, gps, _, v = point_source_geodesics(setup, m, d, model; kwargs...)
    # filter only those that intersected, and sort radially
    I = [i.status == StatusCodes.IntersectedWithGeometry for i in gps]
    points = gps[I]
    δs = δs[I]
    rs = [_equatorial_project(i.x) for i in points]
    J = sortperm(rs)
    rs_sorted = rs[J]
    points = points[J]
    δs = δs[J]

    r, ε, _ = _point_source_emissivity(m, d, setup.spectrum, v, rs_sorted, δs, points)
    t = [i.x[1] for i in points]

    RadialDiscProfile(r, t, ε)
end

function _point_source_emissivity(
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    spec::AbstractLocalSpectrum,
    source_velocity,
    r,
    δs,
    points::AbstractVector{<:AbstractGeodesicPoint{T}},
) where {T}
    # function for obtaining keplerian velocities
    _disc_velocity = _keplerian_velocity_projector(m, d)

    all_g = Vector{T}(undef, length(points))
    ε = map(enumerate(points)) do (i, p)
        v_disc = _disc_velocity(p.x)
        gs = energy_ratio(m, p, source_velocity, v_disc)
        γ = lorentz_factor(m, p.x, v_disc)

        all_g[i] = gs

        i1, i2, i3, i4 = if i == 1
            1, 2, 1, 2
        elseif i != lastindex(points)
            i, i + 1, i, i - 1
        else
            i, i - 1, i, i - 1
        end

        Δr = (abs(r[i1] - r[i2]) + abs(r[i3] - r[i4])) / 2
        weight = (abs(δs[i1] - δs[i2]) + abs(δs[i3] - δs[i4])) / 4

        A = _proper_area(m, p.x) * Δr
        weight * point_source_equatorial_disc_emissivity(spec, δs[i], gs, A, γ)
    end

    r, ε, all_g
end

function emissivity_profile(
    setup::EmissivityProfileSetup{true},
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    model::Union{<:LampPostModel,<:BeamedPointSource};
    kwargs...,
)
    _point_source_symmetric_emissivity_profile(setup, m, d, model; kwargs...)
end

export LampPostModel, BeamedPointSource
