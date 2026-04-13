export tracecorona

function sample_position_direction_velocity(
    m::AbstractMetric,
    model::AbstractCoronaModel{T},
    sampler::AbstractDirectionSampler,
    N::Int,
) where {T}
    xs = Vector{SVector{4,T}}(undef, N)
    vs = Vector{SVector{4,T}}(undef, N)
    vs_source = Vector{SVector{4,T}}(undef, N)

    rmin = inner_radius(m)
    Threads.@threads for i = 1:N
        x, v = sample_position_velocity(m, model, sampler, i, N)
        while x[2] < rmin * 1.9
            x, v = sample_position_velocity(m, model, sampler, i, N)
        end

        # avoid coordinate singularities
        if x[3] < 1e-3
            x = SVector(x[1], x[2], 1e-3, x[4])
        end
        if x[3] > π - 1e-3
            x = SVector(x[1], x[2], π - 1e-3, x[4])
        end

        xs[i] = x
        vs_source[i] = v
        vs[i] = sample_local_velocity(m, sampler, x, v, i, N)
    end

    xs, vs, vs_source
end

"""
    sample_position_velocity(m::AbstractMetric, model::AbstractCoronaModel)
    sample_position_velocity(
        m::AbstractMetric,
        model::AbstractCoronaModel,
        ::AbstractDirectionSampler,
        i,
        N,
    )

Sample a source position and velocity pair from the
[`AbstractCoronaModel`](@ref), optionally specifying the sampler, the sample
index `i`, and total number of samples `N`. The latter is used when uniform
samples are needed, but will invoke the prior if not implemented.

Currently, these functions should make use of `random` if they have underlying
position and/or velocity distributions, allowing higher order methods, such as
[`tracecorona`](@ref) to approximate a Monte-Carlo sampling technique. The user
is required to ensure that the distributions have the desired properties.

This function must return a pair of `SVector{4,T}`, where the type must match
the parametric type of the coronal model, corresponding to the source position
and velocity of that position.

The velocity vector must be appropriately normalised for the source (see
[`propernorm`](@ref) for help).

## Example

Here we implement a new [`AbstractCoronaModel`](@ref) that is extended over a
region at constant height above the black hole. Since we desire the
distribution of points to be even over this disc, we must sample as

```math
\\phi \\sim 2\\pi \\mathcal{U},
\\quad \\text{and} \\quad
r \\sim \\sqrt{R^2 \\mathcal{U}},
```

where ``\\mathcal{U}`` is a uniform random variable in ``[0, 1]``, and ``R`` is
the radial extent of the coronal source. Implemented, this is

```julia
struct ExtendedCorona{T} <: Gradus.AbstractCoronaModel{T}
    h::T
    R::T
end

function Gradus.sample_position_velocity(m::AbstractMetric, model::ExtendedCorona{T}) where {T}
    ϕ = rand(T) * 2π
    R = √(rand(T) * model.R^2)

    # geometry to translate to global coordinates
    r = √(model.h^2 + R^2)
    θ = atan(R, model.h)

    # ensure velocity is normalized
    g = metric_components(m, SVector(r, θ))
    v = inv(√(-g[1])) * SVector(1, 0, 0, 0)
    SVector(0, r, θ, ϕ), v
end
```
"""
function sample_position_velocity(
    m::AbstractMetric,
    model::AbstractCoronaModel,
    ::AbstractDirectionSampler,
    i,
    N,
)
    sample_position_velocity(m, model)
end
function sample_position_velocity(::AbstractMetric, model::AbstractCoronaModel)
    error(
        "This functions needs to be implemented for $(typeof(model)). See the documentation for this function for instructions.",
    )
end

"""
function sample_local_velocity(
    m::AbstractMetric,
    sampler::AbstractDirectionSampler,
    x,
    v,
    index,
    N,
)

Sample a single initial (un-normalised) velocity vector in the local
coordinates of at `x`, according to the [`AbstractDirectionSampler`](@ref)
algorithm and domain, given an initial 4-vector velocity `v`.

This function is metric generic and is implemented in the following way:
- Sample angles ``\\alpha`` and ``\\beta`` in the local sky.
- Map these angles to a spherical polar vector (interally first to Cartesian
  and then using a Jacobian transformation to spherical polar).
- Calculate the tetrad given position and velocity `x` and `v`.
- Use this tetrad to map the local vector to the global coordinates.
"""
function sample_local_velocity(
    m::AbstractMetric{T},
    sampler::AbstractDirectionSampler,
    x,
    v,
    index,
    N,
) where {T}
    i = geti(sampler, index, N)
    θ, ϕ = sample_angles(sampler, i, N)
    sky_angles_to_velocity(m, x, v, θ, ϕ)
end

# bootstrap tracing function for convenience
function tracegeodesics(
    m::AbstractMetric,
    model::AbstractCoronaModel,
    args...;
    n_samples = 1024,
    sampler = EvenSampler(BothHemispheres(), GoldenSpiralGenerator()),
    kwargs...,
)
    xs, vs, _ = sample_position_direction_velocity(m, model, sampler, n_samples)
    tracegeodesics(m, xs, vs, args...; kwargs...)
end

struct CoronaGeodesics{T,M,G,C,P,V}
    trace::T
    metric::M
    geometry::G
    model::C
    geodesic_points::P
    source_velocity::V
end

function tracecorona(
    m::AbstractMetric,
    g::AbstractAccretionGeometry,
    model::AbstractCoronaModel;
    λmax = 10_000,
    n_samples = 1024,
    sampler = EvenSampler(domain = BothHemispheres(), generator = RandomGenerator()),
    trace = TraceGeodesic(),
    callback = domain_upper_hemisphere(),
    kwargs...,
)
    xs, vs, source_vels = sample_position_direction_velocity(m, model, sampler, n_samples)
    gps = tracegeodesics(
        m,
        xs,
        vs,
        g,
        λmax;
        trace = trace,
        save_on = false,
        ensemble = EnsembleEndpointThreads(),
        callback = callback,
        kwargs...,
    )
    mask = [i.status == StatusCodes.IntersectedWithGeometry for i in gps]
    CoronaGeodesics(trace, m, g, model, gps[mask], source_vels[mask])
end

"""
    local_velocity(m::AbstractMetric, x, v, component::Int)

Compute the component of the velocity `v` in the LNRF given by `component`,
where in Boyer-Lindquist `(1, 2, 3, 4)` corresponds to `(r, t, θ, ϕ)`.

Evaluates Bardeen+73 (3.9),

```math
\\mathscr{V}^{(i)} = \\frac{v^\\mu e_\\mu^{(i)}}{v^\\nu e_\\nu^{(t)}}.
```
"""
function local_velocity(m::AbstractMetric, x, v, component::Int)
    es = lnrbasis(m, x)
    𝒱t = _fast_dot(es[1], v)
    𝒱i = _fast_dot(es[component], v)
    𝒱i / 𝒱t
end

"""
    lorentz_factor(m::AbstractMetric, g::AbstractAccretionGeometry, x)

Calculate the Lorentz factor of a patch of the accretion geometry `g` at
position `x` via

```math
\\left( 1 - v^{\\phi}^2 \\right)^{\\frac{-1}{2}}
```
"""
lorentz_factor(m::AbstractMetric, ::AbstractAccretionGeometry, x; kwargs...) =
    lorentz_factor(m, x, CircularOrbits.fourvelocity(m, _equatorial_project(x)); kwargs...)
function lorentz_factor(m::AbstractMetric, x, v; component = 4)
    𝒱ϕ = local_velocity(m, x, v, component)
    inv(√(1 - 𝒱ϕ^2))
end

energy_ratio(m::AbstractMetric, gp::GeodesicPoint, v_src) =
    energy_ratio(m, gp, v_src, CircularOrbits.fourvelocity(m, _equatorial_project(gp.x)))
function energy_ratio(m::AbstractMetric, gp::GeodesicPoint, v_src, v_disc)
    # at the source
    g_src = metric(m, gp.x_init)
    e_src = dotproduct(g_src, gp.v_init, v_src)
    # at the disc
    g_disc = metric(m, gp.x)
    e_disc = dotproduct(g_disc, gp.v, v_disc)

    # TODO: this should be the otherway round to be consistent with g = E_end /
    # E_start but changing this will require making sure all 1/g are replaced
    # with g and vice versa in the emissivity calculations.
    e_src / e_disc
end
