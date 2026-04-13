export EmissivityProfileSetup, emissivity_profile

struct EmissivityProfileSetup{TryToOptimize,T,SamplerType,SpectrumType}
    λmax::T
    δmin::T
    δmax::T
    sampler::SamplerType
    spectrum::SpectrumType
    n_samples::Int
end

function EmissivityProfileSetup(
    T,
    spectrum;
    λmax = T(10000),
    δmin = T(0.01),
    δmax = T(179.99),
    sampler = nothing,
    n_samples = 1000,
    other_kwargs...,
)
    _sampler = if !isnothing(sampler)
        sampler
    else
        EvenSampler(BothHemispheres(), GoldenSpiralGenerator())
    end
    setup = EmissivityProfileSetup{isnothing(sampler),T,typeof(_sampler),typeof(spectrum)}(
        λmax,
        δmin,
        δmax,
        _sampler,
        spectrum,
        n_samples,
    )
    other_kwargs, setup
end

"""
    function emissivity_profile(
        m::AbstractMetric,
        d::AbstractAccretionGeometry,
        model::AbstractCoronaModel;
        spectrum = PowerLawSpectrum(2),
        kwargs...,
    end

    function emissivity_profile(
        setup::EmissivityProfileSetup,
        m::AbstractMetric,
        d::AbstractAccretionGeometry,
        model::AbstractCoronaModel;
        kwargs...,
    end

Calculate the reflection emissivity profile of an accretion disc `d` around the
spacetime `m` for an illuminating coronal model `model`. The return type is
dependent on the type of coronal model given to the function, but will be an
[`AbstractEmissivityProfile`](@ref).

This function will attempt to automatically switch to use a better scheme to
calculate the emissivity profiles if one is available. If not, the default
algorithm is a Monte-Carlo sample to estimate photon count ``N`` and calculate
the emissivity with [`source_to_disc_emissivity`](@ref).

Please consult the documentation of a specific model (e.g.
[`LampPostModel`](@ref) or [`RingCorona`](@ref)) to see algorithm specific
keywords that may be passed.

All other keyword arguments are forwarded to [`tracegeodesics`](@ref).

# Example

```julia
m = KerrMetric()
d = ThinDisc(Gradus.isco(m), 1000.0)
model = LampPostModel(h = 10.0)

profile = emissivity_profile(m, d, model)

# visualise as a function of disc radius
using Plots
plot(profile.radii, profile.ε)
```

# Details

The emissivity profile is equivalently the illumination profile of the disc (up
to some constant of normalisation), since in the reprocessing is assumed to
occur instantaneously. Mathematically, this is
```math
F_\\text{i} (E) = \\int_\\Omega I_\\text{i} (E) \\text{d}\\Omega,
```
where ``\\Omega`` is the solid angle on the sky of the corona. This integration
is conventionally done over the disc, and therefore a change-of-variables is
performed:
```math
F_\\text{i} (E) = \\int_\\Omega I_\\text{i} (E) \\left\\lvert \\frac{\\text{d}\\Omega}{\\text{d}A} \\right\\rvert \\text{d} A,
```
where the Jacobian term is nominally the term that encapsulates all general
relativistic effects and must be calcualated using the ray-tracing technique.
The area element is further corrected using the relativistic correction factors
```math
A_\\text{corr} = A \\gamma^{(\\phi)} \\sqrt{g_{rr} g_{\\phi\\phi}}.
```
See e.g. Wilkins & Fabian (2012) or Baker & Young (2025) for the details.

## Notes

The sampling is performed using an [`AbstractDirectionSampler`](@ref), which
samples angles on the emitters sky along which a geodesic is traced.  The
effects of the spacetime and the observer's velocity are taken into account by
using [`tetradframe`](@ref) and the corresponding coordinate transformation for
local to global coordinates.

This function assumes axis symmetry, and therefore always interpolates the
emissivity as a function of the radial coordinate on the disc. If non-symmetric
profiles are desired, consider using [`tracecorona`](@ref) with a profile
constructor.
"""
function emissivity_profile(
    m::AbstractMetric{T},
    d::AbstractAccretionGeometry,
    model::AbstractCoronaModel,
    spectrum = PowerLawSpectrum(2),
    ;
    kwargs...,
) where {T}
    solver_kwargs, setup = EmissivityProfileSetup(T, spectrum; kwargs...)
    emissivity_profile(setup, m, d, model; solver_kwargs...)
end

function emissivity_profile(
    setup::EmissivityProfileSetup,
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    model::AbstractCoronaModel,
    ;
    grid = GeometricGrid(),
    n_radii = 100,
    kwargs...,
)
    RadialDiscProfile(
        tracecorona(
            m,
            d,
            model;
            sampler = setup.sampler,
            λmax = setup.λmax,
            n_samples = setup.n_samples,
            kwargs...,
        ),
        setup.spectrum;
        grid = grid,
        n_radii = n_radii,
    )
end

"""
    source_to_disc_emissivity(
        m::AbstractStaticAxisSymmetric,
        spec::AbstractLocalSpectrum,
        N,
        A,
        x,
        g,
        v_disc,
    )

Compute the emissivity of a disc element with (proper) area `A` at coordinates
`x` with metric `m` and coronal spectrum `spec`. Since the emissivity is
dependent on the incident flux, the photon (geodesic) count `N` must be
specified, along with the ratio of energies `g` (computed with
[`energy_ratio`](@ref)) and the spectrum `spec`.

The mathematical definition is
```math
\\varepsilon = \\frac{N}{A g^\\Gamma \\gamma},
```

where ``\\gamma`` is the Lorentz factor due to the velocity of the local disc
frame. The velocity is currently always considered to be the Keplerian
velocity.

Wilkins & Fabian (2012) and Gonzalez et al. (2017).
"""
function source_to_disc_emissivity(
    m::AbstractStaticAxisSymmetric,
    spec::AbstractLocalSpectrum,
    N,
    A,
    x,
    g,
    v_disc,
)
    # account for relativistic effects in area due to lorentz shift
    γ = lorentz_factor(m, x, v_disc)
    # divide by area to get number density
    I = local_spectrum(spec, g)
    N * I / (A * γ)
end

"""
    point_source_equatorial_disc_emissivity(θ, g, A, γ, spec)

Calculate the emissivity of a point illuminating source on the spin axis for an
annulus of the equatorial accretion disc with (proper) area `A`. The precise
formulation follows from Dauser et al. (2013), with the emissivity calculated
as
```math
\\varepsilon = \\frac{\\sin \\theta}{A g^\\Gamma \\gamma}
```
where ``\\gamma`` is the Lorentz factor due to the velocity of the local disc
frame.  The ratio of energies is `g` (computed with [`energy_ratio`](@ref)),
with `spec` being the abstract coronal spectrum and  ``\\theta`` is the angle
from the spin axis in the emitters from at which the geodesic was directed.
`local_spectrum` function is used to calculate the spectrum of the corona by
taking `g` to the power of `Γ`, allowing further modification of spectrum if
needed, based on the value of the photon index.

The ``\\sin \\theta`` term appears to extend the result to three dimensions,
since the Jacobian of the spherical coordinates (with ``r`` fixed) yields a
factor ``\\sin \\theta`` in order to maintain point density. It may be regarded
as the PDF that samples ``\\theta`` uniformly.

Dauser et al. (2013)
"""
function point_source_equatorial_disc_emissivity(spec::AbstractLocalSpectrum, θ, g, A, γ)
    abs(sin(θ)) * local_spectrum(spec, g) / (A * γ)
end

function polar_angle_to_velfunc(m::AbstractMetric, x, v, δs; ϕ = zero(eltype(x)))
    function _polar_angle_velfunc(i)
        sky_angles_to_velocity(m, x, v, δs[i], ϕ)
    end
end

"""
    rotated_polar_angle_to_velfunc(m::AbstractMetric, x, v, δs, β; θ₀ = 0, ϕ = 0)

Similar to [`polar_angle_to_velfunc`](@ref), except now the generator function
returned will rotate and offset all velocity vectors. This is equivalent to
reorientating the local sky in the global coordinates.

The `θ₀` parameter is the offset from the global `θ=0` direction, and `β` is
the amount to rotate the sky by around it's zenith.
"""
function rotated_polar_angle_to_velfunc(
    m::AbstractMetric,
    x,
    v,
    δs,
    β;
    θ₀ = zero(eltype(x)),
)
    k = _cart_local_direction(θ₀, zero(eltype(x)))
    function _polar_angle_velfunc(i)
        q = _cart_local_direction(δs[i] + θ₀, zero(eltype(x)))

        b = rodrigues_rotate(k, q, β)

        # convert back to spherical coordinates
        ph = atan(b[2], b[1])
        th = atan(sqrt(b[2]^2 + b[1]^2), b[3])

        sky_angles_to_velocity(m, x, v, th, ph)
    end
end

# A concrete implementation for a radial disc profile where the function forms
# are known ahead of time

struct AnalyticRadialDiscProfile{E,T} <: AbstractEmissivityProfile
    ε::E
    t::T

    function AnalyticRadialDiscProfile(emissivity, time)
        new{typeof(emissivity),typeof(time)}(emissivity, time)
    end
end

function AnalyticRadialDiscProfile(emissivity, cg::CoronaGeodesics)
    J = sortperm(cg.geodesic_points; by = i -> _equatorial_project(i.x))
    radii = @views [_equatorial_project(i.x) for i in cg.geodesic_points[J]]
    times = @views [i.x[1] for i in cg.geodesic_points[J]]
    t = _make_interpolation(radii, times)
    AnalyticRadialDiscProfile(emissivity, t)
end

function emissivity_at(prof::AnalyticRadialDiscProfile, r::Number)
    r_bounded = _enforce_interpolation_bounds(r, prof)
    prof.ε(r)
end
emissivity_at(prof::AnalyticRadialDiscProfile, gp::AbstractGeodesicPoint) =
    emissivity_at(prof, _equatorial_project(gp.x))

function coordtime_at(prof::AnalyticRadialDiscProfile, r::Number)
    r_bounded = _enforce_interpolation_bounds(r, prof)
    prof.t(r_bounded)
end
coordtime_at(prof::AnalyticRadialDiscProfile, gp::AbstractGeodesicPoint) =
    coordtime_at(prof, _equatorial_project(gp.x)) + gp.x[1]

function _enforce_interpolation_bounds(r::Number, prof::AnalyticRadialDiscProfile)
    r_min = first(prof.t.t)
    r_max = last(prof.t.t)
    _enforce_interpolation_bounds(r, r_min, r_max)
end

# A concrete implementation for a radial disc profile interpolated from data
# calculated using one of the available methods

struct RadialDiscProfile{V<:AbstractVector,I} <: AbstractEmissivityProfile
    radii::V
    ε::V
    t::V
    interp_ε::I
    interp_t::I
end

@inline function _enforce_interpolation_bounds(r::Number, prof::RadialDiscProfile)
    r_min = first(prof.radii)
    r_max = last(prof.radii)
    return _enforce_interpolation_bounds(r, r_min, r_max)
end

function emissivity_at(prof::RadialDiscProfile, r::Number)
    r_bounded = _enforce_interpolation_bounds(r, prof)
    prof.interp_ε(r_bounded)
end
emissivity_at(prof::RadialDiscProfile, gp::AbstractGeodesicPoint) =
    emissivity_at(prof, _equatorial_project(gp.x))

function coordtime_at(prof::RadialDiscProfile, r::Number)
    r_bounded = _enforce_interpolation_bounds(r, prof)
    prof.interp_t(r_bounded)
end
coordtime_at(prof::RadialDiscProfile, gp::AbstractGeodesicPoint) =
    coordtime_at(prof, _equatorial_project(gp.x)) + gp.x[1]

function RadialDiscProfile(r, t, ε)
    interp_t = _make_interpolation(view(r, :), view(t, :))
    interp_ε = _make_interpolation(view(r, :), view(ε, :))
    RadialDiscProfile(r, ε, t, interp_ε, interp_t)
end

_get_grouped_intensity(T::Type, groupings, ::Nothing) = @. convert(T, length(groupings))
_get_grouped_intensity(::Type, groupings, intensity) =
    [@views(sum(intensity[grouping])) for grouping in groupings]

function _build_radial_profile(
    m::AbstractMetric,
    spec::AbstractLocalSpectrum,
    radii,
    times,
    source_velocities,
    points::AbstractVector{<:AbstractGeodesicPoint{T}},
    intensity;
    grid = GeometricGrid(),
    n_radii = 100,
) where {T}
    @assert size(radii) == size(source_velocities)
    @assert size(points) == size(source_velocities)

    # function for obtaining keplerian velocities
    _disc_velocity = _keplerian_velocity_projector(m)

    bins = grid(extrema(radii)..., n_radii) |> collect

    # find the grouping with an index bucket
    # that is, the points with the same r in Δr on the disc
    ibucket = Buckets.IndexBucket(Int, size(radii), length(bins))
    bucket!(ibucket, Buckets.Simple(), radii, bins)
    groupings = Buckets.unpack_bucket(ibucket)

    # need to interpolate the redshifts, so calculate those first
    gs = map(groupings) do grouping
        g_total = zero(T)
        for i in grouping
            gp = points[i]
            v_disc = _disc_velocity(gp.x)
            g_total += energy_ratio(m, gp, source_velocities[i], v_disc)
        end
        # get the mean
        g_total / length(grouping)
    end

    g_interp = _make_interpolation(bins, gs)
    grouped_I = _get_grouped_intensity(T, groupings, intensity)

    for (i, I) in enumerate(grouped_I)
        R = bins[i]
        r = i == 1 ? 0 : bins[i-1]
        dr = R - r
        x = SVector(0, R, π / 2, 0)
        v_disc = _disc_velocity(x)
        A = dr * _proper_area(m, x)
        # now stores emissivity
        grouped_I[i] = source_to_disc_emissivity(m, spec, I, A, x, g_interp(R), v_disc)
    end

    ts = [@views(mean(times[grouping])) for grouping in groupings]

    bins, ts, grouped_I
end

function RadialDiscProfile(
    m::AbstractMetric,
    model::AbstractCoronaModel,
    spec::AbstractLocalSpectrum,
    points::AbstractVector{<:AbstractGeodesicPoint},
    source_velocities::AbstractVector;
    intensity = nothing,
    kwargs...,
)
    radii = map(i -> _equatorial_project(i.x), points)
    # ensure sorted: let the user sort so that everything is sure to be
    # in order
    if !issorted(radii)
        error(
            "geodesic points (and therefore also source velocities) must be sorted by radii: use `sortperm(points; by = i -> i.x[2])` to get the sorting permutation for both",
        )
    end

    times = map(i -> i.x[1], points)
    r, t, ε = _build_radial_profile(
        m,
        spec,
        radii,
        times,
        source_velocities,
        points,
        intensity;
        kwargs...,
    )
    RadialDiscProfile(r, t, ε)
end

function RadialDiscProfile(rdp::RadialDiscProfile)
    Base.depwarn(
        "This function is deprecated. Note that `emissivity_profile` now returns a `RadialDiscProfile`.",
        :RadialDiscProfile,
    )
    rdp
end

function RadialDiscProfile(cg::CoronaGeodesics, spec::AbstractLocalSpectrum; kwargs...)
    J = sortperm(cg.geodesic_points; by = i -> _equatorial_project(i.x))
    @views RadialDiscProfile(
        cg.metric,
        cg.model,
        spec,
        cg.geodesic_points[J],
        cg.source_velocity[J];
        kwargs...,
    )
end

function RadialDiscProfile(
    cg::CoronaGeodesics{<:TraceRadiativeTransfer},
    spec::AbstractLocalSpectrum;
    kwargs...,
)
    J = sortperm(cg.geodesic_points; by = i -> _equatorial_project(i.x))
    @views RadialDiscProfile(
        cg.metric,
        cg.model,
        spec,
        cg.geodesic_points[J],
        cg.source_velocity[J];
        intensity = [i.aux[1] for i in cg.geodesic_points[J]],
        kwargs...,
    )
end

"""
    TimeDependentRadialDiscProfile{T}

Time dependent radial disc profile. Each entry in the radii, emissivity, and
time function maps to a specific contribution, along with the appropriate
weighting when summing the emissivities at similar times together.
"""
struct TimeDependentRadialDiscProfile{T} <: AbstractEmissivityProfile
    weights::Vector{T}
    radii::Vector{Vector{T}}
    t::Vector{Vector{T}}
    ε::Vector{Vector{T}}
end

# effectively time averaged values
function emissivity_at(prof::TimeDependentRadialDiscProfile{T}, ρ) where {T}
    sum(eachindex(prof.radii)) do i
        radii = prof.radii[i]
        if (ρ >= radii[1]) && (ρ <= radii[end])
            _make_interpolation(radii, prof.ε[i])(ρ)
        else
            zero(T)
        end
    end
end

function emissivity_interp(prof::TimeDependentRadialDiscProfile{T}, ρ) where {T}
    ts = zeros(T, length(prof.weights))
    εs = zeros(T, length(prof.weights))
    # interpolate the time curve at ρ until we have some (ρ, t)
    for i in eachindex(prof.radii)
        radii = prof.radii[i]
        if ρ >= radii[1] && ρ <= radii[end]
            t_interp = _make_interpolation(radii, prof.t[i])
            ε_interp = _make_interpolation(radii, prof.ε[i])
            ts[i] = t_interp(ρ)
            εs[i] = ε_interp(ρ)
        else
            ts[i] = NaN
            εs[i] = NaN
        end
    end

    J = sortperm(ts)
    _make_interpolation(ts[J], εs[J])
end

function emissivity_interp_limits(prof::TimeDependentRadialDiscProfile{T}, ρ) where {T}
    ts = map(eachindex(prof.radii)) do i
        radii = prof.radii[i]
        if ρ >= radii[1] && ρ <= radii[end]
            t_interp = _make_interpolation(radii, prof.t[i])
            t_interp(ρ)
        else
            NaN
        end
    end
    filter!(!isnan, ts)
    if length(ts) > 0
        extrema(ts)
    else
        (zero(T), zero(T))
    end
end

# TODO: is this still used?
"""
    struct RingCoronaProfile{T} <: AbstractEmissivityProfile

A specialised disc profile that can stores the various
[`TimeDependentRadialDiscProfile`](@ref) for the ring-like extended corona.
"""
struct RingCoronaProfile{T} <: AbstractEmissivityProfile
    left_arm::TimeDependentRadialDiscProfile{T}
    right_arm::TimeDependentRadialDiscProfile{T}
end

function emissivity_at(prof::RingCoronaProfile{T}, ρ) where {T}
    emissivity_at(prof.left_arm, ρ) + emissivity_at(prof.right_arm, ρ)
end

function emissivity_interp(prof::RingCoronaProfile{T}, ρ) where {T}
    left_arm = emissivity_interp(prof.left_arm, ρ)
    right_arm = emissivity_interp(prof.right_arm, ρ)
    function _add_arms(t)
        left = if t >= left_arm.t[1] && t <= left_arm.t[end]
            left_arm(t)
        else
            zero(T)
        end
        right = if t >= right_arm.t[1] && t <= right_arm.t[end]
            right_arm(t)
        else
            zero(T)
        end

        left + right
    end
end

function emissivity_interp_limits(prof::RingCoronaProfile, ρ)
    l_min, l_max = emissivity_interp_limits(prof.left_arm, ρ)
    r_min, r_max = emissivity_interp_limits(prof.right_arm, ρ)
    min(l_min, r_min), max(l_max, r_max)
end

# TODO: rework this
struct DiscCoronaProfile{T,F} <: AbstractEmissivityProfile
    radii::Vector{T}
    rings::Vector{RingCoronaProfile{T}}
    propagation_velocity::F
end

function with_propagation_velocity(d::DiscCoronaProfile, func::Function)
    DiscCoronaProfile(d.radii, d.rings, func)
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(prof::DiscCoronaProfile))
    print(
        io,
        """DiscCoronaProfile
  . N rings      : $(length(prof.radii))
  . r (min, max) : $(extrema(prof.radii))
""",
    )
end

function _ring_weighting(prof::DiscCoronaProfile, i)
    δr = prof.radii[2] - prof.radii[1]
    prof.radii[i] * δr
end

function emissivity_at(prof::DiscCoronaProfile{T}, ρ) where {T}
    sum(eachindex(prof.radii)) do i
        emissivity_at(prof.rings[i], ρ) * _ring_weighting(prof, i)
    end
end

function emissivity_interp(prof::DiscCoronaProfile{T}, ρ) where {T}
    funcs = [emissivity_interp(ring, ρ) for ring in prof.rings]
    dts = [prof.propagation_velocity(radius) for radius in prof.radii]
    function _sum_ring_contributions(t)
        sum(eachindex(funcs)) do i
            funcs[i](t - dts[i]) * _ring_weighting(prof, i)
        end
    end
end

function emissivity_interp_limits(prof::DiscCoronaProfile, ρ)
    function _limits_with_velocity(radius, ring)
        dt = prof.propagation_velocity(radius)
        _min, _max = emissivity_interp_limits(ring, ρ)
        (_min + dt, _max + dt)
    end

    _min, _max = _limits_with_velocity(prof.radii[1], prof.rings[1])
    for i = 2:lastindex(prof.rings)
        l_min, l_max = _limits_with_velocity(prof.radii[i], prof.rings[i])
        _min = min(_min, l_min)
        _max = max(_max, l_max)
    end
    (_min, _max)
end
