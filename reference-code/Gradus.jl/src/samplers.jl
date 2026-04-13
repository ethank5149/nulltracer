"""
    abstract type AbstractDirectionSampler{SkyDomain,Generator}

Used to provide an abstract way of sampling directions on a (local) sky. The
sampler provides the projection, which covers the [`AbstractSkyDomain`](@ref)
using an [`AbstractGenerator`](@ref) to draw samples.

There are two sampler implementations available
- [`EvenSampler`](@ref)
- [`WeierstrassSampler`](@ref)

These implement the following methods:
- [`geti`](@ref)
- [`sample_azimuthal`](@ref)
- [`sample_elevation`](@ref)

For convenience, there is also a [`sample_angles`](@ref) function, which need
not be implemented other than for certain cases.

## Notes

The available domains are:
- [`BothHemispheres`](@ref)
- [`LowerHemisphere`](@ref)

The available generators are:
- [`RandomGenerator`](@ref), for Monte-Carlo methods
- [`GoldenSpiralGenerator`](@ref), also for Monte-Carlo methods, using the
  Golden-Spiral method of sampling evenly on a sphere.

## Examples

```julia
sampler = EvenSampler()
```
"""
abstract type AbstractDirectionSampler{SkyDomain,Generator} end

"""
    geti(sm::AbstractDirectionSampler, j, N)

Generate a new draw, given that this is the `j`th draw of `N`
"""
geti(sm::AbstractDirectionSampler, j, N) = error("Not implemented for $(typeof(sm)).")

"""
    sample_azimuthal(sm::AbstractDirectionSampler, i)

Return the azimuthal angle for the `i`th sample.
"""
sample_azimuthal(sm::AbstractDirectionSampler, i) =
    error("Not implemented for $(typeof(sm)).")

"""
    sample_elevation(sm::AbstractDirectionSampler, i)

Return the elevation angle for the `i`th sample.
"""
sample_elevation(sm::AbstractDirectionSampler, i) =
    error("Not implemented for $(typeof(sm)).")

"""
    sample_angles(sm::AbstractDirectionSampler, i, N)

Return both the elevation and azimuthal angle for the `i`th sample of `N`.
"""
sample_angles(sm::AbstractDirectionSampler, i, N) =
    (sample_elevation(sm, i / N), mod2pi(sample_azimuthal(sm, i)))

"""
Sample only the lower hemisphere ``\\theta \\in [0, \\pi / 2]``.
"""
struct LowerHemisphere <: AbstractSkyDomain end
"""
Sample both hemispheres ``\\theta \\in [0, \\pi)``.
"""
struct BothHemispheres <: AbstractSkyDomain end

struct RandomGenerator <: AbstractGenerator end
struct GoldenSpiralGenerator <: AbstractGenerator end
struct EvenGenerator <: AbstractGenerator end

"""
    EvenSampler(
        domain = BothHemispheres(),
        generator = GoldenSpiralGenerator(),
    )

Sample the full sky evenly over the sphere. This performs the Jacobian
correction ``\\sin(\\theta)`` to the probability density function.
"""
struct EvenSampler{D,G} <: AbstractDirectionSampler{D,G}
    EvenSampler(domain::AbstractSkyDomain, generator::AbstractGenerator) =
        new{typeof(domain),typeof(generator)}()
    EvenSampler(;
        domain::AbstractSkyDomain = BothHemispheres(),
        generator::AbstractGenerator = GoldenSpiralGenerator(),
    ) = EvenSampler(domain, generator)
end

"""
    EvenSampler(
        domain = BothHemispheres(),
        generator = GoldenSpiralGenerator(),
    )

Sample the sphere according to the Weierstrass projection, that is, such that
the points are evenly distributed on the (flat) projective plane instead of on
the sky sphere.
"""
struct WeierstrassSampler{D,G} <: AbstractDirectionSampler{D,G}
    resolution::Float64
    WeierstrassSampler(res, domain::AbstractSkyDomain, generator::AbstractGenerator) =
        new{typeof(domain),typeof(generator)}(res)
    WeierstrassSampler(;
        res = 100.0,
        domain::AbstractSkyDomain = BothHemispheres(),
        generator::AbstractGenerator = GoldenSpiralGenerator(),
    ) = WeierstrassSampler(res, domain, generator)
end

@inline geti(::AbstractDirectionSampler{D,EvenGenerator}, i, N) where {D} = i / N
@inline geti(::AbstractDirectionSampler{D,GoldenSpiralGenerator}, i, N) where {D} = i
@inline geti(::AbstractDirectionSampler{D,RandomGenerator}, i, N) where {D} =
    rand(Float64) * N

@inline sample_azimuthal(::AbstractDirectionSampler{D,EvenGenerator}, i) where {D} = 2π * i
@inline sample_azimuthal(::AbstractDirectionSampler{D,GoldenSpiralGenerator}, i) where {D} =
    π * (1 + √5) * i
@inline sample_azimuthal(::AbstractDirectionSampler{D,RandomGenerator}, i) where {D} =
    2π * i

@inline sample_angles(sm::WeierstrassSampler, i, N) =
    (sample_elevation(sm, i), mod2pi(sample_azimuthal(sm, i)))

@inline sample_elevation(::EvenSampler{LowerHemisphere}, i) = acos(1 - i)
@inline sample_elevation(::EvenSampler{BothHemispheres}, i) = acos(1 - 2i)
@inline sample_elevation(sm::WeierstrassSampler{LowerHemisphere}, i) =
    2atan(√(sm.resolution / i))
@inline function sample_elevation(sm::WeierstrassSampler{BothHemispheres}, i)
    ϕ = 2atan(√(sm.resolution / i))
    if iseven(i)
        ϕ
    else
        π - ϕ
    end
end

# TODO: check these aren't missing a sin(θ) term in the last row?
function _cart_to_spher_jacobian(θ, ϕ)
    @SMatrix [
        sin(θ)*cos(ϕ) sin(θ)*sin(ϕ) cos(θ)
        cos(θ)*cos(ϕ) cos(θ)*sin(ϕ) -sin(θ)
        -sin(ϕ) cos(ϕ) 0
    ]
end

function _spher_to_cart_jacobian(θ, ϕ, r)
    @SMatrix [
        sin(θ)*cos(ϕ) r*cos(θ)*cos(ϕ) -r*sin(θ)*sin(ϕ)
        sin(θ)*sin(ϕ) r*cos(θ)*sin(ϕ) r*sin(θ)*cos(ϕ)
        cos(θ) -r*sin(θ) 0
    ]
end

function _cart_local_direction(θ, ϕ)
    @SVector [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
end

@inbounds function sky_angles_to_velocity(
    m::AbstractMetric{T},
    x,
    v_source,
    θ,
    ϕ;
    E₀ = one(T),
) where {T}
    # multiply by -1 for consitency with `LowerHemisphere`
    hat = -1 * _cart_local_direction(θ, ϕ)
    # to spherical coordinates
    J = _cart_to_spher_jacobian(x[3], x[4])
    k = J * hat

    p = SVector(E₀, E₀ * k[1], E₀ * k[2], E₀ * k[3])

    B = tetradframe_matrix(m, x, v_source)
    (B * p)
end

export LowerHemisphere,
    BothHemispheres, RandomGenerator, GoldenSpiralGenerator, EvenSampler, WeierstrassSampler
