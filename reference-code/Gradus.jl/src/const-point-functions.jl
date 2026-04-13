"""
    module ConstPointFunctions

Module defining a number of `const` [`Gradus.AbstractPointFunction`](@ref), serving different utility
or common purposes for analysis.
"""
module ConstPointFunctions
using ..Gradus:
    PointFunction,
    FilterPointFunction,
    redshift_function,
    StatusCodes,
    KerrMetric,
    KerrSpacetimeFirstOrder,
    AbstractMetric,
    interpolate_redshift
# for doc bindings
import ..Gradus

"""
    filter_early_term(m::AbstractMetric, gp::AbstractGeodesicPoint, max_time)

A [`FilterPointFunction`](@ref) that filters geodesics that termined early (i.e., did not reach maximum integration time or effective infinity).
Default: `NaN`.
"""
function filter_early_term(T::Type = Float64)
    FilterPointFunction((m, gp, max_time) -> gp.λ_max < max_time, T(NaN))
end

"""
    filter_intersected(m::AbstractMetric, gp::AbstractGeodesicPoint, max_time)

A [`FilterPointFunction`](@ref) that filters geodesics which intersected with the accretion
disc. Default: `NaN`.
"""
function filter_intersected(T::Type = Float64)
    FilterPointFunction(
        (m, gp, max_time) -> gp.status == StatusCodes.IntersectedWithGeometry,
        T(NaN),
    )
end

"""
    affine_time(m::AbstractMetric, gp::AbstractGeodesicPoint, max_time)

A [`PointFunction`](@ref) returning the affine integration time at the endpoint of the geodesic.
"""
function affine_time()
    PointFunction((m, gp, max_time) -> gp.λ_max)
end

"""
    shadow(m::AbstractMetric, gp::AbstractGeodesicPoint, max_time)

A [`PointFunction`](@ref) which colours the shadow of the black hole for any disc-less render.
Equivalent to `ConstPointFunctions.affine_time ∘ ConstPointFunctions.filter_early_term`.
"""
function shadow(T::Type = Float64)
    affine_time() ∘ filter_early_term(T)
end

"""
    redshift(m::AbstractMetric, x_obs)

Returns a [`PointFunction`](@ref) that calculates the redshift of a medium as
observed by an observer at `x_obs`.

Calculate the analytic redshift at a given geodesic point, assuming equatorial,
geometrically thin accretion disc. Implementation depends on the metric type,
with non-Kerr metrics using an interpolation scheme within the plunging region
via [`interpolate_redshift`](@ref).

Calls [`redshift_function`](@ref) to dispatch different implementations.
"""
redshift(m::KerrMetric, x; kwargs...) = PointFunction(redshift_function(m, x; kwargs...))
redshift(m::KerrSpacetimeFirstOrder, x; kwargs...) =
    PointFunction(redshift_function(m, x; kwargs...))
redshift(m::AbstractMetric, x; kwargs...) =
    PointFunction(interpolate_redshift(m, x; kwargs...))

end # module

export ConstPointFunctions
