"""
    PhotonFractions{T}

Stores photon fraction counts, with the following fields:

    "Fraction of photons intersecting the disc (above_mark + below_mark)."
    disc::T
    "Fraction of photons escaping to infinity."
    infinity::T
    "Fraction of photons fallen into the black hole."
    black_hole::T
    "Fraction of photons above a marked radius on the disc."
    above_mark::T
    "Fraction of photons below a marked radius on the disc."
    below_mark::T
    "The marked radius in units of rg."
    mark::T
"""
Base.@kwdef struct PhotonFractions{T}
    "Fraction of photons intersecting the disc (above_mark + below_mark)."
    disc::T
    "Fraction of photons escaping to infinity."
    infinity::T
    "Fraction of photons fallen into the black hole."
    black_hole::T
    "Fraction of photons above a marked radius on the disc."
    above_mark::T
    "Fraction of photons below a marked radius on the disc."
    below_mark::T
    "The marked radius in units of rg."
    mark::T
end

"""
    photon_fractions(
        m::AbstractMetric,
        d::AbstractAccretionGeometry,
        corona::AbstractCorona;
        kwargs...
    )

Calculate the illumination fractions (also known as _reflection fractions_) for
a particular disc and coronal geometry.

The `kwargs` are generally similar to the respective
[`emissivity_profile`](@ref) function. Additionally this function accepts the
keyword `mark::Real`, that can be used to set a radius above and below which
the photon fractions are split in the disc illumination.

The return is an instance of [`PhotonFractions`](@ref).

Depending on the geometry used, different dispatches may be more optimal. For
example, if calculating emissivity fractions in tandem for a
[`RingCorona`](@ref), consider the following recipe:

    slices = Gradus.corona_slices(m, d, corona)
    fractions = photon_fractions(slices; mark = Gradus.isco(m))
    prof = make_approximation(m, slices)
"""
photon_fractions

function _count_pointlike_fractions(angles::Vector{T}, gps; mark = zero(T)) where {T}
    above_mark = 0
    below_mark = 0
    black_hole = 0
    infinity = 0
    # need to weight them by their angle
    @assert issorted(angles)
    for (ang, g) in zip(angles, gps)
        w = sin(ang)
        if g.status == StatusCodes.IntersectedWithGeometry
            if g.x[2] >= mark
                above_mark += w
            else
                below_mark += w
            end
        elseif g.status == StatusCodes.WithinInnerBoundary
            black_hole += w
        elseif g.status == StatusCodes.NoStatus || g.status == StatusCodes.OutOfDomain
            infinity += w
        else
            throw("unreachable")
        end
    end
    disc = above_mark + below_mark
    total = disc + black_hole + infinity
    PhotonFractions(;
        disc = disc / total,
        black_hole = black_hole / total,
        infinity = infinity / total,
        below_mark = below_mark / total,
        above_mark = above_mark / total,
        mark = mark,
    )
end

function photon_fractions(
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    corona::LampPostModel;
    mark = Gradus.isco(m),
    kwargs...,
)
    geod_info = point_source_geodesics(m, d, corona; kwargs...)
    _count_pointlike_fractions(geod_info.δs, geod_info.gps)
end

function photon_fractions(
    m::AbstractMetric,
    d::AbstractAccretionGeometry,
    corona::RingCorona;
    mark = Gradus.isco(m),
    kwargs...,
)
    slices = corona_slices(m, d, corona; kwargs...)
    photon_fractions(slices; mark = mark)
end

"""
    photon_fractions(slices::RingPointApproximationSlices; kwargs...)

Calculate the illumination fraction from the [`corona_slices`](@ref) of a
[`RingCorona`](@ref).
"""
function photon_fractions(
    slices::RingPointApproximationSlices{T};
    mark = zero(T),
    kwargs...,
) where {T}
    above_mark::T = 0
    below_mark::T = 0
    black_hole::T = 0
    infinity::T = 0

    @assert issorted(slices.βs)

    for i = 1:(lastindex(slices.βs)-1)
        fracs = photon_fraction_for_trace(slices.traces[i]; mark = mark, kwargs...)
        dβ = slices.βs[i+1] - slices.βs[i]

        above_mark += fracs.above_mark * dβ
        below_mark += fracs.below_mark * dβ
        black_hole += fracs.black_hole * dβ
        infinity += fracs.infinity * dβ
    end

    disc = above_mark + below_mark
    total = disc + black_hole + infinity

    PhotonFractions(;
        disc = disc / total,
        black_hole = black_hole / total,
        infinity = infinity / total,
        below_mark = below_mark / total,
        above_mark = above_mark / total,
        mark = mark,
    )
end

export photon_fractions
