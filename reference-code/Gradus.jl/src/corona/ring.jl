const INVPHI = (sqrt(5) - 1) / 2

"""
A thread local cache for doing the calculations related to tracing the arms of
a point-approximation of a ring corona.

One is created, which is then copied once for each thread.
"""
struct _RingCoronaCache{T,S<:EmissivityProfileSetup{true},A}
    setup::S
    "Position of a point in the ring."
    x::SVector{4,T}
    "Velocity of the point in the ring."
    v::SVector{4,T}
    "(Local) poloidal angles that have been calculated."
    angles::Vector{T}
    "Geodesic points calculated"
    gps::Vector{GeodesicPoint{T,A}}
    "An optimising numeric buffer that is the same length of `gps` for storing
    partial results. Only used during processing, not during value accumulation."
    _buffer::Vector{T}
    "Used to track which angles and gps we have written so far. `_index == 0` means the array is empty."
    _index::Ref{Int}
    "Number of geodesics for all the arms"
    N::Int
    "Maximal number of iterations that the extremiser should make"
    extrema_iter::Int
end

function Base.copy(cache::_RingCoronaCache)
    _RingCoronaCache(
        cache.setup,
        cache.x,
        cache.v,
        deepcopy(cache.angles),
        deepcopy(cache.gps),
        deepcopy(cache._buffer),
        Ref(cache._index[]),
        cache.N,
        cache.extrema_iter,
    )
end

function _add_to_cache!(
    cache::_RingCoronaCache{T,S,A},
    ang::T,
    gp::GeodesicPoint{T,A},
) where {T,S,A}
    cache._index[] += 1
    i::Int = cache._index[]

    if i > lastindex(cache.angles)
        # TODO: some kind of warning?
        @warn "Not enough pre-allocated space, ignoring values!"
        return
    end

    cache.angles[i] = ang
    cache.gps[i] = gp
end

function _RingCoronaCache(
    setup::EmissivityProfileSetup,
    m::AbstractMetric,
    model::RingCorona;
    h = 1e-7,
    extrema_iter = 80,
)
    x, v = sample_position_velocity(m, model)

    # TODO: infer this
    GPType = GeodesicPoint{eltype(x),Nothing}

    N = (setup.n_samples - 2 * extrema_iter)
    if (N <= 0)
        error(
            "Too few samples for selected method. Pass `n_samples` of more than $(2 * extrema_iter)",
        )
    end
    if !iseven(N)
        error("`n_samples` must be an even number for this method (given $N)")
    end

    angles = zeros(eltype(x), setup.n_samples)
    _buffer = zeros(eltype(x), setup.n_samples)
    gps = Vector{GPType}(undef, setup.n_samples)

    num_per_arm = div(N, 2)

    # setup the angles we are going to sample evenly over the sky
    left_arm_angles = range(h, π - h, num_per_arm)
    for (i, a) in enumerate(left_arm_angles)
        angles[i] = a
        angles[i+num_per_arm] = mod2pi(a + π)
    end
    sort!(@views(angles[1:N]))

    _RingCoronaCache(setup, x, v, angles, gps, _buffer, Ref(0), N, extrema_iter)
end

"""
Returns a function that takes position, velocity, local angle, and the slice
angle and returns the initial velocity vector for the geodesics.
"""
function rotatorfunctor(m::AbstractMetric{T}, x::SVector, v::SVector, β) where {T}
    angle_of_axis = x[3]
    k = Gradus._cart_local_direction(angle_of_axis, zero(T))

    function _rotator(θ; φ = β)
        q = _cart_local_direction(θ + angle_of_axis, zero(T))
        b = rodrigues_rotate(k, q, φ)
        # convert back to spherical coordinates
        ph = atan(b[2], b[1])
        th = atan(sqrt(b[2]^2 + b[1]^2), b[3])

        sky_angles_to_velocity(m, x, v, th, ph)
    end
end

"""
    _determine_bracket(N, angles, gps, comparator)

Determines the initial bracketing interval for a finding the minima or maxima
of a function. Unlike a regular bracketing method, this function tries to
ignore those geodesics that did not intersect with the disc.

The function uses the equatorial projected radius on the disc, and returns the
bracketing angles, along with the best estimate of the radius. A minimal delta
is applied to the angles to broaden the range of possible values.
"""
function _determine_bracket(
    N::Int,
    angles::Vector,
    gps::Vector{<:GeodesicPoint{T}},
    comparator::Function,
) where {T}
    a = first(angles)
    b = first(angles)
    estimate::Union{Nothing,T} = nothing
    for i = 1:N
        gp = gps[i]

        if gp.status == StatusCodes.IntersectedWithGeometry
            rho = _equatorial_project(gp.x)

            if isnothing(estimate) || comparator(rho, estimate)
                estimate = rho
                a = b = angles[i]

                if i < N && gps[i+1].status != StatusCodes.IntersectedWithGeometry
                    # this will likely be a better upper limit
                    b = angles[i+1]
                end

                if i > 1 && gps[i-1].status != StatusCodes.IntersectedWithGeometry
                    # this will likely be a better lower limit
                    a = angles[i-1]
                end
            end
        end
    end

    best_estimate::T = estimate
    delta = π / min(N, 32)
    a - 2 * delta, b + 2 * delta, best_estimate
end

"""
Given an already traced set of angles, use the best estimate of where the maxima
as and then bisect to try and find a better maxima.
"""
function _golden_bracket!(
    cache::_RingCoronaCache{T},
    tracer::Function,
    target::Val{Target};
    kwargs...,
) where {T,Target}
    comparator = if Target == :minima
        (a, b) -> a < b
    elseif Target == :maxima
        (a, b) -> a > b
    else
        error("Unknown target: $Target")
    end

    a, b, best_estimate = _determine_bracket(cache.N, cache.angles, cache.gps, comparator)
    a_init, b_init = a, b

    function _objective(θ)
        gp = tracer(θ)
        if gp.status != StatusCodes.IntersectedWithGeometry
            # just return the best estimate, as this will (hopefully) be worse
            # than whatever we've calculated so far
            gp, best_estimate
        else
            gp, _equatorial_project(gp.x)
        end
    end

    # take a bracketing interval
    c_value = 0.0

    n = 0
    iters = 0
    too_many_iters = false
    while (n < cache.extrema_iter)
        c = b - (b - a) * INVPHI
        d = a + (b - a) * INVPHI

        c_gp, c_value = _objective(c)
        d_gp, d_value = _objective(d)

        if comparator(c_value, d_value)
            if comparator(c_value, best_estimate)
                n += 1
                _add_to_cache!(cache, c, c_gp)
            elseif too_many_iters
                n += 1
            end
            b = d
        else
            if comparator(d_value, best_estimate)
                n += 1
                _add_to_cache!(cache, d, d_gp)
            elseif too_many_iters
                n += 1
            end
            a = c
        end

        iters += 1
        if !too_many_iters && (iters > cache.extrema_iter)
            @warn "Too many iterations solving for $Target" maxlog = 1
            too_many_iters = true
        end
    end
end

"""
    function _ring_arm_traces!(
        cache::_RingCoronaCache,
        m::AbstractMetric,
        d::AbstractAccretionGeometry,
        β_angle;
    )

Calculates all of the base geodesics and optimizer geodesics for a given
point-approximation of a ring corona.
"""
function _ring_arm_traces!(
    cache::_RingCoronaCache,
    m::AbstractMetric{T},
    d::AbstractAccretionGeometry,
    β_angle;
    solver_opts...,
) where {T}
    _velfunc = rotatorfunctor(m, cache.x, cache.v, β_angle)

    # init a reusable integrator
    integ = _init_integrator(
        m,
        cache.x,
        _velfunc(0.0),
        d,
        # TODO: make this a parameter
        1_000_000.0;
        save_on = false,
        callback = domain_upper_hemisphere(),
        chart = chart_for_metric(m, 1_000_000.0),
        integrator_verbose = false,
        solver_opts...,
    )

    function _tracer(θ)
        init_v = _velfunc(θ)
        _solve_reinit!(integ, vcat(cache.x, init_v))
    end

    for i = 1:cache.N
        cache.gps[i] = _tracer(cache.angles[i])
    end
    # update the index
    cache._index[] = cache.N

    # do the minima and maxima extrema optimiser
    _golden_bracket!(cache, _tracer, Val{:minima}())
    _golden_bracket!(cache, _tracer, Val{:maxima}())
end

function unpack_traces(cache::_RingCoronaCache)
    last_index::Int = cache._index[]
    @views cache.angles[1:last_index], cache.gps[1:last_index]
end

function canonical_orders!(slices, rs)
    for I in slices
        sort!(I, by = i -> rs[i])
    end
    slices
end

# TODO: these are all incredibly allocating, when we could pre-allocate a
# buffer and reuse it
function _split_branches_further(inds, rs; cutoff_r = 1e3, delta = 0.01, min_length = 3)
    N = lastindex(inds)
    prev, first_i::Int = @views findmin(rs[inds])
    splits = Int[1, first_i]
    increasing::Bool = true

    for j = 0:(N-1)
        k = (j + first_i) % N + 1
        r = rs[inds[k]]

        if r > cutoff_r
            increasing = false
            prev = r
            continue
        end
        if increasing && (r < prev - delta)
            increasing = false
            push!(splits, k > 1 ? k - 1 : lastindex(inds))
        elseif !increasing && (r > prev + delta)
            increasing = true
            push!(splits, k > 1 ? k - 1 : lastindex(inds))
        end
        prev = r
    end

    sort!(splits)

    if length(splits) > 2
        # merge slices that are smaller than the minimum length together
        slices = Vector{Int}[]
        i1 = first(splits)
        for i2 in splits[2:end]
            if i2 - i1 >= min_length
                if (N - i2) <= min_length
                    i2 = N
                end
                push!(slices, inds[i1:i2])
                i1 = i2
            end
        end
        if i1 < N
            push!(slices, inds[i1:end])
        end
        slices
    else
        [inds]
    end
end

function _split_arms_indices(angles, ρs)
    # split the sky into a left and right side, such that each side has strictly
    # sortable and monotonic r as a function of θ on the local sky

    # some values may be NaN so need to avoid those
    _start_index = findfirst(!isnan, ρs)

    _min_ρ_index = _start_index
    _max_ρ_index = _start_index
    _r_min = ρs[_start_index]
    _r_max = ρs[_start_index]

    for (i, r) in enumerate(ρs)
        r == NaN && continue
        if r < _r_min
            _min_ρ_index = i
            _r_min = r
        end
        if r > _r_max
            _max_ρ_index = i
            _r_max = r
        end
    end

    min_ρ_index, max_ρ_index =
        min(_min_ρ_index, _max_ρ_index), max(_min_ρ_index, _max_ρ_index)

    r1 = vcat(collect((max_ρ_index+1):lastindex(ρs)), collect(1:(min_ρ_index-1)))
    r2 = collect((min_ρ_index+1):max_ρ_index)

    l1 = canonical_orders!(_split_branches_further(r1, ρs), ρs)
    l2 = canonical_orders!(_split_branches_further(r2, ρs), ρs)
    vcat(l1, l2)
end

"""
    PointSlice

A struct containing the raw values from a given ``\\beta`` slice trace.
"""
struct PointSlice{T}
    "Status codes of each photon"
    statuses::Vector{StatusCodes.T}
    "Latitudal angles"
    θ::Vector{T}
    "Redshift"
    g::Vector{T}
    "Lorentz factor"
    γ::Vector{T}
    "Radial coordinate"
    r::Vector{T}
    "Disc azimuthal coordinate"
    φ::Vector{T}
    "Jacobian"
    drdθ::Vector{T}
    "Corona to disc time"
    t::Vector{T}
end

struct TimeDependentEmissivityBranch{T}
    "Latitudal angles"
    θ::Vector{T}
    "Radial disc coordinate"
    r::Vector{T}
    "Azimuthal disc coordinate"
    φ::Vector{T}
    "Corona to disc time."
    t::Vector{T}
    "Emissivity"
    ε::Vector{T}
end

function _branch_emissivity(m::AbstractMetric, slice::PointSlice, I::Vector{Int}, spectrum)
    map(I) do i
        Jinv = slice.drdθ[i]
        A = _proper_area(m, slice.r[i], π/2) * Jinv
        point_source_equatorial_disc_emissivity(
            spectrum,
            slice.θ[i],
            slice.g[i],
            A,
            slice.γ[i],
        )
    end
end

"""
    split_into_branches(m::AbstractMetric, slice::PointSlice, spectrum)

Split a given [`PointSlice`](@ref) into a number of
[`TimeDependentEmissivityBranch`](@ref) by cutting the curves of `(θ, r)` into
bijective branches.

Returns a vector of [`TimeDependentEmissivityBranch`](@ref).
"""
function split_into_branches(m::AbstractMetric, slice::PointSlice, spectrum)
    splits = _split_arms_indices(slice.θ, slice.r)
    map(splits) do I
        em = _branch_emissivity(m, slice, I, spectrum)
        TimeDependentEmissivityBranch(slice.θ[I], slice.r[I], slice.φ[I], slice.t[I], em)
    end
end

function empty_point_slice(; n = 1000, T = Float64)
    PointSlice(
        Vector{StatusCodes.T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
    )
end

"""
    RingPointApproximationSlices

Used to hold all of the different [`PointSlice`](@ref) slices for their
respective values of ``\\beta``.
"""
struct RingPointApproximationSlices{T}
    "The slice angle."
    βs::Vector{T}
    "The slice of traces corresponding to a given β"
    traces::Vector{PointSlice{T}}
end

"""
    function interpolate_slice(
        slice::PointSlice{T},
        pas::RingPointApproximationSlices{T},
        β,
    )

Interpolate a new slice for ``\\beta`` given a set of calculated slices in
[`RingPointApproximationSlices`](@ref).
"""
function interpolate_slice(pas::RingPointApproximationSlices, β::T) where {T}
    idx = clamp(searchsortedlast(pas.βs, β), 1, length(pas.βs) - 1)
    x1, x2 = pas.βs[idx], pas.βs[idx+1]
    # interpolation weight
    w = (β - x1) / (x2 - x1)

    s1 = pas.traces[idx]
    s2 = pas.traces[idx+1]

    new_thetas = sort!(unique!(vcat(s1.θ, s2.θ)))

    slice = empty_point_slice(; n = length(new_thetas), T = T)

    # interpolate the slice over the common support
    g1 = Gradus.NaNLinearInterpolator(s1.θ, s1.g, NaN)
    g2 = Gradus.NaNLinearInterpolator(s2.θ, s2.g, NaN)

    γ1 = Gradus.NaNLinearInterpolator(s1.θ, s1.γ, NaN)
    γ2 = Gradus.NaNLinearInterpolator(s2.θ, s2.γ, NaN)

    r1 = Gradus.NaNLinearInterpolator(s1.θ, s1.r, NaN)
    r2 = Gradus.NaNLinearInterpolator(s2.θ, s2.r, NaN)

    φ1 = Gradus.NaNLinearInterpolator(s1.θ, s1.φ, NaN)
    φ2 = Gradus.NaNLinearInterpolator(s2.θ, s2.φ, NaN)

    t1 = Gradus.NaNLinearInterpolator(s1.θ, s1.t, NaN)
    t2 = Gradus.NaNLinearInterpolator(s2.θ, s2.t, NaN)

    drdθ1 = Gradus.NaNLinearInterpolator(s1.θ, s1.drdθ, NaN)
    drdθ2 = Gradus.NaNLinearInterpolator(s2.θ, s2.drdθ, NaN)

    i::Int = 1
    for th in new_thetas
        slice.θ[i] = th
        slice.g[i] = _linear_interpolate(g1(th), g2(th), w)
        slice.γ[i] = _linear_interpolate(γ1(th), γ2(th), w)
        slice.r[i] = _linear_interpolate(r1(th), r2(th), w)
        slice.φ[i] = _linear_interpolate(φ1(th), φ2(th), w)
        slice.t[i] = _linear_interpolate(t1(th), t2(th), w)
        slice.drdθ[i] = _linear_interpolate(drdθ1(th), drdθ2(th), w)
        i += 1
    end

    slice
end

function _drdθ(
    cache::_RingCoronaCache,
    m::AbstractMetric,
    d::AbstractAccretionDisc,
    θs,
    mask,
    β;
    solver_opts...,
)
    integ, ps = _make_emissivity_tracer(m, cache.x, cache.v, d; solver_opts...)

    _velfunc = Gradus.rotatorfunctor(m, cache.x, cache.v, β)

    # this inner function calculates the Jacobian J = ∂(r, ϕ) / ∂(Θ, β)
    function _trace_r(th::T) where {T}
        th_dual = _make_emiss_Dual(th; d1 = one(T))
        beta_dual = _make_emiss_Dual(β; d2 = one(T))
        # do the trace
        v = _velfunc(th_dual; φ = beta_dual)
        gp = _solve_reinit!(integ, vcat(ps.xinit, v))

        y_dual = SVector{2,typeof(th_dual)}(gp.x[2], gp.x[4])

        res = ForwardDiff.value.(Tag_Make_Emissivity, y_dual)
        jac = _extract_jacobian(Tag_Make_Emissivity, y_dual, SVector{2}(th_dual, beta_dual))
    end

    map(eachindex(θs)) do i
        if !mask[i]
            NaN
        else
            J = _trace_r(θs[i])
            abs(det(J))
        end
    end
end

function arrange_slice!(
    cache::_RingCoronaCache,
    m::AbstractMetric,
    d::AbstractAccretionDisc,
    β;
    slice_max_r = 1e5,
    solver_opts...,
)
    _all_angles, _all_gps = unpack_traces(cache)

    # calculate the projected radial coordinate into the buffer
    for (i, gp) in enumerate(_all_gps)
        cache._buffer[i] = _equatorial_project(gp.x)
    end
    _all_rs = cache._buffer[1:length(_all_gps)]

    J = sortperm(_all_angles)
    unique!(i -> _all_rs[i], J)

    all_angles = _all_angles[J]
    all_rs = _all_rs[J]
    all_gps = _all_gps[J]

    # function for obtaining keplerian velocities
    _disc_velocity = _keplerian_velocity_projector(m, d)

    gammas = similar(all_rs)
    gs = map(enumerate(all_gps)) do dat
        i, p = dat
        v_disc = _disc_velocity(p.x)
        gammas[i] = lorentz_factor(m, p.x, v_disc)
        energy_ratio(m, p, cache.v, v_disc)
    end

    all_φs = [gp.x[4] for gp in all_gps]

    mask = map(all_gps) do i
        status_good = i.status == StatusCodes.IntersectedWithGeometry
        # above a certain radius pathologies are bound to happen in the
        # Jacobain, because values get so small. In this domain the results are
        # all classical at any rate, so it's fine to curtail at some particular
        # disc radis
        radius_good = i.x[2] < slice_max_r
        status_good && radius_good
    end

    statuses = map(all_gps) do gp
        gp.status
    end

    ts = [i.x[1] for i in all_gps]

    derivs = _drdθ(cache, m, d, all_angles, mask, β; solver_opts...)

    # mask the traces that didn't hit the disc
    # but not on angles, since we use those for interpolating
    @. gs[!mask] = all_rs[!mask] = all_φs[!mask] = ts[!mask] = gammas[!mask] = NaN
    PointSlice(statuses, all_angles, gs, gammas, all_rs, all_φs, derivs, ts)
end

"""
    function corona_slices(
        m::AbstractMetric{T},
        d::AbstractAccretionDisc,
        model::RingCorona;
        kwargs...
    )

    function corona_slices(
        setup::EmissivityProfileSetup{true},
        m::AbstractMetric,
        d::AbstractAccretionDisc,
        model::RingCorona,
        βs;
        solver_kwargs...
    )

Calculates a [`RingPointApproximationSlices`](@ref) for a [`RingCorona`](@ref) for a
given set or range of ``\\beta`` angles. Here, ``\\beta`` is the angle relative
to the radial coordinate vector of the slice of geodesics being calculated (the
'slices of the orange' or 'beachball').

The first function signature is for general use to quickly calculate some
coronal slices. The second is used within the context of an emissivity profile
calculation, where an [`EmissivityProfileSetup`](@ref) is defined.

This function parallelises over CPU threads.
"""
function corona_slices(
    m::AbstractMetric{T},
    d::AbstractAccretionDisc,
    model::RingCorona;
    n_slices = 100,
    βs = DEFAULT_β_ANGLES(n_slices),
    spectrum = PowerLawSpectrum(2.0),
    kwargs...,
) where {T}
    solver_kwargs, setup = EmissivityProfileSetup(T, spectrum; kwargs...)
    corona_slices(setup, m, d, model, βs; solver_kwargs...)
end
function corona_slices(
    setup::EmissivityProfileSetup{true},
    m::AbstractMetric,
    d::AbstractAccretionDisc,
    model::RingCorona,
    βs;
    verbose = false,
    solver_opts...,
)
    cache = _RingCoronaCache(setup, m, model)
    # copy one cache for each thread
    caches = [copy(cache) for i = 1:(Threads.nthreads()-1)]
    push!(caches, cache)

    n_threads = Threads.nthreads()

    progress_bar = init_progress_bar("β slices: ", length(βs), verbose)
    function _func(β)
        thread_cache = caches[_thread_id(n_threads)]

        _ring_arm_traces!(thread_cache, m, d, β; solver_opts...)

        slice = arrange_slice!(thread_cache, m, d, β; solver_opts...)

        # reset thread local for next iteration
        thread_cache._index[] = 0
        ProgressMeter.next!(progress_bar)

        slice
    end

    traces = Gradus._threaded_map(_func, βs)
    RingPointApproximationSlices(βs, traces)
end


struct RingApproximation{T} <: AbstractEmissivityProfile
    βs::Vector{T}
    branches::Vector{Vector{TimeDependentEmissivityBranch{T}}}
end

# TODO: issue a warning that this is slow?
emissivity_at(ra::RingApproximation, r) = emissivity_at(ra, [r])

function emissivity_at(ra::RingApproximation, rs::AbstractVector)
    # need to do a linear interpolation in log space
    # so convert all the branch emissivities to logarithms
    all_branches = reduce(vcat, ra.branches)
    log_branches = map(all_branches) do br
        (; r = log2.(br.r), em = log2.(br.ε), φ = br.φ)
    end

    phi_grid = collect(range(0, 2π, 5000))
    phi_counts = zeros(Int, length(phi_grid) - 1)
    flux = zeros(Float64, length(phi_grid) - 1)

    counters = zeros(Int, size(rs))

    vals = map(enumerate(rs)) do I
        # zero the counter arrays
        phi_counts .= 0
        flux .= 0

        i, r = I
        lr = log2(r)
        for branch in log_branches
            r_min, r_max = extrema(branch.r)
            if (lr >= r_min) && (lr <= r_max)
                exp = _make_interpolation(branch.r, branch.em)(lr)
                ang = _make_interpolation(branch.r, branch.φ)(lr)

                i = searchsortedlast(phi_grid, mod2pi(ang))
                if i > 0
                    phi_counts[i] += 1
                    flux[i] += 2^exp
                end
            end
        end

        # normalise
        @. flux = flux / max(1, phi_counts)
        fill_linear!(flux)
        sum(flux) / length(phi_counts)
    end

    vals
end

function fill_linear!(a::AbstractVector)
    ts = collect(range(0, 1, length(a)))
    mask = @. !iszero(a)

    if count(mask) < 2
        return a
    end

    X = @views ts[mask]
    Y = @views a[mask]

    itp = DataInterpolations.LinearInterpolation(
        Y,
        X;
        extrapolation = DataInterpolations.ExtrapolationType.Constant,
    )
    for i in eachindex(a)
        if !mask[i]
            a[i] = itp(ts[i])
        end
    end
    a
end

function make_approximation(
    m::AbstractMetric,
    slices::RingPointApproximationSlices,
    spectrum;
    βs = collect(range(extrema(slices.βs)..., 1000)),
)
    branches = map(βs) do beta
        slice = interpolate_slice(slices, beta)
        slice_branches = split_into_branches(m, slice, spectrum)
        # remove all nans in r
        filter(slice_branches) do b
            mask = @. isnan(b.r)
            deleteat!(b.r, mask)
            deleteat!(b.φ, mask)
            deleteat!(b.t, mask)
            deleteat!(b.ε, mask)
            deleteat!(b.θ, mask)
            length(b.r) > 1
        end
    end
    RingApproximation(βs, branches)
end

# TODO: move me
struct TimeDependentEmissivityProfile{T} <: AbstractEmissivityProfile
    rs::Vector{T}
    ts::Vector{T}
    matrix::Matrix{T}
end

function emissivity_interp(prof::TimeDependentEmissivityProfile, r)
    ri = min(searchsortedfirst(prof.rs, r), lastindex(prof.rs))
    @views Gradus._make_interpolation(prof.ts, prof.matrix[:, ri])
end

function emissivity_interp_limits(prof::TimeDependentEmissivityProfile, r)
    ri = min(searchsortedfirst(prof.rs, r), lastindex(prof.rs))
    slice = @views prof.matrix[:, ri]
    first_i = findfirst(!isnan, slice)
    last_i = findlast(!isnan, slice)
    if isnothing(first_i) || isnothing(last_i)
        (zero(r), zero(r))
    else
        (prof.ts[first_i], prof.ts[last_i])
    end
end

"""
    photon_fraction_for_trace(trace::PointSlice; mark = 0)

Calculate the photon fraction contributions for a [`PointSlice`](@ref).
Returns a named tuple, containing the (weighted) angular fractions
contributions to `disc`, `black_hole` and `infinity`. Also returned are
`above_mark` and `below_mark`, where the `mark` is a keyword argument that can
be used to mark a particular radius on the disc.

The intended use of `mark` is to set it equal to the ISCO.
"""
function photon_fraction_for_trace(trace::PointSlice{T}; mark = 0) where {T}
    above_mark::T = 0
    below_mark::T = 0
    black_hole::T = 0
    infinity::T = 0

    @assert issorted(trace.θ)
    for i = 1:(lastindex(trace.statuses)-1)
        s = trace.statuses[i]
        if s == StatusCodes.IntersectedWithGeometry
            # if not the same status code, then count the contribution different
            s = trace.statuses[i+1]
        end
        θ = trace.θ[i]
        dθ = trace.θ[i+1] - θ
        w = abs(sin(θ) * dθ)

        if s == StatusCodes.WithinInnerBoundary
            black_hole += w
        elseif s == StatusCodes.IntersectedWithGeometry
            if isnan(trace.r[i]) || trace.r[i] >= mark
                # nans can happen far away on the disc, since in those regions
                # it's difficult to know if the values are sensible. but for
                # photon fractions, it's safe to add them to be above the
                # marked radius, provided the mark is not too large (> 10^4 rg
                # or so).
                above_mark += w
            else
                below_mark += w
            end
        elseif s == StatusCodes.NoStatus || s == StatusCodes.OutOfDomain
            infinity += w
        else
            throw("unreachable")
        end
    end

    disc = above_mark + below_mark
    total = disc + black_hole + infinity

    (;
        disc = disc / total,
        black_hole = black_hole / total,
        infinity = infinity / total,
        below_mark = below_mark / total,
        above_mark = above_mark / total,
    )
end


# TODO: refactor the time-integration to make things like below possible without copy pasting the function
function _integrate_transfer_problem!(
    output::AbstractMatrix,
    setup::IntegrationSetup{T,Profile},
    transfer_function_radial_interpolation,
    r_limits,
    g_grid,
    t_grid;
    g_scale = 1,
) where {
    T,
    Profile<:Union{<:RingCoronaProfile,DiscCoronaProfile,TimeDependentEmissivityProfile},
}
    g_grid_view = @views g_grid[1:(end-1)]

    r_itterator = collect(Grids._geometric_grid(r_limits..., setup.n_radii))
    r2 = first(iterate(r_itterator, 2))
    # prime the first r_prev so that the bin width is r2 - r1
    r_prev = r_limits[1] - (r2 - r_limits[1])

    N_t_steps = 100

    for rₑ in r_itterator
        branch = transfer_function_radial_interpolation(rₑ)
        S_lower = _lower_branch(setup, branch)
        S_upper = _upper_branch(setup, branch)

        Δrₑ = rₑ - r_prev
        # integration weight for this annulus
        θ = Δrₑ * rₑ * π / (branch.gmax - branch.gmin)

        # interpolate the emissivity as function of time
        t_ε_interp = emissivity_interp(setup.profile, rₑ)
        t_ε_limts = emissivity_interp_limits(setup.profile, rₑ)
        δt = (t_ε_limts[2] - t_ε_limts[1]) / N_t_steps

        @inbounds for j in eachindex(g_grid_view)
            glo = clamp(g_grid[j] / g_scale, branch.gmin, branch.gmax)
            ghi = clamp(g_grid[j+1] / g_scale, branch.gmin, branch.gmax)
            # skip if bin not relevant
            if glo == ghi
                continue
            end

            Δg = (ghi - glo) / setup.g_grid_upscale

            for i = 1:setup.g_grid_upscale
                g_fine_lo = glo + (i - 1) * Δg
                g_fine_hi = g_fine_lo + Δg

                k1 = integrate_bin(
                    setup,
                    S_lower,
                    g_fine_lo,
                    g_fine_hi,
                    branch.gmin,
                    branch.gmax,
                )
                k2 = integrate_bin(
                    setup,
                    S_upper,
                    g_fine_lo,
                    g_fine_hi,
                    branch.gmin,
                    branch.gmax,
                )

                # find which bin to dump in
                (tl1, tl2), (tu1, tu2) = _time_bins(setup, branch, g_fine_lo, g_fine_hi)
                t_lower_branch = (tl1 + tl2) / 2
                t_upper_branch = (tu1 + tu2) / 2

                imax = lastindex(t_grid)
                # loop over all times and find the offsets to dump flux into
                for time in range(t_ε_limts..., N_t_steps)
                    tlower = t_lower_branch + time - setup.t0
                    i1 = @views searchsortedfirst(t_grid, tlower)

                    tupper = t_upper_branch + time - setup.t0
                    i2 = @views searchsortedfirst(t_grid, tupper)

                    em = t_ε_interp(time)
                    if i1 <= imax
                        output[j, i1] += k1 * θ * em * δt
                    end
                    if i2 <= imax
                        output[j, i2] += k2 * θ * em * δt
                    end
                end
            end
        end

        r_prev = rₑ
    end
end
