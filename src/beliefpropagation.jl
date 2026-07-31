# BP message schedules
# ---------------------
#
# A `BPSchedule` controls the *order* in which messages are updated within a
# single `AI.step!`. The default [`SpanningTreeSchedule`](@ref) walks a fresh
# random BFS order twice, recomputing all of a vertex's outgoing messages in one
# batch per visit, so later updates see earlier ones and information crosses the
# whole network within a single sweep. [`SynchronousSchedule`](@ref) is the
# opposite extreme: every message is recomputed from the previous iterate and the
# whole set is swapped in at once, so no update sees a freshly-computed neighbor.

"""
    BPSchedule

Abstract supertype for belief-propagation message-update schedules. Concrete
subtypes ([`SpanningTreeSchedule`](@ref) — the default,
[`SynchronousSchedule`](@ref), [`ResidualSchedule`](@ref),
[`ResidualSplashSchedule`](@ref)) are stored in [`BeliefPropagation`](@ref) and
dispatched on by `update_messages!`.

Only [`SynchronousSchedule`](@ref) leaves the input messages intact: it builds a
fresh message container each sweep. Every other schedule updates messages in
place, so the [`BPMessages`](@ref) returned by [`belief_propagation`](@ref)
*aliases* the one passed in and the caller's messages are overwritten.
"""
abstract type BPSchedule end

# BP Algorithm
# ------------
struct BeliefPropagation{Sched <: BPSchedule, S <: AI.StoppingCriterion, TO, B, A} <: AI.Algorithm
    schedule::Sched
    stopping_criterion::S
    timer::TO
    backend::B
    allocator::A
end
BeliefPropagation(
        stopping_criterion::AI.StoppingCriterion;
        schedule::BPSchedule = SpanningTreeSchedule(), timer = nothing,
        backend = DefaultBackend(), allocator = _default_allocator(),
    ) = BeliefPropagation(schedule, stopping_criterion, timer, backend, allocator)

struct BPProblem{N} <: AI.Problem
    network::N
end

mutable struct BPState{M, S, V, L} <: AI.State
    iterate::M
    residuals::Dictionary{DirectedEdge{V}, Float64}
    last_used::L
    iteration::Int
    stopping_criterion_state::S
end

# Per-schedule initialization of the `last_used` message snapshots. The fallback
# returns `nothing`: only schedules that maintain incremental residuals against a
# snapshot (e.g. the residual-splash schedule) need one.
init_last_used(::BPSchedule, messages) = nothing

function AI.initialize_state(
        problem::BPProblem, algorithm::BeliefPropagation;
        messages::BPMessages = BPMessages(problem.network), kwargs...,
    )
    residuals = map(Returns(Inf), keys(messages.messages))
    last_used = init_last_used(algorithm.schedule, messages)
    stopping_state = AI.initialize_state(problem, algorithm, algorithm.stopping_criterion; kwargs...)
    return BPState(messages, residuals, last_used, 0, stopping_state)
end

function AI.initialize_state!(
        problem::BPProblem, algorithm::BeliefPropagation, state::BPState;
        messages::Union{BPMessages, Nothing} = nothing, kwargs...,
    )
    state.iterate = something(messages, BPMessages(problem.network))
    map!(state.residuals, Inf)
    state.last_used = init_last_used(algorithm.schedule, state.iterate)
    state.iteration = 0
    state.stopping_criterion_state = AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state;
        kwargs...,
    )
    return state
end

function AI.step!(problem::BPProblem, alg::BeliefPropagation, state::BPState)
    @maybe_timeit alg.timer "bp_iteration" begin
        update_messages!(alg.schedule, problem, alg, state)
    end
    return state
end

# Shared message primitives
# -------------------------
# Recompute the (normalized, hermitian-projected) message along `edge` from the
# current `msgs`, without storing it.
function recompute_message(
        msgs::BPMessages, network, edge::DirectedEdge, backend, allocator; timer = nothing,
    )
    return @maybe_timeit timer "compute_message" begin
        normalize!(project_hermitian!(compute_message(msgs, network, edge, backend, allocator)))
    end
end

# Batch form: recompute the normalized messages along `edges` (all leaving the
# same source vertex) via the vertex-centric kernel, returned in `edges` order.
function recompute_messages(
        msgs::BPMessages, network, edges::AbstractVector, backend, allocator; timer = nothing,
    )
    return @maybe_timeit timer "compute_message" begin
        out = compute_message(msgs, network, edges, backend, allocator)
        # In place: `map(f, out)` would allocate a second `Vector` per call, and
        # `f` is itself in-place on the message tensors.
        map!(m -> normalize!(project_hermitian!(m)), out, out)
    end
end

# Recompute the message along `edge`, store it back into `msgs` in place, and
# return the trace distance to the old message.
function update_message!(
        msgs::BPMessages, network, edge::DirectedEdge, backend, allocator; timer = nothing,
    )
    new_msg = recompute_message(msgs, network, edge, backend, allocator; timer)
    res = tr_distance(msgs[edge], new_msg; is_hermitian = true)
    msgs.messages[edge] = new_msg
    return res
end

"""
    update_messages_at!(msgs, network, targets::AbstractVector{<:DirectedEdge}, backend, allocator;
                        residuals = nothing, reference = nothing, snapshot = nothing, timer = nothing)
    update_messages_at!(msgs, network, v, backend, allocator; exclude = (), kwargs...)

Recompute every message in `targets` — which must all leave the *same* source
vertex — in one vertex-centric batch and store them back into `msgs`. The second
form batches all of `v`'s outgoing edges, minus any neighbor in `exclude`.

The batch is exact in the sense that matters for schedules: `compute_message`
reads one snapshot of the source's *incoming* messages and a message *out of* `v`
is never an input *at* `v`, so batching is indistinguishable from a per-edge loop
over `targets` in any order.

If `residuals` is given, `residuals[e]` is updated for every target. Which
reference the change is measured against depends on the (mutually exclusive)
`reference` / `snapshot` keywords:

* neither — against the value `msgs[e]` held on entry (what
  [`update_message!`](@ref) does);
* `reference` — against `reference[e]`, **non-destructively**; for the
  incremental "changed since its consumer last used it" bookkeeping of
  [`ResidualSplashSchedule`](@ref);
* `snapshot` — against `snapshot[e]`, **destructively**: `snapshot[e]` is
  consumed by `tr_distance!` and then re-pointed at the new message, so a
  `BPMessages` snapshot taken at the start of a sweep measures the *full-sweep*
  change at zero copies. Only valid when `snapshot[e]` is dead at that instant.

Returns `msgs`.
"""
function update_messages_at!(
        msgs::BPMessages, network, targets::AbstractVector{<:DirectedEdge}, backend, allocator;
        residuals = nothing, reference = nothing, snapshot = nothing, timer = nothing,
    )
    isempty(targets) && return msgs
    isnothing(reference) || isnothing(snapshot) ||
        throw(ArgumentError("`reference` and `snapshot` are mutually exclusive"))
    new_msgs = recompute_messages(msgs, network, targets, backend, allocator; timer)
    for (e, new_msg) in zip(targets, new_msgs)
        if !isnothing(residuals)
            residuals[e] = if !isnothing(snapshot)
                # Destructive on `snapshot[e]`: `add!!(A, B, …)` writes into `A`,
                # so `new_msg` is untouched.
                tr_distance!(snapshot[e], new_msg; is_hermitian = true)
            elseif !isnothing(reference)
                tr_distance(new_msg, reference[e]; is_hermitian = true)
            else
                tr_distance(msgs[e], new_msg; is_hermitian = true)
            end
        end
        msgs.messages[e] = new_msg
        isnothing(snapshot) || (snapshot.messages[e] = new_msg)
    end
    return msgs
end

function update_messages_at!(
        msgs::BPMessages, network, v, backend, allocator; exclude = (), kwargs...,
    )
    targets = collect(outgoing_edges(network, v; exclude))
    return update_messages_at!(msgs, network, targets, backend, allocator; kwargs...)
end

# Convergence criterion
# ---------------------
struct StopWhenStable <: AI.StoppingCriterion
    tol::Float64
end

mutable struct StopWhenStableState <: AI.StoppingCriterionState
    at_iteration::Int
    delta::Float64
end

function AI.initialize_state(::AI.Problem, ::AI.Algorithm, c::StopWhenStable; kwargs...)
    return StopWhenStableState(-1, NaN)
end

function AI.initialize_state!(
        ::AI.Problem, ::AI.Algorithm, stop_when::StopWhenStable, st::StopWhenStableState;
        kwargs...
    )
    st.at_iteration = -1
    st.delta = NaN
    return st
end

function AI.is_finished!(
        ::AI.Problem, ::AI.Algorithm, state::AI.State, c::StopWhenStable, st::StopWhenStableState
    )
    k = state.iteration
    k == 0 && return false

    st.delta = maximum(values(state.residuals))
    if st.delta < c.tol
        st.at_iteration = k
        return true
    end
    return false
end

function AI.is_finished(
        ::AI.Problem, ::AI.Algorithm, state::AI.State, c::StopWhenStable, ::StopWhenStableState
    )
    state.iteration == 0 && return false
    return maximum(values(state.residuals)) < c.tol
end

function belief_propagation(
        messages::BPMessages, state::TensorNetworkState;
        maxiter::Int, tol::Real = 0, schedule::BPSchedule = SpanningTreeSchedule(),
        timer = nothing, backend = DefaultBackend(), allocator = default_allocator(state),
    )
    stopping = AI.StopAfterIteration(maxiter)
    tol > 0 && (stopping = stopping | StopWhenStable(tol))
    alg = BeliefPropagation(stopping; schedule, timer, backend, allocator)
    @debug "belief_propagation entry" isempty = buffer_isempty(allocator) stats = buffer_stats(allocator) gc_live_bytes = Base.gc_live_bytes() maxrss = Sys.maxrss()
    return @maybe_timeit timer "belief_propagation" begin
        res = AI.solve(BPProblem(state), alg; messages)
        @debug "belief_propagation exit" isempty = buffer_isempty(allocator) stats = buffer_stats(allocator) gc_live_bytes = Base.gc_live_bytes() maxrss = Sys.maxrss()
        res
    end
end

# A `TensorNetworkOperator` runs through its fused state view — BP itself is unchanged, and
# the resulting messages are directly usable by `apply!` on the operator. The view is built
# here rather than cached because it aliases `op`'s storage.
belief_propagation(messages::BPMessages, op::TensorNetworkOperator; kwargs...) =
    belief_propagation(messages, TensorNetworkState(op); kwargs...)
