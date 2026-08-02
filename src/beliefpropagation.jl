# BP message schedules
# ---------------------
#
# A `BPSchedule` controls the *order* in which messages are updated within a
# single `AI.step!`. The synchronous default recomputes every message from the
# previous iterate and swaps them in at once (no update sees a freshly-computed
# neighbor); the other schedules update messages sequentially / in batches so
# that later updates do see earlier ones.

"""
    BPSchedule

Abstract supertype for belief-propagation message-update schedules. Concrete
subtypes ([`SynchronousSchedule`](@ref), [`SpanningTreeSchedule`](@ref),
[`ResidualSchedule`](@ref)) are stored in [`BeliefPropagation`](@ref) and
dispatched on by `update_messages!`.
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
        schedule::BPSchedule = SynchronousSchedule(), timer = nothing,
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
        map(
            m -> normalize!(project_hermitian!(m)),
            compute_message(msgs, network, edges, backend, allocator),
        )
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
        maxiter::Int, tol::Real = 0, schedule::BPSchedule = SynchronousSchedule(),
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
