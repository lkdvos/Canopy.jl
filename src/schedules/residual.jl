"""
    BPSampler

Abstract supertype for [`ResidualSchedule`](@ref) batch samplers. A sampler is a
callable mapping the per-edge residuals to the batch of directed edges to update
this iteration: `sampler(residuals) -> AbstractVector{<:DirectedEdge}`. Any
callable obeying that contract may be used; [`GreedySampler`](@ref) and
[`WeightedSampler`](@ref) are provided.
"""
abstract type BPSampler end

"""
    GreedySampler(batchsize)

Select the `batchsize` directed edges with the largest residuals.
"""
struct GreedySampler <: BPSampler
    batchsize::Int
end
function (s::GreedySampler)(residuals)
    ks = collect(keys(residuals))
    vs = collect(values(residuals))
    n = min(s.batchsize, length(ks))
    return ks[partialsortperm(vs, 1:n; rev = true)]
end

"""
    WeightedSampler(batchsize; rng = Random.default_rng())

Sample `batchsize` directed edges without replacement, each with weight
proportional to its residual (Efraimidis–Spirakis A-Res: draw key `u^(1/w)` per
edge and take the largest keys).
"""
struct WeightedSampler{RNG} <: BPSampler
    batchsize::Int
    rng::RNG
end
WeightedSampler(batchsize::Int; rng = Random.default_rng()) =
    WeightedSampler(batchsize, rng)
function (s::WeightedSampler)(residuals)
    ks = collect(keys(residuals))
    vs = collect(values(residuals))
    n = min(s.batchsize, length(ks))
    es_keys = map(vs) do w
        u = rand(s.rng)
        w > 0 ? u^inv(w) : zero(u)
    end
    return ks[partialsortperm(es_keys, 1:n; rev = true)]
end

"""
    ResidualSchedule(sampler)

Each iteration the `sampler` selects a batch of directed edges from the current
residuals (e.g. [`GreedySampler`](@ref) or [`WeightedSampler`](@ref)); that batch
is recomputed in parallel from the current messages and applied together, after
which the residuals of the edges that consume an updated message are refreshed.
The first iteration seeds the residuals with one synchronous sweep.
"""
struct ResidualSchedule{F} <: BPSchedule
    sampler::F
end

# Residual schedule: priorities propagate the magnitude of recent message
# changes to the downstream edges they feed, so the sampler targets the edges
# whose inputs moved most — "the neighbors of whatever changed most". Each
# iteration only computes the messages it applies; no message is ever computed
# merely to measure a residual.
#
# `state.residuals[e]` accumulates the changes of `e`'s inputs since `e` was last
# updated (0 ⇔ `e` is consistent with the current messages), which both drives
# selection and certifies convergence (`max → 0` at the fixed point). Note this
# is an input-change surrogate, not the exact message residual, so its `tol`
# scale differs from the other schedules'.
function update_messages!(
        sched::ResidualSchedule, problem::BPProblem, alg::BeliefPropagation, state::BPState
    )
    network = problem.network
    msgs = state.iterate
    # Seed on the first iteration: one synchronous sweep records each edge's true
    # change, which we then propagate into downstream priorities.
    if all(isinf, values(state.residuals))
        update_messages!(SynchronousSchedule(), problem, alg, state)
        deltas = state.residuals
        priorities = map(Returns(0.0), keys(msgs.messages))
        for e in keys(msgs.messages)
            propagate_change!(priorities, network, e, deltas[e])
        end
        state.residuals = priorities
        return state
    end
    batch = sched.sampler(state.residuals)
    # Compute the batch from the current messages (order-independent), capturing
    # each edge's change, then apply together.
    updates = map(batch) do e
        new_msg = recompute_message(msgs, network, e, alg.backend, alg.allocator; timer = alg.timer)
        return (e, new_msg, tr_distance(msgs[e], new_msg; is_hermitian = true))
    end
    for (e, new_msg, _) in updates
        msgs.messages[e] = new_msg
        state.residuals[e] = 0.0
    end
    # Propagate each applied change to the downstream edges it feeds.
    for (e, _, δ) in updates
        propagate_change!(state.residuals, network, e, δ)
    end
    return state
end
