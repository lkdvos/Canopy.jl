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

"""
    SynchronousSchedule()

Default schedule: every message is recomputed from the previous iterate and the
whole set is swapped in at once. No update within a sweep sees a freshly-computed
neighbor message. Exact on trees in a single sweep per tree depth.
"""
struct SynchronousSchedule <: BPSchedule end

"""
    SpanningTreeSchedule(; rng = Random.default_rng())

Each iteration sample a random spanning tree of the network, solve it exactly
with a sequential leaves→root then root→leaves sweep (so updates see freshly
computed neighbors), and finally update the remaining loop-closing (cotree)
edges from the just-updated messages. The tree is resampled every iteration.
"""
struct SpanningTreeSchedule{RNG} <: BPSchedule
    rng::RNG
end
SpanningTreeSchedule(; rng = Random.default_rng()) = SpanningTreeSchedule(rng)

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

mutable struct BPState{M, S, V} <: AI.State
    iterate::M
    residuals::Dictionary{DirectedEdge{V}, Float64}
    iteration::Int
    stopping_criterion_state::S
end

function AI.initialize_state(
        problem::BPProblem, algorithm::BeliefPropagation;
        messages::BPMessages = BPMessages(problem.network), kwargs...,
    )
    residuals = map(Returns(Inf), keys(messages.messages))
    stopping_state = AI.initialize_state(problem, algorithm, algorithm.stopping_criterion; kwargs...)
    return BPState(messages, residuals, 0, stopping_state)
end

function AI.initialize_state!(
        problem::BPProblem, algorithm::BeliefPropagation, state::BPState;
        messages::Union{BPMessages, Nothing} = nothing, kwargs...,
    )
    state.iterate = something(messages, BPMessages(problem.network))
    map!(state.residuals, Inf)
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

# Recompute the message along `edge` (normalized, hermitian-projected), store it
# back into `msgs` in place, and return the trace distance to the old message.
function update_message!(
        msgs::BPMessages, network, edge::DirectedEdge, backend, allocator; timer = nothing,
    )
    new_msg = @maybe_timeit timer "compute_message" begin
        normalize!(project_hermitian!(compute_message(msgs, network, edge, backend, allocator)))
    end
    res = tr_distance(msgs[edge], new_msg; is_hermitian = true)
    msgs.messages[edge] = new_msg
    return res
end

# Synchronous (default): recompute every message from the previous iterate and
# swap the whole set in at once.
function update_messages!(
        ::SynchronousSchedule, problem::BPProblem, alg::BeliefPropagation, state::BPState
    )
    old = state.iterate
    new_dict = Dictionary{keytype(old.messages), eltype(old)}()
    for edge in keys(old.messages)
        new_msg = @maybe_timeit alg.timer "compute_message" begin
            normalize!(project_hermitian!(compute_message(old, problem.network, edge, alg.backend, alg.allocator)))
        end
        insert!(new_dict, edge, new_msg)
        state.residuals[edge] = tr_distance(old[edge], new_msg; is_hermitian = true)
    end
    state.iterate = BPMessages(new_dict)
    return state
end

# Randomized BFS spanning tree over the network's own adjacency. Returns the BFS
# vertex order (root first), the `parent` map (root absent), and the cotree
# (loop-closing) undirected edges. Assumes a connected network.
function random_spanning_tree(state::TensorNetworkState, rng)
    V = keytype(state)
    verts = collect(vertices(state))
    root = rand(rng, verts)
    parent = Dictionary{V, V}()
    visited = Set{V}((root,))
    order = V[root]
    tree_edges = Set{UndirectedEdge{V}}()
    head = 1
    while head <= length(order)
        v = order[head]
        head += 1
        nbrs = collect(neighbors(state, v))
        Random.shuffle!(rng, nbrs)
        for n in nbrs
            n in visited && continue
            push!(visited, n)
            insert!(parent, n, v)
            push!(tree_edges, UndirectedEdge(v, n))
            push!(order, n)
        end
    end
    cotree = [e for e in edges(state) if !(e in tree_edges)]
    return order, parent, cotree
end

# Spanning-tree schedule: exact two-pass sweep along a fresh random spanning
# tree, then update the cotree edges from the just-updated messages.
function update_messages!(
        sched::SpanningTreeSchedule, problem::BPProblem, alg::BeliefPropagation, state::BPState
    )
    network = problem.network
    msgs = state.iterate
    order, parent, cotree = random_spanning_tree(network, sched.rng)
    # inward pass: leaves → root
    for v in Iterators.reverse(order)
        haskey(parent, v) || continue
        e = DirectedEdge(v, parent[v])
        state.residuals[e] = update_message!(msgs, network, e, alg.backend, alg.allocator; timer = alg.timer)
    end
    # outward pass: root → leaves
    for v in order
        haskey(parent, v) || continue
        e = DirectedEdge(parent[v], v)
        state.residuals[e] = update_message!(msgs, network, e, alg.backend, alg.allocator; timer = alg.timer)
    end
    # cotree (loop-closing) edges: both directions
    for ce in cotree
        e_fwd = DirectedEdge(ce)
        state.residuals[e_fwd] = update_message!(msgs, network, e_fwd, alg.backend, alg.allocator; timer = alg.timer)
        e_bwd = reverse(e_fwd)
        state.residuals[e_bwd] = update_message!(msgs, network, e_bwd, alg.backend, alg.allocator; timer = alg.timer)
    end
    return state
end

# Propagate a change `δ` along edge `e = (s → r)` to the downstream edges that
# consume `msgs[e]` — the outgoing edges `r → t` with `t ≠ s`.
function propagate_change!(residuals, network, e::DirectedEdge, δ)
    s, r = first(e), last(e)
    for t in neighbors(network, r)
        t == s && continue
        residuals[DirectedEdge(r, t)] += δ
    end
    return residuals
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
        new_msg = @maybe_timeit alg.timer "compute_message" begin
            normalize!(project_hermitian!(compute_message(msgs, network, e, alg.backend, alg.allocator)))
        end
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

function belief_propagation(
        messages::BPMessages, state::TensorNetworkState;
        maxiter::Int, tol::Real = 0, schedule::BPSchedule = SynchronousSchedule(),
        timer = nothing, backend = DefaultBackend(), allocator = default_allocator(state),
    )
    stopping = AI.StopAfterIteration(maxiter)
    tol > 0 && (stopping = stopping | StopWhenStable(tol))
    alg = BeliefPropagation(stopping; schedule, timer, backend, allocator)
    return @maybe_timeit timer "belief_propagation" begin
        AI.solve(BPProblem(state), alg; messages)
    end
end
