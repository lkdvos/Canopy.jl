# --- BPMessages --------------------------------------------------------------
# Directed-edge dictionary of message tensors, over the same vertex set as a
# `TensorNetworkState`. Each directed edge `u → v` carries a TensorMap on the
# receiver `v`'s side of the underlying undirected edge, in the (ket; bra)
# convention `m[ket; bra]`:
#
#   * codomain index = ket-bond index at v's side. It contracts with v's
#     domain (ket bond) leg via the codom-vs-dom @tensor rule (same literal).
#   * domain index = bra-bond index at v's side. It contracts with conj(v)'s
#     domain (bra bond) leg via the dom-vs-dom @tensor rule (dual literal).
#
# Data convention: `m[a; b] = sum_p T_u[p; a, ...] * conj(T_u)[p; b, ...] *
# (incoming messages)` — codomain index drawn from `T_u`'s dom (ket), domain
# index from `conj(T_u)`'s dom (bra). With the target leg permuted to the
# last domain position, the BP contraction outputs `M` already in
# `V_v-side ← V_v-side`, so no `flip` is needed.

const MessageTensor{T <: Number, S <: IndexSpace, A <: DenseVector{T}} =
    TensorMap{T, S, 1, 1, A}

struct BPMessages{T <: Number, S <: IndexSpace, A <: DenseVector{T}, V}
    messages::Dictionary{DirectedEdge{V}, MessageTensor{T, S, A}}
end

"""
    BPMessages(state::TensorNetworkState)

Identity-message initialization: every directed edge carries the identity on
the receiver's side of the underlying undirected edge. This is the natural BP
starting point and the fixed point on a trivial network.
"""
function BPMessages(state::TensorNetworkState)
    K = DirectedEdge{keytype(state)}
    TT = MessageTensor{scalartype(state), spacetype(state), TensorKit.storagetype(state)}
    messages = Dictionary{K, TT}()
    for edge in Graphs.edges(state)
        edge_forwards = DirectedEdge(Tuple(edge)...)
        V_forwards = virtualspace(state, edge_forwards)
        msg = TensorKit.id!(TT(undef, V_forwards ← V_forwards))
        insert!(messages, edge_forwards, msg)
        insert!(messages, reverse(edge_forwards), copy(msg))
    end
    return BPMessages(messages)
end

# Properties
# ----------
Base.eltype(::Type{BPMessages{T, S, A, V}}) where {T, S, A, V} = TensorMap{T, S, 1, 1, A}
Base.keytype(::Type{BPMessages{T, S, A, V}}) where {T, S, A, V} = V

VectorInterface.scalartype(::Type{T}) where {T <: BPMessages} = scalartype(eltype(T))
TensorKit.storagetype(::Type{T}) where {T <: BPMessages} = storagetype(eltype(T))
TensorKit.spacetype(::Type{T}) where {T <: BPMessages} = spacetype(eltype(T))


# Accessors
# ---------
Base.haskey(msgs::BPMessages, key) = haskey(msgs.messages, key)
Base.getindex(msgs::BPMessages, key) = getindex(msgs.messages, key)

Base.length(msgs::BPMessages) = length(msgs.messages)

# --- compatibility check ------------------------------------------------------
"""
    check_consistency(state, messages) -> Bool

Return `true` if `state` is consistent with `messages`:

- every (undirected) edge of the `state` has a pair of `messages`.
- every message runs along the direction of its arrow
"""
function check_consistency(state::TensorNetworkState, msgs::BPMessages)
    edges = Graphs.edges(state)
    2 * length(edges) == length(messages) || return false

    for edge in edges
        edge_fwd = DirectedEdge(Tuple(edge)...)
        (haskey(msgs, edge_fwd) && haskey(msgs, reverse(edge_fwd))) || return false

        V = virtualspace(state, edge)
        space(msgs[edge_fwd]) == V ← V || return false
        space(msgs[reverse(edge_fwd)]) == V ← V || return false
    end
    return true
end

# BP contractions
# ---------------

"""
    attach_messages(state, msgs, site, edges) -> StateTensor

Return a copy of the on-site tensor `state[site]` with the incoming BP
messages on `edges` absorbed into the corresponding virtual legs. Each
element of `edges` must be a `DirectedEdge` whose `last` endpoint is `site`;
the message `msgs[e]` is contracted (codomain / ket side) against the
virtual ket leg of `state[site]` at position `leg_index(state, reverse(e))`,
leaving the message's bra-side index in its place.

Because every message has space `V ← V`, the result has the same
`TensorMap` space as `state[site]` and the same type as `eltype(state)`.
Legs not covered by `edges` — including padded unit-space legs — are passed
through unchanged.
"""
function attach_messages(
        state::TensorNetworkState, msgs::BPMessages, site, edges
    )
    T = state[site]
    N = numind(T)

    tensors = Any[T]
    T_idx = Int[-i for i in 1:N]
    indices = Vector{Int}[T_idx]
    next_label = 0
    for e in edges
        last(e) == site || throw(ArgumentError(lazy"edge $e does not terminate at $site"))
        k = leg_index(state, reverse(e))
        T_idx[k + 1] = (next_label += 1)
        push!(tensors, msgs[e])
        push!(indices, [T_idx[k + 1], -(k + 1)])
    end

    raw = ncon(tensors, indices)
    return permute(raw, ((1,), ntuple(i -> i + 1, numin(T))))::eltype(state)
end

# Compute the new message for a single directed edge WITHOUT mutating msgs.
# Note that we perform an in-place permutation at the end to restore type stability
compute_message(msgs::BPMessages, state::TensorNetworkState, edge::DirectedEdge) =
    compute_message!(similar(msgs[edge]), msgs, state, edge)

function compute_message!(msg, msgs::BPMessages, state::TensorNetworkState, edge::DirectedEdge)
    site = first(edge)
    target = leg_index(state, edge)
    T = state[site]
    N = numind(T)

    incoming = (
        DirectedEdge(n, site) for n in neighbors(state, site)
            if n != last(edge)
    )
    Tm = attach_messages(state, msgs, site, incoming)

    Tm_idx = replace(1:N, (target + 1) => -1)
    Td_idx = replace(vcat(2:N, [1]), (target + 1) => -2)

    tensors = Any[Tm, T']
    indices = Vector{Int}[Tm_idx, Td_idx]
    p = isless(Tuple(edge)...) ? ((1,), (2,)) : ((2,), (1,))

    return permute!(msg, ncon(tensors, indices), p)::typeof(msg)
end

function tr_distance(
        A::MessageTensor, B::MessageTensor;
        p::Real = 1, is_hermitian::Bool = false
    )
    diff = add(A, B, -inv(tr(B)), inv(tr(A)))
    return norm(is_hermitian ? eigh_vals!(diff) : svd_vals!(diff), p)
end
function tr_distance!(
        A::MessageTensor, B::MessageTensor;
        p::Real = 1, is_hermitian::Bool = false
    )
    diff = add!!(A, B, -inv(tr(B)), inv(tr(A)))
    return norm(is_hermitian ? eigh_vals!(diff) : svd_vals!(diff), p)
end

# BP Algorithm
# ------------
struct BeliefPropagation{S <: AI.StoppingCriterion} <: AI.Algorithm
    stopping_criterion::S
end

struct BPProblem{N} <: AI.Problem
    network::N
end

mutable struct BPState{M, S} <: AI.State
    iterate::M
    iteration::Int
    stopping_criterion_state::S
end

function AI.initialize_state(problem::BPProblem, algorithm::BeliefPropagation; kwargs...)
    messages = BPMessages(problem.network)
    stopping_state = AI.initialize_state(problem, algorithm, algorithm.stopping_criterion)
    return BPState(messages, 0, stopping_state)
end

function AI.initialize_state!(problem::BPProblem, algorithm::BeliefPropagation, state::BPState; kwargs...)
    state.iterate = BPMessages(problem.network)
    state.stopping_criterion_state = AI.initialize_state!(problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state)
    return BPState(messages, 0, stopping_state)
end

function AI.step!(problem::BPProblem, ::BeliefPropagation, state::BPState)
    messages = map(keys(state.iterate.messages)) do edge
        return compute_message(state.iterate, problem.network, edge)
    end
    state.iterate = BPMessages(messages)
    return state
end

function belief_propagation(messages, state::TensorNetworkState; maxiter::Int)
    stopping_criterion = AI.StopAfterIteration(maxiter)
    alg = BeliefPropagation(stopping_criterion)
    return AI.solve(BPProblem(state), alg)
end

iterate_difference!(prev_messages::BPMessages, messages::BPMessages) =
    maximum(edge -> tr_distance!(prev_messages[edge], messages[edge]), keys(prev_messages))
