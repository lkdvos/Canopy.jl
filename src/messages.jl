# --- BP message conventions ---------------------------------------------------
#
# Belief-propagation messages live on the *directed* version of the state's
# graph. For every undirected edge `(u, v)` of `state` there are two directed
# edges and two messages:
#
#     DirectedEdge(u, v)   "message from sender u to receiver v"
#     DirectedEdge(v, u)   "message from sender v to receiver u"
#
# ## Geometry
#
# A directed edge `e = (s, r)` is read as `sender → receiver`. The message
# `msgs[e]` lives on the **receiver's** side of the underlying bond, i.e. on
# the virtual leg of `state[r]` that points to `s`. With
# `V_r = virtualspace(state, DirectedEdge(r, s))` (the bond space as seen
# from r), every message has TensorKit space `V_r ← V_r`.
#
# ## Codomain / domain
#
# Treating a message as a linear map `V_r → V_r`:
#
# - the **domain** (ket-layer) leg contracts against the ket virtual leg of
#   `state[r]` at this bond — this is what [`attach_messages`](@ref) does;
# - the **codomain** (bra-layer) leg takes the place of that ket virtual
#   leg in the modified site tensor, so a subsequent contraction with the
#   bra `state[r]'` closes the BP environment.
#
# In the double-layer picture, a message is the partial trace of the
# environment outside the receiver projected onto the bond: codomain = bra
# index, domain = ket index. On a tree the BP fixed point coincides with
# this exact environment; on graphs with loops it is the loop-free
# (Bethe) approximation.
#
# ## Normalization
#
# Messages are defined up to overall scale (BP only constrains them up to a
# multiplicative constant per directed edge). The convention used here is
# trace normalization: [`compute_message`](@ref) returns the new message
# normalized by its trace, and [`tr_distance`](@ref) compares two messages
# after trace-normalising both — i.e. it ignores any overall rescaling.
#
# ## Identity initialization
#
# `BPMessages(state)` initialises every message to the identity on its
# receiver-side space. This is the natural starting point for BP and is
# *exact* on trees: a single sweep along the tree reaches the fixed point.

const MessageTensor{T <: Number, S <: IndexSpace, A <: DenseVector{T}} =
    TensorMap{T, S, 1, 1, A}

"""
    BPMessages{T, S, A, V}

Container for BP messages over a [`TensorNetworkState`](@ref) with vertex
keys of type `V`. Each undirected state edge contributes two entries — one
per direction — stored in a `Dictionary` keyed by `DirectedEdge{V}`.

Every message is a `TensorMap{T, S, 1, 1, A}` of space `V_r ← V_r`, where
`V_r` is the virtual-leg space of the *receiver* at that bond. See the
file header for the geometric convention (sender → receiver, codomain =
bra, domain = ket) used by [`attach_messages`](@ref) and
[`compute_message`](@ref).
"""
struct BPMessages{T <: Number, S <: IndexSpace, A <: DenseVector{T}, V}
    messages::Dictionary{DirectedEdge{V}, MessageTensor{T, S, A}}
end

"""
    BPMessages(state::TensorNetworkState)

Allocate a `BPMessages` for `state` with every directed edge initialized to
the identity on the receiver's side of the underlying undirected edge.

This is the standard BP starting point and is *exact* on trees (one sweep
reaches the fixed point). See the file header for the message-space
convention.
"""
function BPMessages(state::TensorNetworkState)
    K = DirectedEdge{keytype(state)}
    TT = MessageTensor{scalartype(state), spacetype(state), TensorKit.storagetype(state)}
    messages = Dictionary{K, TT}()
    for edge in edges(state)
        edge_fwd = DirectedEdge(edge)
        V_fwd = virtualspace(state, reverse(edge_fwd))
        msg_fwd = TensorKit.id!(TT(undef, V_fwd ← V_fwd))
        insert!(messages, edge_fwd, msg_fwd)

        edge_bwd = reverse(edge_fwd)
        V_bwd = virtualspace(state, reverse(edge_bwd))
        msg_bwd = TensorKit.id!(TT(undef, V_bwd ← V_bwd))
        insert!(messages, edge_bwd, msg_bwd)
    end
    return BPMessages(messages)
end

# Properties
# ----------
Base.eltype(::Type{BPMessages{T, S, A, V}}) where {T, S, A, V} = TensorMap{T, S, 1, 1, A}

Base.keytype(msgs::BPMessages) = keytype(typeof(msgs))
Base.keytype(::Type{BPMessages{T, S, A, V}}) where {T, S, A, V} = V

VectorInterface.scalartype(::Type{T}) where {T <: BPMessages} = scalartype(eltype(T))
TensorKit.storagetype(::Type{T}) where {T <: BPMessages} = storagetype(eltype(T))
TensorKit.spacetype(::Type{T}) where {T <: BPMessages} = spacetype(eltype(T))


# Accessors
# ---------
Base.haskey(msgs::BPMessages, key) = haskey(msgs.messages, key)
Base.getindex(msgs::BPMessages, key) = getindex(msgs.messages, key)

Base.length(msgs::BPMessages) = length(msgs.messages)

# Deep snapshot: copies every message `TensorMap` so mutating one container does
# not touch the other. Used by schedules that keep a `last_used` snapshot.
Base.copy(msgs::BPMessages) = BPMessages(map(copy, msgs.messages))

# --- compatibility check ------------------------------------------------------
"""
    check_consistency(state, messages) -> Bool

Return `true` if `messages` is structurally compatible with `state`:

- every undirected edge of `state` has both directed messages present;
- each message has space `V_r ← V_r`, where `V_r` is the virtual-leg
  space of the receiver (`last(edge)`) at that bond — i.e. the messages
  follow the sender→receiver convention documented at the top of this file.

Does not check that the messages are a BP fixed point; only that their
spaces line up with `state`.
"""
function check_consistency(state::TensorNetworkState, msgs::BPMessages)
    es = edges(state)
    2 * length(es) == length(msgs.messages) || return false

    for edge in es
        edge_fwd = DirectedEdge(edge)
        (haskey(msgs, edge_fwd) && haskey(msgs, reverse(edge_fwd))) || return false

        V_recv_fwd = virtualspace(state, reverse(edge_fwd))
        V_recv_bwd = virtualspace(state, edge_fwd)
        space(msgs[edge_fwd]) == (V_recv_fwd ← V_recv_fwd) || return false
        space(msgs[reverse(edge_fwd)]) == (V_recv_bwd ← V_recv_bwd) || return false
    end
    return true
end

# BP contractions
# ---------------

# `k` is a *virtual* (domain) leg index; the corresponding tensor slot is `NP + k`, where
# `NP = numout` is the number of physical legs (1 for a state, 2 for an operator).
function _mul_leg!(
        dst::TensorMap{<:Any, S, NP, N}, src::TensorMap{<:Any, S, NP, N}, L, k::Int,
        backend, allocator,
    ) where {S, NP, N}
    M = NP + N
    d = NP + k
    oindA = TupleTools.deleteat(ntuple(identity, M), d)
    pB = ((2,), (1,))                                  # contract L's codomain, keep its domain
    pAB = (ntuple(j -> j < d ? j : (j == d ? M : j - 1), M), ())  # restore the new leg to slot `d`
    dualleg = isdual(space(src, d))
    cp = allocator_checkpoint!(allocator)
    for (f₁, f₂) in fusiontrees(dst)
        σ = f₂.uncoupled[k]
        α = dualleg ? convert(eltype(dst), twist(σ)) : One()
        tensorcontract!(
            dst[f₁, f₂], src[f₁, f₂], (oindA, (d,)), false, block(L, conj(σ)), pB, false, pAB,
            α, Zero(), backend, allocator,
        )
    end
    allocator_reset!(allocator, cp)
    return dst
end

function _absorb_legs(T::TensorMap{Tn, S, NP, N, A}, leg_factors, backend, allocator) where {Tn, S, NP, N, A}
    factors = collect(leg_factors)
    isempty(factors) && return copy(T)
    M = NP + N
    cp = allocator_checkpoint!(allocator)
    out = T
    for i in eachindex(factors)
        k, L = factors[i]
        d = NP + k
        oindA = TupleTools.deleteat(ntuple(identity, M), d)
        pA = (oindA, (d,))
        pB = ((2,), (1,))
        # physical legs 1:NP are untouched (d > NP always); the new leg goes back to slot d
        pAB = (
            ntuple(identity, NP),
            ntuple(j -> (jj = NP + j; jj < d ? jj : (jj == d ? M : jj - 1)), N),
        )
        TC = promote_contract(scalartype(out), scalartype(L))
        new_out = tensoralloc_contract(TC, out, pA, false, L, pB, false, pAB, Val(i != lastindex(factors)), allocator)
        # `_mul_leg!` indexes the *domain* fusion tree with `k`, so it is only valid for a
        # virtual leg — `d > NP` is guaranteed here, but assert it so a future slot-addressed
        # caller cannot silently route a physical leg through this fast path.
        if d > NP && space(L, 1) == space(out, d)
            _mul_leg!(new_out, out, L, k, backend, allocator)          # space-preserving (BP message)
        else
            tensorcontract!(new_out, out, pA, false, L, pB, false, pAB, One(), Zero(), backend, allocator)  # space-flipping (gauge √)
        end
        out === T || tensorfree!(out, allocator)
        out = new_out
    end
    allocator_reset!(allocator, cp)
    return out
end

# Contract `L`'s leg `3 - keep` into tensor slot `slot` of `T`, re-emitting L's `keep` leg
# at `slot` so the result has `T`'s leg layout with that one space replaced. Unlike
# `_absorb_legs` this addresses an *absolute* slot, so it also reaches physical legs — which
# is what a one-site gate needs. Always the plain contraction path: TensorKit inserts the
# right braiding for a general slot, whereas `_mul_leg!` is a fast path valid only for
# space-preserving *domain* absorptions.
function _absorb_slot(T::AbstractTensorMap, slot::Int, L, keep::Int, backend, allocator)
    M = numind(T)
    np = numout(T)
    oindA = TupleTools.deleteat(ntuple(identity, M), slot)
    pA = (oindA, (slot,))
    pB = ((3 - keep,), (keep,))
    # pairwise output is `(oindA..., L's kept leg)`, so slot `slot` sits last at position M
    pos(i) = i == slot ? M : (i < slot ? i : i - 1)
    pAB = (ntuple(pos, np), ntuple(j -> pos(np + j), numin(T)))
    TC = promote_contract(scalartype(T), scalartype(L))
    out = tensoralloc_contract(TC, T, pA, false, L, pB, false, pAB, Val(false))
    cp = allocator_checkpoint!(allocator)
    tensorcontract!(out, T, pA, false, L, pB, false, pAB, One(), Zero(), backend, allocator)
    allocator_reset!(allocator, cp)
    return out
end

"""
    attach_messages(state, msgs, site, edges) -> StateTensor

Return a copy of the on-site tensor `state[site]` with the *incoming* BP
messages on `edges` absorbed into the corresponding virtual legs. Each
element of `edges` must be a `DirectedEdge` ending at `site` (i.e.
`last(e) == site`, so `site` is the receiver and `msgs[e]` lives on this
side of the bond).

For each such edge `e`, the leg `k = leg_index(state, reverse(e))` of
`state[site]` is replaced as follows: the message's **domain** (ket) is
contracted against the ket virtual leg of `state[site]` at position `k`,
and the message's **codomain** (bra) takes its place. Because every
message has space `V_r ← V_r`, the result has the same `TensorMap` space
as `state[site]` and the same concrete type as `eltype(state)`. Legs not
listed in `edges` — including padded unit-space legs — pass through
unchanged.
"""
function attach_messages(
        state::TensorNetworkState, msgs::BPMessages, site, edges,
        backend = DefaultBackend(), allocator = default_allocator(state),
    )
    leg_factors = map(edges) do e
        last(e) == site || throw(ArgumentError(lazy"edge $e does not terminate at $site"))
        return (leg_index(state, reverse(e)), msgs[e])
    end
    return _absorb_legs(state[site], leg_factors, backend, allocator)::eltype(state)
end

"""
    attach_all_messages(state, msgs, site) -> StateTensor

Convenience wrapper for [`attach_messages`](@ref): absorbs *every* incoming
BP message at `site` into the corresponding virtual leg of `state[site]`.
Equivalent to `attach_messages(state, msgs, site, incoming_edges(state, site))`.
"""
attach_all_messages(
    state::TensorNetworkState, msgs::BPMessages, site,
    backend = DefaultBackend(), allocator = default_allocator(state),
) = attach_messages(state, msgs, site, incoming_edges(state, site), backend, allocator)

"""
    compute_message(msgs, state, edge::DirectedEdge) -> MessageTensor
    compute_message!(out, msgs, state, edge::DirectedEdge) -> out

Compute the updated BP message along `edge = (s, r)` from the current
`msgs`. The sender's on-site tensor `state[s]` is closed with its bra by
absorbing every incoming message on `s` *except* the one coming from
`r` ([`attach_messages`](@ref)), and contracted with `state[s]'`; the
remaining open virtual legs — codomain = bra side at `r`, domain = ket
side at `r` — form the new message in space `V_r ← V_r`.

`compute_message` allocates a fresh `MessageTensor`; `compute_message!`
writes into `out`, which must have the same space as `msgs[edge]`. Neither
mutates `msgs`, and neither performs trace normalization — callers (e.g.
[`BeliefPropagation`](@ref)) are expected to normalise the returned
message.
"""
compute_message(
    msgs::BPMessages, state::TensorNetworkState, edge::DirectedEdge,
    backend = DefaultBackend(), allocator = default_allocator(state),
) = compute_message!(similar(msgs[edge]), msgs, state, edge, backend, allocator)

function compute_message!(
        msg, msgs::BPMessages, state::TensorNetworkState, edge::DirectedEdge,
        backend = DefaultBackend(), allocator = default_allocator(state),
    )
    site = first(edge)
    target = leg_index(state, edge)
    T = state[site]
    N = numind(T)

    Tm = attach_messages(
        state, msgs, site, incoming_edges(state, site; exclude = (last(edge),)),
        backend, allocator,
    )

    tc = target + 1
    cA = TupleTools.deleteat(ntuple(identity, N), tc)
    pA = ((tc,), cA)
    pB = (cA, (tc,))
    cp = allocator_checkpoint!(allocator)
    tensorcontract!(msg, Tm, pA, false, T, pB, true, ((2,), (1,)), One(), Zero(), backend, allocator)
    allocator_reset!(allocator, cp)

    return msg::typeof(msg)
end

"""
    compute_message(msgs, state, edges::AbstractVector{<:DirectedEdge}) -> Vector{MessageTensor}
    compute_message!(out, msgs, state, edges::AbstractVector{<:DirectedEdge}) -> out

Compute the updated BP messages along several `edges` that all leave the **same**
source vertex `v = first(first(edges))`, sharing work across them. Every edge must
satisfy `first(e) == v`; the targets `last(e)` may be any subset of `v`'s
neighbors, in any order (e.g. `outgoing_edges(state, v)` for all of them).
`out[i]` is the message along `edges[i]`, with the same space the single-edge
[`compute_message`](@ref) produces.

All messages out of `v` close `state[v]` against `state[v]'` with the same
incoming messages absorbed leave-one-out. A ket prefix chain and a bra suffix
chain share those absorptions (each incoming message absorbed about twice in
total rather than once per output); only the chain segments spanning the
requested targets are built, so a small subset costs proportionally less. Like
the single-edge method, neither variant mutates `msgs` nor normalises the result;
`compute_message` allocates the output vector, `compute_message!` fills `out`.
"""
function compute_message(
        msgs::BPMessages, state::TensorNetworkState, edges::AbstractVector{<:DirectedEdge},
        backend = DefaultBackend(), allocator = default_allocator(state),
    )
    out = map(e -> similar(msgs[e]), edges)
    return compute_message!(out, msgs, state, edges, backend, allocator)
end

# Chain links carry their leg ids (`1` = physical, `2:M` = virtual leg `id-1`) in
# axis order, the first `ncod` forming the codomain. `_absorb` contracts `msg` into
# leg `absorbed` of `link` and re-emits it directly in the target order/partition
# `(newlegs, ncod)`, folding the reorder and codomain/domain split into the
# contraction's one output pass so the active leg stays matrix-form for the next
# step. `_repartition` does the same for a bare on-site tensor (native order, no msg).
function _absorb(link, legs, absorbed::Int, msg, newlegs, ncod::Int, backend, allocator)
    M = numind(link)
    ax = findfirst(==(absorbed), legs)
    kept = TupleTools.deleteat(ntuple(identity, M), ax)
    order = (ntuple(i -> legs[kept[i]], M - 1)..., absorbed)   # pairwise-output leg ids
    pA = (kept, (ax,))
    pB = ((2,), (1,))
    pAB = (
        ntuple(i -> findfirst(==(newlegs[i]), order), ncod),
        ntuple(i -> findfirst(==(newlegs[ncod + i]), order), M - ncod),
    )
    TC = promote_contract(scalartype(link), scalartype(msg))
    result = tensoralloc_contract(TC, link, pA, false, msg, pB, false, pAB, Val(true), allocator)
    cp = allocator_checkpoint!(allocator)
    tensorcontract!(result, link, pA, false, msg, pB, false, pAB, One(), Zero(), backend, allocator)
    allocator_reset!(allocator, cp)
    return result
end

function _repartition(tensor, newlegs, ncod::Int, backend, allocator)
    M = numind(tensor)
    pC = (ntuple(i -> newlegs[i], ncod), ntuple(i -> newlegs[ncod + i], M - ncod))
    result = tensoralloc_add(scalartype(tensor), tensor, pC, false, Val(true), allocator)
    cp = allocator_checkpoint!(allocator)
    tensoradd!(result, tensor, pC, false, One(), Zero(), backend, allocator)
    allocator_reset!(allocator, cp)
    return result
end

# Shared leave-one-out: `prefix[k]` absorbs virtual legs 1..k-1, `suffix[k]`
# absorbs k+1..d (adjoint messages), and the closing contracts them over the
# physical leg and every virtual leg ≠ k — the `compute_message!` sandwich. Links
# are pre-partitioned so each closing is a direct block GEMM with no repartition,
# and bump-allocated under one checkpoint freed at the end. Only the prefix up to
# the largest target and the suffix down to the smallest are built, so a clustered
# subset costs proportionally less than all `d` outputs.
function compute_message!(
        out, msgs::BPMessages, state::TensorNetworkState, edges::AbstractVector{<:DirectedEdge},
        backend = DefaultBackend(), allocator = default_allocator(state),
    )
    isempty(edges) && return out
    v = first(first(edges))
    T = state[v]
    M = numind(T)
    nbrs = neighbors(state, v)
    d = length(nbrs)
    target_legs = map(edges) do e
        first(e) == v || throw(ArgumentError(lazy"edge $e does not leave the shared source $v"))
        return leg_index(state, e)
    end
    legmin, legmax = extrema(target_legs)
    incoming = map(n -> msgs[DirectedEdge(n, v)], nbrs)
    others(k) = TupleTools.deleteat(ntuple(identity, M), k + 1)   # every leg except target k
    prefix_legs(k) = (k + 1, others(k)...)                        # open leg k+1 alone in codomain
    suffix_legs(k) = (others(k)..., k + 1)                        # open leg k+1 alone in domain

    cp = allocator_checkpoint!(allocator)

    # ket prefix chain (built up to the largest target). The link type is read off
    # the first link, so the container is concrete and GPU-portable.
    prefix1 = _repartition(T, prefix_legs(1), 1, backend, allocator)
    prefix = Vector{typeof(prefix1)}(undef, d)
    prefix[1] = prefix1
    for k in 2:legmax
        prefix[k] = _absorb(prefix[k - 1], prefix_legs(k - 1), k, incoming[k - 1], prefix_legs(k), 1, backend, allocator)
    end

    # bra suffix chain (built down to the smallest target, adjoint messages)
    suffixd = _repartition(T, suffix_legs(d), M - 1, backend, allocator)
    suffix = Vector{typeof(suffixd)}(undef, d)
    suffix[d] = suffixd
    for k in (d - 1):-1:legmin
        suffix[k] = _absorb(suffix[k + 1], suffix_legs(k + 1), k + 2, incoming[k + 1]', suffix_legs(k), M - 1, backend, allocator)
    end

    pA = ((1,), ntuple(i -> i + 1, M - 1))
    pB = (ntuple(identity, M - 1), (M,))
    for (i, k) in enumerate(target_legs)
        cp_close = allocator_checkpoint!(allocator)
        tensorcontract!(
            out[i], prefix[k], pA, false, suffix[k], pB, true,
            ((2,), (1,)), One(), Zero(), backend, allocator
        )
        allocator_reset!(allocator, cp_close)
    end

    allocator_reset!(allocator, cp)
    return out
end

"""
    tr_distance(A, B; p=1, is_hermitian=false) -> Real
    tr_distance!(A, B; p=1, is_hermitian=false) -> Real

Schatten-`p` distance between two messages after trace normalization,

    ‖ A/tr(A) − B/tr(B) ‖_p .

This is the natural BP convergence diagnostic: BP messages are only
defined up to an overall scale, so comparing raw messages is meaningless;
comparing trace-normalised ones is. With `p = 1` (the default) this is
the trace distance.

Singular values of the difference are used by default; pass
`is_hermitian = true` to use eigenvalues instead, which is cheaper and
appropriate when both messages are known to be Hermitian (e.g. positive
density-matrix-like messages). The bang version may overwrite `A` and `B`.
"""
function tr_distance(
        A::MessageTensor, B::MessageTensor;
        p::Real = 1, is_hermitian::Bool = false
    )
    diff = add(A, B, -inv(tr(B)), inv(tr(A)))

    # for small `diff`, need to `project_hermitian!` to make `eigh_vals` happy
    return norm(is_hermitian ? eigh_vals!(project_hermitian!(diff)) : svd_vals!(diff), p)
end
function tr_distance!(
        A::MessageTensor, B::MessageTensor;
        p::Real = 1, is_hermitian::Bool = false
    )
    diff = add!!(A, B, -inv(tr(B)), inv(tr(A)))
    # for small `diff`, need to `project_hermitian!` to make `eigh_vals` happy
    return norm(is_hermitian ? eigh_vals!(project_hermitian!(diff)) : svd_vals!(diff), p)
end

iterate_difference!(prev_messages::BPMessages, messages::BPMessages) =
    maximum(edge -> tr_distance!(prev_messages[edge], messages[edge]), keys(prev_messages))
