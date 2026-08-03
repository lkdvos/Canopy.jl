# BP messages live on the *directed* graph: each undirected edge `(u, v)` carries
# `DirectedEdge(u, v)` and `DirectedEdge(v, u)`. A directed edge `(s, r)` is read
# `sender → receiver`; `msgs[e]` lives on the receiver's side of the bond, with space
# `V_r ← V_r`, codomain = bra index and domain = ket index.
#
# The geometric convention, its double-layer meaning, the trace-normalization
# convention and the two `compute_message!` formulations are documented in
# `docs/src/design.md` ("BP messages" and "The vertex-batched message kernel").

const MessageTensor{T <: Number, S <: IndexSpace, A <: DenseVector{T}} =
    TensorMap{T, S, 1, 1, A}

"""
    BPMessages{T, S, A, V}

Container for BP messages over a [`TensorNetworkState`](@ref) with vertex
keys of type `V`. Each undirected state edge contributes two entries — one
per direction — stored in a `Dictionary` keyed by `DirectedEdge{V}`.

Every message is a `TensorMap{T, S, 1, 1, A}` of space `V_r ← V_r`, where
`V_r` is the virtual-leg space of the *receiver* at that bond, with codomain =
bra index and domain = ket index.
"""
struct BPMessages{T <: Number, S <: IndexSpace, A <: DenseVector{T}, V}
    messages::Dictionary{DirectedEdge{V}, MessageTensor{T, S, A}}
end

"""
    BPMessages(state::TensorNetworkState)

Allocate a `BPMessages` for `state` with every directed edge initialized to
the identity on the receiver's side of the underlying undirected edge.

This is the standard BP starting point and is *exact* on trees (one sweep
reaches the fixed point).
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

# A `TensorNetworkOperator` reaches BP through its fused state view: message spaces involve
# only virtual legs, which the view shares, so the messages are interchangeable between the
# two objects. See `TensorNetworkState(::TensorNetworkOperator)`.
BPMessages(op::TensorNetworkOperator) = BPMessages(TensorNetworkState(op))

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
  space of the receiver (`last(edge)`) at that bond.

Does not check that the messages are a BP fixed point; only that their
spaces line up with `state`.
"""
check_consistency(op::TensorNetworkOperator, msgs::BPMessages) =
    check_consistency(TensorNetworkState(op), msgs)

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
    backend = inner_backend(backend)
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
    backend = inner_backend(backend)
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
        state::AbstractTensorNetwork, msgs::BPMessages, site, edges,
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
    state::AbstractTensorNetwork, msgs::BPMessages, site,
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
    backend = inner_backend(backend)
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

Two formulations of the same contraction are implemented: the *pairwise* one described
above, and the `Layout(k)` *blocked* one further down this file, which has fewer copy
passes and folds the fermionic signs into the (χ²-sized) messages. **The blocked one runs
unconditionally**; [`PairwiseBackend`](@ref) forces the pairwise one, which is how the two
are differentially tested against each other. See `docs/src/design.md`.
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
    backend = inner_backend(backend)
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
    backend = inner_backend(backend)
    M = numind(tensor)
    pC = (ntuple(i -> newlegs[i], ncod), ntuple(i -> newlegs[ncod + i], M - ncod))
    result = tensoralloc_add(scalartype(tensor), tensor, pC, false, Val(true), allocator)
    cp = allocator_checkpoint!(allocator)
    tensoradd!(result, tensor, pC, false, One(), Zero(), backend, allocator)
    allocator_reset!(allocator, cp)
    return result
end

function compute_message!(
        out, msgs::BPMessages, state::TensorNetworkState, edges::AbstractVector{<:DirectedEdge},
        backend = DefaultBackend(), allocator = default_allocator(state),
    )
    isempty(edges) && return out
    return _blocked_message!(out, msgs, state, edges, inner_backend(backend), allocator)
end

function compute_message!(
        out, msgs::BPMessages, state::TensorNetworkState,
        edges::AbstractVector{<:DirectedEdge}, backend::PairwiseBackend,
        allocator = default_allocator(state),
    )
    isempty(edges) && return out
    return _pairwise_message!(out, msgs, state, edges, inner_backend(backend), allocator)
end

# Shared leave-one-out: `prefix[k]` absorbs virtual legs 1..k-1, `suffix[k]` absorbs
# k+1..d (adjoint messages), and the closing contracts them over the physical leg and
# every virtual leg ≠ k. Only the prefix up to the largest target and the suffix down to
# the smallest are built, so a clustered subset costs proportionally less.
#
# The differential-test oracle, reached only through `PairwiseBackend`.
function _pairwise_message!(
        out, msgs::BPMessages, state::TensorNetworkState,
        edges::AbstractVector{<:DirectedEdge}, backend, allocator,
    )
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

# BP requires **symmetric braiding**, and this is the one place that has to say so out
# loud. Every primitive the blocked chain uses succeeds under anyonic braiding, but the
# fermionic-sign derivation folds `twist(σ)` factors using `twist(σ)² = 1`, which is false
# there — so without this check the kernel would run to completion and return **wrong
# numbers silently**. The pairwise kernel is safe by accident: TensorKit's
# `tensorcontract!` refuses non-symmetric braiding outright.
function _require_symmetric_braiding(t)
    I = sectortype(t)
    BraidingStyle(I) isa SymmetricBraiding || throw(
        SectorMismatch(
            lazy"belief propagation requires symmetric braiding, got $(BraidingStyle(I)) \
                 for sector type $I: the message kernel's fermionic-sign derivation uses \
                 twist(σ)² = 1, which only holds for symmetric braiding"
        )
    )
    return nothing
end

# --- blocked (`Layout(k)`) vertex-batched kernel -------------------------------
#
# Same mathematics as the pairwise kernel above, one layout family for both chains with
# the target leg alone in the domain:
#
#     Layout(k) = (phys, l1 ... lk-hat ... lN) <- (lk)
#
# so a relayout is a single transposition, every absorption is a composition (one `gemm`
# per coupled sector, no repartition of the link), and the closing writes the output
# message in its natural partition. Fermionic signs fold into the (chi^2-sized) transposed
# messages and the ket entry copy instead of costing `twist!` passes over the chain links.
#
# **Read `docs/src/design.md` before changing the sign handling or the chain step.** It
# carries the sign derivation and three reformulations that were measured and rejected;
# each looks like an obvious improvement and each is slower.

# `twist!(mt, (1,))` for a message-shaped (`1 ← 1`) tensor, without the per-subblock
# hashed lookup: a `1 ← 1` tensor has one fusion tree pair per coupled sector and a
# single-leg tree's uncoupled sector *is* its coupled one, so scaling `blocks(mt)` by
# `twist(c)` is the same operation with one structure lookup for the whole tensor rather
# than the two `GlobalLRUCache` lookups `t[f₁, f₂]` costs per sector.
function _twist_message!(mt)
    for (c, b) in blocks(mt)
        θ = twist(c)
        isone(θ) || scale!(b, θ)
    end
    return mt
end

# Incoming message `j` in composable form: transposed to `pmsg = ((2,), (1,))`
# so that composition contracts its ket index and leaves its bra index as the
# new leg, and twisted on its codomain leg — the per-coupled-sector `twist(σⱼ)`
# derived above, at χ² cost.
function _transposed_message(msg, pmsg, backend, allocator)
    mt = tensoralloc_add(scalartype(msg), msg, pmsg, false, Val(true), allocator)
    tensoradd!(mt, msg, pmsg, false, One(), Zero(), backend, allocator)
    return _twist_message!(mt)
end

# The transposed messages, hoisted out of both chains: `msgt[j]` is absorbed by the ket
# chain (`j < legmax`) and `adjoint(msgt[j])` by the bra chain (`j > legmin`). `nothing`
# when neither chain takes a step (`d == 1`). Cost of holding them is `d` χ²-sized buffers
# against the `≈2d` links of `dim(P)·χ^d` the chains already hold live. Why one tensor can
# be shared between the chains, and why hoisting beats transposing inside them:
# `docs/src/design.md`.
function _transposed_messages(incoming, legmin, legmax, pmsg, backend, allocator)
    d = length(incoming)
    absorbed(j) = j < legmax || j > legmin
    j0 = findfirst(absorbed, 1:d)
    isnothing(j0) && return nothing
    mt = _transposed_message(incoming[j0], pmsg, backend, allocator)
    msgt = Vector{typeof(mt)}(undef, d)
    msgt[j0] = mt
    for j in (j0 + 1):d
        absorbed(j) || continue
        msgt[j] = _transposed_message(incoming[j], pmsg, backend, allocator)
    end
    return msgt
end

# One chain step: absorb the composable message `mt` (or its adjoint, on the bra chain)
# into the sole domain leg of the `Layout` link and re-emit in the neighbouring layout, `p`
# being the axis transposition. One `gemm` per coupled sector with no copy of `link` (`pid`
# is `link`'s own partition and only sizes the output buffer); `tmp` is taken above a
# checkpoint and released before returning, so only the chain links stay live.
#
# `mul!` cannot write into a differently-partitioned destination, hence the separate
# re-permute. **Do not try to fuse them into one `tensorcontract!`** — measured 0.3-9.8%
# slower at every point; `docs/src/design.md` has the reason.
function _blocked_step(link, mt, pid, p, backend, allocator)
    result = tensoralloc_add(scalartype(link), link, p, false, Val(true), allocator)
    cp = allocator_checkpoint!(allocator)
    tmp = tensoralloc_add(scalartype(link), link, pid, false, Val(true), allocator)
    mul!(tmp, link, mt)
    tensoradd!(result, tmp, p, false, One(), Zero(), backend, allocator)
    allocator_reset!(allocator, cp)
    return result
end

# `T` relaid out into `Layout(k)` — the entry link of either chain.
function _chain_entry(T, p, backend, allocator)
    link = tensoralloc_add(scalartype(T), T, p, false, Val(true), allocator)
    tensoradd!(link, T, p, false, One(), Zero(), backend, allocator)
    return link
end

function _blocked_message!(
        out, msgs::BPMessages, state::TensorNetworkState,
        edges::AbstractVector{<:DirectedEdge}, backend, allocator,
    )
    v = first(first(edges))
    T = state[v]
    _require_symmetric_braiding(T)
    M = numind(T)
    np = numout(T)
    nbrs = neighbors(state, v)
    d = length(nbrs)
    target_legs = map(edges) do e
        first(e) == v || throw(ArgumentError(lazy"edge $e does not leave the shared source $v"))
        return leg_index(state, e)
    end
    legmin, legmax = extrema(target_legs)

    # Every space query is hoisted here: the loops below touch only tensors.
    #
    # Virtual leg `k` sits at tensor slot `np + k`, and a `Layout` has `M - 1` codomain
    # axes and one domain axis whatever `np` is, so only the slot arithmetic depends on
    # the physical arity. In `Layout(k)`'s own axis order the leg a chain step moves into
    # the domain is at position `np + k` for both chains: on the ket side the legs before
    # `k + 1` skip `k`, and on the bra side (source `Layout(k+1)`) nothing before `k` is.
    #
    # `np > 1` is threaded through but **not exercised** — nothing can currently supply a
    # two-physical-leg tensor. See `docs/src/design.md`, "Physical arity".
    allinds = ntuple(identity, M)
    layout(k) = (TupleTools.deleteat(allinds, np + k), (np + k,))   # Layout(k)
    pid = (ntuple(identity, M - 1), (M,))                          # a layout's own partition
    pswap(j) = (ntuple(i -> ifelse(i == j, M, i), M - 1), (j,))     # Layout(j-1) ↔ Layout(j)
    pmsg = ((2,), (1,))                                            # message → composable form
    # `Z`'s physical factor: one `twist(σ_p)` per *dual* physical leg. `twist!` applies the
    # product of the given legs in a single pass.
    dual_phys = filter(p -> isdual(space(T, p)), ntuple(identity, np))

    incoming = map(n -> msgs[DirectedEdge(n, v)], nbrs)

    cp = allocator_checkpoint!(allocator)

    # the messages both chains absorb, in composable form, built once per leg
    msgt = _transposed_messages(incoming, legmin, legmax, pmsg, backend, allocator)

    # ket chain, `Layout(1)` up to `Layout(legmax)`: `ket[k]` has messages 1 … k-1
    # absorbed. Both chains share a link type, read off the first entry so the
    # containers are concrete and GPU-portable.
    ket1 = _chain_entry(T, layout(1), backend, allocator)
    isempty(dual_phys) || twist!(ket1, dual_phys)   # the physical legs' `Z` factor
    ket = Vector{typeof(ket1)}(undef, d)
    ket[1] = ket1
    for k in 1:(legmax - 1)
        ket[k + 1] = _blocked_step(ket[k], msgt[k], pid, pswap(np + k), backend, allocator)
    end

    # bra chain, `Layout(d)` down to `Layout(legmin)`: `bra[k]` has messages
    # k+1 … d absorbed, as adjoints — conjugated by the closing.
    bra = Vector{typeof(ket1)}(undef, d)
    bra[d] = _chain_entry(T, layout(d), backend, allocator)
    for k in (d - 1):-1:legmin
        bra[k] = _blocked_step(
            bra[k + 1], adjoint(msgt[k + 1]), pid, pswap(np + k), backend, allocator
        )
    end

    # closing: one `gemm('C', 'N')` per coupled sector, straight into `out[i]`.
    for (i, k) in enumerate(target_legs)
        mul!(out[i], adjoint(bra[k]), ket[k])
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
