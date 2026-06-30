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

function _mul_leg!(
        dst::TensorMap{<:Any, S, 1, N}, src::TensorMap{<:Any, S, 1, N}, L, k::Int,
        backend, allocator,
    ) where {S, N}
    M = N + 1
    d = k + 1
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

function _absorb_legs(T::TensorMap{Tn, S, 1, N, A}, leg_factors, backend, allocator) where {Tn, S, N, A}
    factors = collect(leg_factors)
    isempty(factors) && return copy(T)
    M = N + 1
    cp = allocator_checkpoint!(allocator)
    out = T
    for i in eachindex(factors)
        k, L = factors[i]
        d = k + 1
        oindA = TupleTools.deleteat(ntuple(identity, M), d)
        pA = (oindA, (d,))
        pB = ((2,), (1,))
        pAB = ((1,), ntuple(j -> j + 1 < d ? j + 1 : (j + 1 == d ? M : j), N))  # new leg back to slot d
        TC = promote_contract(scalartype(out), scalartype(L))
        new_out = tensoralloc_contract(TC, out, pA, false, L, pB, false, pAB, Val(i != lastindex(factors)), allocator)
        if space(L, 1) == space(out, d)
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
    compute_outgoing_messages(msgs, state, v, backend, allocator) -> Vector{MessageTensor}
    compute_outgoing_messages!(out, msgs, state, v, backend, allocator) -> out

Compute *every* outgoing BP message from vertex `v` at once: one message per
neighbor `n`, along `DirectedEdge(v, n)`. The `d = degree(state, v)` outputs all
derive from the same `state[v]`, `state[v]'`, and the same set of incoming
messages, so computing them together shares work and improves data locality.

The result is a `Vector{MessageTensor}` in **neighbor order**: `out[k]` is the
message along `DirectedEdge(v, neighbors(state, v)[k])`, with space `V_n ← V_n`
where `V_n = virtualspace(state, DirectedEdge(n, v))` (receiver `n`'s side, as in
[`compute_message`](@ref)). Map an output back to its edge via
`neighbors(state, v)`.

Like [`compute_message`](@ref), neither variant mutates `msgs` nor performs
trace normalization / hermitian projection — callers normalise the returned
messages. `compute_outgoing_messages` allocates the output vector;
`compute_outgoing_messages!` fills `out`, whose `k`-th entry must already have
space `V_n ← V_n` for `n = neighbors(state, v)[k]`.
"""
function compute_outgoing_messages(
        msgs::BPMessages, state::TensorNetworkState, v,
        backend = DefaultBackend(), allocator = default_allocator(state),
    )
    TT = MessageTensor{scalartype(state), spacetype(state), TensorKit.storagetype(state)}
    out = map(neighbors(state, v)) do n
        V_n = virtualspace(state, DirectedEdge(n, v))
        return TT(undef, V_n ← V_n)
    end
    return compute_outgoing_messages!(out, msgs, state, v, backend, allocator)
end

function compute_outgoing_messages!(
        out, msgs::BPMessages, state::TensorNetworkState, v,
        backend = DefaultBackend(), allocator = default_allocator(state),
    )
    return _outgoing_doublelayer!(out, msgs, state, v, backend, allocator)
end

# Reference path: the obvious per-output loop over the existing single-edge
# kernel. Fermion-trivially correct (it *is* `compute_message!`, batched) and
# used as the golden oracle for the optimized double-layer path.
function _outgoing_naive!(out, msgs::BPMessages, state::TensorNetworkState, v, backend, allocator)
    for (k, n) in enumerate(neighbors(state, v))
        compute_message!(out[k], msgs, state, DirectedEdge(v, n), backend, allocator)
    end
    return out
end

# Double-layer leave-one-out. The message `v→nbrs[k]` needs incoming messages on
# every leg except `k`. We share that work across the `d` outputs with a ket
# *prefix* chain and a bra *suffix* chain:
#
#   Kpre[k] = state[v] with messages on legs 1..k-1 absorbed   (ket)
#   Ks      = state[v] with messages on legs k+1..d absorbed   (bra, via conjB)
#
# and close `Kpre[k]` against `Ks` (physical + all legs ≠ k traced) exactly as
# `compute_message!` closes `Tm` against the bra. Each incoming message is then
# absorbed only twice (2(d-1) absorptions vs the naive d(d-1)).
#
# The suffix is built on a ket-shaped tensor (so the verified `_mul_leg!` path
# applies and twists are inherited, not re-derived), absorbing the *adjoint*
# message `m†`: with the closing's `conjB=true`, this reproduces the sandwich
# `Σ T·mⱼ·conj(T)` that `compute_message!` produces with `mⱼ` on the ket — see
# the scalar identity `mⱼ[i,j'] = conj(mⱼ†[j',i])`.
function _outgoing_doublelayer!(out, msgs::BPMessages, state::TensorNetworkState, v, backend, allocator)
    nbrs = neighbors(state, v)
    d = length(nbrs)
    T = state[v]
    N = numind(T)
    m = [msgs[DirectedEdge(n, v)] for n in nbrs]   # incoming message on each leg

    # Ket prefix chain: Kpre[k] carries messages on legs 1..k-1.
    Kpre = Vector{typeof(T)}(undef, d)
    Kpre[1] = T
    for j in 2:d
        Kpre[j] = _absorb_legs(Kpre[j - 1], (j - 1 => m[j - 1],), backend, allocator)
    end

    # Sweep k = d..1, rolling the bra suffix Ks (messages on legs k+1..d).
    Ks = T
    for k in d:-1:1
        tc = k + 1
        cA = TupleTools.deleteat(ntuple(identity, N), tc)
        pA = ((tc,), cA)
        pB = (cA, (tc,))
        cp = allocator_checkpoint!(allocator)
        tensorcontract!(out[k], Kpre[k], pA, false, Ks, pB, true, ((2,), (1,)), One(), Zero(), backend, allocator)
        allocator_reset!(allocator, cp)
        if k > 1
            Ks = _absorb_legs(Ks, (k => copy(m[k]'),), backend, allocator)
        end
    end
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
