# --- Expectation values from BP messages -------------------------------------
# Bethe-approximated single- and two-site expectation values:
#
#   <op>_v ≈ <ψ|op_v|ψ> / <ψ|ψ>
#
# evaluated by closing the local environment at v (or the bond environment at edge e) with the converged BP messages.
#
# One pair of contractions serves both network types. A state has a single physical leg per
# site, which is the open one; an operator has two, and the second — the bra slot of the
# vectorization — is closed against its own adjoint, i.e. *traced out*. That makes the operator
# result the physical marginal of `X X†` where `X` is the network read as a purification, which
# is exactly the double-layer object belief propagation already converges the environment of.
# The only thing that differs between the two is how many labels sit between the open physical
# leg and the virtual legs, so `num_physical` supplies it and the index bookkeeping is shared.

@doc """
    reduced_density_matrix(sites, state::TensorNetworkState, messages) -> TensorMap
    reduced_density_matrix(sites, op::TensorNetworkOperator, messages) -> TensorMap

Bethe-approximated reduced density matrix on a path of vertices `sites`.

`sites` must form a *path* in the graph: all vertices must be distinct and
every consecutive pair `(sites[i], sites[i+1])` must be an edge of `state`.

Returns a `TensorMap` with space `(P_{v_1} ⊗ … ⊗ P_{v_n}) ← (P_{v_1} ⊗ … ⊗ P_{v_n})`,
where `P_{v_i} = physicalspace(state, sites[i])`.
The result is **normalized** and **positive definite**: `tr(ρ) ≈ 1` and every eigenvalue is
positive, so expectation values can be computed as `tr(O * ρ)`.

!!! note
    `isposdef(ρ)` nevertheless returns `false`: it begins with an *exact* hermiticity test,
    which a contraction result satisfies only to rounding. Check `ρ ≈ ρ'` and then
    `isposdef((ρ + ρ') / 2)`, whose argument is Hermitian bit-for-bit.

On a [`TensorNetworkOperator`](@ref) the network is read as a **purification**: slot 1 is the
physical leg and slot 2 the ancilla, and the ancilla is traced out. Writing `X` for the operator
the network represents, the result is the marginal of

    X X† / tr(X X†)

on `sites` — so `P_{v_i}` is `physicalspace(op, sites[i], 1)`, the *ket* space, and the returned
matrix is Hermitian and positive even though `X` itself need not be either.

!!! note "Which β this measures"
    For `X = exp(-βH/2)`, built by evolving [`identity_operator`](@ref) one-sided
    ([`LeftAction`](@ref)), `X X†` is `exp(-βH)` and this is the thermal state at `β`. For a
    Hermitian `ρ = exp(-βH)` built by evolving two-sided ([`SandwichAction`](@ref)), `ρ ρ†` is
    `exp(-2βH)` and this is the thermal state at `2β`. This is the same factor of two that makes
    one `SandwichAction` step of size `dτ` advance `β` by `2dτ`; see the operator docs.

    A genuine single-layer `tr(ρ O) / tr(ρ)` is *not* what this computes and is not provided —
    it needs vector-valued messages and a different fixed point.
""" reduced_density_matrix

function reduced_density_matrix(
        sites::NTuple{1, V}, net::AbstractTensorNetwork, messages::BPMessages;
        backend = DefaultBackend(), allocator = default_allocator(net),
    ) where {V}
    backend = inner_backend(backend)
    site = only(sites)
    np = num_physical(net)
    Tm = _twist_physical!(attach_all_messages(net, messages, site, backend, allocator), np)
    Td = net[site]'
    N = numin(Tm)
    virt = collect(1:N)                      # shared ket/bra virtual labels
    anc = collect((N + 1):(N + np - 1))      # traced physical slots — empty for a state
    indices = [vcat([-1], anc, virt), vcat(virt, [-2], anc)]
    ρ = repartition(ncon(Any[Tm, Td], indices), 1, 1)
    return scale!(ρ, inv(tr(ρ)))
end

function reduced_density_matrix(
        sites::NTuple{2, V}, net::AbstractTensorNetwork, messages::BPMessages;
        backend = DefaultBackend(), allocator = default_allocator(net),
    ) where {V}
    backend = inner_backend(backend)
    has_edge(net, sites...) || error("not implemented")
    s₁, s₂ = sites
    np = num_physical(net)

    e₁ = incoming_edges(net, s₁; exclude = (s₂,))
    e₂ = incoming_edges(net, s₂; exclude = (s₁,))
    T₁ = _twist_physical!(attach_messages(net, messages, s₁, e₁, backend, allocator), np)
    T₂ = _twist_physical!(attach_messages(net, messages, s₂, e₂, backend, allocator), np)
    N = numin(T₁)

    virt₁ = collect(1:N)
    virt₂ = collect((N + 1):(2N))
    anc₁ = collect((2N + 1):(2N + np - 1))
    anc₂ = collect((2N + np):(2N + 2np - 2))
    indices = [
        vcat([-1], anc₁, virt₁),
        vcat([-2], anc₂, virt₂),
        vcat(virt₁, [-4], anc₁),
        vcat(virt₂, [-3], anc₂),
    ]

    edge = DirectedEdge(sites...)
    leg_id₁ = leg_index(net, edge)
    leg_id₂ = leg_index(net, reverse(edge))

    # Stitch the shared bond: one label joins the two ket layers, another the two bra layers.
    # Virtual leg `k` sits at position `np + k` in the ket labels and at position `k` in the
    # bra (adjoint) labels, where the virtual legs come first.
    i₁, i₂ = indices[1][np + leg_id₁], indices[2][np + leg_id₂]
    indices[1][np + leg_id₁] = indices[2][np + leg_id₂] = i₁
    indices[3][leg_id₁] = indices[4][leg_id₂] = i₂

    tensors = Any[T₁, T₂, net[s₁]', net[s₂]']
    ρ = repartition(ncon(tensors, indices), 2, 2)
    return scale!(ρ, inv(tr(ρ)))
end

# Every physical leg of the *ket* layer carries a twist. For a state that is the single open
# leg, and twisting it on the site tensor is the same as twisting the corresponding leg of the
# result — the twist is diagonal in that leg's sector, so it commutes with the contraction. For
# an operator it also covers the *traced* ancilla leg, which is the fermionic partial trace: a
# leg closed back onto the same layer needs its parity factor, exactly as a `dual` virtual leg
# does in `_mul_leg!`. Without it the operator density matrix is not even positive.
_twist_physical!(T, np::Int) = twist!(T, ntuple(identity, np))

"""
    expect(state, msgs, op, sites) -> Number

Bethe-approximated expectation value `⟨ψ|op|ψ⟩ / ⟨ψ|ψ⟩` of `op` over
`sites`, computed as `tr(op * reduced_density_matrix(sites, state, msgs))`.

`sites` may be:
- a single vertex token `v` — equivalent to `(v,)`,
- an `NTuple{N, V}` of vertex tokens for a single-site (`N=1`), two-site
  (`N=2`), or path region (general `N`),
- an `UndirectedEdge` — equivalent to `(first(e), last(e))`.

`op` must have matching `TensorMap` space — see [`reduced_density_matrix`](@ref)
for the leg convention.

`state` may also be a [`TensorNetworkOperator`](@ref), in which case this is the thermal /
purification expectation value `tr(O · X X†) / tr(X X†)` — see [`reduced_density_matrix`](@ref)
for what that means and, in particular, which `β` it corresponds to.
"""
expect(net::AbstractTensorNetwork, msgs::BPMessages, op, sites::Tuple; kwargs...) =
    tr(op * reduced_density_matrix(sites, net, msgs; kwargs...))
expect(net::AbstractTensorNetwork, msgs::BPMessages, op, v; kwargs...) =
    expect(net, msgs, op, (v,); kwargs...)
expect(net::AbstractTensorNetwork, msgs::BPMessages, op, e::UndirectedEdge; kwargs...) =
    expect(net, msgs, op, (first(e), last(e)); kwargs...)
