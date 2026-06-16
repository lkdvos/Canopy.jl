# --- Expectation values from BP messages -------------------------------------
# Bethe-approximated single- and two-site expectation values:
#
#   <op>_v ≈ <ψ|op_v|ψ> / <ψ|ψ>
#
# evaluated by closing the local environment at v (or the bond environment at edge e) with the converged BP messages.

@doc """
    reduced_density_matrix(sites, state, messages) -> TensorMap

Bethe-approximated reduced density matrix on a path of vertices `sites`.

`sites` must form a *path* in the graph: all vertices must be distinct and
every consecutive pair `(sites[i], sites[i+1])` must be an edge of `state`.

Returns a `TensorMap` with space `(P_{v_1} ⊗ … ⊗ P_{v_n}) ← (P_{v_1} ⊗ … ⊗ P_{v_n})`,
where `P_{v_i} = physicalspace(state, sites[i])`.
The result is **normalized** and **positive definite**: `tr(ρ) ≈ 1` and `isposdef(ρ)` on the region, such that
expectation values can be computed as `tr(O * ρ)`.
""" reduced_density_matrix

function reduced_density_matrix(
        sites::NTuple{1, V}, state::TensorNetworkState, messages::BPMessages;
        backend = DefaultBackend(), allocator = _default_allocator(),
    ) where {V}
    site = only(sites)
    Tm = attach_all_messages(state, messages, site, backend, allocator)
    Td = state[site]'
    tensors = Any[Tm, Td]
    indices = [vcat([-1], 2:numind(Tm)), vcat(2:numind(Td), [-2])]
    ρ = twist!(repartition(ncon(tensors, indices), 1, 1), 1)
    return scale!(ρ, inv(tr(ρ)))
end

function reduced_density_matrix(
        sites::NTuple{2, V}, state::TensorNetworkState, messages::BPMessages;
        backend = DefaultBackend(), allocator = _default_allocator(),
    ) where {V}
    has_edge(state, sites...) || error("not implemented")

    T₁ = attach_messages(state, messages, sites[1], incoming_edges(state, sites[1]; exclude = (sites[2],)), backend, allocator)
    T₂ = attach_messages(state, messages, sites[2], incoming_edges(state, sites[2]; exclude = (sites[1],)), backend, allocator)
    tensors = Any[T₁, T₂, state[sites[1]]', state[sites[2]]']
    indices = [
        vcat([-1], 2:numind(T₁)),
        vcat([-2], (2:numind(T₂)) .+ numind(T₁)),
        vcat(2:numind(T₁), [-4]),
        vcat((2:numind(T₂)) .+ numind(T₁), [-3]),
    ]

    edge = DirectedEdge(sites...)
    leg_id₁ = leg_index(state, edge)
    leg_id₂ = leg_index(state, reverse(edge))

    i₁, i₂ = indices[1][leg_id₁ + 1], indices[2][leg_id₂ + 1]
    indices[1][leg_id₁ + 1] = indices[2][leg_id₂ + 1] = i₁
    indices[3][leg_id₁] = indices[4][leg_id₂] = i₂

    ρ = twist!(repartition(ncon(tensors, indices), 2, 2), (1, 2))
    return scale!(ρ, inv(tr(ρ)))
end

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
"""
expect(state::TensorNetworkState, msgs::BPMessages, op, sites::Tuple; kwargs...) =
    tr(op * reduced_density_matrix(sites, state, msgs; kwargs...))
expect(state::TensorNetworkState, msgs::BPMessages, op, v; kwargs...) =
    expect(state, msgs, op, (v,); kwargs...)
expect(state::TensorNetworkState, msgs::BPMessages, op, e::UndirectedEdge; kwargs...) =
    expect(state, msgs, op, (first(e), last(e)); kwargs...)
