# --- Expectation values from BP messages -------------------------------------
# Bethe-approximated single- and two-site expectation values:
#
#   <op>_v ≈ <ψ|op_v|ψ> / <ψ|ψ>
#
# evaluated by closing the local environment at v (or the bond environment at
# edge e) with the converged BP messages.
#
# Contraction conventions follow `messages.jl`:
#   * The TensorMap at vertex v has codomain = physical space `P` and domain =
#     `V_1 ⊗ ... ⊗ V_d` matching `state.adjacency[v]`.
#   * Incoming messages `m_{w→v}` live on v's side of the shared edge in
#     `V_k ← V_k`.
#   * Single-body `op` must have space `P ← P`.
#   * Two-body `op_e` must have space `(P_u ⊗ P_v) ← (P_u ⊗ P_v)`.

@doc """
    reduced_density_matrix(sites, state, messages) -> TensorMap

Bethe-approximated reduced density matrix on a path of vertices `sites`.
Each site contributes its ket tensor and its bra conjugate; bonds between two
consecutive sites are contracted directly (ket–ket, bra–bra), and every bond
crossing the boundary is closed with the BP message arriving at the inside
site from the outside neighbor.

`sites` must form a *path* in the graph: all vertices must be distinct and
every consecutive pair `(sites[i], sites[i+1])` must be an edge of `state`.

Returns a `TensorMap` with space
`(P_{v_1} ⊗ … ⊗ P_{v_n}) ← (P_{v_1} ⊗ … ⊗ P_{v_n})`, where
`P_{v_i} = physicalspace(state, sites[i])`. The result is **unnormalized**:
`tr(ρ) ≈ <ψ|ψ>_Bethe` on the region. Divide by `tr(ρ)` for a probability-
normalized density matrix.

Dispatches internally on `length(sites)` to a single-site, two-site, or
generic path implementation.
""" reduced_density_matrix

function reduced_density_matrix(
        sites::NTuple{1, V}, state::TensorNetworkState, messages::BPMessages,
    ) where {V}
    site = only(sites)
    Tm = attach_messages(state, messages, site, (DirectedEdge(n, site) for n in neighbors(state, site)))
    Td = state[site]'
    tensors = Any[Tm, Td]
    indices = [vcat([-1], 2:numind(Tm)), vcat(2:numind(Td), [-2])]
    ρ = twist!(repartition(ncon(tensors, indices), 1, 1), 1)
    return scale!(ρ, inv(tr(ρ)))
end

function reduced_density_matrix(
        sites::NTuple{2, V}, state::TensorNetworkState, messages::BPMessages,
    ) where {V}
    haskey(Graphs.edges(state), UndirectedEdge(sites...)) || error("not implemented")

    T₁ = attach_messages(
        state, messages, sites[1],
        (DirectedEdge(n, sites[1]) for n in neighbors(state, sites[1]) if n != sites[2])
    )
    T₂ = attach_messages(
        state, messages, sites[2],
        (DirectedEdge(n, sites[2]) for n in neighbors(state, sites[2]) if n != sites[1])
    )
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
