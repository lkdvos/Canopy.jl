using Canopy
using Canopy: UndirectedEdge, belief_propagation, reduced_density_matrix, expect
using TensorKit
using TensorKitTensors.SpinOperators: σᶻ, S_z_S_z
using Graphs: path_graph
using Dictionaries
using Random: Random, randn!
using Test


@testset "expect agrees with tr(op * reduced_density_matrix)" begin
    Random.seed!(0)
    L = 4
    P = ComplexSpace(2)
    V = ComplexSpace(3)
    pspaces = Dictionary(collect(1:L), fill(P, L))
    edges = [UndirectedEdge(i, i + 1) for i in 1:(L - 1)]
    vspaces = Dictionary(edges, fill(V, L - 1))
    state = randn!(TensorNetworkState{ComplexF64}(undef, pspaces, vspaces))

    msgs = BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter = 10)

    sz = σᶻ(ComplexF64, Trivial)
    szsz = S_z_S_z(ComplexF64, Trivial)

    # Single-vertex form (scalar v vs (v,) tuple).
    for v in 1:L
        a = expect(state, msgs, sz, v)
        b = expect(state, msgs, sz, (v,))
        c = tr(sz * reduced_density_matrix((v,), state, msgs))
        @test a ≈ b ≈ c
    end

    # Bond form (NTuple{2} vs UndirectedEdge).
    e = UndirectedEdge(1, 2)
    a = expect(state, msgs, szsz, e)
    b = expect(state, msgs, szsz, (1, 2))
    c = tr(szsz * reduced_density_matrix((1, 2), state, msgs))
    @test a ≈ b ≈ c
end
