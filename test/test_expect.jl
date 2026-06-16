using Canopy
using Canopy: UndirectedEdge, belief_propagation, reduced_density_matrix, expect
using TensorKit
using TensorKitTensors.SpinOperators: σᶻ, S_z_S_z
using TensorKitTensors.FermionOperators: fermion_space, f_num, f_hop
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

@testset "reduced_density_matrix on a fermionic (fℤ₂) state" begin
    # A path graph is a tree, so belief propagation is exact: single-site
    # marginals and the marginals of two-site density matrices must agree.
    # This pins the fermionic twist/sign conventions, which are no-ops under
    # `Trivial` symmetry and therefore invisible to the test above.
    Random.seed!(42)
    L = 4
    P = fermion_space(Trivial)            # Vect[fℤ₂](0 => 1, 1 => 1)
    V = Vect[fℤ₂](0 => 2, 1 => 2)
    pspaces = Dictionary(collect(1:L), fill(P, L))
    edges = [UndirectedEdge(i, i + 1) for i in 1:(L - 1)]
    vspaces = Dictionary(edges, fill(V, L - 1))
    state = randn!(TensorNetworkState{ComplexF64}(undef, pspaces, vspaces))

    msgs = BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter = 50)

    n = f_num(ComplexF64)
    nn = n ⊗ id(P)                        # site-1 number operator on a two-site region
    nn′ = id(P) ⊗ n                       # site-2 number operator on a two-site region
    hop = f_hop(ComplexF64)

    ρ1 = [reduced_density_matrix((v,), state, msgs) for v in 1:L]
    for v in 1:L
        @test tr(ρ1[v]) ≈ 1               # probability-normalized
        @test ρ1[v] ≈ ρ1[v]'              # Hermitian
        @test imag(tr(n * ρ1[v])) ≈ 0 atol = 1e-10
        # `expect` wrapper agrees with the explicit trace under twists.
        @test expect(state, msgs, n, v) ≈ tr(n * ρ1[v])
    end

    for i in 1:(L - 1)
        ρ2 = reduced_density_matrix((i, i + 1), state, msgs)
        @test tr(ρ2) ≈ 1
        @test ρ2 ≈ ρ2'
        @test expect(state, msgs, hop, (i, i + 1)) ≈ tr(hop * ρ2)
        # The two- and single-site constructions must give the same local
        # number expectation (exact on a tree); a fermionic twist/sign error in
        # either density matrix would break this.
        @test tr(nn * ρ2) ≈ tr(n * ρ1[i])
        @test tr(nn′ * ρ2) ≈ tr(n * ρ1[i + 1])
    end
end
