using Canopy
using Canopy: TensorNetworkState, BPMessages, UndirectedEdge, LocalGate,
              CompositeGate, Circuit, apply!
using TensorKit
using MatrixAlgebraKit: truncrank
using Dictionaries
using Random
using Test


@testset "CompositeGate rejects overlapping sites" begin
    P = ComplexSpace(2)
    g = id(ComplexF64, P ⊗ P)
    g1 = LocalGate((1, 2), g)
    g2 = LocalGate((2, 3), g)  # shares vertex 2 with g1
    @test_throws ArgumentError CompositeGate([g1, g2])

    # Disjoint gates compose fine.
    g3 = LocalGate((3, 4), g)
    cg = CompositeGate([g1, g3])
    @test length(cg.gatelist) == 2
end


@testset "CompositeGate applies disjoint gates in parallel" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([1, 2, 3, 4], fill(P, 4))
    vspaces = Dictionary(
        [UndirectedEdge(1, 2), UndirectedEdge(2, 3), UndirectedEdge(3, 4)],
        fill(V, 3),
    )
    state = TensorNetworkState{ComplexF64}(undef, pspaces, vspaces)
    Random.seed!(0); Random.randn!(state)
    msgs = BPMessages(state)

    g = id(ComplexF64, P ⊗ P)
    cg = CompositeGate([LocalGate((1, 2), g), LocalGate((3, 4), g)])
    _, _, info = apply!(state, msgs, cg; trunc = truncrank(8), normp = 2)
    @test info.ϵ ≥ 0
    @test isfinite(info.logλ)
end


@testset "Circuit accumulates ϵ and logλ across gates" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([1, 2, 3], fill(P, 3))
    vspaces = Dictionary(
        [UndirectedEdge(1, 2), UndirectedEdge(2, 3)], fill(V, 2),
    )
    state = TensorNetworkState{ComplexF64}(undef, pspaces, vspaces)
    Random.seed!(0); Random.randn!(state)
    msgs = BPMessages(state)

    g = id(ComplexF64, P ⊗ P)
    circuit = Circuit([LocalGate((1, 2), g), LocalGate((2, 3), g)])
    _, _, info = apply!(state, msgs, circuit; trunc = truncrank(8), normp = 2)
    @test info.ϵ ≥ 0
    @test isfinite(info.logλ)
end
