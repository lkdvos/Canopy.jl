using Canopy
using Canopy: TensorNetworkState, BPMessages, UndirectedEdge, LocalGate, apply!
using TensorKit
using Dictionaries
using Random
using Test


@testset "LocalGate constructor: structural checks" begin
    P = ComplexSpace(2)
    g₂ = id(ComplexF64, P ⊗ P)

    # Mismatched leg / site count: enforced at the type level via
    # `TT <: AbstractTensorMap{T, S, N, N}` — no matching method, so MethodError.
    @test_throws MethodError LocalGate((1,), g₂)                  # 2 phys legs, 1 site
    @test_throws MethodError LocalGate((1, 2), id(ComplexF64, P)) # 1 phys leg, 2 sites

    # Codom and dom have different leg counts: same — type-level rejection.
    rect = randn(ComplexF64, P, P ⊗ P)                            # 1 codom, 2 dom
    @test_throws MethodError LocalGate((1, 2), rect)

    # Codomain ≠ domain (same leg count) is allowed — e.g. swap-like or
    # basis-changing gates that change the on-site physical space.
    nonsquare = randn(ComplexF64, ComplexSpace(3), P)             # ℂ³ ← P
    lg = LocalGate((1,), nonsquare)
    @test domain(lg.tensor) == ProductSpace(P)
    @test codomain(lg.tensor) == ProductSpace(ComplexSpace(3))

    # Well-formed square gates round-trip through the constructor.
    g_local = LocalGate((1, 2), g₂)
    @test g_local.sites == (1, 2)
    @test space(g_local.tensor) == (P ⊗ P ← P ⊗ P)
end


@testset "LocalGate compatibility checks against state" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([1, 2, 3], fill(P, 3))
    vspaces = Dictionary([UndirectedEdge(1, 2), UndirectedEdge(2, 3)], [V, V])
    state = TensorNetworkState{ComplexF64}(undef, pspaces, vspaces)
    Random.seed!(0); Random.randn!(state)
    msgs = BPMessages(state)

    # Wrong physical space.
    bad = id(ComplexF64, ComplexSpace(3) ⊗ ComplexSpace(3))
    @test_throws SpaceMismatch apply!(state, msgs, LocalGate((1, 2), bad))

    # Sites that don't share an edge.
    g = id(ComplexF64, P ⊗ P)
    @test_throws ArgumentError apply!(state, msgs, LocalGate((1, 3), g))

    # Site that isn't in the state.
    @test_throws KeyError apply!(state, msgs, LocalGate((42, 2), g))

    # Single-site wrong space.
    @test_throws SpaceMismatch apply!(state, msgs, LocalGate((1,), id(ComplexSpace(3))))
end
