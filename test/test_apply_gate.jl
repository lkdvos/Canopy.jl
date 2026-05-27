using Canopy
using Canopy: TensorNetworkState, BPMessages, UndirectedEdge, DirectedEdge,
              LocalGate, apply!, physicalspace, virtualspace
using TensorKit
using MatrixAlgebraKit: truncrank, trunctol
using Dictionaries
using Random
using Test

# Build a fresh state from an explicit (pspaces, vspaces) dictionary pair, with
# randn-filled tensors. Returns (state, msgs).
function _build(pspaces, vspaces; seed=0)
    state = TensorNetworkState{ComplexF64}(undef, pspaces, vspaces)
    Random.seed!(seed); Random.randn!(state)
    return state, BPMessages(state)
end

# Snapshot of `state`'s on-site tensors into a fresh state with the same
# adjacency, so we can compare dense materializations before / after a gate.
function _snapshot(pspaces, vspaces, state)
    snap = TensorNetworkState{ComplexF64}(undef, pspaces, vspaces)
    for v in keys(state.vertices)
        snap.vertices[v] = copy(state[v])
    end
    return snap
end

_overlap(ψ, ϕ) = abs(dot(ψ, ϕ)) / (norm(ψ) * norm(ϕ))

# A representative two-site unitary, generated from a random Hermitian.
function _random_unitary_gate(P; seed=0, τ=0.05)
    Random.seed!(seed)
    h = randn(ComplexF64, P ⊗ P, P ⊗ P)
    h = (h + h') / 2
    return exp(-im * τ * h)
end

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

@testset "apply! compatibility checks against state" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([1, 2, 3], fill(P, 3))
    vspaces = Dictionary([UndirectedEdge(1, 2), UndirectedEdge(2, 3)], [V, V])
    state, msgs = _build(pspaces, vspaces; seed=4)

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

@testset "Non-square single-site gate replaces physical space" begin
    P = ComplexSpace(2); Q = ComplexSpace(3); V = ComplexSpace(2)
    pspaces = Dictionary([1, 2], [P, P])
    vspaces = Dictionary([UndirectedEdge(1, 2)], [V])
    state, msgs = _build(pspaces, vspaces; seed=7)

    g = randn(ComplexF64, Q, P)                # Q ← P
    @test physicalspace(state, 1) == P
    apply!(state, msgs, LocalGate((1,), g))
    @test physicalspace(state, 1) == Q
end

@testset "Single-site identity gate is a no-op" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([1, 2, 3], fill(P, 3))
    vspaces = Dictionary([UndirectedEdge(1, 2), UndirectedEdge(2, 3)], [V, V])
    state, msgs = _build(pspaces, vspaces; seed=5)

    for v in 1:3
        T_before = copy(state[v])
        apply!(state, msgs, LocalGate((v,), id(P)))
        @test isapprox(state[v], T_before; atol=1e-14)
    end
end

# The two-site unitary round-trip is the central correctness check: applying
# `g` and then `g'` on the same edge should leave the dense wave-function
# unchanged up to numerical noise. We exercise three topologies (chain, PBC
# cycle, degree-4 star) plus both edge orientations on the star to hit the
# `k = 1` and `k = Nd` boundary cases in the leg-permutation code.
@testset "Two-site unitary g ∘ g' round-trip (chain L=3)" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([1, 2, 3], fill(P, 3))
    vspaces = Dictionary([UndirectedEdge(1, 2), UndirectedEdge(2, 3)], [V, V])
    state, msgs = _build(pspaces, vspaces; seed=1)
    g = _random_unitary_gate(P; seed=1)

    ψ_before = TensorMap(_snapshot(pspaces, vspaces, state))
    apply!(state, msgs, LocalGate((1, 2), g);  trunc=truncrank(8))
    apply!(state, msgs, LocalGate((1, 2), g'); trunc=truncrank(8))
    ψ_after = TensorMap(state)

    @test _overlap(ψ_before, ψ_after) ≈ 1.0 atol=1e-12
    @test norm(ψ_before - ψ_after) < 1e-12
end

@testset "Two-site unitary g ∘ g' round-trip (PBC cycle L=4)" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([1, 2, 3, 4], fill(P, 4))
    vspaces = Dictionary(
        [UndirectedEdge(1, 2), UndirectedEdge(2, 3),
         UndirectedEdge(3, 4), UndirectedEdge(1, 4)],
        fill(V, 4),
    )
    state, msgs = _build(pspaces, vspaces; seed=2)
    g = _random_unitary_gate(P; seed=2)

    ψ_before = TensorMap(_snapshot(pspaces, vspaces, state))
    apply!(state, msgs, LocalGate((2, 3), g);  trunc=truncrank(6))
    apply!(state, msgs, LocalGate((2, 3), g'); trunc=truncrank(6))
    ψ_after = TensorMap(state)

    @test _overlap(ψ_before, ψ_after) ≈ 1.0 atol=1e-12
    @test norm(ψ_before - ψ_after) < 1e-12
end

@testset "Two-site unitary round-trip (degree-4 star, both edge boundaries)" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([0, 1, 2, 3, 4], fill(P, 5))
    vspaces = Dictionary(
        [UndirectedEdge(0, 1), UndirectedEdge(0, 2),
         UndirectedEdge(0, 3), UndirectedEdge(0, 4)],
        fill(V, 4),
    )
    state, msgs = _build(pspaces, vspaces; seed=11)
    g = _random_unitary_gate(P; seed=11)
    ψ_before = TensorMap(_snapshot(pspaces, vspaces, state))

    # Edge (0,1) hits k₁ = 1; edge (0,4) hits k₁ = Nd.
    for (u, v) in ((0, 1), (0, 4))
        apply!(state, msgs, LocalGate((u, v), g);  trunc=truncrank(8))
        apply!(state, msgs, LocalGate((u, v), g'); trunc=truncrank(8))
    end
    ψ_after = TensorMap(state)
    @test _overlap(ψ_before, ψ_after) ≈ 1.0 atol=1e-11
    @test norm(ψ_before - ψ_after) < 1e-11
end

@testset "Reverse-order sites with swapped gate match canonical order" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([0, 1, 2, 3, 4], fill(P, 5))
    vspaces = Dictionary(
        [UndirectedEdge(0, 1), UndirectedEdge(0, 2),
         UndirectedEdge(0, 3), UndirectedEdge(0, 4)],
        fill(V, 4),
    )
    state_fwd, msgs_fwd = _build(pspaces, vspaces; seed=3)
    state_rev = _snapshot(pspaces, vspaces, state_fwd)
    msgs_rev = BPMessages(state_rev)

    g = _random_unitary_gate(P; seed=3)
    g_swap = permute(g, ((2, 1), (4, 3)))   # swap the two physical slots
    apply!(state_fwd, msgs_fwd, LocalGate((0, 2), g);      trunc=truncrank(6))
    apply!(state_rev, msgs_rev, LocalGate((2, 0), g_swap); trunc=truncrank(6))

    ψ_fwd, ψ_rev = TensorMap(state_fwd), TensorMap(state_rev)
    @test _overlap(ψ_fwd, ψ_rev) ≈ 1.0 atol=1e-13
    @test norm(ψ_fwd - ψ_rev) < 1e-13
end

@testset "Truncation shrinks the bond dimension" begin
    P = ComplexSpace(2); V = ComplexSpace(4)
    pspaces = Dictionary([1, 2], fill(P, 2))
    vspaces = Dictionary([UndirectedEdge(1, 2)], [V])
    state, msgs = _build(pspaces, vspaces; seed=2)
    @test dim(virtualspace(state, DirectedEdge(1, 2))) == 4

    apply!(state, msgs, LocalGate((1, 2), id(P ⊗ P)); trunc=truncrank(2))
    @test dim(virtualspace(state, DirectedEdge(1, 2))) ≤ 2
end

@testset "apply! reports two methods" begin
    # Sanity: only the NTuple{1} and NTuple{2} specializations exist.
    @test length(methods(apply!)) == 2
end
