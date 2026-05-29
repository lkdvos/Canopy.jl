using Canopy
using Canopy: TensorNetworkState, BPMessages, UndirectedEdge, DirectedEdge,
              LocalGate, apply!, physicalspace, virtualspace
using TensorKit
using TensorKitTensors.FermionOperators: fermion_space
using MatrixAlgebraKit: truncrank, trunctol, notrunc
using Graphs: path_graph, cycle_graph, star_graph, complete_graph, grid,
              edges, src, dst
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
    # normp=0 keeps the raw SVD magnitudes so the exact wavefunction is preserved.
    apply!(state, msgs, LocalGate((1, 2), g);  trunc=truncrank(8), normp=0)
    apply!(state, msgs, LocalGate((1, 2), g'); trunc=truncrank(8), normp=0)
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
    apply!(state, msgs, LocalGate((2, 3), g);  trunc=truncrank(6), normp=0)
    apply!(state, msgs, LocalGate((2, 3), g'); trunc=truncrank(6), normp=0)
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
        apply!(state, msgs, LocalGate((u, v), g);  trunc=truncrank(8), normp=0)
        apply!(state, msgs, LocalGate((u, v), g'); trunc=truncrank(8), normp=0)
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
    apply!(state_fwd, msgs_fwd, LocalGate((0, 2), g);      trunc=truncrank(6), normp=0)
    apply!(state_rev, msgs_rev, LocalGate((2, 0), g_swap); trunc=truncrank(6), normp=0)

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

@testset "Bond-message normalization with `normp`" begin
    P = ComplexSpace(2); V = ComplexSpace(3)
    pspaces = Dictionary([1, 2, 3], fill(P, 3))
    vspaces = Dictionary([UndirectedEdge(1, 2), UndirectedEdge(2, 3)], [V, V])

    # Unitary gate, normp=2: bond message gets unit 2-norm; the encoded
    # wavefunction matches the normp=0 result up to the rescaling exp(logλ).
    state, msgs = _build(pspaces, vspaces; seed=21)
    g = _random_unitary_gate(P; seed=21)
    _, _, info = apply!(state, msgs, LocalGate((1, 2), g); trunc=truncrank(8), normp=2)
    ψ_after = TensorMap(state)
    @test norm(msgs[DirectedEdge(1, 2)]) ≈ 1.0 atol=1e-13
    @test norm(msgs[DirectedEdge(2, 1)]) ≈ 1.0 atol=1e-13
    state_raw, msgs_raw = _build(pspaces, vspaces; seed=21)
    apply!(state_raw, msgs_raw, LocalGate((1, 2), g); trunc=truncrank(8), normp=0)
    ψ_raw = TensorMap(state_raw)
    @test norm(exp(info.logλ) * ψ_after - ψ_raw) < 1e-12

    # g ∘ g': accumulated logλ tracks the total absorbed norm, and the
    # compensated wavefunction matches the original.
    state, msgs = _build(pspaces, vspaces; seed=22)
    g = _random_unitary_gate(P; seed=22)
    ψ_before = TensorMap(_snapshot(pspaces, vspaces, state))
    _, _, i1 = apply!(state, msgs, LocalGate((1, 2), g);  trunc=truncrank(8), normp=2)
    _, _, i2 = apply!(state, msgs, LocalGate((1, 2), g'); trunc=truncrank(8), normp=2)
    ψ_after = TensorMap(state)
    @test norm(exp(i1.logλ + i2.logλ) * ψ_after - ψ_before) < 1e-12

    # Non-unitary gate exp(-τ h): logλ matches log‖σ‖₂ of θ, hand-computed by
    # repeating the same path with normp=0.
    state, msgs = _build(pspaces, vspaces; seed=23)
    Random.seed!(23)
    h = randn(ComplexF64, P ⊗ P, P ⊗ P); h = (h + h') / 2
    τ = 0.1
    g_imag = exp(-τ * h)
    state_ref, msgs_ref = _build(pspaces, vspaces; seed=23)
    _, _, info_norm = apply!(state, msgs, LocalGate((1, 2), g_imag);
                              trunc=truncrank(8), normp=2)
    _, _, info_raw  = apply!(state_ref, msgs_ref, LocalGate((1, 2), g_imag);
                              trunc=truncrank(8), normp=0)
    α_ref = norm(msgs_ref[DirectedEdge(1, 2)])
    @test info_norm.logλ ≈ log(α_ref) atol=1e-13
    @test info_raw.logλ == 0
    @test norm(msgs[DirectedEdge(1, 2)]) ≈ 1.0 atol=1e-13
end

# Diagnostic: compare `apply!` against direct dense gate contraction across
# (geometry × on-site space). With `trunc=notrunc()` and `normp=0`, `apply!`
# should reproduce the dense result to numerical precision. The geometry
# axis is graded by loop content (tree → single loop → multi-loop) so a
# failure pattern in the report directly says whether the bug is loop-related.

# Build (state, msgs) on a Graphs.jl graph with uniform spaces.
function _state_on(g, P, V; seed)
    state = TensorNetworkState{ComplexF64}(undef, g, P, V)
    Random.seed!(seed); Random.randn!(state)
    return state, BPMessages(state)
end

# Dense application of a 2-site gate G to the codomain wavefunction ψ at
# leg positions p₁, p₂ (which match vertex iteration order in TensorMap(state)).
function _dense_apply_2site(ψ, p1::Int, p2::Int, G)
    L = numout(ψ)
    @assert numin(ψ) == 0
    ψ_labels = Int[k == p1 ? 1 : k == p2 ? 2 : -k for k in 1:L]
    G_labels = Int[-p1, -p2, 1, 2]
    return ncon([G, ψ], [G_labels, ψ_labels])
end

const _DIAG_GEOMETRIES = (
    ("chain L=3",     path_graph(3)),                # tree, degrees 1–2
    ("star deg 4",    star_graph(5)),                # tree, central degree 4
    ("cycle L=4",     cycle_graph(4)),               # single length-4 loop
    ("K₄",            complete_graph(4)),            # multi-loop, girth 3
    ("3×3 PBC grid",  grid((3, 3); periodic=true)),  # multi-loop, girth 4
)

const _DIAG_SPACES = (
    ("bosonic",   ComplexSpace(2),        ComplexSpace(2)),
    ("fermionic", fermion_space(Trivial), Vect[fℤ₂](0 => 1, 1 => 1)),
)

@testset "Dense-equivalence diagnostic" begin
for (gname, g) in _DIAG_GEOMETRIES, (sname, P, V) in _DIAG_SPACES
    @testset "$gname ($sname)" begin
        e_rep = first(edges(g))
        u, v = src(e_rep), dst(e_rep)
        seed = hash((gname, sname))
        rtol = 1.0e-10

        state, msgs = _state_on(g, P, V; seed)
        ψ_init = TensorMap(state)

        # (A) Single-site identity is a no-op.
        s, m = _state_on(g, P, V; seed)
        apply!(s, m, LocalGate((u,), id(P)))
        @test isapprox(TensorMap(s), ψ_init; rtol)

        # (B) Two-site identity is a no-op.
        s, m = _state_on(g, P, V; seed)
        apply!(s, m, LocalGate((u, v), id(P ⊗ P)); trunc=notrunc(), normp=0)
        @test isapprox(TensorMap(s), ψ_init; rtol)

        # (C) Untruncated random 2-site unitary matches dense reference.
        G = _random_unitary_gate(P; seed=hash((seed, "C")), τ=0.1)
        ψ_ref = _dense_apply_2site(ψ_init, u, v, G)

        s, m = _state_on(g, P, V; seed)
        apply!(s, m, LocalGate((u, v), G); trunc=notrunc(), normp=0)
        @test isapprox(TensorMap(s), ψ_ref; rtol)

        # (C') Reverse-orientation form must give the same dense result via
        # the s₁ > s₂ canonical-ordering branch inside apply!.
        G_swap = permute(G, ((2, 1), (4, 3)))
        s, m = _state_on(g, P, V; seed)
        apply!(s, m, LocalGate((v, u), G_swap); trunc=notrunc(), normp=0)
        @test isapprox(TensorMap(s), ψ_ref; rtol)

        # (D) g ∘ g† round-trip restores ψ_init.
        s, m = _state_on(g, P, V; seed)
        apply!(s, m, LocalGate((u, v), G);  trunc=notrunc(), normp=0)
        apply!(s, m, LocalGate((u, v), G'); trunc=notrunc(), normp=0)
        @test isapprox(TensorMap(s), ψ_init; rtol)
    end
end
end