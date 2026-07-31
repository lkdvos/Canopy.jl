using Canopy
using Canopy: TensorNetworkOperator, TensorNetworkState, BPMessages, UndirectedEdge, DirectedEdge,
              LocalGate, LeftAction, RightAction, SandwichAction, CompositeGate, Circuit,
              identity_operator, randn_operator, isvectorized, check_consistency,
              physicalspace, physicalspaces, virtualspace, degree, apply!, belief_propagation
using TensorKit
using TensorKitTensors.FermionOperators: fermion_space
using MatrixAlgebraKit: truncrank, trunctol, notrunc
using Graphs: path_graph, cycle_graph, star_graph, complete_graph, grid, edges, src, dst
using Dictionaries
using Random
using Test

# --- helpers ------------------------------------------------------------------

# `TensorMap(op)` orders its legs so that this is the operator `⊗P ← ⊗P`, with vertex `i` at
# slot `i` on both sides (`i` indexing `vertices(op)`).
_dense(op) = repartition(TensorMap(op), length(op), length(op))

# Embed a gate acting on `ps` (indices into `vertices`) into the full `⊗P ← ⊗P` operator, by
# padding with identities. Same leg-labelling convention as `TensorMap(op)`: ket of site `i`
# gets `-i` and bra of site `i` gets `-(2n + 1 - i)`, which is what makes `repartition(·, n, n)`
# put both sides in vertex order.
function _embed(G, n, ps, pspaces)
    tensors = Any[G]
    ket(i) = -i
    bra(i) = -(2n + 1 - i)
    indices = Vector{Int}[[map(ket, ps)..., map(bra, ps)...]]
    for i in 1:n
        i in ps && continue
        push!(tensors, id(pspaces[i]))
        push!(indices, [ket(i), bra(i)])
    end
    return repartition(ncon(tensors, indices), n, n)
end

_random_hermitian(P; seed = 0) = (Random.seed!(seed); h = randn(ComplexF64, P, P); (h + h') / 2)

function _random_unitary_gate(P; seed = 0, τ = 0.05)
    Random.seed!(seed)
    h = randn(ComplexF64, P ⊗ P, P ⊗ P)
    h = (h + h') / 2
    return exp(-im * τ * h)
end

# Fresh identity operator plus its (exact, since all bonds are 1-dimensional) messages.
function _fresh(topology, P)
    ρ = identity_operator(ComplexF64, topology, P)
    return ρ, BPMessages(ρ)
end

# A random operator whose dense form is well-conditioned enough to compare against.
function _fresh_random(topology, P, V; seed = 0)
    Random.seed!(seed)
    ρ = randn_operator(ComplexF64, topology, P, V)
    return ρ, BPMessages(ρ)
end

# Gate application is not yet validated for fermionic braiding, so the sweep is bosonic.
#
# Geometries for the dense-reference sweep: tree → single loop → multi-loop, with degree 1
# (padded trailing legs), degree 3 and degree 4 all represented. Deliberately capped at 5
# vertices: a dense *operator* costs `dim(P)^(2n)`, so the 3×3 periodic grid used by the state
# diagnostic is 512× more expensive here. That geometry is still covered, structurally rather
# than against a dense reference, by "Large-coordination geometry" below.
const _OP_GEOMETRIES = [
    ("chain-3", path_graph(3)),
    ("star-5", star_graph(5)),
    ("cycle-4", cycle_graph(4)),
    ("K4", complete_graph(4)),
]
const _OP_GATE_SPACES = [
    ("bosonic", ComplexSpace(2), ComplexSpace(2)),
    ("U1", Vect[U1Irrep](0 => 1, 1 => 1), Vect[U1Irrep](-1 => 1, 0 => 1, 1 => 1)),
]

# --- refusals -----------------------------------------------------------------

@testset "Fermionic gate application on an operator is refused, not silently wrong" begin
    P = fermion_space(Trivial)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    ρ, msgs = _fresh(es, P)
    g = id(P)
    G = id(P ⊗ P)
    @test_throws ArgumentError apply!(ρ, msgs, LocalGate((1,), g); action = SandwichAction)
    @test_throws ArgumentError apply!(ρ, msgs, LocalGate((1, 2), G); action = LeftAction)
    # a state on the same space still works — the guard is operator-only
    ψ = randn_state(ComplexF64, es, P, Vect[fℤ₂](0 => 1, 1 => 1))
    @test apply!(ψ, BPMessages(ψ), LocalGate((1,), g)) isa Tuple
end

@testset "SandwichAction requires a vectorized operator" begin
    P, V = ComplexSpace(2), ComplexSpace(2)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    ψ = randn_state(ComplexF64, es, P, V)
    purification = TensorNetworkOperator(ψ)          # trivial ancilla, so slot 2 ≠ dual(slot 1)
    @test !isvectorized(purification)
    msgs = BPMessages(purification)
    @test_throws SpaceMismatch apply!(purification, msgs, LocalGate((1,), id(P)); action = SandwichAction)
    # ... but a one-sided gate on the ket leg is perfectly fine on a purification
    @test apply!(purification, msgs, LocalGate((1,), id(P)); action = LeftAction) isa Tuple
end

@testset "`action` on a state is rejected, not ignored" begin
    P, V = ComplexSpace(2), ComplexSpace(2)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    ψ = randn_state(ComplexF64, es, P, V)
    msgs = BPMessages(ψ)
    g, G = id(P), id(P ⊗ P)
    for action in (LeftAction, RightAction, SandwichAction)
        @test_throws ArgumentError apply!(ψ, msgs, LocalGate((1,), g); action)
        @test_throws ArgumentError apply!(ψ, msgs, LocalGate((1, 2), G); action)
    end
end

@testset "Gate space mismatches are caught" begin
    P, Q, V = ComplexSpace(2), ComplexSpace(3), ComplexSpace(2)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    ρ, msgs = _fresh(es, P)
    @test_throws SpaceMismatch apply!(ρ, msgs, LocalGate((1,), id(Q)); action = LeftAction)
    @test_throws KeyError apply!(ρ, msgs, LocalGate((7,), id(P)); action = LeftAction)
    # two-site gate on a non-edge
    @test_throws ArgumentError apply!(ρ, msgs, LocalGate((1, 3), id(P ⊗ P)); action = SandwichAction)
end

# --- one-site gates -----------------------------------------------------------

@testset "One-site gate actions against dense — $sname" for (sname, P, _) in _OP_GATE_SPACES
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    n = 3
    pspaces = fill(P, n)
    g = _random_hermitian(P; seed = 3)
    𝒢 = _embed(g, n, (2,), pspaces)

    for (action, reference) in (
            (LeftAction, ρd -> 𝒢 * ρd),
            (RightAction, ρd -> ρd * 𝒢),
            (SandwichAction, ρd -> 𝒢 * ρd * 𝒢'),
        )
        ρ, msgs = _fresh(es, P)
        before = _dense(ρ)
        apply!(ρ, msgs, LocalGate((2,), g); action)
        @test check_consistency(ρ)
        @test isvectorized(ρ)
        @test _dense(ρ) ≈ reference(before) rtol = 1e-10
    end
end

@testset "One-site gate on a two-vertex operator (empty QR codomain)" begin
    # The smallest possible network: every leg of each site is either physical or the shared
    # bond, so `Q` has an empty codomain. Exercised for states elsewhere; check it at np = 2.
    P = ComplexSpace(2)
    es = [UndirectedEdge(1, 2)]
    g = _random_hermitian(P; seed = 4)
    𝒢 = _embed(g, 2, (1,), fill(P, 2))
    for (action, reference) in (
            (LeftAction, ρd -> 𝒢 * ρd), (RightAction, ρd -> ρd * 𝒢), (SandwichAction, ρd -> 𝒢 * ρd * 𝒢'),
        )
        ρ, msgs = _fresh(es, P)
        before = _dense(ρ)
        apply!(ρ, msgs, LocalGate((1,), g); action)
        @test _dense(ρ) ≈ reference(before) rtol = 1e-10
    end
end

@testset "One-site LeftAction can change the physical space" begin
    P, Q = ComplexSpace(2), ComplexSpace(3)
    es = [UndirectedEdge(1, 2)]
    ρ, msgs = _fresh(es, P)
    g = randn(ComplexF64, Q, P)                       # Q ← P
    apply!(ρ, msgs, LocalGate((1,), g); action = LeftAction)
    @test physicalspaces(ρ, 1) == (Q, dual(P))
    @test !isvectorized(ρ)                            # Gρ for rectangular G is not square
    # a right action on the same site restores squareness
    apply!(ρ, msgs, LocalGate((1,), g'); action = RightAction)    # ρ ↦ ρ g', consuming codomain(g') = P
    @test physicalspaces(ρ, 1) == (Q, dual(Q))
    @test isvectorized(ρ)
end

# --- two-site gates: the dense-equivalence diagnostic -------------------------

@testset "Two-site identity gates are no-ops — $sname / $gname" for
        (sname, P, V) in _OP_GATE_SPACES, (gname, g) in _OP_GEOMETRIES
    # Compared on the *dense* operator, not tensor-by-tensor: the SVD re-gauges the bond, and
    # with `notrunc()` a `SandwichAction` legitimately grows it to `d²χ` (the extra Schmidt
    # values are zero). Neither changes the operator the network represents.
    G = id(P ⊗ P)
    for action in (LeftAction, RightAction, SandwichAction)
        ρ, msgs = _fresh_random(g, P, V; seed = 21)
        before = _dense(ρ)
        e = first(edges(ρ))
        apply!(ρ, msgs, LocalGate((first(e), last(e)), G); action, trunc = notrunc(), normp = 0)
        @test check_consistency(ρ)
        @test check_consistency(ρ, msgs)
        @test _dense(ρ) ≈ before rtol = 1e-10
    end
end

@testset "Two-site gate actions against dense — $sname / $gname" for
        (sname, P, V) in _OP_GATE_SPACES, (gname, g) in _OP_GEOMETRIES
    G = _random_unitary_gate(P; seed = 5)
    ρ0, _ = _fresh_random(g, P, V; seed = 22)
    verts = collect(vertices(ρ0))
    n = length(verts)
    pspaces = fill(P, n)
    e = first(edges(ρ0))
    u, v = first(e), last(e)
    ps = (findfirst(==(u), verts), findfirst(==(v), verts))
    𝒢 = _embed(G, n, ps, pspaces)

    for (action, reference) in (
            (LeftAction, ρd -> 𝒢 * ρd),
            (RightAction, ρd -> ρd * 𝒢),
            (SandwichAction, ρd -> 𝒢 * ρd * 𝒢'),
        )
        ρ, msgs = _fresh_random(g, P, V; seed = 22)
        before = _dense(ρ)
        apply!(ρ, msgs, LocalGate((u, v), G); action, trunc = notrunc(), normp = 0)
        @test check_consistency(ρ)
        @test isvectorized(ρ)
        @test _dense(ρ) ≈ reference(before) rtol = 1e-10

        # reverse site order with the correspondingly permuted gate must agree — this is what
        # validates the canonical-orientation swap for two physical legs
        ρr, msgsr = _fresh_random(g, P, V; seed = 22)
        apply!(
            ρr, msgsr, LocalGate((v, u), permute(G, ((2, 1), (4, 3))));
            action, trunc = notrunc(), normp = 0,
        )
        @test _dense(ρr) ≈ _dense(ρ) rtol = 1e-10
    end
end

@testset "SandwichAction round-trip g then g† restores the operator — $sname / $gname" for
        (sname, P, V) in _OP_GATE_SPACES, (gname, g) in _OP_GEOMETRIES
    G = _random_unitary_gate(P; seed = 6)
    ρ, msgs = _fresh_random(g, P, V; seed = 23)
    before = _dense(ρ)
    e = first(edges(ρ))
    u, v = first(e), last(e)
    apply!(ρ, msgs, LocalGate((u, v), G); action = SandwichAction, trunc = notrunc(), normp = 0)
    apply!(ρ, msgs, LocalGate((u, v), G'); action = SandwichAction, trunc = notrunc(), normp = 0)
    @test _dense(ρ) ≈ before rtol = 1e-10
end

@testset "SandwichAction ≡ LeftAction then RightAction — $sname / $gname" for
        (sname, P, V) in _OP_GATE_SPACES, (gname, g) in _OP_GEOMETRIES
    # Dense-reference-free: the joint two-sided kernel must agree with applying the two
    # one-sided actions in sequence, since the intermediate SVD is exact and the gauge factors
    # on the untouched bonds do not change.
    G = _random_unitary_gate(P; seed = 7)
    ρa, msgsa = _fresh_random(g, P, V; seed = 24)
    ρb, msgsb = _fresh_random(g, P, V; seed = 24)
    e = first(edges(ρa))
    u, v = first(e), last(e)

    apply!(ρa, msgsa, LocalGate((u, v), G); action = SandwichAction, trunc = notrunc(), normp = 0)
    apply!(ρb, msgsb, LocalGate((u, v), G); action = LeftAction, trunc = notrunc(), normp = 0)
    apply!(ρb, msgsb, LocalGate((u, v), G'); action = RightAction, trunc = notrunc(), normp = 0)
    @test _dense(ρa) ≈ _dense(ρb) rtol = 1e-10
end

@testset "SandwichAction preserves hermiticity and trace — $gname" for (gname, g) in _OP_GEOMETRIES
    P, V = ComplexSpace(2), ComplexSpace(2)
    # Starting from 𝟙, `ρ ↦ GρG†` must stay Hermitian for *any* G, and preserve the trace when
    # G is unitary. A missing conjugation or transpose on the bra side breaks this immediately,
    # with no dense gate reference needed.
    for (label, G) in (
            ("unitary", _random_unitary_gate(P; seed = 8)),
            ("non-hermitian", (Random.seed!(9); randn(ComplexF64, P ⊗ P, P ⊗ P))),
        )
        ρ, msgs = _fresh(g, P)
        e = first(edges(ρ))
        tr_before = tr(_dense(ρ))
        apply!(ρ, msgs, LocalGate((first(e), last(e)), G); action = SandwichAction, trunc = notrunc(), normp = 0)
        d = _dense(ρ)
        @test d ≈ d' rtol = 1e-10
        label == "unitary" && @test tr(d) ≈ tr_before rtol = 1e-10
    end
end

@testset "Large-coordination geometry: structural checks on a 3×3 periodic grid" begin
    # Coordination 4 everywhere, so no vertex has padded trailing legs and every gate hits an
    # interior bond of a multi-loop graph. A dense reference is out of reach here (2^18 for the
    # operator), so this checks the invariants that do not need one: the leg bookkeeping must
    # round-trip through QR/SVD/reconstruct, the vectorized structure must survive, BP must
    # still converge on the result, and a unitary sandwich must preserve the Frobenius norm.
    P, V = ComplexSpace(2), ComplexSpace(2)
    g = grid((3, 3); periodic = true)
    G = _random_unitary_gate(P; seed = 13)
    ρ, msgs = _fresh_random(g, P, V; seed = 26)
    @test all(v -> degree(ρ, v) == 4, vertices(ρ))

    e = first(edges(ρ))
    u, v = first(e), last(e)
    for action in (LeftAction, RightAction, SandwichAction)
        _, _, info = apply!(ρ, msgs, LocalGate((u, v), G); action, trunc = truncrank(4), normp = 0)
        @test check_consistency(ρ)
        @test check_consistency(ρ, msgs)
        @test isvectorized(ρ)
        @test isfinite(info.ϵ)
    end
    msgs = belief_propagation(msgs, ρ; maxiter = 200, tol = 1e-10)
    @test check_consistency(ρ, msgs)

    # Everything here stays local on purpose. Nothing dense is materialized: with 9 sites and
    # two physical legs each the wavefunction is `dim(P)^18`, and a periodic grid gives `ncon`
    # no good contraction order — the norm/trace invariants are checked on the small geometries
    # instead ("SandwichAction preserves hermiticity and trace").
    ρ2, msgs2 = _fresh(g, P)                       # 1-dimensional bonds ⇒ BP is exact here
    apply!(ρ2, msgs2, LocalGate((u, v), G); action = SandwichAction, trunc = notrunc(), normp = 0)
    @test check_consistency(ρ2)
    @test check_consistency(ρ2, msgs2)
    @test isvectorized(ρ2)
    # the sandwich grows the touched bond from 1 to dim(P)^2 and leaves the rest alone
    @test dim(virtualspace(ρ2, DirectedEdge(u, v))) == dim(P)^2
    for e2 in edges(ρ2)
        e2 == UndirectedEdge(u, v) && continue
        @test dim(virtualspace(ρ2, DirectedEdge(first(e2), last(e2)))) == 1
    end
end

# --- truncation ---------------------------------------------------------------

@testset "Truncation shrinks the bond and reports an error" begin
    P, V = ComplexSpace(2), ComplexSpace(4)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    G = _random_unitary_gate(P; seed = 10)
    ρ, msgs = _fresh_random(es, P, V; seed = 25)
    _, _, info = apply!(ρ, msgs, LocalGate((1, 2), G); action = SandwichAction, trunc = truncrank(3))
    @test dim(virtualspace(ρ, DirectedEdge(1, 2))) == 3
    @test info.ϵ > 0
    @test check_consistency(ρ)
    @test check_consistency(ρ, msgs)

    # a weak gate barely truncates: the result stays close to the untouched operator
    ρ2, msgs2 = _fresh_random(es, P, V; seed = 25)
    ref = _dense(ρ2)
    Gweak = _random_unitary_gate(P; seed = 10, τ = 1.0e-6)
    apply!(ρ2, msgs2, LocalGate((1, 2), Gweak); action = SandwichAction, trunc = truncrank(16))
    @test _dense(ρ2) / norm(_dense(ρ2)) ≈ ref / norm(ref) rtol = 1e-4
end

# --- composition with aggregates and Trotter --------------------------------

@testset "The action threads through CompositeGate and Circuit" begin
    P = ComplexSpace(2)
    es = [UndirectedEdge(1, 2), UndirectedEdge(3, 4)]
    ρ, msgs = _fresh(es, P)
    G = _random_unitary_gate(P; seed = 11)
    layer = CompositeGate([LocalGate((1, 2), G), LocalGate((3, 4), G)])
    _, _, info = apply!(ρ, msgs, layer; action = SandwichAction, trunc = notrunc(), normp = 0)
    @test check_consistency(ρ)
    @test isfinite(info.logλ)

    # the keyword reaches every gate in an aggregate, so a Trotter circuit is reusable verbatim.
    # `trotterize` takes an `AbstractDict`, which `Dictionaries.Dictionary` is not.
    bond_hams = Dict(e => _random_hermitian(P ⊗ P; seed = 12) for e in es)
    circuit = trotterize(bond_hams, 0.01, Strang())
    ρ2, msgs2 = _fresh(es, P)
    apply!(ρ2, msgs2, circuit; action = SandwichAction, trunc = notrunc(), normp = 0)
    @test check_consistency(ρ2)
    @test isvectorized(ρ2)

    # ... and a one-sided action on the same circuit halves the physical legs it touches
    ρ3, msgs3 = _fresh(es, P)
    apply!(ρ3, msgs3, circuit; action = LeftAction, trunc = notrunc(), normp = 0)
    @test check_consistency(ρ3)
    @test isvectorized(ρ3)                          # G is square, so Gρ stays square
end

# --- imaginary-time evolution -------------------------------------------------

@testset "Imaginary-time evolution of 𝟙 reproduces exp(-βH)" begin
    # The physical end-to-end test, and the one that pins down the β bookkeeping: a
    # `SandwichAction` applies exp(-dτ H) on *both* sides, so a step of size dτ advances β by 2dτ.
    P = ComplexSpace(2)
    n = 3
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    X = TensorMap(ComplexF64[0 1; 1 0], P, P)
    Z = TensorMap(ComplexF64[1 0; 0 -1], P, P)
    # transverse-field Ising bond term, on-site field split over the bond's endpoints
    deg(v) = v == 2 ? 2 : 1
    h(u, v) = -(Z ⊗ Z) - (1 / deg(u)) * (X ⊗ id(P)) - (1 / deg(v)) * (id(P) ⊗ X)
    bond_hams = Dict(e => h(first(e), last(e)) for e in es)   # trotterize wants an AbstractDict

    H = sum(_embed(bond_hams[e], n, (first(e), last(e)), fill(P, n)) for e in es)

    dτ = 0.005
    nsteps = 20
    β = 2 * dτ * nsteps                    # two-sided: each step advances β by 2dτ
    # explicit coloring: `Dict` key order is hash-seed-dependent, and so would the Trotter
    # layer order be
    circuit = trotterize(bond_hams, dτ, Strang([[es[1]], [es[2]]]))
    ρ, msgs = _fresh(es, P)
    for _ in 1:nsteps
        apply!(ρ, msgs, circuit; action = SandwichAction, trunc = truncrank(16), normp = 0)
    end

    exact = exp(-β * H)
    got = _dense(ρ)
    @test got / tr(got) ≈ exact / tr(exact) rtol = 1e-4
end
