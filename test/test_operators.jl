using Canopy
using Canopy: TensorNetworkState, TensorNetworkOperator, BPMessages, UndirectedEdge, DirectedEdge,
              identity_operator, randn_operator, rand_operator, isvectorized, check_consistency,
              physicalspace, physicalspaces, virtualspace, num_physical, neighbors, degree,
              leg_index, belief_propagation, _fuse_physical
using TensorKit
using TensorKitTensors.FermionOperators: fermion_space
using Graphs: path_graph, cycle_graph, star_graph, grid, edges
using Dictionaries
using MatrixAlgebraKit: eigh_vals
using Random
using Test

# The three sectortypes the operator path is expected to work on, plus `fℤ₂` — gate
# application refuses fermions for now, but the *data layout* the fused view depends on is
# sign-agnostic, so it is checked here too.
const _OP_SPACES = [
    ("trivial", ComplexSpace(2), ComplexSpace(3)),
    ("U1", Vect[U1Irrep](0 => 1, 1 => 2), Vect[U1Irrep](-1 => 1, 0 => 2, 1 => 1)),
    ("SU2", Vect[SU2Irrep](0 => 1, 1 // 2 => 1), Vect[SU2Irrep](0 => 2, 1 // 2 => 1)),
    ("fermionic", fermion_space(Trivial), Vect[fℤ₂](0 => 2, 1 => 2)),
]

@testset "Operator layout and duality invariants — $name" for (name, P, V) in _OP_SPACES
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3), UndirectedEdge(1, 3)]
    Random.seed!(4)
    op = randn_operator(ComplexF64, es, P, V)

    @test num_physical(op) == 2
    @test check_consistency(op)
    @test isvectorized(op)
    for v in vertices(op)
        t = op[v]
        @test numout(t) == 2
        @test physicalspace(op, v) == P            # default is the ket space
        @test physicalspace(op, v, 1) == P
        @test physicalspace(op, v, 2) == dual(P)
        @test physicalspaces(op, v) == (P, dual(P))
        # virtual leg k lives at tensor slot k + 2, not k + 1
        for (k, n) in enumerate(neighbors(op, v))
            @test virtualspace(op, DirectedEdge(v, n)) == space(t, k + 2)
            @test leg_index(op, DirectedEdge(v, n)) == k
        end
    end
    # the bond-duality invariant, oriented
    for e in edges(op)
        d = DirectedEdge(e)
        @test isdual(virtualspace(op, d))
        @test virtualspace(op, d) == dual(virtualspace(op, reverse(d)))
    end
end

@testset "The fused state view is a genuine zero-copy reinterpretation — $name" for
        (name, P, V) in _OP_SPACES
    # This is the one place Canopy leans on TensorKit's internal block layout: the fusion
    # trees of `P₁ ⊗ P₂ → c` must enumerate the degeneracy basis of `fuse(P₁, P₂)` in sector
    # `c`, in the same order, so that reinterpreting the data vector *is* multiplying by the
    # fusion isomorphism. If TensorKit ever changes that, this test must be the thing that
    # fails — not belief propagation, silently.
    Random.seed!(5)
    for P₂ in (dual(P), P)                        # vectorized operator, and a purification
        t = randn(ComplexF64, P ⊗ P₂, V ⊗ dual(V))
        W = unitary(fuse(codomain(t)) ← codomain(t))
        f = _fuse_physical(t)

        @test space(f) == (fuse(codomain(t)) ← domain(t))
        @test f ≈ W * t
        @test dim(space(f)) == dim(space(t))
        @test f.data === t.data                   # shares storage, copies nothing
        @test norm(f) ≈ norm(t)
    end

    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    op = randn_operator(ComplexF64, es, P, V)
    view = TensorNetworkState(op)
    @test check_consistency(view)
    @test num_physical(view) == 1
    for v in vertices(op)
        @test physicalspace(view, v) == fuse(P ⊗ dual(P))
        @test view[v].data === op[v].data
        # the view shares virtual legs verbatim, which is why messages are interchangeable
        for n in neighbors(op, v)
            @test virtualspace(view, DirectedEdge(v, n)) == virtualspace(op, DirectedEdge(v, n))
        end
    end
end

@testset "identity_operator is the β = 0 anchor — $name" for (name, P, _) in _OP_SPACES
    for (gname, topology, n) in (
            ("chain", [UndirectedEdge(1, 2), UndirectedEdge(2, 3)], 3),
            ("cycle", cycle_graph(4), 4),
            ("star", star_graph(4), 4),
        )
        ρ = identity_operator(ComplexF64, topology, P)
        @test check_consistency(ρ)
        @test isvectorized(ρ)
        @test length(ρ) == n
        # every bond is one-dimensional
        for e in edges(ρ)
            @test dim(virtualspace(ρ, DirectedEdge(e))) == 1
        end
        dense = repartition(TensorMap(ρ), n, n)
        @test dense ≈ id(P^n)
        @test tr(dense) ≈ dim(P)^n
    end
end

@testset "identity_operator accepts per-vertex spaces" begin
    P, Q = ComplexSpace(2), ComplexSpace(3)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    ρ = identity_operator(ComplexF64, es, Dictionary([1, 2, 3], [P, Q, P]))
    @test check_consistency(ρ)
    @test physicalspaces(ρ, 2) == (Q, dual(Q))
    @test repartition(TensorMap(ρ), 3, 3) ≈ id(P ⊗ Q ⊗ P)
end

@testset "TensorMap(op) round-trips through the vectorization bend" begin
    P, V = ComplexSpace(2), ComplexSpace(2)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    Random.seed!(7)
    op = randn_operator(ComplexF64, es, P, V)
    dense = repartition(TensorMap(op), 3, 3)
    @test space(dense) == (P^3 ← P^3)
    # the same contraction read as a *state* on the fused physical space must agree
    fused = TensorMap(TensorNetworkState(op))
    @test norm(fused) ≈ norm(dense)
end

@testset "randn_operator / rand_operator defaults" begin
    P, V = ComplexSpace(2), ComplexSpace(3)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    for f in (randn_operator, rand_operator)
        @test scalartype(f(es, P, V)) === Float64
        @test scalartype(f(ComplexF64, es, P, V)) === ComplexF64
        @test scalartype(f(MersenneTwister(1234), ComplexF64, es, P, V)) === ComplexF64
        @test check_consistency(f(ComplexF64, path_graph(4), P, V))
        pspaces, vspaces = Dictionary([1, 2, 3], fill(P, 3)), Dictionary(es, fill(V, 2))
        @test check_consistency(f(ComplexF64, pspaces, vspaces))
    end
end

@testset "Lifting a state gives a purification with a trivial ancilla" begin
    P, V = ComplexSpace(2), ComplexSpace(3)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3), UndirectedEdge(1, 3)]
    Random.seed!(11)
    ψ = randn_state(ComplexF64, es, P, V)
    op = TensorNetworkOperator(ψ)

    @test check_consistency(op)
    @test num_physical(op) == 2
    @test !isvectorized(op)                       # ancilla, not a square operator
    @test physicalspace(op, 1, 1) == P
    @test dim(physicalspace(op, 1, 2)) == 1

    # A trivial second leg fuses to nothing, so the lift's BP messages must be *equal* to the
    # state's own. This isolates the fused view from the rest of the machinery.
    msgs_ψ = belief_propagation(BPMessages(ψ), ψ; maxiter = 100, tol = 1e-12)
    msgs_op = belief_propagation(BPMessages(op), op; maxiter = 100, tol = 1e-12)
    for e in keys(msgs_ψ.messages)
        @test msgs_op[e] ≈ msgs_ψ[e]
    end
end

@testset "Belief propagation runs on an operator — $name" for (name, P, V) in _OP_SPACES
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3), UndirectedEdge(3, 4)]
    Random.seed!(13)
    op = randn_operator(ComplexF64, es, P, V)
    msgs = BPMessages(op)
    @test check_consistency(op, msgs)

    msgs = belief_propagation(msgs, op; maxiter = 200, tol = 1e-12)
    @test check_consistency(op, msgs)

    view = TensorNetworkState(op)
    n = length(op)
    verts = collect(vertices(op))
    vidx = 2                                      # interior site: degree 2, so the RDM is full rank
    ρ = reduced_density_matrix((verts[vidx],), view, msgs)
    @test tr(ρ) ≈ 1
    @test ρ ≈ ρ'
    # `isposdef` is not usable here: the contraction leaves `ρ` Hermitian only to ~1e-16, and
    # `isposdef` demands exact hermiticity (this is equally true of the state path). Check the
    # physically meaningful statement instead.
    @test all(>(0), eigh_vals((ρ + ρ') / 2))
    # the fused single-site RDM lives on the doubled physical space
    @test space(ρ) == (fuse(P ⊗ dual(P)) ← fuse(P ⊗ dual(P)))

    # The geometry is a path, so BP is *exact*: the message-based RDM must equal the one from
    # the fully contracted wavefunction. This is the end-to-end check that the fused view
    # feeds BP the right tensors — BP on a two-leg network closes both physical legs between
    # ket and bra, and here that is compared against an independent dense contraction.
    # Restricted to bosonic braiding, where `reduced_density_matrix`'s ket-leg twist is the
    # identity and the naive dense contraction below matches its convention.
    if BraidingStyle(sectortype(P)) isa Bosonic
        ψ = TensorMap(view)
        lk = [i == vidx ? -1 : i for i in 1:n]
        lb = [i == vidx ? -2 : i for i in 1:n]
        ρ_exact = repartition(ncon([ψ, ψ], [lk, lb], [false, true]), 1, 1)
        ρ_exact = ρ_exact / tr(ρ_exact)
        @test ρ ≈ ρ_exact
    end
end

@testset "Operator constructor error paths" begin
    P, V = ComplexSpace(2), ComplexSpace(3)
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    # dual virtual spaces are rejected, as for states
    @test_throws Exception TensorNetworkOperator{ComplexF64}(
        undef, Dictionary([1, 2, 3], fill(P, 3)), Dictionary(es, fill(dual(V), 2))
    )
    # mismatched vertex set
    @test_throws ArgumentError TensorNetworkOperator{ComplexF64}(
        undef, Dictionary([1, 2], fill(P, 2)), Dictionary(es, fill(V, 2))
    )
end
