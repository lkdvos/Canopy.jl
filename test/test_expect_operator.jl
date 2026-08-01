using Canopy
using Canopy: TensorNetworkOperator, TensorNetworkState, BPMessages, UndirectedEdge,
              LocalGate, LeftAction, RightAction, SandwichAction,
              identity_operator, randn_operator, randn_state, belief_propagation,
              reduced_density_matrix, expect, physicalspace
using TensorKit
using TensorKitTensors.FermionOperators: fermion_space, f_num, f_hopping
using Graphs: path_graph, star_graph, edges, src, dst
using Random
using Test

# --- helpers ------------------------------------------------------------------

# `TensorMap(op)` orders its legs so that this is the operator `⊗P ← ⊗P`, with vertex `i` at
# slot `i` on both sides (`i` indexing `vertices(op)`).
_dense(op) = repartition(TensorMap(op), length(op), length(op))

# Embed a gate acting on `ps` (indices into `vertices`) into the full `⊗P ← ⊗P` operator, by
# padding with identities — same leg-labelling convention as `TensorMap(op)`.
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

# The reference the BP result must reproduce: the network read as a purification `X`, its
# physical density matrix `X X† / tr(X X†)`, and expectation values against it. Only *full*
# traces and products appear, never a partial trace, so the reference carries no convention
# of its own — which is what makes it a valid oracle for the fermionic rows.
_reference(op) = (X = _dense(op); R = X * X'; R / tr(R))
_expect_ref(R, O, n, ps, pspaces) = tr(_embed(O, n, ps, pspaces) * R)

_hermitian(P; seed = 0) = (Random.seed!(seed); h = randn(ComplexF64, P, P); (h + h') / 2)
_hermitian2(P; seed = 0) = (Random.seed!(seed); h = randn(ComplexF64, P ⊗ P, P ⊗ P); (h + h') / 2)

# `isposdef` begins with an *exact* hermiticity test, which a contraction result satisfies only
# to ~1e-16 (equally true of the state path). So check hermiticity approximately first, then
# project onto the Hermitian part — `(ρ + ρ')/2` is Hermitian bit-for-bit — and let `isposdef`
# judge the projection. Without the ancilla twist the operator density matrix has genuinely
# negative eigenvalues, so this is a real gate, not a formality.
_ispositive(ρ) = ρ ≈ ρ' && isposdef((ρ + ρ') / 2)

# Trees only: belief propagation is *exact* there, so the Bethe density matrix must reproduce
# the dense reference to machine precision rather than approximately.
const _TREES = [("chain-3", path_graph(3)), ("star-5", star_graph(5))]
const _SPACES = [
    ("bosonic", ComplexSpace(2), ComplexSpace(2)),
    ("U1", Vect[U1Irrep](0 => 1, 1 => 1), Vect[U1Irrep](-1 => 1, 0 => 1, 1 => 1)),
    ("fermionic", fermion_space(Trivial), Vect[fℤ₂](0 => 1, 1 => 1)),
    ("fermionic-U1", fermion_space(U1Irrep),
     Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1, (1, -1) => 1)),
]

function _converged(op; maxiter = 500, tol = 1.0e-13)
    return belief_propagation(BPMessages(op), op; maxiter, tol)
end

# --- the density matrix itself ------------------------------------------------

@testset "Operator density matrix is a state — $sname / $gname" for
        (sname, P, V) in _SPACES, (gname, g) in _TREES
    Random.seed!(hash((sname, gname)))
    op = randn_operator(ComplexF64, g, P, V)
    msgs = _converged(op)
    verts = collect(vertices(op))

    for v in verts
        ρ = reduced_density_matrix((v,), op, msgs)
        @test space(ρ) == (physicalspace(op, v, 1) ← physicalspace(op, v, 1))
        @test tr(ρ) ≈ 1
        @test ρ ≈ ρ'
        @test _ispositive(ρ)
    end
    for e in edges(op)
        ρ = reduced_density_matrix((first(e), last(e)), op, msgs)
        @test tr(ρ) ≈ 1
        @test ρ ≈ ρ'
        @test _ispositive(ρ)
    end
end

@testset "Operator density matrix against dense X X† — $sname / $gname" for
        (sname, P, V) in _SPACES, (gname, g) in _TREES
    Random.seed!(hash((sname, gname)))
    op = randn_operator(ComplexF64, g, P, V)
    msgs = _converged(op)
    verts = collect(vertices(op))
    n = length(verts)
    pspaces = fill(P, n)
    R = _reference(op)

    h1 = _hermitian(P; seed = 1)
    h2 = _hermitian2(P; seed = 2)

    for (i, v) in enumerate(verts)
        got = expect(op, msgs, h1, v)
        ref = _expect_ref(R, h1, n, (i,), pspaces)
        @test got ≈ ref rtol = 1.0e-10
        # the wrapper must agree with the explicit trace, under whatever twists the RDM carries
        @test got ≈ tr(h1 * reduced_density_matrix((v,), op, msgs))
        @test expect(op, msgs, h1, (v,)) ≈ got
    end

    for e in edges(op)
        u, w = first(e), last(e)
        iu, iw = findfirst(==(u), verts), findfirst(==(w), verts)
        got = expect(op, msgs, h2, (u, w))
        ref = _expect_ref(R, h2, n, (iu, iw), pspaces)
        @test got ≈ ref rtol = 1.0e-10
        @test expect(op, msgs, h2, e) ≈ got          # UndirectedEdge form
    end
end

@testset "Single-site and two-site marginals agree — $sname / $gname" for
        (sname, P, V) in _SPACES, (gname, g) in _TREES
    # Exact on a tree, and independent of any dense reference: a sign or twist error in either
    # construction breaks the agreement. Mirrors the state-path test in `test_expect.jl`.
    Random.seed!(hash((sname, gname, :marginal)))
    op = randn_operator(ComplexF64, g, P, V)
    msgs = _converged(op)
    h = _hermitian(P; seed = 3)
    ρ1 = Dict(v => reduced_density_matrix((v,), op, msgs) for v in vertices(op))
    for e in edges(op)
        u, w = first(e), last(e)
        ρ2 = reduced_density_matrix((u, w), op, msgs)
        @test tr((h ⊗ id(P)) * ρ2) ≈ tr(h * ρ1[u])
        @test tr((id(P) ⊗ h) * ρ2) ≈ tr(h * ρ1[w])
    end
end

# --- anchors against the (independently validated) state path -----------------

@testset "A state lifted to an operator measures like the state — $sname" for
        (sname, P, V) in _SPACES
    # `TensorNetworkOperator(ψ)` is `|ψ⟩` with a one-dimensional ancilla, so `X X† = |ψ⟩⟨ψ|`
    # and every expectation value must equal the state's own. This ties the new operator
    # contraction to the fermion-validated state contraction with no dense reference at all.
    Random.seed!(hash((sname, :lift)))
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    ψ = randn_state(ComplexF64, es, P, V)
    X = TensorNetworkOperator(ψ)
    msgs_ψ = belief_propagation(BPMessages(ψ), ψ; maxiter = 500, tol = 1.0e-13)
    msgs_X = _converged(X)

    h1 = _hermitian(P; seed = 4)
    h2 = _hermitian2(P; seed = 5)
    for v in 1:3
        @test expect(X, msgs_X, h1, v) ≈ expect(ψ, msgs_ψ, h1, v) rtol = 1.0e-10
    end
    for e in es
        @test expect(X, msgs_X, h2, e) ≈ expect(ψ, msgs_ψ, h2, e) rtol = 1.0e-10
    end
end

@testset "β = 0: the identity operator is the infinite-temperature state — $sname" for
        (sname, P, _) in _SPACES
    # `identity_operator` has one-dimensional bonds, so BP is exact and `X X† = 𝟙`: every
    # single-site marginal must be exactly `id(P)/dim(P)`.
    es = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]
    ρ = identity_operator(ComplexF64, es, P)
    msgs = BPMessages(ρ)
    for v in 1:3
        @test reduced_density_matrix((v,), ρ, msgs) ≈ id(P) / dim(P) rtol = 1.0e-12
    end
    ρ2 = reduced_density_matrix((1, 2), ρ, msgs)
    @test ρ2 ≈ id(P ⊗ P) / dim(P ⊗ P) rtol = 1.0e-12
end

# --- fermionic observables with physical meaning -------------------------------

@testset "Fermionic occupations and hopping on a purification — $sname" for
        (sname, P) in (("fℤ₂", fermion_space(Trivial)), ("fℤ₂⊠U1", fermion_space(U1Irrep)))
    # The observables the finite-temperature tests actually use, checked against dense here so
    # that a failure there is unambiguously about the evolution and not about the measurement.
    V = sname == "fℤ₂" ? Vect[fℤ₂](0 => 1, 1 => 1) :
        Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1, (1, -1) => 1)
    symm = sname == "fℤ₂" ? Trivial : U1Irrep
    Random.seed!(hash((sname, :fermionic)))
    g = path_graph(4)
    op = randn_operator(ComplexF64, g, P, V)
    msgs = _converged(op)
    n = 4
    pspaces = fill(P, n)
    R = _reference(op)

    nop = f_num(ComplexF64, symm)
    hop = f_hopping(ComplexF64, symm)

    for v in 1:n
        got = expect(op, msgs, nop, v)
        @test imag(got) ≈ 0 atol = 1.0e-12
        @test 0 ≤ real(got) ≤ 1
        @test got ≈ _expect_ref(R, nop, n, (v,), pspaces) rtol = 1.0e-10
    end
    for v in 1:(n - 1)
        got = expect(op, msgs, hop, (v, v + 1))
        @test imag(got) ≈ 0 atol = 1.0e-12
        @test got ≈ _expect_ref(R, hop, n, (v, v + 1), pspaces) rtol = 1.0e-10
    end
end
