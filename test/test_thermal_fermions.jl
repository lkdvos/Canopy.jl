using Canopy
using Canopy: UndirectedEdge, identity_operator, BPMessages, belief_propagation, apply!,
              expect, reduced_density_matrix, LeftAction, SandwichAction,
              Strang, trotterize, edge_coloring, virtualspace, DirectedEdge
using TensorKit
using TensorKitTensors.FermionOperators: fermion_space, f_num, f_hopping
using MatrixAlgebraKit: truncrank, trunctol
using LinearAlgebra: Diagonal
using Random
using Test

# Finite-temperature free fermions on an **open chain**, against the exact single-particle
# result at the same finite `L`.
#
# The chain is a tree, so belief propagation is exact and the only approximations left are the
# Trotter step and the bond truncation. `L = 6` is chosen so that the exact purification fits
# in bond dimension `4^3 = 64` — with `truncrank(64)` there is *no* truncation error at all,
# and the comparison is purely a statement about the Trotter step and about the fermionic
# signs of the operator path.

const L = 6
const T_HOP = 1.0

# `isposdef` demands *exact* hermiticity, which a contraction result never has: check it
# approximately first, then project onto the Hermitian part and let `isposdef` judge that.
_ispositive(ρ) = ρ ≈ ρ' && isposdef((ρ + ρ') / 2)

# --- exact reference ----------------------------------------------------------

# Single-particle eigenmodes of the open tight-binding chain:
#   ε_k = -2t cos(πk/(L+1)) - μ,   φ_k(j) = √(2/(L+1)) sin(πkj/(L+1)),   k = 1 … L
# In the grand-canonical ensemble ρ = exp(-βH)/Z (μ already inside H) Wick's theorem gives
#   ⟨c†_i c_j⟩ = Σ_k φ_k(i) φ_k(j) f(ε_k),   f(ε) = 1/(exp(βε) + 1).
function exact_correlations(L::Int, β::Real; t::Real = 1.0, μ::Real = 0.0)
    ε = [-2t * cos(π * k / (L + 1)) - μ for k in 1:L]
    φ = [sqrt(2 / (L + 1)) * sin(π * k * j / (L + 1)) for j in 1:L, k in 1:L]
    f = [1 / (exp(β * εk) + 1) for εk in ε]
    return φ * Diagonal(f) * transpose(φ), ε, f
end

# --- model --------------------------------------------------------------------

_chain_edges(L) = [UndirectedEdge(j, j + 1) for j in 1:(L - 1)]
_degree(j, L) = (j == 1 || j == L) ? 1 : 2

# h_e = -t·hop - (μ/deg u)·(n⊗I) - (μ/deg v)·(I⊗n), so that Σ_e h_e = H exactly.
function bond_hamiltonians(symm, L; t::Real = 1.0, μ::Real = 0.0)
    P = fermion_space(symm)
    hop = f_hopping(ComplexF64, symm)
    nop = f_num(ComplexF64, symm)
    I1 = id(P)
    return Dict(
        e => -t * hop - (μ / _degree(first(e), L)) * (nop ⊗ I1) -
            (μ / _degree(last(e), L)) * (I1 ⊗ nop)
            for e in _chain_edges(L)
    )
end

# Evolve `identity_operator` into a thermal network. `action = LeftAction` builds
# `X = exp(-nsteps·dτ·H)`, whose measured ensemble (`X X†`) is at `β = 2·nsteps·dτ`;
# `action = SandwichAction` builds `exp(-2·nsteps·dτ·H)`, measured at `β = 4·nsteps·dτ`.
function thermal_network(symm, L, dτ, nsteps; action = LeftAction, χ = 64, t = 1.0, μ = 0.0)
    P = fermion_space(symm)
    es = _chain_edges(L)
    hams = bond_hamiltonians(symm, L; t, μ)
    circuit = trotterize(hams, dτ, Strang(edge_coloring(es)))
    trunc = truncrank(χ) & trunctol(; atol = 1.0e-13)

    ρ = identity_operator(ComplexF64, es, P)
    msgs = BPMessages(ρ)
    ϵmax = 0.0
    for i in 1:nsteps
        _, _, info = apply!(ρ, msgs, circuit; action, trunc, normp = 2)
        ϵmax = max(ϵmax, info.ϵ)
        iszero(mod(i, 10)) && (msgs = belief_propagation(msgs, ρ; maxiter = 200, tol = 1.0e-12))
    end
    msgs = belief_propagation(msgs, ρ; maxiter = 500, tol = 1.0e-13)
    return ρ, msgs, ϵmax
end

# --- β = 0 ---------------------------------------------------------------------

@testset "β = 0 is the infinite-temperature ensemble — $sname" for
        (sname, symm) in (("fℤ₂", Trivial), ("fℤ₂⊠U1", U1Irrep))
    P = fermion_space(symm)
    es = _chain_edges(L)
    ρ = identity_operator(ComplexF64, es, P)
    msgs = BPMessages(ρ)
    nop = f_num(ComplexF64, symm)
    hop = f_hopping(ComplexF64, symm)
    C, _, _ = exact_correlations(L, 0.0)
    for j in 1:L
        @test real(expect(ρ, msgs, nop, j)) ≈ C[j, j] atol = 1.0e-12       # = 1/2
    end
    for j in 1:(L - 1)
        @test real(expect(ρ, msgs, hop, (j, j + 1))) ≈ 2C[j, j + 1] atol = 1.0e-12   # = 0
    end
end

# --- the analytic finite-temperature comparison --------------------------------

@testset "Free fermions at β = $β against exact Fermi–Dirac — $sname" for
        (sname, symm) in (("fℤ₂", Trivial), ("fℤ₂⊠U1", U1Irrep)), β in (1.0, 2.0)
    μ = 0.3
    dτ = 0.01
    nsteps = round(Int, (β / 2) / dτ)           # one-sided: β = 2·nsteps·dτ
    @assert 2 * nsteps * dτ ≈ β

    ρ, msgs, ϵmax = thermal_network(symm, L, dτ, nsteps; action = LeftAction, t = T_HOP, μ)
    # χ = 64 is exact for L = 6, so nothing may be truncated away
    @test ϵmax < 1.0e-8

    C, ε, f = exact_correlations(L, β; t = T_HOP, μ)
    nop = f_num(ComplexF64, symm)
    hop = f_hopping(ComplexF64, symm)

    # With χ = 64 exact and BP exact on a tree, the remaining error is the second-order Trotter step:
    # O(dτ²·τ) ≈ 1e-4 here. The tolerances are set an order of magnitude above that; anything
    # a wrong fermionic sign would do is O(1).
    #
    # Site-resolved occupations — an open chain has a nontrivial profile, so this is a real
    # check and not just a check of the average.
    for j in 1:L
        @test real(expect(ρ, msgs, nop, j)) ≈ C[j, j] atol = 1.0e-3
    end
    # Nearest-neighbour hopping ⟨c†_j c_{j+1} + c†_{j+1} c_j⟩ = 2 C[j, j+1]
    for j in 1:(L - 1)
        @test real(expect(ρ, msgs, hop, (j, j + 1))) ≈ 2C[j, j + 1] atol = 1.0e-3
    end
    # Total energy Σ_e ⟨h_e⟩ = Σ_k ε_k f(ε_k)
    hams = bond_hamiltonians(symm, L; t = T_HOP, μ)
    E = sum(real(expect(ρ, msgs, h, (first(e), last(e)))) for (e, h) in hams)
    @test E ≈ sum(ε .* f) atol = 5.0e-3
end

# --- convention checks that need no analytics ----------------------------------

@testset "One-sided at dτ ≡ two-sided at dτ/2 (the factor of two)" begin
    # A `SandwichAction` step applies the gate on both sides, so it advances the measured β
    # twice as fast as a `LeftAction` step of the same size: one-sided at `dτ` and two-sided at
    # `dτ/2` reach the same ensemble in the same number of steps.
    #
    # They are two *different* Strang discretizations of it, though, so they agree only to the
    # Trotter order — O(dτ²·τ) ≈ 2e-4 here, and observed ~1e-5. `atol` is set to match that, not
    # to machine precision. Getting the factor of two wrong would move these by O(1).
    dτ, nsteps = 0.02, 25
    ρ1, m1, _ = thermal_network(Trivial, L, dτ, nsteps; action = LeftAction, μ = 0.3)
    ρ2, m2, _ = thermal_network(Trivial, L, dτ / 2, nsteps; action = SandwichAction, μ = 0.3)
    nop = f_num(ComplexF64, Trivial)
    hop = f_hopping(ComplexF64, Trivial)
    for j in 1:L
        @test real(expect(ρ1, m1, nop, j)) ≈ real(expect(ρ2, m2, nop, j)) atol = 1.0e-4
    end
    for j in 1:(L - 1)
        @test real(expect(ρ1, m1, hop, (j, j + 1))) ≈
            real(expect(ρ2, m2, hop, (j, j + 1))) atol = 1.0e-4
    end
end

@testset "Charge-resolved fermions agree with plain fℤ₂" begin
    # `fermion_space(U1Irrep)` resolves particle number on top of the parity grading. The
    # ensemble is the same, so every observable must match — a symmetry-dependent sign error
    # in the operator path would show up here.
    dτ, nsteps = 0.01, 50                      # β = 1
    ρa, ma, _ = thermal_network(Trivial, L, dτ, nsteps; μ = 0.3)
    ρb, mb, _ = thermal_network(U1Irrep, L, dτ, nsteps; μ = 0.3)
    for (symm, ρ, m) in ((Trivial, ρa, ma), (U1Irrep, ρb, mb))
        @test all(v -> real(expect(ρ, m, f_num(ComplexF64, symm), v)) ≥ 0, 1:L)
    end
    for j in 1:L
        @test real(expect(ρa, ma, f_num(ComplexF64, Trivial), j)) ≈
            real(expect(ρb, mb, f_num(ComplexF64, U1Irrep), j)) atol = 1.0e-8
    end
    for j in 1:(L - 1)
        @test real(expect(ρa, ma, f_hopping(ComplexF64, Trivial), (j, j + 1))) ≈
            real(expect(ρb, mb, f_hopping(ComplexF64, U1Irrep), (j, j + 1))) atol = 1.0e-8
    end
end

@testset "The thermal density matrix is a state" begin
    dτ, nsteps = 0.01, 50
    ρ, msgs, _ = thermal_network(Trivial, L, dτ, nsteps; μ = 0.3)
    for j in 1:L
        r = reduced_density_matrix((j,), ρ, msgs)
        @test tr(r) ≈ 1
        @test r ≈ r'
        @test _ispositive(r)
    end
    for j in 1:(L - 1)
        r = reduced_density_matrix((j, j + 1), ρ, msgs)
        @test tr(r) ≈ 1
        @test r ≈ r'
        @test _ispositive(r)
    end
end
