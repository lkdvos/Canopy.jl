using Canopy
using Canopy: UndirectedEdge, DirectedEdge, belief_propagation, reduced_density_matrix, expect
using TensorKit
using LinearAlgebra: isposdef
using TensorKitTensors.SpinOperators: σᶻ, S_z_S_z, S_exchange
using TensorKitTensors.FermionOperators: f_num, f_hopping, fermion_space
using Graphs
using Test
using Dictionaries
using Random: Random, randn!


# Exact reduced density matrix of the dense wavefunction `psi` on `sites`,
# matching the leg convention of `reduced_density_matrix(sites, ...)`:
# codomain = ⊗_k P_{sites[k]}, domain = ⊗_k P_{sites[k]}. `sites` are
# positions in the codomain of `psi` (== iteration order of `vertices(state)`).
function exact_rdm(psi::AbstractTensorMap, sites::NTuple{N, Int}) where {N}
    L = numout(psi)
    @assert numin(psi) == 0
    ket = Vector{Int}(undef, L)
    bra = Vector{Int}(undef, L)
    next_contract = 0
    for v in 1:L
        idx = findfirst(==(v), sites)
        if idx === nothing
            next_contract += 1
            ket[v] = next_contract
            bra[v] = next_contract
        else
            # ncon orders output legs ascending |label|; the first N go to
            # codomain (ket of sites[1..N]) and the next N to domain. TensorKit
            # stores domain legs in *reverse* physical order, so to match the
            # leg convention of `reduced_density_matrix((sites...,), ...)`
            # we put bra of sites[k] at label -(2N+1-k).
            ket[v] = -idx
            bra[v] = -(2N + 1 - idx)
        end
    end
    ρ = ncon([psi, psi'], [ket, bra])
    ρ = twist!(repartition(ρ, N, N), ntuple(identity, N))
    @assert ishermitian(ρ; rtol = eps(real(scalartype(ρ)))^(3 / 4))
    return scale!(project_hermitian!(ρ), inv(tr(ρ)))
end

function chain_state(::Type{T}, L::Int, P, V) where {T}
    pspaces = Dictionary(collect(1:L), fill(P, L))
    edges = [UndirectedEdge(i, i + 1) for i in 1:(L - 1)]
    vspaces = Dictionary(edges, fill(V, L - 1))
    return randn!(TensorNetworkState{T}(undef, pspaces, vspaces))
end

function ring_state(::Type{T}, L::Int, P, V) where {T}
    pspaces = Dictionary(collect(1:L), fill(P, L))
    edges = vcat(
        [UndirectedEdge(i, i + 1) for i in 1:(L - 1)],
        [UndirectedEdge(1, L)],
    )
    vspaces = Dictionary(edges, fill(V, L))
    return randn!(TensorNetworkState{T}(undef, pspaces, vspaces))
end

# `M`×`N` torus (periodic 2D grid). Vertices are labeled in row-major order:
# `v(r, c) = (r - 1) * N + c`. Each vertex has 4 neighbors.
function torus_state(::Type{T}, M::Int, N::Int, P, V) where {T}
    L = M * N
    pspaces = Dictionary(collect(1:L), fill(P, L))
    edges = UndirectedEdge{Int}[]
    for r in 1:M, c in 1:N
        v = (r - 1) * N + c
        cnext = c == N ? 1 : c + 1
        rnext = r == M ? 1 : r + 1
        push!(edges, UndirectedEdge(v, (r - 1) * N + cnext))
        push!(edges, UndirectedEdge(v, (rnext - 1) * N + c))
    end
    vspaces = Dictionary(edges, fill(V, length(edges)))
    return randn!(TensorNetworkState{T}(undef, pspaces, vspaces))
end

function check_rdm(state, msgs, psi, sites; rtol, atol = rtol)
    ρ_bp = reduced_density_matrix(sites, state, msgs)
    @test ishermitian(ρ_bp; rtol = eps(real(scalartype(ρ_bp)))^(3 / 4))
    @test isposdef!(project_hermitian(ρ_bp))

    ρ_ex = exact_rdm(psi, sites)
    @test isapprox(ρ_bp, ρ_ex; rtol, atol)
    return ρ_bp, ρ_ex
end

function check_op(ρ_bp, ρ_ex, op; rtol, atol = rtol)
    return @test isapprox(tr(ρ_bp * op), tr(ρ_ex * op); rtol, atol)
end


@testset "Graphs.degree matches neighbors length" begin
    Random.seed!(0)
    state = chain_state(ComplexF64, 4, ComplexSpace(2), ComplexSpace(3))
    for v in 1:4
        @test Graphs.degree(state, v) == length(Canopy.neighbors(state, v))
    end
    state2 = torus_state(ComplexF64, 3, 3, ComplexSpace(2), ComplexSpace(2))
    for v in 1:9
        @test Graphs.degree(state2, v) == 4
    end
end


@testset "Chain — ComplexSpace + spin operators (BP exact on tree)" begin
    Random.seed!(0)
    L = 4
    P = ComplexSpace(2)
    V = ComplexSpace(4)
    state = chain_state(ComplexF64, L, P, V)
    psi = TensorMap(state)

    msgs = BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter = L + 2)

    sz = σᶻ(ComplexF64, Trivial)
    szsz = S_z_S_z(ComplexF64, Trivial)
    ss = S_exchange(ComplexF64, Trivial)

    rtol = 1.0e-10
    for v in 1:L
        ρ_bp, ρ_ex = check_rdm(state, msgs, psi, (v,); rtol)
        check_op(ρ_bp, ρ_ex, sz; rtol)
    end
    for i in 1:(L - 1)
        sites = (i, i + 1)
        ρ_bp, ρ_ex = check_rdm(state, msgs, psi, sites; rtol)
        check_op(ρ_bp, ρ_ex, szsz; rtol)
        check_op(ρ_bp, ρ_ex, ss; rtol)
    end
end


@testset "Chain — fermionic space + fermion operators (BP exact on tree)" begin
    Random.seed!(1)
    L = 4
    P = fermion_space(Trivial)
    V = Vect[fℤ₂](0 => 2, 1 => 2)
    state = chain_state(ComplexF64, L, P, V)
    psi = TensorMap(state)

    msgs = BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter = L + 2)

    n_op = f_num(ComplexF64, Trivial)
    hop = f_hopping(ComplexF64, Trivial)

    rtol = 1.0e-10
    for v in 1:L
        ρ_bp, ρ_ex = check_rdm(state, msgs, psi, (v,); rtol)
        check_op(ρ_bp, ρ_ex, n_op; rtol)
    end
    for i in 1:(L - 1)
        sites = (i, i + 1)
        ρ_bp, ρ_ex = check_rdm(state, msgs, psi, sites; rtol)
        check_op(ρ_bp, ρ_ex, hop; rtol)
    end
end


@testset "Ring — ComplexSpace + spin operators (BP approximate)" begin
    Random.seed!(2)
    L = 10
    P = ComplexSpace(2)
    V = ComplexSpace(2)
    state = ring_state(ComplexF64, L, P, V)
    psi = TensorMap(state)

    msgs = BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter = 300)

    sz = σᶻ(ComplexF64, Trivial)
    szsz = S_z_S_z(ComplexF64, Trivial)
    ss = S_exchange(ComplexF64, Trivial)

    # BP is approximate on a ring; expect a few-% loop-correction error.
    rtol = 1.0e-1
    atol = 1.0e-1
    for v in 1:L
        ρ_bp, ρ_ex = check_rdm(state, msgs, psi, (v,); rtol, atol)
        check_op(ρ_bp, ρ_ex, sz; rtol, atol)
    end
    for sites in ((1, 2), (5, 6), (1, L))  # adjacent pairs including ring closure
        ρ_bp, ρ_ex = check_rdm(state, msgs, psi, sites; rtol, atol)
        check_op(ρ_bp, ρ_ex, szsz; rtol, atol)
        check_op(ρ_bp, ρ_ex, ss; rtol, atol)
    end
end


@testset "Ring — fermionic space + fermion operators (BP approximate)" begin
    Random.seed!(3)
    L = 10
    P = fermion_space(Trivial)
    V = Vect[fℤ₂](0 => 1, 1 => 1)
    state = ring_state(ComplexF64, L, P, V)
    psi = TensorMap(state)

    msgs = BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter = 300)

    n_op = f_num(ComplexF64, Trivial)
    hop = f_hopping(ComplexF64, Trivial)

    rtol = 1.0e-1
    atol = 1.0e-1
    for v in 1:L
        ρ_bp, ρ_ex = check_rdm(state, msgs, psi, (v,); rtol, atol)
        check_op(ρ_bp, ρ_ex, n_op; rtol, atol)
    end
    for sites in ((1, 2), (5, 6), (1, L))
        ρ_bp, ρ_ex = check_rdm(state, msgs, psi, sites; rtol, atol)
        check_op(ρ_bp, ρ_ex, hop; rtol, atol)
    end
end

@testset "Edge conversions round-trip" begin
    u = UndirectedEdge(2, 5)
    d = DirectedEdge(u)
    @test d isa DirectedEdge{Int}
    @test (first(d), last(d)) == (2, 5)
    @test UndirectedEdge(d) == u

    # Reverse-orientation DirectedEdge collapses to the canonical UndirectedEdge
    d_rev = DirectedEdge(5, 2)
    @test UndirectedEdge(d_rev) == u

    # Convert methods agree with constructors
    @test convert(DirectedEdge{Int}, u) == d
    @test convert(UndirectedEdge{Int}, d) == u
end


@testset "Graphs.has_edge / has_vertex" begin
    Random.seed!(0)
    state = chain_state(ComplexF64, 4, ComplexSpace(2), ComplexSpace(3))
    @test Graphs.has_vertex(state, 1)
    @test Graphs.has_vertex(state, 4)
    @test !Graphs.has_vertex(state, 99)

    @test Graphs.has_edge(state, 1, 2)
    @test Graphs.has_edge(state, 2, 1)  # symmetric for undirected
    @test Graphs.has_edge(state, UndirectedEdge(2, 3))
    @test !Graphs.has_edge(state, 1, 4)
    @test !Graphs.has_edge(state, UndirectedEdge(1, 4))
    @test !Graphs.has_edge(state, 99, 100)
end


@testset "incoming_edges iterator" begin
    Random.seed!(0)
    state = chain_state(ComplexF64, 4, ComplexSpace(2), ComplexSpace(3))
    v = 2
    expected = [DirectedEdge(n, v) for n in Canopy.neighbors(state, v)]
    @test collect(Canopy.incoming_edges(state, v)) == expected
    @test collect(Canopy.incoming_edges(state, v; exclude=(1,))) ==
          [DirectedEdge(n, v) for n in Canopy.neighbors(state, v) if n != 1]
    # exclude=() (default) should yield everything
    @test collect(Canopy.incoming_edges(state, v; exclude=())) == expected
end


@testset "Graph-based TensorNetworkState constructor" begin
    Random.seed!(0)
    g = path_graph(4)
    P = ComplexSpace(2)
    V = ComplexSpace(3)
    state = TensorNetworkState{ComplexF64}(undef, g, P, V)
    @test collect(vertices(state)) == 1:4
    @test length(edges(state)) == 3
    @test physicalspace(state, 1) == P
    # Edge (1,2): non-dual side is the smaller vertex, so the leg as seen from
    # vertex 2 (the larger) is V itself.
    @test virtualspace(state, DirectedEdge(2, 1)) == V

    # Default T = Float64
    state_f = TensorNetworkState(undef, g, P, V)
    @test scalartype(state_f) == Float64

    # Cycle graph adds the wrap-around edge
    gc = cycle_graph(5)
    state_c = TensorNetworkState{ComplexF64}(undef, gc, P, V)
    @test length(edges(state_c)) == 5
    @test Canopy.has_edge(state_c, 1, 5)
end


# Helper: vertex coordination numbers of the graph spanned by `edges`.
coordinations(es) = collect(length.(values(Canopy.adjacency(Indices(es)))))

@testset "Lattice constructors" begin
    # --- square ---
    @test length(square_lattice(3, 3)) == 12                       # open: 2*m*(n-1) bonds
    @test length(square_lattice(3, 3; periodic = (false, true))) == 15
    @test length(square_lattice(3, 3; periodic = (true, false))) == 15
    @test length(square_lattice(3, 3; periodic = (true, true))) == 18
    @test all(==(4), coordinations(square_lattice(4, 4; periodic = (true, true))))
    @test maximum(coordinations(square_lattice(3, 3))) == 4         # open: bulk still 4
    @test length(keys(Canopy.adjacency(Indices(square_lattice(3, 4))))) == 12  # vertex count

    # --- triangular: bulk coordination 6 on the torus ---
    @test all(==(6), coordinations(triangular_lattice(4, 4; periodic = (true, true))))

    # --- hexagonal: coordination 3, two sites per cell ---
    @test all(==(3), coordinations(hexagonal_lattice(2, 2; periodic = (true, true))))
    @test length(keys(Canopy.adjacency(Indices(hexagonal_lattice(2, 3))))) == 12

    # --- periodic-size guards ---
    @test_throws ArgumentError square_lattice(2, 3; periodic = (true, false))
    @test_throws ArgumentError triangular_lattice(3, 2; periodic = (false, true))
    @test_throws ArgumentError hexagonal_lattice(1, 3; periodic = (true, false))

    # --- builds a (random) state directly from lattice edges ---
    es = square_lattice(2, 3; periodic = (false, true))
    state = randn!(TensorNetworkState{ComplexF64}(undef, es, ComplexSpace(2), ComplexSpace(2)))
    @test length(state) == 6
    @test issetequal(vertices(state), keys(Canopy.adjacency(Indices(es))))
end

@testset "Product state — bosonic dense equivalence" begin
    P = ComplexSpace(2)
    coeffs = [0.3, 0.4]          # unnormalized, definite (trivial) charge
    for es in (square_lattice(2, 3), triangular_lattice(2, 2))
        state = product_state(es, P, Trivial() => coeffs)
        @test Canopy.scalartype(state) == Float64
        # all deduced bonds are one-dimensional
        @test all(e -> dim(virtualspace(state, e)) == 1, edges(state))
        # dense wavefunction is the (order-independent, uniform) product of local kets
        psi = reshape(convert(Array, TensorMap(state)), :)
        ref = foldl((a, _) -> kron(a, coeffs), 2:length(state); init = coeffs)
        @test isapprox(psi, ref; rtol = 1e-12)
    end
    # ComplexF64 promotion from complex coefficients
    statec = product_state(square_lattice(2, 2), P, Trivial() => ComplexF64[1, im])
    @test Canopy.scalartype(statec) == ComplexF64
end

@testset "Product state — fermionic charge deduction" begin
    Random.seed!(0)
    es = square_lattice(2, 2)
    Pf = fermion_space(Trivial)
    verts = collect(keys(Canopy.adjacency(Indices(es))))
    occ = Dictionary(verts, [isodd(v[1] + v[2]) ? 1 : 0 for v in verts])  # checkerboard, neutral
    ls = Dictionary(verts, [fℤ₂(occ[v]) => [1.0] for v in verts])
    ps = Dictionary(verts, fill(Pf, length(verts)))
    state = product_state(es, ps, ls)

    @test all(e -> dim(virtualspace(state, e)) == 1, edges(state))

    msgs = Canopy.BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter = 50)
    nop = f_num(Float64, Trivial)
    for v in verts
        @test isapprox(real(expect(state, msgs, nop, (v,))), occ[v]; atol = 1e-8)
    end
end

@testset "Product state — errors" begin
    es = square_lattice(2, 2)
    Pf = fermion_space(Trivial)
    verts = collect(keys(Canopy.adjacency(Indices(es))))
    ps = Dictionary(verts, fill(Pf, length(verts)))

    # non-neutral total charge (three odd, one even → odd total)
    bad = Dictionary(verts, [fℤ₂(i == 1 ? 0 : 1) => [1.0] for i in eachindex(verts)])
    @test_throws ArgumentError product_state(es, ps, bad)

    # coefficient count must match the sector degeneracy
    wrongdim = Dictionary(verts, [fℤ₂(0) => [1.0, 0.0] for _ in verts])
    @test_throws ArgumentError product_state(es, ps, wrongdim)

    # non-abelian symmetry is rejected
    Vsu2 = Vect[SU2Irrep](1 // 2 => 1)
    @test_throws ArgumentError product_state(es, Vsu2, SU2Irrep(1 // 2) => [1.0])
end
