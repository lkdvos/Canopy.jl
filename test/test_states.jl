using Canopy
using Canopy: UndirectedEdge, DirectedEdge, belief_propagation, reduced_density_matrix, expect,
              auxiliary_vertex, check_consistency
using TensorKit
using LinearAlgebra: isposdef
using TensorKitTensors.SpinOperators: σᶻ, S_z_S_z, S_exchange
using TensorKitTensors.FermionOperators: f_num, f_hopping, fermion_space
using Graphs
using Test
using Dictionaries
using Random: Random, randn!, rand!, MersenneTwister


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
        @test check_consistency(state)
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
    @test check_consistency(state)

    msgs = Canopy.BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter = 50)
    nop = f_num(Float64, Trivial)
    for v in verts
        @test isapprox(real(expect(state, msgs, nop, (v,))), occ[v]; atol = 1e-8)
    end
end

@testset "Product state — local-state shorthands" begin
    es = square_lattice(2, 2)
    verts = vertices(es)

    # a bare coefficient vector means the trivial sector
    P = ComplexSpace(2)
    explicit = product_state(es, P, Trivial() => [0.3, 0.4])
    bare = product_state(es, P, [0.3, 0.4])
    @test all(v -> explicit[v] ≈ bare[v], verts)
    @test scalartype(bare) === Float64

    # a bare sector means unit coefficients, hence `Bool` → `Float64`
    Pf = fermion_space(Trivial)
    occ = Dictionary(verts, [isodd(v[1] + v[2]) ? 1 : 0 for v in verts])   # neutral parity
    explicit_f = product_state(es, Pf, map(n -> fℤ₂(n) => [1.0], occ))
    bare_f = product_state(es, Pf, map(fℤ₂, occ))
    @test all(v -> explicit_f[v] ≈ bare_f[v], verts)
    @test scalartype(bare_f) === Float64
    @test scalartype(product_state(ComplexF64, es, Pf, map(fℤ₂, occ))) === ComplexF64

    # anything convertible to the sectortype works — `1` for `U1Irrep`. Also the
    # mixed form: a uniform physical space with per-vertex local states.
    Pu = Vect[U1Irrep](-1 => 1, 1 => 1)
    charges = Dictionary(verts, [isodd(v[1] + v[2]) ? 1 : -1 for v in verts])  # sums to 0
    explicit_u = product_state(es, Pu, map(c -> U1Irrep(c) => [1.0], charges))
    bare_u = product_state(es, Pu, charges)
    @test all(v -> explicit_u[v] ≈ bare_u[v], verts)
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

    # a bare coefficient vector has no definite charge under a graded space
    @test_throws ArgumentError product_state(es, Pf, [1.0, 0.0])

    # a bare sector only selects a state if that sector is one-dimensional
    @test_throws ArgumentError product_state(es, ComplexSpace(2), Trivial())

    # the sector's *type* must be that of the physical space
    @test_throws ArgumentError product_state(es, ComplexSpace(2), fℤ₂(1) => [1.0])

    # the sector must appear in the physical space at all
    @test_throws ArgumentError product_state(es, Vect[U1Irrep](0 => 1, 1 => 1), U1Irrep(5) => [1.0])

    # `Dictionary` keys must span exactly the vertices of the topology
    @test_throws ArgumentError product_state(square_lattice(2, 3), ps, wrongdim)
    @test_throws ArgumentError product_state(es, ps, Dictionary(verts[1:3], fill(fℤ₂(0), 3)))

    # non-abelian symmetry is rejected
    Vsu2 = Vect[SU2Irrep](1 // 2 => 1)
    @test_throws ArgumentError product_state(es, Vsu2, SU2Irrep(1 // 2) => [1.0])
end

# Maximum coordination number of a state: every on-site tensor carries exactly `N`
# domain legs, unit-padded beyond the vertex's neighbors.
maxcoordination(state) = numin(state[first(vertices(state))])

@testset "Product state — charge bath" begin
    es = square_lattice(2, 3)          # checkerboard → 3 particles, so not neutral
    verts = vertices(es)
    Pu = fermion_space(U1Irrep)
    I = sectortype(Pu)
    occ(v) = isodd(v[1] + v[2]) ? 1 : 0
    ls = Dictionary(verts, [fℤ₂(occ(v)) ⊠ U1Irrep(occ(v)) for v in verts])
    Q = sum(occ, verts)
    qtot = fℤ₂(mod(Q, 2)) ⊠ U1Irrep(Q)
    @test Q == 3

    @test_throws ArgumentError product_state(es, Pu, ls)                     # bath is opt-in
    @test_throws ArgumentError product_state(es, Pu, ls; total_charge = fℤ₂(1) ⊠ U1Irrep(5))
    @test_throws ArgumentError product_state(es, Pu, ls; total_charge = one(I))
    # neutral local charges cannot absorb a nontrivial total charge either
    @test_throws ArgumentError product_state(
        square_lattice(2, 2), fermion_space(Trivial), fℤ₂(0); total_charge = fℤ₂(1)
    )

    aux = auxiliary_vertex(verts)
    @test aux == (0, 0)
    state = product_state(ComplexF64, es, Pu, ls; total_charge = qtot)
    @test length(state) == length(verts) + 1
    @test aux in vertices(state)
    @test physicalspace(state, aux) == Vect[I](dual(qtot) => 1)
    @test length(Canopy.neighbors(state, aux)) == 1
    @test check_consistency(state)

    # the bath attaches to a vertex of minimal degree, tie-broken by vertex order
    @test only(Canopy.neighbors(state, aux)) == (1, 1)

    # ... which is equivalent to spelling the bath out by hand, as
    # `benchmark/realtime_timing/run_timings.jl` used to do
    augmented = vcat(verts, [aux])
    manual = product_state(
        ComplexF64, vcat(es, [UndirectedEdge(aux, (1, 1))]),
        Dictionary(augmented, vcat(fill(Pu, length(verts)), [Vect[I](dual(qtot) => 1)])),
        Dictionary(augmented, vcat([ls[v] => [1.0] for v in verts], [dual(qtot) => [1.0]])),
    )
    @test all(v -> state[v] ≈ manual[v], vertices(state))

    @test (-5, -5) in vertices(
        product_state(ComplexF64, es, Pu, ls; total_charge = qtot, auxiliary = (-5, -5))
    )
    @test_throws ArgumentError product_state(
        ComplexF64, es, Pu, ls; total_charge = qtot, auxiliary = (1, 1)
    )

    # BP is exact on a product state — 1-dim bonds make every message rank-1 — so
    # the occupations are exact even on a loopy lattice
    msgs = belief_propagation(BPMessages(state), state; maxiter = 50)
    nop = f_num(ComplexF64, U1Irrep)
    @test all(v -> isapprox(real(expect(state, msgs, nop, (v,))), occ(v); atol = 1.0e-8), verts)
    @test isapprox(real(sum(expect(state, msgs, nop, (v,)) for v in verts)), Q; atol = 1.0e-8)

    # attaching at minimal degree keeps `N` at the bath-free value
    hex = hexagonal_lattice(2, 2)
    L = length(vertices(hex))
    hst = product_state(
        ComplexF64, hex, Pu, fℤ₂(1) ⊠ U1Irrep(1);
        total_charge = fℤ₂(mod(L, 2)) ⊠ U1Irrep(L),
    )
    plain = randn_state(ComplexF64, hex, Pu, Vect[I](one(I) => 1))
    @test maxcoordination(hst) == maxcoordination(plain) == 3
end

@testset "Product state — graph topology" begin
    state = product_state(cycle_graph(6), ComplexSpace(2), [1.0, 0.0])
    @test length(state) == 6
    @test issetequal(vertices(state), 1:6)
    @test check_consistency(state)
end

@testset "vertices(edges)" begin
    es = square_lattice(2, 3)
    verts = vertices(es)
    @test length(verts) == 6
    @test issetequal(verts, [(i, j) for i in 1:2, j in 1:3])
    # the order a state built on `es` iterates its vertices in
    @test verts == collect(vertices(product_state(es, ComplexSpace(2), [1.0, 0.0])))
end

@testset "randn_state / rand_state" begin
    es = square_lattice(2, 3)
    P = fermion_space(Trivial)
    V = Vect[fℤ₂](0 => 2, 1 => 2)

    state = randn_state(MersenneTwister(0), ComplexF64, es, P, V)
    reference = randn!(MersenneTwister(0), TensorNetworkState{ComplexF64}(undef, es, P, V))
    @test all(v -> state[v] == reference[v], vertices(state))
    @test check_consistency(state)

    # the per-vertex/per-edge form draws the same numbers
    @test all(v -> state[v] == randn_state(MersenneTwister(0), ComplexF64, Canopy._spaces(es, P, V)...)[v],
              vertices(state))

    rstate = rand_state(MersenneTwister(0), ComplexF64, es, P, V)
    @test all(v -> rstate[v] == rand!(MersenneTwister(0), TensorNetworkState{ComplexF64}(undef, es, P, V))[v],
              vertices(rstate))

    # `rng` and `T` are independently optional, `T` defaults to `Float64`
    @test scalartype(randn_state(MersenneTwister(0), es, P, V)) === Float64
    @test scalartype(randn_state(ComplexF64, es, P, V)) === ComplexF64
    @test scalartype(randn_state(es, P, V)) === Float64
    @test check_consistency(rand_state(cycle_graph(6), ComplexSpace(2), ComplexSpace(3)))

    # --- charge bath on a chain, where BP is exact ---
    Pu = fermion_space(U1Irrep)
    I = sectortype(Pu)
    # the virtual charges must span both signs, or the target sector cannot flow
    # through the network and the state comes out all-zero
    Vu = Vect[I](
        (fℤ₂(0) ⊠ U1Irrep(-2)) => 1, (fℤ₂(1) ⊠ U1Irrep(-1)) => 1, (fℤ₂(0) ⊠ U1Irrep(0)) => 2,
        (fℤ₂(1) ⊠ U1Irrep(1)) => 1, (fℤ₂(0) ⊠ U1Irrep(2)) => 1,
    )
    chain = [UndirectedEdge(i, i + 1) for i in 1:5]
    qtot = fℤ₂(1) ⊠ U1Irrep(3)
    st = randn_state(MersenneTwister(0), ComplexF64, chain, Pu, Vu; total_charge = qtot)
    @test auxiliary_vertex(vertices(chain)) == 0
    @test length(st) == 7
    @test physicalspace(st, 0) == Vect[I](dual(qtot) => 1)
    @test length(Canopy.neighbors(st, 0)) == 1
    @test check_consistency(st)
    @test maxcoordination(st) == 2

    # a trivial total charge needs no bath
    @test length(randn_state(ComplexF64, chain, Pu, Vu; total_charge = one(I))) == 6

    msgs = belief_propagation(BPMessages(st), st; maxiter = 200)
    ntot = sum(expect(st, msgs, f_num(ComplexF64, U1Irrep), (v,)) for v in 1:6)
    @test isapprox(real(ntot), 3; atol = 1.0e-8)
end
