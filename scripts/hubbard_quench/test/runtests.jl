# Tests for the Hubbard quench physics layer.
#
#   julia --project=scripts/hubbard_quench scripts/hubbard_quench/test/runtests.jl
#
# The two load-bearing testsets are:
#   - "local-state table": pins the (sector, degeneracy) → physical state map, which is
#     undocumented upstream and version-dependent. Derived here from first principles
#     (diagonal blocks of the number operators) rather than restating the table.
#   - "U = 0 agreement": end-to-end, all four symmetries against the exact free-fermion
#     result. This is what actually certifies gates + charge bath + Trotter + measurement.

using Test
using LinearAlgebra: BLAS, Diagonal, diag, norm
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "HubbardQuench.jl"))
using .HubbardQuench
using Canopy
using Canopy: belief_propagation
using TensorKit
using TensorKitTensors.HubbardOperators: hubbard_space, u_num, d_num, ud_num

const SYMS = (
    ("trivial", "trivial"), ("trivial", "u1"), ("u1", "trivial"), ("u1", "u1"),
)

@testset "HubbardQuench" begin

    @testset "sectortypes" begin
        @test sectortypes("trivial", "trivial") == (Trivial, Trivial)
        @test sectortypes("u1", "u1") == (U1Irrep, U1Irrep)
        @test_throws ArgumentError sectortypes("su2", "trivial")
        @test_throws ArgumentError sectortypes("trivial", "su2")
        @test_throws ArgumentError sectortypes("z2", "trivial")
    end

    @testset "occupation" begin
        @test occupation(:emp) == (0, 0)
        @test occupation(:up) == (1, 0)
        @test occupation(:dn) == (0, 1)
        @test occupation(:updn) == (1, 1)
        @test_throws ArgumentError occupation(:nonsense)
    end

    # The number operators are diagonal in the basis `hubbard_space` uses, so for a basis
    # state living in sector `c` at degeneracy index `k` the occupations are exactly
    # `block(op, c)[k, k]`. Re-deriving the table this way means a change in
    # TensorKitTensors' internal ordering fails here rather than silently producing the
    # wrong physical state.
    @testset "local-state table ($pn,$sn)" for (pn, sn) in SYMS
        P, S = sectortypes(pn, sn)
        V = hubbard_space(P, S)
        nu, nd, du = u_num(ComplexF64, P, S), d_num(ComplexF64, P, S), ud_num(ComplexF64, P, S)

        # every basis state is an eigenstate of all three (else `sector => coeffs` would be
        # ambiguous)
        for op in (nu, nd, du), c in sectors(V)
            B = block(op, c)
            size(B, 1) > 1 && @test norm(B - Diagonal(diag(B))) < 1.0e-12
        end

        for which in (:emp, :up, :dn, :updn)
            spec = localstate(P, S, which)
            c, coeffs = first(spec), last(spec)
            @test length(coeffs) == dim(V, c)          # degeneracy matches the space
            @test count(!iszero, coeffs) == 1          # a single basis state
            k = findfirst(!iszero, coeffs)
            nup, ndn = occupation(which)
            @test real(block(nu, c)[k, k]) ≈ nup
            @test real(block(nd, c)[k, k]) ≈ ndn
            @test real(block(du, c)[k, k]) ≈ nup * ndn
        end
        @test_throws ArgumentError localstate(P, S, :nonsense)
    end

    @testset "lattice" begin
        hex = lattice("hex", 3, 3)
        @test length(hex) == 2 * 3 * 3
        @test length(hex.edges) == 21
        @test all(hex.sublattice[e.src] == -hex.sublattice[e.dst] for e in hex.edges)

        sq = lattice("square", 4, 4)
        @test length(sq) == 16
        @test length(sq.edges) == 24
        @test all(sq.sublattice[e.src] == -sq.sublattice[e.dst] for e in sq.edges)

        @test_throws ArgumentError lattice("triangular", 3, 3)
        @test_throws ArgumentError lattice("kagome", 3, 3)
        # an odd periodic extent closes an odd cycle and breaks bipartiteness
        @test_throws ArgumentError lattice("square", 3, 3; periodic = (true, false))
        @test length(lattice("square", 4, 4; periodic = (true, true)).edges) == 32
    end

    @testset "regions" begin
        # 4x4 open square: corners have degree 2, edges 3, the 2x2 core 4.
        sq = lattice("square", 4, 4)
        @test length(bulk_region(sq, 0)) == 16
        @test bulk_region(sq, 1) == sort([(i, j) for i in 2:3, j in 2:3] |> vec)
        @test_throws ArgumentError bulk_region(sq, 5)
        @test_throws ArgumentError bulk_region(sq, -1)
        # the graph centre of the 4x4 square is its 2x2 core
        @test graph_center(sq) == sort([(i, j) for i in 2:3, j in 2:3] |> vec)
        # a torus has no boundary, so everything is bulk at any depth
        torus = lattice("square", 4, 4; periodic = (true, true))
        @test length(bulk_region(torus, 3)) == 16

        hex = lattice("hex", 3, 3)
        @test length(bulk_region(hex, 0)) == length(hex)
        @test issubset(bulk_region(hex, 1), hex.verts)
    end

    @testset "quench patterns" begin
        hex = lattice("hex", 3, 3)
        cdw = quench_pattern("cdw", hex)
        @test count(==(:up), values(cdw)) == 9
        @test count(==(:dn), values(cdw)) == 9
        @test all(cdw[v] == (hex.sublattice[v] > 0 ? :up : :dn) for v in hex.verts)

        dbl = quench_pattern("doublon", hex)
        @test count(==(:updn), values(dbl)) == 1
        @test count(==(:emp), values(dbl)) == 1
        vc = only(v for v in hex.verts if dbl[v] == :updn)
        vn = only(v for v in hex.verts if dbl[v] == :emp)
        @test vn in hex.adj[vc]                                  # doublon and hole adjacent
        @test vc == first(graph_center(hex))                     # deterministic
        @test quench_pattern("doublon", hex) == dbl              # reproducible

        @test_throws ArgumentError quench_pattern("nonsense", hex)
    end

    @testset "state construction + self-check ($pn,$sn)" for (pn, sn) in SYMS
        P, S = sectortypes(pn, sn)
        ops = operators(P, S)
        lat = lattice("hex", 2, 2)
        for qname in ("cdw", "doublon")
            pat = quench_pattern(qname, lat)
            state, q, bath = build_state(lat, pat, P, S)
            msgs = belief_propagation(BPMessages(state), state; maxiter = 50, tol = 1.0e-13)

            # the self-check is the guard on the local-state table; it must pass
            chk = check_initial_state(state, msgs, lat, pat, ops)
            @test chk.nup_total ≈ sum(occupation(pat[v])[1] for v in lat.verts)
            @test chk.ndn_total ≈ sum(occupation(pat[v])[2] for v in lat.verts)

            # a charge bath is attached exactly when the fused charge is non-trivial, which
            # for these patterns means particle-U(1) (Q = N ≠ 0) and never spin-U(1) (Sz = 0)
            @test bath == (q != one(sectortype(hubbard_space(P, S))))
            @test bath == (P === U1Irrep)
            # ...and when it is, the state really does carry an extra vertex
            @test length(Canopy.vertices(state)) == length(lat) + (bath ? 1 : 0)

            # the staggered order parameter is exactly 1 for the AFM background
            sc, _ = measure(state, msgs, lat, ops, lat.verts; t = -1.0, U = 0.0, energy = false)
            qname == "cdw" && @test sc.m_s_all ≈ 1.0
        end
    end

    @testset "trotter layers" begin
        lat = lattice("hex", 3, 3)
        ops = operators(Trivial, Trivial)
        classes = edge_coloring(lat.edges)
        K = length(classes)

        single, hops, ncolors = build_layers(lat, ops, 4.0, -1.0, 0.01)
        @test ncolors == K
        @test length(hops) == 2K - 1              # symmetric Strang sandwich
        @test !isnothing(single)

        # the interaction layer is the identity at U = 0 and is skipped
        s0, _, _ = build_layers(lat, ops, 0.0, -1.0, 0.01)
        @test isnothing(s0)

        # every edge is covered with total weight dt: the middle class once at dt, all
        # others twice at dt/2
        counts = Dict(e => 0 for e in lat.edges)
        for layer in hops, g in layer.gatelist
            counts[Canopy.UndirectedEdge(g.sites...)] += 1
        end
        @test all(counts[e] == (e in classes[K] ? 1 : 2) for e in lat.edges)
    end

    @testset "truncation guard" begin
        @test truncation(8, 1.0e-10) isa Any
        # a cutoff at or below the gauge tolerance is the silent-corruption trap
        @test_logs (:warn,) truncation(8, 0.0)
        @test_logs (:warn,) truncation(8, 1.0e-14)
        @test_logs truncation(8, default_cutoff())          # no warning
        @test default_cutoff() > Canopy.default_gauge_tol(ComplexF64[])
        @test_throws ArgumentError truncation(0, 1.0e-10)
    end

    @testset "bp_schedule" begin
        @test bp_schedule("sync") isa SynchronousSchedule
        @test bp_schedule("spanningtree") isa SpanningTreeSchedule
        @test bp_schedule("residual") isa ResidualSchedule
        @test bp_schedule("splash") isa ResidualSplashSchedule
        @test_throws ArgumentError bp_schedule("nonsense")
    end

    @testset "free-fermion reference" begin
        lat = lattice("hex", 2, 2)
        pat = quench_pattern("cdw", lat)
        circ, cont = free_fermion_reference(lat, pat, -1.0, 0.02, 10)
        @test length(circ) == length(cont) == 11
        @test circ[1] ≈ 1.0                     # perfect AFM at t = 0
        @test cont[1] ≈ 1.0
        # the two references differ only by Trotter error, which is O(dt^2) per step
        @test maximum(abs, circ .- cont) < 1.0e-3
        # halving dt must shrink the gap towards the continuous result
        c2, k2 = free_fermion_reference(lat, pat, -1.0, 0.01, 20)
        @test maximum(abs, c2 .- k2) < maximum(abs, circ .- cont)
    end

    # The end-to-end correctness gate. On a tree (2 sites) BP and the Bethe RDM are exact
    # and no truncation is needed, so every symmetry must reproduce the exact free-fermion
    # trajectory to machine precision. This simultaneously certifies the local-state table,
    # the charge bath, the hopping gate convention, the Trotter sandwich and `measure`.
    @testset "U = 0 agreement on a tree ($pn,$sn)" for (pn, sn) in SYMS
        P, S = sectortypes(pn, sn)
        ops = operators(P, S)
        lat = lattice("square", 1, 2)
        pat = quench_pattern("cdw", lat)
        t, dt, nsteps = -1.0, 0.05, 8
        ref, _ = free_fermion_reference(lat, pat, t, dt, nsteps)

        state, _, _ = build_state(lat, pat, P, S)
        msgs = belief_propagation(BPMessages(state), state; maxiter = 50, tol = 1.0e-14)
        _, hops, _ = build_layers(lat, ops, 0.0, t, dt)
        trunc = truncation(64, default_cutoff())
        for step in 1:nsteps
            for layer in hops
                state, msgs, _ = apply!(state, msgs, layer; trunc)
            end
            msgs = belief_propagation(msgs, state; maxiter = 50, tol = 1.0e-14)
            sc, _ = measure(state, msgs, lat, ops, lat.verts; t, U = 0.0, energy = false)
            @test sc.m_s_all ≈ ref[step + 1] atol = 1.0e-9
        end
    end

    # Same gate with a loop present: BP is now approximate, so the tolerance is loose, but
    # all four symmetries must still agree with each other and track the exact result.
    @testset "U = 0 agreement on a loop" begin
        lat = lattice("hex", 2, 2)
        pat = quench_pattern("cdw", lat)
        t, dt, nsteps, χ = -1.0, 0.05, 6, 24
        ref, _ = free_fermion_reference(lat, pat, t, dt, nsteps)
        finals = Float64[]
        for (pn, sn) in SYMS
            P, S = sectortypes(pn, sn)
            ops = operators(P, S)
            state, _, _ = build_state(lat, pat, P, S)
            msgs = belief_propagation(BPMessages(state), state; maxiter = 50, tol = 1.0e-12)
            _, hops, _ = build_layers(lat, ops, 0.0, t, dt)
            trunc = truncation(χ, default_cutoff())
            for _ in 1:nsteps
                for layer in hops
                    state, msgs, _ = apply!(state, msgs, layer; trunc)
                end
                msgs = belief_propagation(msgs, state; maxiter = 50, tol = 1.0e-12)
            end
            @test bp_residual(msgs, state, lat) < 1.0e-9
            sc, _ = measure(state, msgs, lat, ops, lat.verts; t, U = 0.0, energy = false)
            push!(finals, sc.m_s_all)
            @test sc.m_s_all ≈ ref[end] atol = 1.0e-4
        end
        @test maximum(finals) - minimum(finals) < 1.0e-4   # symmetries agree with each other
    end

    # At U != 0 there is no exact reference, so energy conservation is the validation. For
    # this quench E_kin(0) = E_int(0) = 0 (a product state has no inter-site coherence and
    # no double occupancy), so the exact E_tot is 0 for all time.
    #
    # A Trotterized circuit conserves its effective Hamiltonian rather than H, so the
    # measured drift is genuinely O(dt²) and asserting a fixed absolute bound would just be
    # fitting a magic number. Instead: evolve to the *same* final time at two step sizes and
    # require the drift to shrink roughly quadratically. That tests the property we actually
    # rely on when using E_tot as a χ/BP diagnostic.
    @testset "energy conservation at U != 0" begin
        lat = lattice("hex", 2, 2)
        pat = quench_pattern("cdw", lat)
        P, S = sectortypes("u1", "u1")
        ops = operators(P, S)
        t, U, tfinal = -1.0, 4.0, 0.2

        function drift(dt)
            nsteps = round(Int, tfinal / dt)
            state, _, _ = build_state(lat, pat, P, S)
            msgs = belief_propagation(BPMessages(state), state; maxiter = 50, tol = 1.0e-12)
            single, hops, _ = build_layers(lat, ops, U, t, dt)
            @test !isnothing(single)
            trunc = truncation(24, default_cutoff())
            sc0, _ = measure(state, msgs, lat, ops, lat.verts; t, U)
            @test sc0.E_tot ≈ 0.0 atol = 1.0e-12
            for _ in 1:nsteps
                state, msgs, _ = apply!(state, msgs, single)
                for layer in hops
                    state, msgs, _ = apply!(state, msgs, layer; trunc)
                end
                state, msgs, _ = apply!(state, msgs, single)
                msgs = belief_propagation(msgs, state; maxiter = 50, tol = 1.0e-12)
            end
            sc, _ = measure(state, msgs, lat, ops, lat.verts; t, U)
            @test sc.n_mean ≈ 1.0 atol = 1.0e-10        # half filling is conserved
            @test sc.docc_mean > 0.0                    # ...but doublons have formed
            return abs(sc.E_tot)
        end

        coarse, fine = drift(0.02), drift(0.01)
        @test coarse < 1.0e-2
        # halving dt must cut the drift by ~4x for a second-order splitting; allow slack for
        # the truncation and BP contributions that do not scale with dt
        @test fine < coarse / 2.5
    end

end
