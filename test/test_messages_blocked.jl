using Canopy
using Canopy: compute_message, compute_message!, DirectedEdge, UndirectedEdge,
              belief_propagation, outgoing_edges, buffer_isempty, buffer_stats,
              reduced_density_matrix, LocalGate, apply!, _spaces
using TensorKit
using TensorKit.TO: DefaultAllocator, DefaultBackend
using TensorKitTensors.FermionOperators: fermion_space, f_num
using TensorKitTensors.HubbardOperators: hubbard_space
import Bumper
using Dictionaries
using Graphs
using Graphs: path_graph, cycle_graph, star_graph, complete_graph, grid
using LinearAlgebra: norm
using MatrixAlgebraKit: notrunc
using Random
using Test

# The vertex-batched message kernel runs the `Layout(k)` formulation **unconditionally**:
# one layout family, the target leg alone in the domain, fermionic signs folded into the
# (χ²) transposed messages instead of paid as `twist!` passes over the chain links. It must
# be *numerically identical* to the pairwise formulation, which is retained only as the
# oracle and is reached only through `PairwiseBackend` — hence this file is one long
# differential test.
#
# `space(blocked[i]) == space(pairwise[i])` is asserted alongside every comparison: a
# kernel that silently drops a zero-dimensional sector still compares `≈` but breaks
# `check_consistency`.
#
# There is no longer a selection *predicate* to assert against. What used to be checked as
# the selection predicate is now checked structurally, by the `tensoralloc`
# fingerprint near the end of this file: the default path must not look like the pairwise
# one. That is the guard against a fallback being reintroduced by accident.

_state_on(g, P, V; seed) = (Random.seed!(seed); randn_state(ComplexF64, g, P, V))

# As `test/test_messages.jl`, plus the graded and non-abelian rows.
#
# The two `SU2Irrep` rows are the **non-abelian** ones, and they are here rather
# than in the fallback testset because the blocked kernel is *not* restricted to
# abelian fusion: every step of the chain is a `TensorMap`-level operation
# (`tensoradd!` permutes, `mul!`, `adjoint`, `twist!`) that handles a general
# fusion style on its own, and the sign derivation needs `twist(σ)² = 1` — i.e.
# `SymmetricBraiding` — not `UniqueFusion`. See [`PairwiseBackend`](@ref).
# `hubbard_space(Trivial, SU2Irrep)` is the spin-SU(2) Hubbard physical space, so the
# row covers a physically meaningful space and not only the fusion style. Note it is
# *not* currently reachable from `scripts/hubbard_quench`, which rejects SU(2) in
# `sectortypes` because `product_state` handles abelian sectors only — so this file is
# the only place the non-abelian path is exercised at all.
const _MSG_SPACES = (
    ("bosonic", ComplexSpace(2), ComplexSpace(3)),
    ("U(1)", Vect[U1Irrep](0 => 1, 1 => 1), Vect[U1Irrep](-1 => 1, 0 => 2, 1 => 1)),
    ("fermionic", fermion_space(Trivial), Vect[fℤ₂](0 => 2, 1 => 2)),
    (
        "fZ2xU1", fermion_space(U1Irrep),
        Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 2, (1, 1) => 1, (1, -1) => 1, (0, 2) => 1),
    ),
    ("SU(2)", Vect[SU2Irrep](0 => 1, 1 // 2 => 1), Vect[SU2Irrep](0 => 2, 1 // 2 => 1)),
    (
        "fZ2xSU2", hubbard_space(Trivial, SU2Irrep),
        Vect[fℤ₂ ⊠ SU2Irrep]((0, 0) => 2, (1, 1 // 2) => 1, (0, 1) => 1),
    ),
)

const _MSG_GEOMETRIES = (
    ("chain L=4", path_graph(4)),       # degrees 1,2 — also `d < N` at the ends
    ("cycle L=6", cycle_graph(6)),      # degree 2
    ("star deg 4", star_graph(5)),      # central degree 4
    ("3x3 grid", grid([3, 3])),         # degrees 2,3,4 — padded legs
    ("K5", complete_graph(5)),          # degree 4, dense
    ("K6", complete_graph(6)),          # degree 5
)

# Which geometries each symmetry row sweeps. Everything sweeps everything except the
# two non-abelian rows, which skip `K6`.
#
# MEASURED, and the reason this exists: adding the two `SU2Irrep` rows took this file
# from 11m39s to 18m05s, and this trim brings it back to **15m44s**. Essentially all of
# that is *compilation*, not fixture work — the whole warm sweep over all 6 geometries
# × 3 graded symmetries runs in ~20 s, and the worst single fixture (`fZ2xSU2` on `K6`)
# is 6.2 s of BP plus 2.1 s of sweep. Compilation is driven by distinct
# `(sector type, numind)` pairs, so the only
# trim with any leverage is dropping the one geometry that contributes an otherwise
# unused `numind`: `K6` is the sole degree-5 (`M = 6`) fixture. Dropping *any* other
# geometry from the non-abelian rows would save run time measured in seconds and no
# compilation at all, because `star`/`grid`/`K5` share `M = 5` and `chain`/`cycle`
# share `M = 3`.
#
# What that costs in coverage: degree 5 (an 8-step chain) is still swept for all four
# abelian symmetries, and the non-abelian rows still reach degrees 1-4 including the
# `d < N` padded-leg case. The blocked chain is generic in `d` — nothing in it is
# special at 5 — so this trades a redundant arity for two thirds of the added CI time.
# If a future change makes the chain arity-dependent, delete this and take the time.
_msg_geometries(sname) =
    sname in ("SU(2)", "fZ2xSU2") ?
    filter(g -> first(g) != "K6", collect(_MSG_GEOMETRIES)) : collect(_MSG_GEOMETRIES)

# Blocked vs pairwise on one vertex batch, returning both so the caller can chain
# further comparisons without recomputing.
function _cmp_batch(msgs, state, edges)
    out_b = compute_message(msgs, state, edges, DefaultBackend(), DefaultAllocator())
    out_p = compute_message(msgs, state, edges, PairwiseBackend(), DefaultAllocator())
    @test length(out_b) == length(edges)
    for i in eachindex(edges)
        @test space(out_b[i]) == space(out_p[i])
        @test out_b[i] ≈ out_p[i]
    end
    return out_b, out_p
end

@testset "blocked ≡ pairwise ≡ per-edge oracle" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in _msg_geometries(sname)
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname)))
            msgs = belief_propagation(
                BPMessages(state), state; maxiter = 5, tol = 0, schedule = SynchronousSchedule()
            )
            for v in vertices(state)
                edges = collect(outgoing_edges(state, v))
                out_b, out_p = _cmp_batch(msgs, state, edges)
                for (i, e) in enumerate(edges)
                    ref = compute_message(msgs, state, e)   # per-edge golden oracle
                    @test space(out_b[i]) == space(ref)
                    @test out_b[i] ≈ ref
                    @test out_p[i] ≈ ref
                end
            end
        end
    end
end

# The decisive testset. BP messages are hermitian after projection, so `m† = m`
# masks a conjugation error in `adjoint(Mt)` / `adjoint(S_k)` — and a chain
# rewrite is exactly where those appear. Generic random messages do not.
@testset "blocked ≡ pairwise (non-hermitian messages)" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in _msg_geometries(sname)
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "nh")))
            msgs = BPMessages(state)
            Random.seed!(hash((sname, gname, "nhmsg")))
            for e in keys(msgs.messages)
                Random.randn!(msgs.messages[e])          # generic, non-hermitian
            end
            for v in vertices(state)
                edges = collect(outgoing_edges(state, v))
                _, out_p = _cmp_batch(msgs, state, edges)
                for (i, e) in enumerate(edges)
                    @test out_p[i] ≈ compute_message(msgs, state, e)
                end
            end
        end
    end
end

# Reordered full span pins output ordering; the clustered middle subset exercises
# the `extrema(target_legs)` partial chains (`kmin > 1` and `kmax < d`). Needs degree
# ≥ 3, so it runs on the two densest geometries — filtered through
# `_msg_geometries`, or the non-abelian rows would pull in the `M = 6` compilation
# here that the sweeps above skip and the trim would buy nothing.
@testset "blocked ≡ pairwise (subset, reordered targets)" begin
    for (sname, P, V) in _MSG_SPACES,
            (gname, g) in filter(
                r -> first(r) in ("star deg 4", "K6"), _msg_geometries(sname)
            )

        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "sub")))
            msgs = BPMessages(state)
            Random.seed!(hash((sname, gname, "submsg")))
            for e in keys(msgs.messages)
                Random.randn!(msgs.messages[e])
            end
            for v in vertices(state)
                all_e = collect(outgoing_edges(state, v))
                length(all_e) < 3 && continue
                subsets = (
                    [all_e[end]; all_e[1:2:(end - 1)]],   # reordered, full span
                    reverse(all_e[2:(end - 1)]),          # clustered middle, reversed
                    [all_e[2], all_e[2]],                 # duplicate target
                )
                for edges in subsets
                    _, out_p = _cmp_batch(msgs, state, edges)
                    for (i, e) in enumerate(edges)
                        @test out_p[i] ≈ compute_message(msgs, state, e)
                    end
                end
            end
        end
    end
end

# `randn_state` never produces a dual physical space, but the physical leg enters
# the `Z` factor, so it is the one sign term the geometry sweep cannot reach.
@testset "blocked ≡ pairwise (dual physical space)" begin
    for (sname, P, V) in _MSG_SPACES
                @testset "$sname" begin
            g = star_graph(5)
            state = _state_on(g, dual(P), V; seed = hash((sname, "dualP")))
            @test isdual(physicalspace(state, 1))
            msgs = BPMessages(state)
            Random.seed!(hash((sname, "dualPmsg")))
            for e in keys(msgs.messages)
                Random.randn!(msgs.messages[e])
            end
            for v in vertices(state)
                edges = collect(outgoing_edges(state, v))
                _, out_p = _cmp_batch(msgs, state, edges)
                for (i, e) in enumerate(edges)
                    @test out_p[i] ≈ compute_message(msgs, state, e)
                end
                # a non-trivial comparison, not two zero tensors
                @test norm(out_p[1]) > 0
            end
        end
    end
end

# `M = numind(state[v]) = N + 1` with `N` the *lattice* maximum coordination, so
# unused domain legs are `oneunit`-padded and still enter the axis ordering.
# `_MSG_GEOMETRIES` reaches `d < N` incidentally (chain ends, grid corners); this
# builds it on purpose, with two padded legs at every vertex.
@testset "blocked ≡ pairwise (oneunit-padded legs, d < N)" begin
    for (sname, P, V) in _MSG_SPACES
                @testset "$sname" begin
            g = path_graph(4)
            pspaces, vspaces = _spaces(g, P, V)
            state = TensorNetworkState{ComplexF64, typeof(P), 4}(undef, pspaces, vspaces)
            Random.seed!(hash((sname, "pad")))
            Random.randn!(state)
            msgs = BPMessages(state)
            Random.seed!(hash((sname, "padmsg")))
            for e in keys(msgs.messages)
                Random.randn!(msgs.messages[e])
            end
            for v in vertices(state)
                d = length(neighbors(state, v))
                @test numin(state[v]) == 4 && d < 4       # genuinely padded
                @test all(i -> isunitspace(domain(state[v])[i]), (d + 1):4)
                edges = collect(outgoing_edges(state, v))
                _, out_p = _cmp_batch(msgs, state, edges)
                for (i, e) in enumerate(edges)
                    @test out_p[i] ≈ compute_message(msgs, state, e)
                end
                @test norm(out_p[1]) > 0
            end
        end
    end
end

# What the selection rule still rejects, and how.
#
# There is nothing left of it: the abelian, `Trivial`, single-physical-leg and
# `Array`-storage conditions were all removed, and the predicate with them, so the
# blocked formulation now runs for every input. Three of the four were retired by
# *measuring* an argument that had only been asserted (`benchmark/reports/backend_ab.md`),
# and in each case the assertion here was **inverted rather than deleted**.
#
# What replaces the predicate is the requirement below, which is a different kind of thing:
# a correctness precondition of belief propagation rather than a choice between kernels.
@testset "belief propagation requires symmetric braiding" begin
    # This is the testset that matters most in this file after the non-Hermitian one,
    # because it is the only place where removing a restriction would have produced
    # *silently wrong numbers* rather than an error. Every primitive the blocked chain
    # uses succeeds under anyonic braiding — `permute`/`tensoradd!` and `mul!` both work
    # on `Vect[FibonacciAnyon]` — but the fermionic-sign derivation folds twists using
    # `twist(σ)² = 1`, which is false there. So the kernel would run to completion.
    #
    # The three assertions before the `@test_throws` are what make this non-vacuous: they
    # pin down that nothing *else* rejects this input, so `_require_symmetric_braiding` is
    # load-bearing. If a future change makes any of them false, this testset explains why
    # the guard exists instead of just failing.
    P = Vect[FibonacciAnyon](:I => 1, :τ => 1)
    V = Vect[FibonacciAnyon](:I => 2, :τ => 1)
    @test !(BraidingStyle(sectortype(P)) isa SymmetricBraiding)
    @test any(c -> !isone(twist(c)^2), sectors(V))
    state = _state_on(star_graph(4), P, V; seed = hash("fib"))   # constructs fine
    msgs = BPMessages(state)
    edges = collect(outgoing_edges(state, 1))
    @test_throws SectorMismatch compute_message(msgs, state, edges)
    @test_throws SectorMismatch compute_message(
        msgs, state, edges, DefaultBackend(), DefaultAllocator()
    )
    # the single-edge kernel reaches it through TensorKit's own refusal
    @test_throws Exception compute_message(msgs, state, first(edges))
end

@testset "blocked: Bumper ≡ default allocator + hygiene" begin
    for (sname, P, V) in _MSG_SPACES
        @testset "$sname" begin
            state = _state_on(star_graph(5), P, V; seed = hash((sname, "alloc")))
            msgs = belief_propagation(
                BPMessages(state), state; maxiter = 5, tol = 0, schedule = SynchronousSchedule()
            )
            buf = Bumper.default_buffer(Bumper.ResizeBuffer)
            for v in vertices(state)
                edges = collect(outgoing_edges(state, v))
                o_def = compute_message(msgs, state, edges, DefaultBackend(), DefaultAllocator())

                # `buffer_stats(...).peak` is a *process-lifetime* high-water mark:
                # reset first, or a previous fixture's peak gets reported here.
                Bumper.reset_buffer!(buf)
                o_bmp = compute_message(msgs, state, edges, DefaultBackend(), buf)
                st_blocked = buffer_stats(buf)
                @test buffer_isempty(buf)
                @test st_blocked.noverflow == 0
                for i in eachindex(o_def)
                    @test o_def[i] ≈ o_bmp[i]
                end

                Bumper.reset_buffer!(buf)
                compute_message(msgs, state, edges, PairwiseBackend(), buf)
                st_pairwise = buffer_stats(buf)
                @test buffer_isempty(buf)
                # the blocked chain holds one temporary per step instead of two
                @test st_blocked.peak ≤ 1.2 * st_pairwise.peak
            end
            Bumper.reset_buffer!(buf)
        end
    end
end

# `BeliefPropagation` stores *one* backend that every kernel sees, so the remaining
# selector must be transparent to every kernel that has no pairwise variant.
@testset "PairwiseBackend is transparent to other kernels" begin
    P = fermion_space(U1Irrep)
    V = Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 2, (1, 1) => 1, (1, -1) => 1)
    g = path_graph(4)
    state = _state_on(g, P, V; seed = 11)
    msgs = belief_propagation(
        BPMessages(state), state; maxiter = 8, tol = 0, schedule = SynchronousSchedule()
    )
    n = f_num(ComplexF64, U1Irrep)

    for v in vertices(state)
        ρ_d = reduced_density_matrix((v,), state, msgs; backend = DefaultBackend())
        ρ_b = reduced_density_matrix((v,), state, msgs; backend = DefaultBackend())
        @test space(ρ_b) == space(ρ_d)
        @test ρ_b ≈ ρ_d
        @test expect(state, msgs, n, v; backend = DefaultBackend()) ≈
            expect(state, msgs, n, v; backend = DefaultBackend())
    end
    for e in edges(state)
        ρ_d = reduced_density_matrix(
            (first(e), last(e)), state, msgs; backend = DefaultBackend()
        )
        ρ_b = reduced_density_matrix(
            (first(e), last(e)), state, msgs; backend = DefaultBackend()
        )
        @test ρ_b ≈ ρ_d
    end

    # `apply!` mutates, so build the same fixture twice rather than copying.
    gate = LocalGate((1, 2), exp(-0.05 * f_num(ComplexF64, U1Irrep) ⊗ f_num(ComplexF64, U1Irrep)))
    st_d = _state_on(g, P, V; seed = 11)
    ms_d = belief_propagation(
        BPMessages(st_d), st_d; maxiter = 8, tol = 0, schedule = SynchronousSchedule()
    )
    st_b = _state_on(g, P, V; seed = 11)
    ms_b = belief_propagation(
        BPMessages(st_b), st_b; maxiter = 8, tol = 0, schedule = SynchronousSchedule()
    )
    apply!(st_d, ms_d, gate; trunc = notrunc(), backend = DefaultBackend())
    apply!(st_b, ms_b, gate; trunc = notrunc(), backend = DefaultBackend())
    for v in vertices(st_d)
        @test space(st_b[v]) == space(st_d[v])
        @test st_b[v] ≈ st_d[v]
    end
    for e in keys(ms_d.messages)
        @test ms_b[e] ≈ ms_d[e]
    end
end

# --- which formulation actually runs -------------------------------------------
#
# An ordinary backend (`DefaultBackend()`, and whatever else reaches the kernel) runs the
# blocked formulation unconditionally; `PairwiseBackend` forces the oracle.
#
# Testing that by comparing *results* would be worthless twice over. The two formulations
# are supposed to agree, so agreement proves nothing about which ran; and on these fixtures
# they frequently agree **bitwise**, so not even `===` on the output data discriminates.
# What does discriminate is the sequence of temporaries: the pairwise chain repartitions the
# on-site tensor and closes with a `tensorcontract!` that needs a `copyC` buffer for the
# `((2,),(1,))` repartition of `out`, while the blocked chain transposes the (χ²) message
# and closes straight into `out` with `mul!`. Logging every `TO.tensoralloc` through a
# custom allocator therefore fingerprints the path — measured: 20 vs 24 allocations at a
# degree-4 `fℤ₂ ⊠ U1Irrep` vertex, 2 vs 3 at a leaf.
mutable struct LoggingAllocator
    log::Vector{Any}
end
LoggingAllocator() = LoggingAllocator(Any[])

function TensorKit.TO.tensoralloc(
        ::Type{A}, structure, v::Val{istemp}, a::LoggingAllocator
    ) where {A <: AbstractArray, istemp}
    push!(a.log, (A, structure, istemp))
    return TensorKit.TO.tensoralloc(A, structure, v)
end

function _path_fingerprint(msgs, state, edges, backend)
    a = LoggingAllocator()
    out = map(e -> similar(msgs[e]), edges)
    compute_message!(out, msgs, state, edges, backend, a)
    return a.log
end

@testset "the default path is the blocked kernel, not the pairwise one" begin
    # With the selection rule gone this is structural rather than conditional: there is
    # one production path and it must not look like the pairwise formulation. The test is
    # kept — rather than deleted as tautological — because it is the guard against a
    # fallback being reintroduced by accident, which a result comparison could never catch
    # (the two kernels agree, and on these fixtures often agree bitwise).
    #
    # WHERE THE FINGERPRINT IS BLIND, AND WHY THAT IS ASSERTED RATHER THAN SKIPPED.
    # It discriminates only when the two formulations issue different allocations, and at a
    # **degree-1** vertex with a **`Trivial`** sector type they do not: neither chain takes
    # a step, so both reduce to two entry copies plus a closing, and `has_array_view` means
    # the pairwise closing does not need the `copyC` buffer that separates them for a graded
    # sector. Both then allocate exactly two `Vector{ComplexF64}` of length 6.
    #
    # `discriminates` is therefore compared against an expectation, keeping this two-sided:
    # if the fingerprint goes blind anywhere else — or stops being blind here — the testset
    # fails and says so. Skipping the check for `Trivial` instead would let a future change
    # that made it degenerate everywhere pass silently, which is the failure mode this
    # assertion exists to prevent.
    g = star_graph(5)     # degree 4 at the centre, degree 1 at the leaves
    for (sname, P, V) in _MSG_SPACES
        @testset "$sname" begin
            state = _state_on(g, P, V; seed = hash((sname, "sel")))
            msgs = BPMessages(state)
            Random.seed!(hash((sname, "selmsg")))
            for e in keys(msgs.messages)
                Random.randn!(msgs.messages[e])
            end
            for v in vertices(state)
                edges = collect(outgoing_edges(state, v))
                fp_default = _path_fingerprint(msgs, state, edges, DefaultBackend())
                fp_pairwise = _path_fingerprint(msgs, state, edges, PairwiseBackend())
                expect_discriminates =
                    !(sectortype(P) === Trivial && length(neighbors(state, v)) == 1)
                @test (fp_default != fp_pairwise) == expect_discriminates
            end
        end
    end
end

# End to end: the default (vertex-batched) schedule, so the blocked kernel is
# genuinely on the path, at fixed `maxiter` and a fixed schedule RNG.
@testset "belief_propagation blocked ≡ pairwise" begin
    for (sname, P, V) in _MSG_SPACES
        @testset "$sname" begin
            state = _state_on(grid([3, 3]), P, V; seed = hash((sname, "bp")))
            m_p = belief_propagation(
                BPMessages(state), state; maxiter = 6, tol = 0,
                schedule = SpanningTreeSchedule(), backend = PairwiseBackend(),
            )
            m_b = belief_propagation(
                BPMessages(state), state; maxiter = 6, tol = 0,
                schedule = SpanningTreeSchedule(), backend = DefaultBackend(),
            )
            # No `backend` at all: the whole default route, selection included.
            m_d = belief_propagation(
                BPMessages(state), state; maxiter = 6, tol = 0,
                schedule = SpanningTreeSchedule(),
            )
            for e in keys(m_p.messages)
                @test space(m_b[e]) == space(m_p[e])
                @test m_b[e] ≈ m_p[e]
                # bitwise, not `≈`: the default must be the blocked kernel, not merely a
                # numerically equivalent one. `m_b` is itself `DefaultBackend`, so this is
                # a reproducibility check on top of the differential one against `m_p`.
                @test m_d[e].data == m_b[e].data
            end
        end
    end
end
