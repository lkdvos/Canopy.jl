using Canopy
using Canopy: compute_message, compute_message!, belief_propagation, outgoing_edges,
              buffer_isempty, buffer_stats, reduced_density_matrix, LocalGate, apply!,
              _spaces
using TensorKit
using TensorKit.TO: DefaultAllocator, DefaultBackend
using TensorKitTensors.FermionOperators: fermion_space, f_num
using TensorKitTensors.HubbardOperators: hubbard_space
import Bumper
using Graphs
using Graphs: path_graph, cycle_graph, star_graph, complete_graph, grid
using LinearAlgebra: norm
using MatrixAlgebraKit: notrunc
using Random
using Test

# The vertex-batched message kernel runs the `Layout(k)` formulation unconditionally, and
# must be numerically identical to the pairwise formulation, which is retained only as the
# oracle and reached only through `PairwiseBackend` — hence this file is one long
# differential test. `space(blocked[i]) == space(pairwise[i])` is asserted alongside every
# comparison: a kernel that silently drops a zero-dimensional sector still compares `≈` but
# breaks `check_consistency`.
#
# There is no selection *predicate* left to assert against. What replaces it is the
# `tensoralloc` fingerprint at the end of this file: the default path must not look like the
# pairwise one. That is the guard against a fallback being reintroduced by accident.

_state_on(g, P, V; seed) = (Random.seed!(seed); randn_state(ComplexF64, g, P, V))

# As `test/test_messages.jl`, plus the graded and non-abelian rows.
#
# The two `SU2Irrep` rows are the non-abelian ones. They belong here rather than in a
# fallback testset because the blocked kernel is not restricted to abelian fusion: every
# chain step is a `TensorMap`-level operation that handles a general fusion style on its
# own, and the sign derivation needs `SymmetricBraiding`, not `UniqueFusion`.
# `hubbard_space(Trivial, SU2Irrep)` is the spin-SU(2) Hubbard physical space; note it is
# not reachable from `scripts/hubbard_quench`, which rejects SU(2) because `product_state`
# handles abelian sectors only — so this file is the only place that path is exercised.
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

# Everything sweeps everything except the two non-abelian rows, which skip `K6`.
#
# MEASURED: compilation here is driven by distinct `(sector type, numind)` pairs, and `K6`
# is the sole degree-5 (`M = 6`) fixture, so it is the only geometry whose removal buys any
# compilation at all — dropping any other would save seconds and nothing else. Degree 5 is
# still swept for all four abelian rows, and the non-abelian rows still reach degrees 1-4
# including the padded `d < N` case. The blocked chain is generic in `d`; if a future change
# makes it arity-dependent, delete this and take the time.
_msg_geometries(sname) =
    sname in ("SU(2)", "fZ2xSU2") ?
    filter(g -> first(g) != "K6", collect(_MSG_GEOMETRIES)) : collect(_MSG_GEOMETRIES)

# Generic random (non-hermitian) messages.
#
# Non-hermitian is the decisive input and the default throughout this file. BP messages are
# hermitian after projection, and `m† = m` masks a conjugation error in `adjoint(msgt)` /
# `adjoint(bra[k])` — exactly where a chain rewrite goes wrong. Hermitian messages are a
# measure-zero special case of these, so sweeping them too would add CI time and no
# discriminating power; the converged regime is covered end to end at the bottom of the file.
function _random_messages(state; seed)
    msgs = BPMessages(state)
    Random.seed!(seed)
    for e in keys(msgs.messages)
        Random.randn!(msgs.messages[e])
    end
    return msgs
end

# Blocked vs pairwise vs the per-edge golden oracle, on one vertex batch.
function _cmp_batch(msgs, state, edges)
    out_b = compute_message(msgs, state, edges, DefaultBackend(), DefaultAllocator())
    out_p = compute_message(msgs, state, edges, PairwiseBackend(), DefaultAllocator())
    @test length(out_b) == length(edges)
    for (i, e) in enumerate(edges)
        ref = compute_message(msgs, state, e)
        @test space(out_b[i]) == space(out_p[i]) == space(ref)
        @test out_b[i] ≈ out_p[i]
        @test out_b[i] ≈ ref
    end
    @test norm(out_b[1]) > 0    # a non-trivial comparison, not two zero tensors
    return out_b
end

# ...over every vertex of `state`, batching all of its outgoing edges.
function _cmp_vertices(msgs, state)
    for v in vertices(state)
        _cmp_batch(msgs, state, collect(outgoing_edges(state, v)))
    end
end


@testset "blocked ≡ pairwise ≡ per-edge oracle" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in _msg_geometries(sname)
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname)))
            _cmp_vertices(_random_messages(state; seed = hash((sname, gname, "m"))), state)
        end
    end
end


# Reordered full span pins output ordering; the clustered middle subset exercises the
# `extrema(target_legs)` partial chains (`legmin > 1` and `legmax < d`). Needs degree ≥ 3,
# so it runs on the two densest geometries — filtered through `_msg_geometries`, or the
# non-abelian rows would pull in the `M = 6` compilation the sweep above skips.
@testset "blocked ≡ pairwise (subset, reordered targets)" begin
    for (sname, P, V) in _MSG_SPACES,
            (gname, g) in filter(
                r -> first(r) in ("star deg 4", "K6"), _msg_geometries(sname)
            )

        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "sub")))
            msgs = _random_messages(state; seed = hash((sname, gname, "subm")))
            for v in vertices(state)
                all_e = collect(outgoing_edges(state, v))
                length(all_e) < 3 && continue
                subsets = (
                    [all_e[end]; all_e[1:2:(end - 1)]],   # reordered, full span
                    reverse(all_e[2:(end - 1)]),          # clustered middle, reversed
                    [all_e[2], all_e[2]],                 # duplicate target
                )
                for edges in subsets
                    _cmp_batch(msgs, state, edges)
                end
            end
        end
    end
end


# `randn_state` never produces a dual physical space, but the physical leg enters the `Z`
# factor, so it is the one sign term the geometry sweep cannot reach.
@testset "blocked ≡ pairwise (dual physical space)" begin
    for (sname, P, V) in _MSG_SPACES
        @testset "$sname" begin
            state = _state_on(star_graph(5), dual(P), V; seed = hash((sname, "dualP")))
            @test isdual(physicalspace(state, 1))
            _cmp_vertices(_random_messages(state; seed = hash((sname, "dualPm"))), state)
        end
    end
end


# `M = numind(state[v]) = N + 1` with `N` the *lattice* maximum coordination, so unused
# domain legs are `oneunit`-padded and still enter the axis ordering. `_MSG_GEOMETRIES`
# reaches `d < N` incidentally (chain ends, grid corners); this builds it on purpose, with
# two padded legs at every vertex.
@testset "blocked ≡ pairwise (oneunit-padded legs, d < N)" begin
    for (sname, P, V) in _MSG_SPACES
        @testset "$sname" begin
            pspaces, vspaces = _spaces(path_graph(4), P, V)
            state = TensorNetworkState{ComplexF64, typeof(P), 4}(undef, pspaces, vspaces)
            Random.seed!(hash((sname, "pad")))
            Random.randn!(state)
            for v in vertices(state)
                d = length(neighbors(state, v))
                @test numin(state[v]) == 4 && d < 4       # genuinely padded
                @test all(i -> isunitspace(domain(state[v])[i]), (d + 1):4)
            end
            _cmp_vertices(_random_messages(state; seed = hash((sname, "padm"))), state)
        end
    end
end


# The only place where removing a restriction would produce *silently wrong numbers* rather
# than an error. Every primitive the blocked chain uses succeeds under anyonic braiding —
# `permute`/`tensoradd!` and `mul!` both work on `Vect[FibonacciAnyon]` — but the
# fermionic-sign derivation folds twists using `twist(σ)² = 1`, which is false there, so the
# kernel would run to completion.
#
# The three assertions before the `@test_throws` are what make this non-vacuous: they pin
# down that nothing *else* rejects this input, so `_require_symmetric_braiding` is
# load-bearing. If a future change makes any of them false, this explains why the guard
# exists instead of just failing.
@testset "belief propagation requires symmetric braiding" begin
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
            msgs = _random_messages(state; seed = hash((sname, "allocm")))
            buf = Bumper.default_buffer(Bumper.ResizeBuffer)
            for v in vertices(state)
                edges = collect(outgoing_edges(state, v))
                o_def = compute_message(msgs, state, edges, DefaultBackend(), DefaultAllocator())

                # `buffer_stats(...).peak` is a *process-lifetime* high-water mark: reset
                # first, or a previous fixture's peak gets reported here.
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


# `BeliefPropagation` stores *one* backend that every kernel sees, so the selector must be
# transparent to every kernel that has no pairwise variant — i.e. each must unwrap it via
# `inner_backend` rather than hand a `PairwiseBackend` to TensorOperations.
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
        ρ_p = reduced_density_matrix((v,), state, msgs; backend = PairwiseBackend())
        @test space(ρ_p) == space(ρ_d)
        @test ρ_p ≈ ρ_d
        @test expect(state, msgs, n, v; backend = PairwiseBackend()) ≈
            expect(state, msgs, n, v; backend = DefaultBackend())
    end
    for e in edges(state)
        sites = (first(e), last(e))
        @test reduced_density_matrix(sites, state, msgs; backend = PairwiseBackend()) ≈
            reduced_density_matrix(sites, state, msgs; backend = DefaultBackend())
    end

    # `apply!` mutates, so build the same fixture twice rather than copying.
    gate = LocalGate((1, 2), exp(-0.05 * f_num(ComplexF64, U1Irrep) ⊗ f_num(ComplexF64, U1Irrep)))
    fixture() = let st = _state_on(g, P, V; seed = 11)
        st, belief_propagation(
            BPMessages(st), st; maxiter = 8, tol = 0, schedule = SynchronousSchedule()
        )
    end
    st_d, ms_d = fixture()
    st_p, ms_p = fixture()
    apply!(st_d, ms_d, gate; trunc = notrunc(), backend = DefaultBackend())
    apply!(st_p, ms_p, gate; trunc = notrunc(), backend = PairwiseBackend())
    for v in vertices(st_d)
        @test space(st_p[v]) == space(st_d[v])
        @test st_p[v] ≈ st_d[v]
    end
    for e in keys(ms_d.messages)
        @test ms_p[e] ≈ ms_d[e]
    end
end


# --- which formulation actually runs -------------------------------------------
#
# Comparing *results* would be worthless twice over: the two formulations are supposed to
# agree, so agreement proves nothing about which ran, and on these fixtures they frequently
# agree **bitwise**, so not even `===` on the output data discriminates. What does
# discriminate is the sequence of temporaries — the pairwise chain repartitions the on-site
# tensor and closes with a `tensorcontract!` needing a `copyC` buffer for the `((2,),(1,))`
# repartition of `out`, while the blocked chain transposes the (χ²) message and closes
# straight into `out` with `mul!`. Logging every `TO.tensoralloc` therefore fingerprints the
# path: measured 20 vs 24 allocations at a degree-4 `fℤ₂ ⊠ U1Irrep` vertex, 2 vs 3 at a leaf.
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
    # WHERE THE FINGERPRINT IS BLIND, AND WHY THAT IS ASSERTED RATHER THAN SKIPPED. It
    # discriminates only when the two formulations issue different allocations, and at a
    # **degree-1** vertex with a **`Trivial`** sector type they do not: neither chain takes a
    # step, so both reduce to two entry copies plus a closing, and `has_array_view` means the
    # pairwise closing does not need the `copyC` buffer that separates them for a graded
    # sector. Both then allocate exactly two `Vector{ComplexF64}` of length 6.
    #
    # `discriminates` is therefore compared against an expectation, keeping this two-sided:
    # if the fingerprint goes blind anywhere else — or stops being blind here — this fails
    # and says so. Skipping the check for `Trivial` instead would let a future change that
    # made it degenerate everywhere pass silently.
    g = star_graph(5)     # degree 4 at the centre, degree 1 at the leaves
    for (sname, P, V) in _MSG_SPACES
        @testset "$sname" begin
            state = _state_on(g, P, V; seed = hash((sname, "sel")))
            msgs = _random_messages(state; seed = hash((sname, "selm")))
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


# End to end: the default (vertex-batched) schedule, so the blocked kernel is genuinely on
# the path, at fixed `maxiter` and a fixed schedule RNG. This is also where converged,
# hermitian messages are exercised — the regime the per-vertex sweeps above deliberately
# skip in favour of the strictly more general non-hermitian ones.
@testset "belief_propagation blocked ≡ pairwise" begin
    for (sname, P, V) in _MSG_SPACES
        @testset "$sname" begin
            state = _state_on(grid([3, 3]), P, V; seed = hash((sname, "bp")))
            bp(; kwargs...) = belief_propagation(
                BPMessages(state), state;
                maxiter = 6, tol = 0, schedule = SpanningTreeSchedule(), kwargs...,
            )
            m_p = bp(; backend = PairwiseBackend())
            m_b = bp(; backend = DefaultBackend())
            m_d = bp()      # no `backend` at all: the whole default route
            for e in keys(m_p.messages)
                @test space(m_b[e]) == space(m_p[e])
                @test m_b[e] ≈ m_p[e]
                # bitwise, not `≈`: the default must be the blocked kernel, not merely a
                # numerically equivalent one.
                @test m_d[e].data == m_b[e].data
            end
        end
    end
end
