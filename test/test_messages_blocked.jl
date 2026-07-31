using Canopy
using Canopy: compute_message, compute_message!, DirectedEdge, UndirectedEdge,
              belief_propagation, outgoing_edges, buffer_isempty, buffer_stats,
              reduced_density_matrix, LocalGate, apply!, _spaces
using TensorKit
using TensorKit.TO: DefaultAllocator, DefaultBackend
using TensorKitTensors.FermionOperators: fermion_space, f_num
import Bumper
using Dictionaries
using Graphs
using Graphs: path_graph, cycle_graph, star_graph, complete_graph, grid
using LinearAlgebra: norm
using MatrixAlgebraKit: notrunc
using Random
using Test

# `BlockedBackend` re-runs the vertex-batched message kernel in the `Layout(k)`
# formulation: one layout family, the target leg alone in the domain, fermionic
# signs folded into the (χ²) transposed messages instead of paid as `twist!`
# passes over the chain links. It must be *numerically identical* to the pairwise
# kernel, which stays the oracle — hence this file is one long differential test.
#
# Two things are deliberately asserted alongside every comparison:
#
#   * `space(blocked[i]) == space(pairwise[i])`. A kernel that silently drops a
#     zero-dimensional sector still compares `≈` but breaks `check_consistency`.
#   * `Canopy.uses_blocked_kernel(state[v])`, i.e. *which path actually ran*.
#     Without it a blocked kernel that always fell back would pass everything.

_state_on(g, P, V; seed) = (Random.seed!(seed); randn_state(ComplexF64, g, P, V))

# As `test/test_messages.jl`, plus the production symmetry as the last row.
const _MSG_SPACES = (
    ("bosonic", ComplexSpace(2), ComplexSpace(3)),
    ("U(1)", Vect[U1Irrep](0 => 1, 1 => 1), Vect[U1Irrep](-1 => 1, 0 => 2, 1 => 1)),
    ("fermionic", fermion_space(Trivial), Vect[fℤ₂](0 => 2, 1 => 2)),
    (
        "fZ2xU1", fermion_space(U1Irrep),
        Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 2, (1, 1) => 1, (1, -1) => 1, (0, 2) => 1),
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

# `Trivial` is excluded from the blocked path on purpose: the pairwise kernel
# short-circuits via `has_array_view` to plain-array TensorOperations and one
# large BLAS call, which the blocked formulation cannot beat.
_expect_blocked(P) = sectortype(P) !== Trivial

# Blocked vs pairwise on one vertex batch, returning both so the caller can chain
# further comparisons without recomputing.
function _cmp_batch(msgs, state, edges)
    out_b = compute_message(msgs, state, edges, BlockedBackend(), DefaultAllocator())
    out_p = compute_message(msgs, state, edges, PairwiseBackend(), DefaultAllocator())
    @test length(out_b) == length(edges)
    for i in eachindex(edges)
        @test space(out_b[i]) == space(out_p[i])
        @test out_b[i] ≈ out_p[i]
    end
    return out_b, out_p
end

@testset "blocked ≡ pairwise ≡ per-edge oracle" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in _MSG_GEOMETRIES
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname)))
            msgs = belief_propagation(
                BPMessages(state), state; maxiter = 5, tol = 0, schedule = SynchronousSchedule()
            )
            for v in vertices(state)
                @test Canopy.uses_blocked_kernel(state[v]) == _expect_blocked(P)
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
    for (sname, P, V) in _MSG_SPACES, (gname, g) in _MSG_GEOMETRIES
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
# the `extrema(target_legs)` partial chains (`kmin > 1` and `kmax < d`).
@testset "blocked ≡ pairwise (subset, reordered targets)" begin
    for (sname, P, V) in _MSG_SPACES,
            (gname, g) in (("star deg 4", star_graph(5)), ("K6", complete_graph(6)))

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
        _expect_blocked(P) || continue
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
                @test Canopy.uses_blocked_kernel(state[v])
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
        _expect_blocked(P) || continue
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

# The fallback must *fire* — asserting the predicate matters, because a silently
# taken blocked path would otherwise pass by accident.
@testset "fallback for unsupported sectors" begin
    fixtures = (
        ("SU(2)", Vect[SU2Irrep](0 => 1, 1 // 2 => 1), Vect[SU2Irrep](0 => 2, 1 // 2 => 1)),
        ("trivial", ComplexSpace(2), ComplexSpace(3)),
    )
    for (sname, P, V) in fixtures
        @testset "$sname" begin
            state = _state_on(star_graph(5), P, V; seed = hash((sname, "fb")))
            msgs = BPMessages(state)
            Random.seed!(hash((sname, "fbmsg")))
            for e in keys(msgs.messages)
                Random.randn!(msgs.messages[e])
            end
            for v in vertices(state)
                @test !Canopy.uses_blocked_kernel(state[v])
                edges = collect(outgoing_edges(state, v))
                _, out_p = _cmp_batch(msgs, state, edges)
                for (i, e) in enumerate(edges)
                    @test out_p[i] ≈ compute_message(msgs, state, e)
                end
            end
        end
    end
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
                o_def = compute_message(msgs, state, edges, BlockedBackend(), DefaultAllocator())

                # `buffer_stats(...).peak` is a *process-lifetime* high-water mark:
                # reset first, or a previous fixture's peak gets reported here.
                Bumper.reset_buffer!(buf)
                o_bmp = compute_message(msgs, state, edges, BlockedBackend(), buf)
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

# `BeliefPropagation` stores *one* backend that every kernel sees, so a selector
# must be transparent to the kernels that have no blocked variant.
@testset "BlockedBackend is transparent to other kernels" begin
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
        ρ_b = reduced_density_matrix((v,), state, msgs; backend = BlockedBackend())
        @test space(ρ_b) == space(ρ_d)
        @test ρ_b ≈ ρ_d
        @test expect(state, msgs, n, v; backend = BlockedBackend()) ≈
            expect(state, msgs, n, v; backend = DefaultBackend())
    end
    for e in edges(state)
        ρ_d = reduced_density_matrix(
            (first(e), last(e)), state, msgs; backend = DefaultBackend()
        )
        ρ_b = reduced_density_matrix(
            (first(e), last(e)), state, msgs; backend = BlockedBackend()
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
    apply!(st_b, ms_b, gate; trunc = notrunc(), backend = BlockedBackend())
    for v in vertices(st_d)
        @test space(st_b[v]) == space(st_d[v])
        @test st_b[v] ≈ st_d[v]
    end
    for e in keys(ms_d.messages)
        @test ms_b[e] ≈ ms_d[e]
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
                schedule = SpanningTreeSchedule(), backend = BlockedBackend(),
            )
            for e in keys(m_p.messages)
                @test space(m_b[e]) == space(m_p[e])
                @test m_b[e] ≈ m_p[e]
            end
        end
    end
end
