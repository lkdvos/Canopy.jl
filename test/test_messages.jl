using Canopy
using Canopy: compute_message, compute_message!, DirectedEdge, belief_propagation,
              outgoing_edges, buffer_isempty
using TensorKit
using TensorKit.TO: DefaultAllocator, DefaultBackend
using TensorKitTensors.FermionOperators: fermion_space
import Bumper
using Graphs
using Graphs: path_graph, cycle_graph, star_graph, complete_graph, grid
using Random
using Test

# The vector form of `compute_message` computes several messages out of one shared
# vertex at once; it must agree, message by message, with the single-edge
# `compute_message` kernel on the raw (pre-normalize) output, so any twist/sign
# discrepancy surfaces directly. The fermionic case is the decisive one (it
# exercises `_mul_leg!`'s `twist`); U(1) checks graded blocking; bosonic checks
# the plain index plumbing.

_state_on(g, P, V; seed) = (Random.seed!(seed); randn_state(ComplexF64, g, P, V))

# (physical, virtual) space pairs spanning trivial / U(1) / fermionic symmetry.
# The last row is the *production* symmetry (`scripts/hubbard_quench`,
# `examples/realtime`): graded fermion parity times U(1) charge, where the blocks
# are many and small rather than few and large.
const _MSG_SPACES = (
    ("bosonic",   ComplexSpace(2),                        ComplexSpace(3)),
    ("U(1)",      Vect[U1Irrep](0 => 1, 1 => 1),          Vect[U1Irrep](-1 => 1, 0 => 2, 1 => 1)),
    ("fermionic", fermion_space(Trivial),                 Vect[fℤ₂](0 => 2, 1 => 2)),
    (
        "fZ2xU1", fermion_space(U1Irrep),
        Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 2, (1, 1) => 1, (1, -1) => 1, (0, 2) => 1),
    ),
)

# Geometries spanning coordination numbers 1..5 and dual/non-dual leg mixes.
const _MSG_GEOMETRIES = (
    ("chain L=4",  path_graph(4)),       # degrees 1,2
    ("cycle L=6",  cycle_graph(6)),      # degree 2
    ("star deg 4", star_graph(5)),       # central degree 4
    ("3x3 grid",   grid([3, 3])),        # degrees 2,3,4
    ("K5",         complete_graph(5)),   # degree 4, dense
    ("K6",         complete_graph(6)),   # degree 5
)

@testset "compute_message (all outgoing) ≡ per-edge compute_message" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in _MSG_GEOMETRIES
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname)))
            msgs = belief_propagation(
                BPMessages(state), state; maxiter = 5, tol = 0, schedule = SynchronousSchedule()
            )
            for v in vertices(state)
                edges = collect(outgoing_edges(state, v))
                out = compute_message(msgs, state, edges)
                @test length(out) == length(edges)
                for (i, e) in enumerate(edges)
                    ref = compute_message(msgs, state, e)
                    @test space(out[i]) == space(ref)
                    @test out[i] ≈ ref
                end
            end
        end
    end
end

# Subsets of outgoing edges, in non-neighbor order, must (a) match the single-edge
# kernel and (b) be returned in the *input* order. The reordered-full-span subset
# checks ordering; the clustered middle subset exercises the partial prefix/suffix
# chains (kmin > 1 and kmax < degree).
@testset "compute_message (subset, reordered) ≡ per-edge" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in (("star deg 4", star_graph(5)), ("K6", complete_graph(6)))
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "sub")))
            msgs = belief_propagation(
                BPMessages(state), state; maxiter = 5, tol = 0, schedule = SynchronousSchedule()
            )
            for v in vertices(state)
                all_e = collect(outgoing_edges(state, v))
                length(all_e) < 3 && continue
                subsets = (
                    [all_e[end]; all_e[1:2:(end - 1)]],   # reordered, full span
                    reverse(all_e[2:(end - 1)]),          # clustered middle, reversed
                )
                for edges in subsets
                    out = compute_message(msgs, state, edges)
                    @test length(out) == length(edges)
                    for (i, e) in enumerate(edges)
                        @test out[i] ≈ compute_message(msgs, state, e)
                    end
                end
            end
        end
    end
end

# Guards the shared kernel against the naive per-edge oracle directly, independent
# of trusting a single `compute_message` call.
@testset "compute_message (all outgoing) ≡ naive oracle" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in (("star deg 4", star_graph(5)), ("K6", complete_graph(6)))
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "naive")))
            msgs = belief_propagation(
                BPMessages(state), state; maxiter = 5, tol = 0, schedule = SynchronousSchedule()
            )
            for v in vertices(state)
                edges = collect(outgoing_edges(state, v))
                opt = compute_message(msgs, state, edges)
                naive = compute_message(msgs, state, edges)  # fresh output vector
                # Reference per-edge loop over the single-edge kernel; the golden oracle.
                for (i, e) in enumerate(edges)
                    compute_message!(naive[i], msgs, state, e, DefaultBackend(), DefaultAllocator())
                end
                for i in eachindex(opt)
                    @test opt[i] ≈ naive[i]
                end
            end
        end
    end
end

# BP messages are hermitian after projection (so m† = m), which would mask a
# conjugation error in the suffix chain's adjoint absorption. Use generic
# *non-hermitian* messages to pin that the shared kernel matches compute_message
# for arbitrary inputs.
@testset "compute_message (vector) matches compute_message (non-hermitian msgs)" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in (("star deg 4", star_graph(5)), ("K6", complete_graph(6)))
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "nh")))
            msgs = BPMessages(state)
            Random.seed!(hash((sname, gname, "nhmsg")))
            for e in keys(msgs.messages)
                Random.randn!(msgs.messages[e])   # generic, non-hermitian
            end
            for v in vertices(state)
                edges = collect(outgoing_edges(state, v))
                out = compute_message(msgs, state, edges)
                for (i, e) in enumerate(edges)
                    @test out[i] ≈ compute_message(msgs, state, e)
                end
            end
        end
    end
end

# Shared source restriction.
@testset "compute_message (vector) rejects mismatched source" begin
    state = _state_on(complete_graph(5), ComplexSpace(2), ComplexSpace(3); seed = 7)
    msgs = belief_propagation(
        BPMessages(state), state; maxiter = 3, tol = 0, schedule = SynchronousSchedule()
    )
    @test_throws ArgumentError compute_message(msgs, state, [DirectedEdge(1, 2), DirectedEdge(3, 4)])
end

@testset "compute_message (vector): Bumper ≡ default + allocator hygiene" begin
    for (sname, P, V) in _MSG_SPACES
        state = _state_on(star_graph(5), P, V; seed = hash((sname, "alloc")))
        msgs = belief_propagation(
            BPMessages(state), state; maxiter = 5, tol = 0, schedule = SynchronousSchedule()
        )
        for v in vertices(state)
            edges = collect(outgoing_edges(state, v))
            o_def = compute_message(msgs, state, edges, DefaultBackend(), DefaultAllocator())
            buf = Bumper.default_buffer(Bumper.ResizeBuffer)
            o_bmp = compute_message(msgs, state, edges, DefaultBackend(), buf)
            @test buffer_isempty(buf)
            for i in eachindex(o_def)
                @test o_def[i] ≈ o_bmp[i]
            end
        end
    end
end
