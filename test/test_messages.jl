using Canopy
using Canopy: compute_outgoing_messages, compute_outgoing_messages!, compute_message,
              DirectedEdge, belief_propagation, _outgoing_naive!, buffer_isempty
using TensorKit
using TensorKit.TO: DefaultAllocator, DefaultBackend
using TensorKitTensors.FermionOperators: fermion_space
import Bumper
using Graphs
using Graphs: path_graph, cycle_graph, star_graph, complete_graph, grid
using Random
using Test

# `compute_outgoing_messages` computes every message out of a vertex at once; it
# must agree, message by message, with the single-edge `compute_message` kernel
# on the raw (pre-normalize) output, so any twist/sign discrepancy surfaces
# directly. The fermionic case is the decisive one (it exercises `_mul_leg!`'s
# `twist`); U(1) checks graded blocking; bosonic checks the plain index plumbing.

_state_on(g, P, V; seed) = (
    s = TensorNetworkState{ComplexF64}(undef, g, P, V);
    Random.seed!(seed); Random.randn!(s); s
)

# (physical, virtual) space pairs spanning trivial / U(1) / fermionic symmetry.
const _MSG_SPACES = (
    ("bosonic",   ComplexSpace(2),                        ComplexSpace(3)),
    ("U(1)",      Vect[U1Irrep](0 => 1, 1 => 1),          Vect[U1Irrep](-1 => 1, 0 => 2, 1 => 1)),
    ("fermionic", fermion_space(Trivial),                 Vect[fℤ₂](0 => 2, 1 => 2)),
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

@testset "compute_outgoing_messages ≡ per-edge compute_message" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in _MSG_GEOMETRIES
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname)))
            msgs = belief_propagation(BPMessages(state), state; maxiter = 5, tol = 0)
            for v in vertices(state)
                out = compute_outgoing_messages(msgs, state, v)
                nbrs = neighbors(state, v)
                @test length(out) == length(nbrs)
                for (k, n) in enumerate(nbrs)
                    ref = compute_message(msgs, state, DirectedEdge(v, n))
                    @test space(out[k]) == space(ref)
                    @test out[k] ≈ ref
                end
            end
        end
    end
end

# Guards the optimized path against the naive oracle directly, independent of
# `compute_message` (meaningful once the public kernel diverges from the loop).
@testset "compute_outgoing_messages ≡ naive oracle" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in (("star deg 4", star_graph(5)), ("K6", complete_graph(6)))
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "naive")))
            msgs = belief_propagation(BPMessages(state), state; maxiter = 5, tol = 0)
            for v in vertices(state)
                opt = compute_outgoing_messages(msgs, state, v)
                naive = compute_outgoing_messages(msgs, state, v)  # fresh output vector
                _outgoing_naive!(naive, msgs, state, v, DefaultBackend(), DefaultAllocator())
                for k in eachindex(opt)
                    @test opt[k] ≈ naive[k]
                end
            end
        end
    end
end

# BP messages are hermitian after projection (so m† = m), which would mask a
# conjugation error in the suffix chain's adjoint absorption. Use generic
# *non-hermitian* messages to pin that the double-layer matches compute_message
# for arbitrary inputs.
@testset "compute_outgoing_messages matches compute_message (non-hermitian msgs)" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in (("star deg 4", star_graph(5)), ("K6", complete_graph(6)))
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "nh")))
            msgs = BPMessages(state)
            Random.seed!(hash((sname, gname, "nhmsg")))
            for e in keys(msgs.messages)
                Random.randn!(msgs.messages[e])   # generic, non-hermitian
            end
            for v in vertices(state)
                out = compute_outgoing_messages(msgs, state, v)
                for (k, n) in enumerate(neighbors(state, v))
                    ref = compute_message(msgs, state, DirectedEdge(v, n))
                    @test out[k] ≈ ref
                end
            end
        end
    end
end

@testset "compute_outgoing_messages: Bumper ≡ default + allocator hygiene" begin
    for (sname, P, V) in _MSG_SPACES
        state = _state_on(star_graph(5), P, V; seed = hash((sname, "alloc")))
        msgs = belief_propagation(BPMessages(state), state; maxiter = 5, tol = 0)
        for v in vertices(state)
            o_def = compute_outgoing_messages(msgs, state, v, DefaultBackend(), DefaultAllocator())
            buf = Bumper.default_buffer(Bumper.ResizeBuffer)
            o_bmp = compute_outgoing_messages(msgs, state, v, DefaultBackend(), buf)
            @test buffer_isempty(buf)
            for k in eachindex(o_def)
                @test o_def[k] ≈ o_bmp[k]
            end
        end
    end
end
