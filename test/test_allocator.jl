using Canopy
using Canopy: BPMessages, belief_propagation, compute_message,
              LocalGate, LeftGate, RightGate, SandwichGate, apply!, physicalspace
using TensorKit
using TensorKit.TO: DefaultAllocator, DefaultBackend
using TensorKitTensors.FermionOperators: fermion_space
using MatrixAlgebraKit: notrunc
import Bumper
using Graphs: grid, edges, src, dst
using Random
using Test

# The Bumper bump allocator only changes *where* contraction intermediates are
# stored; results must be identical to plain heap allocation. These tests pin
# that equivalence for the BP message kernel, a full BP solve, and a two-site
# simple-update gate, including a fermionic case that exercises the dual-leg /
# `twist!` path in the gauge factorization.

# (physical space, virtual space) pairs; fermionic covers dual virtual legs.
const _SPACES = (
    ("bosonic", ComplexSpace(2), ComplexSpace(4)),
    ("fermionic", fermion_space(Trivial), Vect[fℤ₂](0 => 1, 1 => 1)),
)

_state_on(g, P, V; seed) = (
    Random.seed!(seed); s = randn_state(ComplexF64, g, P, V); (s, BPMessages(s))
)

# A representative two-site unitary from a random Hermitian generator.
function _unitary_gate(P; seed)
    Random.seed!(seed)
    h = randn(ComplexF64, P ⊗ P, P ⊗ P)
    return exp(-im * 0.1 * (h + h') / 2)
end

@testset "compute_message: Bumper ≡ default ($sname)" for (sname, P, V) in _SPACES
    state, _ = _state_on(grid([3, 3]), P, V; seed = 0)
    msgs = belief_propagation(BPMessages(state), state; maxiter = 5, tol = 0)
    for edge in keys(msgs.messages)
        m_def = compute_message(msgs, state, edge, DefaultBackend(), DefaultAllocator())
        m_bmp = compute_message(msgs, state, edge, DefaultBackend(), Bumper.default_buffer(Bumper.ResizeBuffer))
        @test space(m_def) == space(m_bmp)
        @test m_def ≈ m_bmp
    end
end

@testset "belief_propagation: Bumper ≡ default ($sname)" for (sname, P, V) in _SPACES
    state, _ = _state_on(grid([3, 3]), P, V; seed = 1)
    r_def = belief_propagation(BPMessages(state), state; maxiter = 8, tol = 0, allocator = DefaultAllocator())
    r_bmp = belief_propagation(BPMessages(state), state; maxiter = 8, tol = 0)  # Bumper default
    for de in keys(r_def.messages)
        @test r_def[de] ≈ r_bmp[de]
    end
end

@testset "apply! 2-site: Bumper ≡ default ($sname)" for (sname, P, V) in _SPACES
    g = grid([2, 3])
    e = first(edges(g))
    u, v = src(e), dst(e)
    gate = LocalGate((u, v), _unitary_gate(P; seed = 7))

    s_def, m_def = _state_on(g, P, V; seed = 2)
    s_bmp, m_bmp = _state_on(g, P, V; seed = 2)
    _, _, info_def = apply!(s_def, m_def, gate; trunc = notrunc(), allocator = DefaultAllocator())
    _, _, info_bmp = apply!(s_bmp, m_bmp, gate; trunc = notrunc())  # Bumper default

    for w in (u, v)
        @test space(s_def[w]) == space(s_bmp[w])
        @test s_def[w] ≈ s_bmp[w]
    end
    for de in keys(m_def.messages)
        @test m_def[de] ≈ m_bmp[de]
    end
    @test info_def.ϵ ≈ info_bmp.ϵ
    @test info_def.logλ ≈ info_bmp.logλ
end

# The operator path adds a second physical leg to every tensor, so `_absorb_legs`' bump
# allocation and the `θ` contraction see different shapes; `SandwichGate` in particular puts
# two physical legs in the `R` factor. Pin the same equivalence there. Bosonic only — gate
# application on an operator refuses fermionic sectors for now.
_operator_on(g, P, V; seed) = (
    Random.seed!(seed); o = randn_operator(ComplexF64, g, P, V); (o, BPMessages(o))
)

@testset "apply! 2-site operator: Bumper ≡ default ($(nameof(W)))" for
        W in (LeftGate, RightGate, SandwichGate)
    P, V = ComplexSpace(2), ComplexSpace(4)
    g = grid([2, 3])
    e = first(edges(g))
    u, v = src(e), dst(e)
    gate = W(LocalGate((u, v), _unitary_gate(P; seed = 7)))

    o_def, m_def = _operator_on(g, P, V; seed = 3)
    o_bmp, m_bmp = _operator_on(g, P, V; seed = 3)
    _, _, info_def = apply!(o_def, m_def, gate; trunc = notrunc(), allocator = DefaultAllocator())
    _, _, info_bmp = apply!(o_bmp, m_bmp, gate; trunc = notrunc())  # Bumper default

    for w in (u, v)
        @test space(o_def[w]) == space(o_bmp[w])
        @test o_def[w] ≈ o_bmp[w]
    end
    for de in keys(m_def.messages)
        @test m_def[de] ≈ m_bmp[de]
    end
    @test info_def.ϵ ≈ info_bmp.ϵ
    @test info_def.logλ ≈ info_bmp.logλ
    @test Canopy.buffer_isempty(Bumper.default_buffer(Bumper.ResizeBuffer))
end
