using Canopy
using Canopy: UndirectedEdge, vertices, _fold, _gate_tensor, _lower_groups, _support
using TensorKit
using LinearAlgebra: I, dot, kron, norm
using MatrixAlgebraKit: truncrank
using Test

const P = ComplexSpace(2)

_zero_state(edges) = product_state(ComplexF64, edges, P, Trivial() => [1.0, 0.0])

# Dense reference
# ---------------
# `TensorMap(state)` puts the physical leg of the `i`-th vertex of
# `vertices(state)` on codomain leg `i`, and that iteration order is *not*
# necessarily sorted (a 4-ring iterates as 1, 2, 4, 3). `_tn_array` therefore
# permutes the legs back into sorted vertex order, after which qubit `q` occupies
# bit `q - 1` of the flattened vector — the little-endian convention the
# embeddings below use, pinned down by the "reference embedding" testset.

_eye(n) = Matrix{ComplexF64}(I, n, n)
_embed1(u, n, q) = kron(_eye(2^(n - q)), u, _eye(2^(q - 1)))

# Two-site gates: `u` is the 4x4 reshape of the gate tensor, whose row index is
# `slot1 + 2 * slot2`, and `a`/`b` are the qubits of slot 1 and slot 2.
function _embed2(u, n, a, b)
    d = 2^n
    out = zeros(ComplexF64, d, d)
    for i in 0:(d - 1)
        ia, ib = (i >> (a - 1)) & 1, (i >> (b - 1)) & 1
        for ja in 0:1, jb in 0:1
            amp = u[ia + 2ib + 1, ja + 2jb + 1]
            iszero(amp) && continue
            j = (i & ~(1 << (a - 1))) | (ja << (a - 1))
            j = (j & ~(1 << (b - 1))) | (jb << (b - 1))
            out[i + 1, j + 1] += amp
        end
    end
    return out
end

_matrix(g) = (k = length(g.qubits); reshape(convert(Array, _gate_tensor(ComplexF64, g)), 2^k, 2^k))

# |0…0⟩ evolved by the circuit's primitives, gate by gate.
function _dense(qc::QASMCircuit)
    n = qc.nqubits
    ψ = zeros(ComplexF64, 2^n)
    ψ[1] = 1
    for stmt in qc.statements, g in stmt.gates
        u = _matrix(g)
        M = length(g.qubits) == 1 ? _embed1(u, n, g.qubits[1]) :
            _embed2(u, n, g.qubits[1], g.qubits[2])
        ψ = M * ψ
    end
    return ψ
end

function _tn_array(state)
    A = convert(Array, TensorMap(state))
    # leg `i` belongs to vertex `order[i]`; reorder so leg `v` belongs to vertex `v`
    return permutedims(A, sortperm(collect(vertices(state))))
end

_tn_vector(state) = reshape(_tn_array(state), :)

function _run(src; layered = false, Dmax = 32, edges = nothing, qubitmap = nothing)
    qc = parse_qasm(src)
    es = isnothing(edges) ? qasm_lattice(qc) : edges
    state = _zero_state(es)
    msgs = BPMessages(state)
    circuit = Circuit(qc, state; layered, qubitmap)
    state, msgs, info = apply!(state, msgs, circuit; trunc = truncrank(Dmax))
    return qc, state, info
end

# Overall normalization is factored out into `logλ`, so compare directions.
function _fidelity(qc, state)
    ψ, φ = _tn_vector(state), _dense(qc)
    return abs(dot(φ, ψ)) / (norm(φ) * norm(ψ))
end

const GHZ4 = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
"""

@testset "reference embedding matches the leg order of TensorMap(state)" begin
    # |10⟩: qubit 1 excited, so the *first* index is 2
    _, state, _ = _run("OPENQASM 2.0;\nqreg q[2];\nx q[0];\ncz q[0], q[1];\n")
    A = _tn_array(state)
    @test A[2, 1] ≈ 1
    # cx with control on qubit 1 takes |10⟩ to |11⟩
    _, state, _ = _run("OPENQASM 2.0;\nqreg q[2];\nx q[0];\ncx q[0], q[1];\n")
    A = _tn_array(state)
    @test A[2, 2] ≈ 1
    # ...and the reference reproduces both
    for src in ("x q[0];\ncz q[0], q[1];\n", "x q[0];\ncx q[0], q[1];\n")
        qc, state, _ = _run("OPENQASM 2.0;\nqreg q[2];\n" * src)
        @test _fidelity(qc, state) ≈ 1
    end
end

@testset "parse_qasm flattens registers and unrolls broadcasts" begin
    qc = parse_qasm(
        """
        OPENQASM 2.0;
        qreg a[2];
        qreg b[3];
        creg c[2];
        h a;
        x b[2];
        cx a[1], b[0];
        """
    )
    @test qc.nqubits == 5
    @test qc.registers == [:a => 2, :b => 3]
    # `h a;` broadcasts over the two qubits of `a`, giving two statements
    @test length(qc.statements) == 4
    @test [only(s.gates).qubits for s in qc.statements] == [[1], [2], [5], [2, 3]]
    @test all(s -> only(s.gates).name === :h, qc.statements[1:2])
end

@testset "parse_qasm evaluates classical expressions" begin
    qc = parse_qasm(
        """
        OPENQASM 2.0;
        qreg q[1];
        rz(pi/2) q[0];
        rx(-sin(pi/4)) q[0];
        ry(2*pi - 1.5) q[0];
        u3(pi, 0, sqrt(4)) q[0];
        """
    )
    params = [only(s.gates).params for s in qc.statements]
    @test params[1] ≈ [π / 2]
    @test params[2] ≈ [-sin(π / 4)]
    @test params[3] ≈ [2π - 1.5]
    @test params[4] ≈ [π, 0.0, 2.0]
end

@testset "parse_qasm inlines user gate declarations" begin
    qc = parse_qasm(
        """
        OPENQASM 2.0;
        qreg q[3];
        gate inner(t) a, b { rz(t) a; cx a, b; }
        gate outer(t) a, b, c { inner(t/2) a, b; inner(t) b, c; h c; }
        outer(pi) q[0], q[1], q[2];
        """
    )
    stmt = only(qc.statements)
    @test stmt.name === :outer
    @test [(g.name, g.qubits) for g in stmt.gates] ==
        [(:rz, [1]), (:cx, [1, 2]), (:rz, [2]), (:cx, [2, 3]), (:h, [3])]
    @test stmt.gates[1].params ≈ [π / 2]
    @test stmt.gates[3].params ≈ [π]
    @test _support(stmt) == [1, 2, 3]
end

@testset "parse_qasm rejects what Canopy cannot represent" begin
    prelude = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\n"
    @test_throws ArgumentError parse_qasm(prelude * "measure q[0] -> c[0];\n")
    @test_throws ArgumentError parse_qasm(prelude * "reset q[0];\n")
    @test_throws ArgumentError parse_qasm(prelude * "if (c == 1) x q[0];\n")
    @test_throws ArgumentError parse_qasm(prelude * "opaque mygate a, b;\n")
    @test_throws ArgumentError parse_qasm("OPENQASM 3.0;\nqreg q[1];\n")
    @test_throws ArgumentError parse_qasm("OPENQASM 2.0;\ninclude \"other.inc\";\n")
    # unknown / malformed gate applications
    @test_throws ArgumentError parse_qasm(prelude * "c4x q[0], q[1];\n")
    @test_throws ArgumentError parse_qasm(prelude * "cx q[0], q[0];\n")
    @test_throws ArgumentError parse_qasm(prelude * "rz q[0];\n")
    @test_throws ArgumentError parse_qasm(prelude * "h q[0], q[1];\n")
    @test_throws ArgumentError parse_qasm(prelude * "x r[0];\n")
    @test_throws ArgumentError parse_qasm(prelude * "x q[5];\n")
    @test_throws ArgumentError parse_qasm("OPENQASM 2.0;\nqreg q[1];\nqreg q[2];\n")
    # `barrier` is accepted and carries no gates
    qc = parse_qasm(prelude * "barrier q;\n")
    @test only(qc.statements).barrier
    @test isempty(only(qc.statements).gates)
end

@testset "gate tensors match their reference matrices" begin
    _m(name, params = Float64[]) = _matrix(QASMGate(name, params, [1]))
    _m2(name, params = Float64[]) = _matrix(QASMGate(name, params, [1, 2]))

    h = 1 / sqrt(2)
    @test _m(:x) ≈ [0 1; 1 0]
    @test _m(:y) ≈ [0 -im; im 0]
    @test _m(:z) ≈ [1 0; 0 -1]
    @test _m(:h) ≈ [h h; h -h]
    @test _m(:s) ≈ [1 0; 0 im]
    @test _m(:sdg) ≈ [1 0; 0 -im]
    @test _m(:t) ≈ [1 0; 0 cis(π / 4)]
    @test _m(:tdg) ≈ [1 0; 0 cis(-π / 4)]
    @test _m(:id) ≈ _eye(2)
    @test _m(:u0, [1.0]) ≈ _eye(2)
    @test _m(:u1, [0.3]) ≈ [1 0; 0 cis(0.3)]

    # sx is a genuine square root of x, which Rx(π/2) is only up to a phase
    @test _m(:sx)^2 ≈ _m(:x)
    @test _m(:sx) * _m(:sxdg) ≈ _eye(2)
    @test _m(:sx) ≉ _m(:rx, [π / 2])
    @test cis(-π / 4) * _m(:sx) ≈ _m(:rx, [π / 2])

    # u3(θ, φ, λ) in the qelib1.inc / Qiskit convention
    θ, φ, λ = 0.7, -0.3, 1.1
    @test _m(:u3, [θ, φ, λ]) ≈ [
        cos(θ / 2) (-cis(λ)*sin(θ / 2));
        (cis(φ)*sin(θ / 2)) (cis(φ + λ)*cos(θ / 2))
    ]
    @test _m(:u2, [φ, λ]) ≈ _m(:u3, [π / 2, φ, λ])
    @test _m(:u3, [0, 0, λ]) ≈ _m(:u1, [λ])
    # and it equals Rz(φ) Ry(θ) Rz(λ) up to the usual global phase
    @test _m(:u3, [θ, φ, λ]) ≈
        cis((φ + λ) / 2) * _m(:rz, [φ]) * _m(:ry, [θ]) * _m(:rz, [λ])

    # two-qubit gates, with slot 1 the fastest-varying index
    @test _m2(:cx) ≈ [1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]
    @test _m2(:cz) ≈ [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
    @test _m2(:cp, [0.4]) ≈ [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 cis(0.4)]
    @test _m2(:swap) ≈ [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    # controlled gates are block-diagonal in the control
    for (name, params, u) in [
            (:crx, [0.4], _m(:rx, [0.4])), (:cry, [0.4], _m(:ry, [0.4])),
            (:crz, [0.4], _m(:rz, [0.4])), (:csx, Float64[], _m(:sx)),
            (:cu3, [θ, φ, λ], _m(:u3, [θ, φ, λ])),
            (:cu, [θ, φ, λ, 0.25], cis(0.25) * _m(:u3, [θ, φ, λ])),
        ]
        M = _m2(name, params)
        # control = slot 1 = fastest index, so the controlled block is the odd/even split
        @test M[[1, 3], [1, 3]] ≈ _eye(2)
        @test M[[2, 4], [2, 4]] ≈ u
        @test all(iszero, M[[1, 3], [2, 4]])
        @test all(iszero, M[[2, 4], [1, 3]])
    end
end

@testset "gate table agrees with the qelib1.inc definitions" begin
    # Each pair defines a builtin the long way round, through the expansion
    # machinery, and compares the folded two-site tensors.
    pairs = [
        ("crz(0.7) q[0], q[1];",
            "gate g(l) a, b { u1(l/2) b; cx a, b; u1(-l/2) b; cx a, b; }\ng(0.7) q[0], q[1];"),
        ("cu1(0.7) q[0], q[1];",
            "gate g(l) a, b { u1(l/2) a; cx a, b; u1(-l/2) b; cx a, b; u1(l/2) b; }\ng(0.7) q[0], q[1];"),
        ("cz q[0], q[1];", "gate g a, b { h b; cx a, b; h b; }\ng q[0], q[1];"),
        ("swap q[0], q[1];", "gate g a, b { cx a, b; cx b, a; cx a, b; }\ng q[0], q[1];"),
        ("cy q[0], q[1];", "gate g a, b { sdg b; cx a, b; s b; }\ng q[0], q[1];"),
        ("ch q[0], q[1];",
            "gate g a, b { h b; sdg b; cx a, b; h b; t b; cx a, b; t b; h b; s b; x b; s a; }\ng q[0], q[1];"),
        ("rzz(0.7) q[0], q[1];", "gate g(t) a, b { cx a, b; u1(t) b; cx a, b; }\ng(0.7) q[0], q[1];"),
    ]
    for (direct, expanded) in pairs
        head = "OPENQASM 2.0;\nqreg q[2];\n"
        a = only(_lower_groups(only(parse_qasm(head * direct).statements)))
        b = only(_lower_groups(only(parse_qasm(head * expanded).statements)))
        ta = Canopy._fold(ComplexF64, a[1], a[2])
        tb = Canopy._fold(ComplexF64, b[1], b[2])
        # qelib1's constructions carry an arbitrary global phase
        ov = dot(convert(Array, ta)[:], convert(Array, tb)[:]) / 4
        @test abs(ov) ≈ 1
    end
end

@testset "qasm_lattice derives the circuit connectivity" begin
    @test qasm_lattice(parse_qasm(GHZ4)) ==
        [UndirectedEdge(1, 2), UndirectedEdge(2, 3), UndirectedEdge(3, 4)]
    # repeated couplings give one edge, and orientation is canonical
    qc = parse_qasm("OPENQASM 2.0;\nqreg q[2];\ncx q[0], q[1];\ncx q[1], q[0];\n")
    @test qasm_lattice(qc) == [UndirectedEdge(1, 2)]
    # a qubit nothing couples to cannot be placed on an edge-defined lattice
    @test_throws ArgumentError qasm_lattice(
        parse_qasm("OPENQASM 2.0;\nqreg q[3];\nh q[0];\ncx q[0], q[1];\n")
    )
end

@testset "lowering reproduces the dense reference" begin
    # GHZ is exactly representable at χ = 2
    qc, state, info = _run(GHZ4)
    @test info.ϵ ≈ 0 atol = 1.0e-12
    @test _fidelity(qc, state) ≈ 1
    A = _tn_array(state)
    A ./= norm(A)
    @test abs(A[1, 1, 1, 1]) ≈ 1 / sqrt(2)
    @test abs(A[2, 2, 2, 2]) ≈ 1 / sqrt(2)
    @test norm(A) ≈ 1

    # a single-qubit-heavy circuit, exercising u1/u2/u3/sx/sdg/tdg folding
    single = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    u3(0.3, -0.7, 1.2) q[0];
    sx q[1];
    tdg q[2];
    u2(0.4, -0.1) q[0];
    sdg q[1];
    u1(0.9) q[2];
    cx q[0], q[1];
    cx q[1], q[2];
    rx(0.5) q[0];
    """
    qc, state, info = _run(single)
    @test _fidelity(qc, state) ≈ 1

    # a brickwork chain with parametric two-qubit gates
    brick = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    h q;
    rzz(0.3) q[0], q[1];
    rzz(0.4) q[2], q[3];
    crz(0.5) q[1], q[2];
    rxx(0.6) q[0], q[1];
    cp(0.7) q[2], q[3];
    cu3(0.2, 0.3, 0.4) q[1], q[2];
    """
    qc, state, info = _run(brick)
    @test _fidelity(qc, state) ≈ 1

    # reversed control/target on the same edge must be permuted, not ignored
    rev = """
    OPENQASM 2.0;
    qreg q[2];
    h q[0];
    x q[1];
    cx q[1], q[0];
    """
    qc, state, _ = _run(rev)
    @test _fidelity(qc, state) ≈ 1

    # a loopy lattice (ring), where the network is no longer a tree
    ring = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    h q;
    cx q[0], q[1];
    cx q[1], q[2];
    cx q[2], q[3];
    cx q[3], q[0];
    rz(0.3) q[2];
    """
    qc, state, _ = _run(ring)
    @test length(qasm_lattice(qc)) == 4
    @test _fidelity(qc, state) ≈ 1
end

@testset "layered and flat lowerings agree" begin
    src = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    h q;
    cx q[0], q[1];
    cx q[2], q[3];
    barrier q;
    rz(0.3) q[1];
    cx q[1], q[2];
    t q[0];
    """
    qc = parse_qasm(src)
    state = _zero_state(qasm_lattice(qc))
    flat = Circuit(qc, state; layered = false)
    layered = Circuit(qc, state; layered = true)
    @test all(g -> g isa LocalGate, flat.gatelist)
    @test all(g -> g isa CompositeGate, layered.gatelist)
    # packing only groups gates, so nothing is lost
    @test sum(g -> length(g.gatelist), layered.gatelist) == length(flat.gatelist)
    @test length(layered.gatelist) < length(flat.gatelist)

    _, s1, _ = _run(src; layered = false)
    _, s2, _ = _run(src; layered = true)
    v1, v2 = _tn_vector(s1), _tn_vector(s2)
    @test abs(dot(v1, v2)) / (norm(v1) * norm(v2)) ≈ 1
    @test _fidelity(qc, s1) ≈ 1
    @test _fidelity(qc, s2) ≈ 1
end

@testset "statements are folded into as few gates as possible" begin
    head = "OPENQASM 2.0;\nqreg q[3];\n"
    _groups(src) = [g[1] for s in parse_qasm(head * src).statements for g in _lower_groups(s)]

    # a chain of single-qubit gates on one qubit collapses to one gate
    @test _groups("h q[0];\nt q[0];\ns q[0];\n") == [[1], [1], [1]]
    @test _groups("gate g a { h a; t a; s a; }\ng q[0];\n") == [[1]]
    # a two-qubit gate sandwiched by single-qubit ones is a single two-site gate
    @test _groups("gate g a, b { h a; cx a, b; t b; }\ng q[0], q[1];\n") == [[1, 2]]
    # ...but a body that never couples its qubits stays factorized, so it needs no edge
    @test _groups("gate g a, b { x a; y b; }\ng q[0], q[1];\n") == [[1], [2]]
    # three-qubit support falls back to emitting primitives one at a time
    @test _groups("gate g a, b, c { cx a, b; h c; cx b, c; }\ng q[0], q[1], q[2];\n") ==
        [[1, 2], [3], [2, 3]]

    # folding a custom two-qubit gate must not change the result
    src = head * "h q[0];\ngate g(t) a, b { rz(t) a; cx a, b; ry(t) b; cx b, a; }\ng(0.4) q[0], q[1];\ncx q[1], q[2];\n"
    qc, state, _ = _run(src)
    @test _fidelity(qc, state) ≈ 1
    @test length(Circuit(qc, _zero_state(qasm_lattice(qc))).gatelist) == 3
end

@testset "lowering validates its target state" begin
    qc = parse_qasm("OPENQASM 2.0;\nqreg q[3];\nh q[0];\ncx q[0], q[2];\ncx q[0], q[1];\n")
    chain = [UndirectedEdge(1, 2), UndirectedEdge(2, 3)]

    # this circuit is a star centred on qubit 1, so `cx q[0], q[2]` is not an
    # edge of the chain under the identity map
    @test_throws ArgumentError Circuit(qc, _zero_state(chain))
    # ...but it is once the centre is relabelled onto the chain's middle vertex
    @test Circuit(qc, _zero_state(chain); qubitmap = [2, 1, 3]) isa Circuit

    @test_throws ArgumentError Circuit(qc, _zero_state(chain); qubitmap = [1, 2])
    @test_throws ArgumentError Circuit(qc, _zero_state(chain); qubitmap = [1, 1, 2])
    @test_throws ArgumentError Circuit(qc, _zero_state(chain); qubitmap = [1, 2, 7])

    # a real state cannot carry complex gates
    real_state = product_state(Float64, chain, P, Trivial() => [1.0, 0.0])
    @test_throws ArgumentError Circuit(qc, real_state)

    # non-Int vertices need an explicit map
    sq = square_lattice(2, 2)
    @test_throws ArgumentError Circuit(qc, _zero_state(sq))
    @test Circuit(qc, _zero_state(sq); qubitmap = [(1, 1), (1, 2), (2, 1)]) isa Circuit

    # three-qubit gates have no application path
    ccx = parse_qasm("OPENQASM 2.0;\nqreg q[3];\ncx q[0], q[1];\ncx q[1], q[2];\nccx q[0], q[1], q[2];\n")
    @test_throws ArgumentError Circuit(ccx, _zero_state(chain))
end

@testset "read_qasm reads from a file" begin
    path = tempname() * ".qasm"
    write(path, GHZ4)
    try
        qc = read_qasm(path)
        @test qc.nqubits == 4
        @test length(qc) == length(parse_qasm(GHZ4))
        @test qasm_lattice(qc) == qasm_lattice(parse_qasm(GHZ4))
        @test occursin("QASMCircuit", sprint(show, MIME"text/plain"(), qc))
    finally
        rm(path; force = true)
    end
end
