# QASMCircuit -> Circuit
# ----------------------
# Turn the symbolic IR into `LocalGate`s on a `TensorNetworkState`. Two things
# constrain the translation: `apply!` only implements one- and two-site gates, and
# a two-site gate must act on an existing edge of the network. Statements
# supported on one or two qubits are therefore contracted into a single tensor,
# which both respects that limit and keeps the number of truncating updates down.

# --- gate tensors ------------------------------------------------------------
# Most gates come straight from `TensorKitTensors.QuantumGates`. The ones defined
# here are those it does not provide, chief among them the `U(θ, φ, λ)` primitive
# of the language itself.

"""
    _u3(T, θ, φ, λ)

The general single-qubit unitary of OpenQASM 2.0, `U(θ, φ, λ)`, in the convention
of `qelib1.inc`'s `u3` (and hence of Qiskit's `UGate`).
"""
function _u3(::Type{T}, θ::Real, φ::Real, λ::Real) where {T <: Complex}
    q = QG.qubit_space()
    s, c = sincos(θ / 2)
    return TensorMap(T[c (-cis(λ) * s); (cis(φ) * s) (cis(φ + λ) * c)], q ← q)
end

# The √X gate, `sx`. Equal to `Rx(π/2)` up to a global phase, which matters for
# amplitudes even though it does not for expectation values.
function _sx(::Type{T}) where {T <: Complex}
    q = QG.qubit_space()
    a, b = (1 + im) / 2, (1 - im) / 2
    return TensorMap(T[a b; b a], q ← q)
end

# Controlled version of a single-qubit gate, with the control on the first slot.
function _control(::Type{T}, u::AbstractTensorMap) where {T <: Complex}
    q = QG.qubit_space()
    return QG.proj_0(T) ⊗ id(T, q) + QG.proj_1(T) ⊗ u
end

# Tensor for one primitive; the name/arity table lives in `_QASM_PRIMITIVES`.
function _gate_tensor(::Type{T}, g::QASMGate) where {T <: Complex}
    q = QG.qubit_space()
    p = g.params
    n = g.name
    # `u0` is an idle instruction, whose argument is a duration
    (n === :id || n === :u0) && return id(T, q)
    n === :x && return QG.pauli_x(T)
    n === :y && return QG.pauli_y(T)
    n === :z && return QG.pauli_z(T)
    n === :h && return QG.hadamard(T)
    n === :s && return QG.s_gate(T)
    n === :sdg && return QG.s_gate(T)'
    n === :t && return QG.t_gate(T)
    n === :tdg && return QG.t_gate(T)'
    n === :sx && return _sx(T)
    n === :sxdg && return _sx(T)'
    n === :u1 && return QG.phase_shift(T; θ = p[1])
    n === :rx && return QG.rotation_x(T; θ = p[1])
    n === :ry && return QG.rotation_y(T; θ = p[1])
    n === :rz && return QG.rotation_z(T; θ = p[1])
    n === :u2 && return _u3(T, π / 2, p[1], p[2])
    n === :u3 && return _u3(T, p[1], p[2], p[3])
    n === :cx && return QG.cnot(T)
    n === :cy && return QG.cy(T)
    n === :cz && return QG.cz(T)
    n === :ch && return QG.ch(T)
    n === :cs && return QG.cs(T)
    n === :swap && return QG.swap(T)
    n === :iswap && return QG.iswap(T)
    n === :dcx && return QG.dcx(T)
    n === :ecr && return QG.ecr(T)
    n === :csx && return _control(T, _sx(T))
    n === :cp && return QG.cphase(T; θ = p[1])
    n === :crx && return _control(T, QG.rotation_x(T; θ = p[1]))
    n === :cry && return _control(T, QG.rotation_y(T; θ = p[1]))
    n === :crz && return _control(T, QG.rotation_z(T; θ = p[1]))
    n === :rxx && return QG.rotation_xx(T; θ = p[1])
    n === :ryy && return QG.rotation_yy(T; θ = p[1])
    n === :rzz && return QG.rotation_zz(T; θ = p[1])
    n === :rzx && return QG.rotation_zx(T; θ = p[1])
    n === :cu3 && return _control(T, _u3(T, p[1], p[2], p[3]))
    n === :cu && return _control(T, cis(p[4]) * _u3(T, p[1], p[2], p[3]))
    throw(ArgumentError(lazy"no gate tensor available for `$n`"))
end

# --- statement grouping ------------------------------------------------------

"""
    _lower_groups(stmt) -> Vector{Tuple{Vector{Int}, Vector{QASMGate}}}

Split a statement into the gates that will be emitted for it, each entry pairing
the qubits acted on with the primitives to be multiplied together on them.

A statement supported on one qubit, or on two with at least one genuinely
two-qubit primitive, collapses into a single gate. A statement whose support is
wider — a custom `gate` on three or more qubits — is emitted primitive by
primitive, as is one that merely happens to touch two qubits without coupling
them (`gate foo a, b { x a; y b; }`), which would otherwise demand an edge it
does not need.
"""
function _lower_groups(stmt::QASMStatement)
    groups = Tuple{Vector{Int}, Vector{QASMGate}}[]
    (stmt.barrier || isempty(stmt.gates)) && return groups
    for g in stmt.gates
        length(g.qubits) ≤ 2 || throw(
            ArgumentError(
                lazy"`$(g.name)` acts on $(length(g.qubits)) qubits, but `apply!` implements only one- and two-site gates; decompose it into one- and two-qubit gates first"
            )
        )
    end
    qs = _support(stmt)
    if length(qs) == 1 || (length(qs) == 2 && any(g -> length(g.qubits) == 2, stmt.gates))
        push!(groups, (qs, stmt.gates))
    elseif length(qs) == 2
        for v in qs
            push!(groups, ([v], filter(g -> g.qubits == [v], stmt.gates)))
        end
    else
        for g in stmt.gates
            push!(groups, (sort(g.qubits), [g]))
        end
    end
    return groups
end

# Multiply the primitives of one group into a single tensor on `length(qs)` sites,
# with slot `i` belonging to qubit `qs[i]`. Gates listed first act first, hence
# the left-multiplication.
function _fold(::Type{T}, qs::Vector{Int}, gates::Vector{QASMGate}) where {T <: Complex}
    q = QG.qubit_space()
    if length(qs) == 1
        t = id(T, q)
        for g in gates
            t = _gate_tensor(T, g) * t
        end
        return t
    end
    t = id(T, q ⊗ q)
    for g in gates
        u = _gate_tensor(T, g)
        if length(g.qubits) == 1
            u = g.qubits[1] == qs[1] ? u ⊗ id(T, q) : id(T, q) ⊗ u
        elseif g.qubits[1] != qs[1]
            u = permute(u, ((2, 1), (4, 3)))
        end
        t = u * t
    end
    return t
end

# --- lattice -----------------------------------------------------------------

"""
    qasm_lattice(qc::QASMCircuit) -> Vector{UndirectedEdge{Int}}

The edge set implied by `qc`: one edge for every pair of qubits coupled by a
two-site gate. Pass it to [`product_state`](@ref) to build a
[`TensorNetworkState`](@ref) whose connectivity matches the circuit exactly, so
that no gate needs routing.

The circuit's connectivity is used verbatim, so a densely connected circuit
yields a densely connected (and for belief propagation, poorly behaved) network.
Throws if a qubit is coupled to nothing at all, since an isolated vertex cannot
be described by an edge list; supply your own lattice and `qubitmap` in that case.
"""
function qasm_lattice(qc::QASMCircuit)
    edges = UndirectedEdge{Int}[]
    for stmt in qc.statements, (qs, _) in _lower_groups(stmt)
        length(qs) == 2 || continue
        e = UndirectedEdge(qs[1], qs[2])
        e in edges || push!(edges, e)
    end
    isolated = setdiff(1:qc.nqubits, (v for e in edges for v in Tuple(e)))
    isempty(isolated) ||
        throw(ArgumentError(lazy"qubits $isolated are not coupled to any other qubit, so this circuit does not span an edge-defined lattice; build the lattice yourself and pass `qubitmap`"))
    return edges
end

# --- lowering ----------------------------------------------------------------

# Qubit index -> vertex token, validated against `state`.
function _site_map(qc::QASMCircuit, state::TensorNetworkState, qubitmap)
    V = keytype(state)
    sites = if isnothing(qubitmap)
        V === Int ||
            throw(ArgumentError(lazy"`state` has vertex type $V, so the qubits cannot be identified with it implicitly; pass `qubitmap` to map 1:$(qc.nqubits) onto its vertices"))
        collect(1:qc.nqubits)
    elseif qubitmap isa AbstractVector
        length(qubitmap) == qc.nqubits ||
            throw(ArgumentError(lazy"`qubitmap` has $(length(qubitmap)) entries but the circuit has $(qc.nqubits) qubits"))
        collect(V, qubitmap)
    else
        V[convert(V, qubitmap[i]) for i in 1:(qc.nqubits)]
    end
    allunique(sites) || throw(ArgumentError("`qubitmap` sends two qubits to the same vertex"))
    for (i, v) in enumerate(sites)
        has_vertex(state, v) ||
            throw(ArgumentError(lazy"qubit $i maps to vertex $v, which `state` does not have"))
    end
    return sites
end

# Greedily pack gates into maximal layers of non-overlapping sites. A gate joins
# the layer under construction only if none of its sites is taken, so a later gate
# can never overtake an earlier one on a shared qubit. `breaks` holds the gate
# counts at which a `barrier` was seen, each forcing a new layer.
function _layer(gates::Vector{G}, breaks::Vector{Int}) where {G <: AbstractGate}
    groups = Vector{Int}[]
    occupied = Set{keytype(G)}()
    for (i, g) in enumerate(gates)
        if isempty(groups) || (i - 1) in breaks || any(in(occupied), g.sites)
            push!(groups, Int[])
            empty!(occupied)
        end
        push!(groups[end], i)
        for s in g.sites
            push!(occupied, s)
        end
    end
    return G[CompositeGate(map(identity, gates[idx])) for idx in groups]
end

"""
    Circuit(qc::QASMCircuit, state::TensorNetworkState; qubitmap = nothing, layered = false)

Lower a parsed OpenQASM circuit onto `state`, giving a [`Circuit`](@ref) of
[`LocalGate`](@ref)s that [`apply!`](@ref) accepts.

`qubitmap` identifies qubit `i` with vertex `qubitmap[i]`; it may be any vector or
index-able mapping, and defaults to `1:nqubits` when `state` has `Int` vertices.
Every two-site gate must land on an existing edge of `state` — see
[`qasm_lattice`](@ref) for an edge set that satisfies this by construction.

With `layered = true` the gates are packed into [`CompositeGate`](@ref) layers of
non-overlapping sites, with `barrier`s forcing a layer boundary; otherwise they
are kept in source order. Both give the same result under `apply!`.
"""
function Circuit(
        qc::QASMCircuit, state::TensorNetworkState;
        qubitmap = nothing, layered::Bool = false,
    )
    T = scalartype(state)
    T <: Complex ||
        throw(ArgumentError(lazy"OpenQASM gates are complex, but `state` has scalar type $T; build it with a complex scalar type such as `ComplexF64`"))
    S = spacetype(typeof(state))
    V = keytype(state)
    sites = _site_map(qc, state, qubitmap)

    gates = AbstractGate{T, S, V}[]
    breaks = Int[]
    for stmt in qc.statements
        if stmt.barrier
            push!(breaks, length(gates))
            continue
        end
        for (qs, gs) in _lower_groups(stmt)
            vs = length(qs) == 1 ? (sites[qs[1]],) : (sites[qs[1]], sites[qs[2]])
            length(vs) == 2 && !has_edge(state, vs[1], vs[2]) &&
                throw(ArgumentError(lazy"`$(stmt.name)` couples qubits $(qs[1]) and $(qs[2]), mapped to vertices $(vs[1]) and $(vs[2]), which are not neighbours in `state`"))
            push!(gates, LocalGate(vs, _fold(T, qs, gs)))
        end
    end

    # `map(identity, ...)` narrows the element type away from `AbstractGate`
    # whenever the gates turn out to be uniform, keeping `apply!` statically
    # dispatched in the common case.
    layered && return Circuit(map(identity, _layer(gates, breaks)))
    return Circuit(map(identity, gates))
end
