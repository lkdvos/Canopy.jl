# OpenQASM 2.0 intermediate representation
# ---------------------------------------
# A physics-agnostic description of a parsed circuit: qubits are plain 1-based
# integers and gates are symbolic names with numeric parameters. Nothing here
# depends on TensorKit, so the front-end is testable without any tensor
# machinery, and the choice of lattice and vertex tokens is deferred entirely to
# the lowering step in `lowering.jl`.

"""
    QASMGate(name, params, qubits)

A single primitive gate application. `name` is a canonical OpenQASM 2.0 gate
name, `params` its evaluated classical arguments, and `qubits` the 1-based global
qubit indices it acts on (in the order written in the source, so for `cx` the
control comes first).
"""
struct QASMGate
    name::Symbol
    params::Vector{Float64}
    qubits::Vector{Int}
end

"""
    QASMStatement(name, gates, barrier = false)

One top-level source statement together with its expansion into primitive
[`QASMGate`](@ref)s. `name` is the gate name as written, retained for error
messages; for a user-declared `gate` it is the declaration's name and `gates`
holds the recursively inlined body. A `barrier` statement carries no gates.
"""
struct QASMStatement
    name::Symbol
    gates::Vector{QASMGate}
    barrier::Bool
end
QASMStatement(name::Symbol, gates::Vector{QASMGate}) = QASMStatement(name, gates, false)

# Sorted, unique qubit indices touched by a statement.
function _support(stmt::QASMStatement)
    qs = Int[]
    for g in stmt.gates, q in g.qubits
        q in qs || push!(qs, q)
    end
    return sort!(qs)
end

"""
    QASMCircuit

An OpenQASM 2.0 program reduced to the gates it applies, as produced by
[`read_qasm`](@ref) / [`parse_qasm`](@ref).

`nqubits` is the total number of qubits over all declared quantum registers,
which are flattened into `1:nqubits` in declaration order; `registers` records
that layout as `name => size` pairs. `statements` are the gate statements in
source order, each already expanded into primitives.

Lower a circuit onto a [`TensorNetworkState`](@ref) with
`Circuit(qc, state)`, and derive a matching lattice with [`qasm_lattice`](@ref).
"""
struct QASMCircuit
    nqubits::Int
    registers::Vector{Pair{Symbol, Int}}
    statements::Vector{QASMStatement}
end

Base.length(qc::QASMCircuit) = length(qc.statements)

function Base.show(io::IO, ::MIME"text/plain", qc::QASMCircuit)
    ngates = sum(stmt -> length(stmt.gates), qc.statements; init = 0)
    regs = join(("$name[$n]" for (name, n) in qc.registers), ", ")
    println(io, "QASMCircuit($(qc.nqubits) qubits: $regs)")
    print(io, "  $(length(qc.statements)) statements, $ngates primitive gates")
    return nothing
end
