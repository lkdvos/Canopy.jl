# OpenQASM 2.0 source -> QASMCircuit
# ----------------------------------
# `OpenQASM.parse` returns a bare syntax tree and nothing more: `include` is not
# resolved, `gate` bodies are not expanded, and classical arguments are left as
# unevaluated token trees (binary operations are plain 3-tuples
# `(lhs, op, rhs)`). All of that semantics lives here.

# Canonical primitives, as `name => (nparams, nqubits)`. These are the names
# `_gate_tensor` in `lowering.jl` knows how to build, so the two tables must stay
# in step. `ccx`/`cswap` are listed so that using them produces the specific
# "no three-site gate" error rather than "unsupported gate".
const _QASM_PRIMITIVES = Dict{Symbol, Tuple{Int, Int}}(
    :id => (0, 1), :u0 => (1, 1),
    :x => (0, 1), :y => (0, 1), :z => (0, 1), :h => (0, 1),
    :s => (0, 1), :sdg => (0, 1), :t => (0, 1), :tdg => (0, 1),
    :sx => (0, 1), :sxdg => (0, 1),
    :u1 => (1, 1), :rx => (1, 1), :ry => (1, 1), :rz => (1, 1),
    :u2 => (2, 1), :u3 => (3, 1),
    :cx => (0, 2), :cy => (0, 2), :cz => (0, 2), :ch => (0, 2), :cs => (0, 2),
    :swap => (0, 2), :iswap => (0, 2), :dcx => (0, 2), :ecr => (0, 2), :csx => (0, 2),
    :cp => (1, 2), :crx => (1, 2), :cry => (1, 2), :crz => (1, 2),
    :rxx => (1, 2), :ryy => (1, 2), :rzz => (1, 2), :rzx => (1, 2),
    :cu3 => (3, 2), :cu => (4, 2),
    :ccx => (0, 3), :cswap => (0, 3),
)

# Spellings that denote an existing primitive exactly.
const _QASM_ALIASES = Dict{Symbol, Symbol}(
    :u => :u3, :p => :u1, :phase => :u1,
    :cphase => :cp, :cu1 => :cp, :cnot => :cx,
)

# Source position of a token, for error messages.
_at(tok::Token) = "line $(tok.lineno): "

# Top-level gate parameters cannot reference identifiers.
const _EMPTY_ENV = Dict{Symbol, Float64}()

# --- classical expressions ---------------------------------------------------

_eval_param(x::Token{:int}, env) = Float64(Base.parse(Int, x.str))
_eval_param(x::Token{:float64}, env) = Base.parse(Float64, x.str)

function _eval_param(x::Token, env)
    x.str == "pi" && return Float64(π)
    haskey(env, Symbol(x.str)) && return env[Symbol(x.str)]
    throw(ArgumentError(lazy"$(_at(x))unknown classical parameter `$(x.str)`"))
end

_eval_param(x::Neg, env) = -_eval_param(x.val, env)

function _eval_param(x::Call, env)
    a = _eval_param(x.args, env)
    x.name === :sin && return sin(a)
    x.name === :cos && return cos(a)
    x.name === :tan && return tan(a)
    x.name === :exp && return exp(a)
    x.name === :ln && return log(a)
    x.name === :sqrt && return sqrt(a)
    throw(ArgumentError(lazy"unknown function `$(x.name)` in a gate parameter"))
end

function _eval_param(x::Tuple, env)
    length(x) == 3 ||
        throw(ArgumentError(lazy"cannot evaluate gate parameter expression $x"))
    a, op, b = _eval_param(x[1], env), x[2], _eval_param(x[3], env)
    op.str == "+" && return a + b
    op.str == "-" && return a - b
    op.str == "*" && return a * b
    op.str == "/" && return a / b
    throw(ArgumentError(lazy"$(_at(op))unknown operator `$(op.str)` in a gate parameter"))
end

# --- qubit arguments ---------------------------------------------------------

# Resolve the qubit arguments of a top-level statement, applying OpenQASM's
# register broadcast: a bare register name stands for all of its qubits, so the
# statement is repeated once per index. Returns one index vector per repetition.
function _broadcast_qargs(qargs, regs, name)
    slots = map(qargs) do bit
        rname = Symbol(bit.name.str)
        haskey(regs, rname) ||
            throw(ArgumentError(lazy"$(_at(bit.name))`$name` refers to undeclared quantum register `$rname`"))
        rng = regs[rname]
        isnothing(bit.address) && return rng
        a = Base.parse(Int, bit.address.str)
        0 ≤ a < length(rng) ||
            throw(ArgumentError(lazy"$(_at(bit.address))index $a is out of range for register `$rname[$(length(rng))]`"))
        return rng[a + 1]:rng[a + 1]
    end
    n = maximum(length, slots)
    all(s -> length(s) == 1 || length(s) == n, slots) ||
        throw(ArgumentError(lazy"`$name` broadcasts over registers of differing sizes"))
    return [[length(s) == 1 ? only(s) : s[k] for s in slots] for k in 1:n]
end

# Resolve a formal qubit argument inside a `gate` body.
function _formal_qarg(bit::Bit, qmap, name)
    isnothing(bit.address) ||
        throw(ArgumentError(lazy"$(_at(bit.name))cannot index a register inside the body of `gate $name`"))
    q = Symbol(bit.name.str)
    haskey(qmap, q) ||
        throw(ArgumentError(lazy"$(_at(bit.name))`gate $name` uses undeclared qubit argument `$q`"))
    return qmap[q]
end

# --- gate expansion ----------------------------------------------------------

"""
    _expand!(gates, name, params, qubits, defs, stack)

Append the primitive expansion of one gate application to `gates`. A name found
in `defs` (a user-declared `gate`) is inlined by substituting its formal
arguments and recursing; anything else must be a known primitive. `stack` carries
the definitions currently being inlined, so that a cyclic declaration is
reported instead of overflowing.
"""
function _expand!(
        gates::Vector{QASMGate}, name::Symbol, params::Vector{Float64},
        qubits::Vector{Int}, defs::Dict{Symbol, Gate}, stack::Vector{Symbol},
    )
    allunique(qubits) ||
        throw(ArgumentError(lazy"`$name` is applied to a repeated qubit"))

    if haskey(defs, name)
        name in stack &&
            throw(ArgumentError(lazy"`gate $name` is defined recursively, which OpenQASM 2.0 does not allow"))
        decl = defs[name].decl
        length(params) == length(decl.cargs) ||
            throw(ArgumentError(lazy"`gate $name` takes $(length(decl.cargs)) parameters, got $(length(params))"))
        length(qubits) == length(decl.qargs) ||
            throw(ArgumentError(lazy"`gate $name` takes $(length(decl.qargs)) qubits, got $(length(qubits))"))
        cenv = Dict{Symbol, Float64}(
            Symbol(c.str) => params[i] for (i, c) in enumerate(decl.cargs)
        )
        qmap = Dict{Symbol, Int}(
            Symbol(q.str) => qubits[i] for (i, q) in enumerate(decl.qargs)
        )
        push!(stack, name)
        for s in defs[name].body
            _expand_body!(gates, s, cenv, qmap, defs, stack, name)
        end
        pop!(stack)
        return gates
    end

    canonical = get(_QASM_ALIASES, name, name)
    haskey(_QASM_PRIMITIVES, canonical) ||
        throw(ArgumentError(lazy"unsupported gate `$name`: it is neither declared in the file nor one of the OpenQASM 2.0 / qelib1.inc gates Canopy knows"))
    nparams, nqubits = _QASM_PRIMITIVES[canonical]
    length(params) == nparams ||
        throw(ArgumentError(lazy"`$name` takes $nparams parameters, got $(length(params))"))
    length(qubits) == nqubits ||
        throw(ArgumentError(lazy"`$name` takes $nqubits qubits, got $(length(qubits))"))
    push!(gates, QASMGate(canonical, params, qubits))
    return gates
end

# One statement from inside a `gate` body, with formal arguments bound by
# `cenv`/`qmap`. Bodies only ever hold gate applications and barriers.
function _expand_body!(gates, stmt, cenv, qmap, defs, stack, name)
    stmt isa Barrier && return gates
    if stmt isa UGate
        params = Float64[_eval_param(e, cenv) for e in (stmt.z1, stmt.y, stmt.z2)]
        return _expand!(gates, :u3, params, [_formal_qarg(stmt.qarg, qmap, name)], defs, stack)
    elseif stmt isa CXGate
        qubits = [_formal_qarg(stmt.ctrl, qmap, name), _formal_qarg(stmt.qarg, qmap, name)]
        return _expand!(gates, :cx, Float64[], qubits, defs, stack)
    elseif stmt isa Instruction
        params = Float64[_eval_param(c, cenv) for c in stmt.cargs]
        qubits = Int[_formal_qarg(b, qmap, name) for b in stmt.qargs]
        return _expand!(gates, Symbol(stmt.name), params, qubits, defs, stack)
    end
    throw(ArgumentError(lazy"unsupported statement $(typeof(stmt)) inside the body of `gate $name`"))
end

# --- top level ---------------------------------------------------------------

"""
    read_qasm(path) -> QASMCircuit
    parse_qasm(src::AbstractString) -> QASMCircuit

Read an OpenQASM 2.0 program into a [`QASMCircuit`](@ref). Quantum registers are
flattened into a single 1-based qubit range in declaration order, classical gate
arguments are evaluated, register broadcasts (`h q;`) are unrolled, and
user-declared `gate`s are recursively inlined into primitives.

`include "qelib1.inc"` is accepted and ignored — its gates are built in — but no
other file is resolved. `barrier` is recorded and otherwise ignored. Because
Canopy has no measurement, collapse or classical-register machinery, `measure`,
`reset`, `if` and `opaque` are rejected with an error rather than silently
dropped; strip them from the file to simulate the state they act on.
"""
read_qasm(path::AbstractString) = parse_qasm(read(path, String))

function parse_qasm(src::AbstractString)
    ast = OpenQASM.parse(String(src))
    ast.version.major == 2 ||
        throw(ArgumentError(lazy"only OpenQASM 2.0 is supported, but this program declares version $(ast.version.major).$(ast.version.minor)"))

    regs = Dict{Symbol, UnitRange{Int}}()
    registers = Pair{Symbol, Int}[]
    defs = Dict{Symbol, Gate}()
    statements = QASMStatement[]
    nqubits = 0

    for stmt in ast.prog
        if stmt isa RegDecl
            # `creg`s are only ever reachable through `measure`, rejected below
            if stmt.type.str == "qreg"
                rname = Symbol(stmt.name.str)
                haskey(regs, rname) &&
                    throw(ArgumentError(lazy"$(_at(stmt.name))quantum register `$rname` is declared twice"))
                size = Base.parse(Int, stmt.size.str)
                regs[rname] = (nqubits + 1):(nqubits + size)
                nqubits += size
                push!(registers, rname => size)
            end
        elseif stmt isa Include
            file = convert(String, stmt.file)
            file == "qelib1.inc" ||
                throw(ArgumentError(lazy"cannot resolve `include \"$file\"`; only \"qelib1.inc\" is recognized, whose gates are built in"))
        elseif stmt isa Gate
            defs[Symbol(stmt.decl.name.str)] = stmt
        elseif stmt isa Barrier
            push!(statements, QASMStatement(:barrier, QASMGate[], true))
        elseif stmt isa UGate || stmt isa CXGate || stmt isa Instruction
            name, cargs, qargs = if stmt isa UGate
                :u3, (stmt.z1, stmt.y, stmt.z2), (stmt.qarg,)
            elseif stmt isa CXGate
                :cx, (), (stmt.ctrl, stmt.qarg)
            else
                Symbol(stmt.name), Tuple(stmt.cargs), Tuple(stmt.qargs)
            end
            params = Float64[_eval_param(c, _EMPTY_ENV) for c in cargs]
            for qubits in _broadcast_qargs(qargs, regs, name)
                gates = QASMGate[]
                _expand!(gates, name, params, qubits, defs, Symbol[])
                push!(statements, QASMStatement(name, gates))
            end
        elseif stmt isa Measure
            throw(ArgumentError("`measure` is not supported: Canopy has no measurement or classical-register machinery"))
        elseif stmt isa Reset
            throw(ArgumentError("`reset` is not supported: Canopy has no state-collapse machinery"))
        elseif stmt isa IfStmt
            throw(ArgumentError("classically controlled statements (`if`) are not supported"))
        elseif stmt isa Opaque
            throw(ArgumentError("`opaque` gate declarations have no definition to apply"))
        else
            throw(ArgumentError(lazy"unsupported OpenQASM statement $(typeof(stmt))"))
        end
    end

    return QASMCircuit(nqubits, registers, statements)
end

@doc (@doc read_qasm) parse_qasm
