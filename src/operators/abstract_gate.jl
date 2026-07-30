"""
    AbstractGate{T <: Number, S <: ElementarySpace, V}

Supertype for gates acting on a [`TensorNetworkState`](@ref) with scalar type
`T`, space type `S`, and vertex-key type `V`. Subtypes must implement
`apply!(state, msgs, gate; kwargs...) -> (state, msgs, info)`, where `info`
is a `NamedTuple` with at least the fields `ϵ` (truncation error) and `logλ`
(log of any norm absorbed during bond-message renormalization).
"""
abstract type AbstractGate{T <: Number, S <: ElementarySpace, V} end

"""
    sites(gate) -> NTuple{N, V}

The vertices `gate` acts on. Defaults to the `sites` field, which
[`LocalGate`](@ref) has; wrappers such as [`SandwichGate`](@ref) forward to the gate they
wrap, so callers should use this rather than reaching for `gate.sites`.
"""
sites(g::AbstractGate) = g.sites

"""
    Canopy._acted_slots(gate) -> Tuple{Vararg{Int}}

Which physical (codomain) slots of an on-site tensor a gate acts on. `(1,)` by default —
the only choice for a state — and `(1,)` / `(2,)` / `(1, 2)` for the sided operator gates
[`LeftGate`](@ref), [`RightGate`](@ref) and [`SandwichGate`](@ref).

This is the single parameter the generic two-site kernel needs: slots *not* acted on stay in
the QR environment, so a one-sided operator gate costs exactly what the same gate costs on a
state.
"""
_acted_slots(::AbstractGate) = (1,)

Base.keytype(g::AbstractGate) = keytype(typeof(g))
Base.keytype(::Type{<:AbstractGate{T, S, V}}) where {T, S, V} = V

VectorInterface.scalartype(::Type{<:AbstractGate{T, S, V}}) where {T, S, V} = T
TensorKit.spacetype(::Type{<:AbstractGate{T, S, V}}) where {T, S, V} = S
