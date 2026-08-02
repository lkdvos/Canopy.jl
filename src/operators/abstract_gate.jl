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
[`LocalGate`](@ref) has; other gate types may compute it, so callers should use this rather
than reaching for `gate.sites`.
"""
sites(g::AbstractGate) = g.sites

Base.keytype(g::AbstractGate) = keytype(typeof(g))
Base.keytype(::Type{<:AbstractGate{T, S, V}}) where {T, S, V} = V

VectorInterface.scalartype(::Type{<:AbstractGate{T, S, V}}) where {T, S, V} = T
TensorKit.spacetype(::Type{<:AbstractGate{T, S, V}}) where {T, S, V} = S
