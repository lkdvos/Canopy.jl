"""
    AbstractGate{T <: Number, S <: ElementarySpace, V}

Supertype for gates acting on a [`TensorNetworkState`](@ref) with scalar type
`T`, space type `S`, and vertex-key type `V`. Subtypes must implement
`apply!(state, msgs, gate; kwargs...) -> (state, msgs, ϵ)`.
"""
abstract type AbstractGate{T <: Number, S <: ElementarySpace, V} end

Base.keytype(g::AbstractGate) = keytype(typeof(g))
Base.keytype(::Type{<:AbstractGate{T, S, V}}) where {T, S, V} = V

VectorInterface.scalartype(::Type{<:AbstractGate{T, S, V}}) where {T, S, V} = T
TensorKit.spacetype(::Type{<:AbstractGate{T, S, V}}) where {T, S, V} = S
