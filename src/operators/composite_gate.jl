"""
    CompositeGate{T, S, V, G <: LocalGate{T, <:Any, S, <:Any, V}} <: AbstractGate{T, S, V}

A set of (commuting) gates acting on non-overlapping sites. Groups gates that
can be applied in any order (or in parallel).
"""
struct CompositeGate{T <: Number, S <: ElementarySpace, V, G <: LocalGate{T, <:Any, S, <:Any, V}} <: AbstractGate{T, S, V}
    gatelist::Vector{G}
end

function CompositeGate(gatelist::Vector{G}) where {G <: LocalGate}
    V = keytype(G)
    seen_vertices = Set{V}()
    for gate in gatelist, site in gate.sites
        site in seen_vertices &&
            throw(ArgumentError(lazy"gatelist contains overlapping gates at site $site"))
        push!(seen_vertices, site)
    end
    return CompositeGate{scalartype(G), spacetype(G), V, G}(gatelist)
end

function apply!(
        state::TensorNetworkState, msgs::BPMessages, gates::CompositeGate;
        timer = nothing, kwargs...,
    )
    return @maybe_timeit timer "apply! CompositeGate" begin
        T = real(scalartype(state))
        ϵ = similar(gates.gatelist, T)
        Threads.@threads for i in eachindex(gates.gatelist)
            state, msgs, ϵ[i] = apply!(state, msgs, gates.gatelist[i]; timer, kwargs...)
        end
        (state, msgs, maximum(ϵ))
    end
end

"""
    Circuit{T, S, V, G <: AbstractGate{T, S, V}}

Ordered sequence of gates applied in turn by [`apply!`](@ref). Pure operator
content — does not include BP reconvergence.
"""
struct Circuit{T <: Number, S <: ElementarySpace, V, G <: AbstractGate{T, S, V}}
    gatelist::Vector{G}
end

function apply!(
        state::TensorNetworkState, msgs::BPMessages, circuit::Circuit;
        timer = nothing, kwargs...,
    )
    return @maybe_timeit timer "apply! Circuit" begin
        T = real(scalartype(state))
        ϵ = zero(T)
        for gate in circuit.gatelist
            state, msgs, ϵ_local = apply!(state, msgs, gate; timer, kwargs...)
            ϵ = max(ϵ, ϵ_local)
        end
        (state, msgs, ϵ)
    end
end
