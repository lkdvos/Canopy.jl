"""
    CompositeGate{T, S, V, G <: AbstractGate{T, S, V}} <: AbstractGate{T, S, V}

A set of (commuting) gates acting on non-overlapping sites. Groups gates that
can be applied in any order (or in parallel).

The element type is any [`AbstractGate`](@ref). Keyword arguments of [`apply!`](@ref) —
including the [`GateAction`](@ref) on an operator — are forwarded to every gate in the list.
"""
struct CompositeGate{T <: Number, S <: ElementarySpace, V, G <: AbstractGate{T, S, V}} <: AbstractGate{T, S, V}
    gatelist::Vector{G}

    # The non-overlap check lives in an *inner* constructor on purpose. As an outer one it
    # would be shadowed by the constructor Julia auto-generates from the field types, which is
    # more specific (it pins `T, S, V`) and would silently skip the validation.
    function CompositeGate{T, S, V, G}(gatelist::Vector{G}) where
        {T <: Number, S <: ElementarySpace, V, G <: AbstractGate{T, S, V}}
        seen_vertices = Set{V}()
        for gate in gatelist, site in sites(gate)
            site in seen_vertices &&
                throw(ArgumentError(lazy"gatelist contains overlapping gates at site $site"))
            push!(seen_vertices, site)
        end
        return new{T, S, V, G}(gatelist)
    end
end

CompositeGate(gatelist::Vector{G}) where {T, S, V, G <: AbstractGate{T, S, V}} =
    CompositeGate{T, S, V, G}(gatelist)

Adapt.adapt_structure(to, gates::CompositeGate) =
    CompositeGate(map(g -> adapt(to, g), gates.gatelist))

function apply!(
        state::AbstractTensorNetwork, msgs::BPMessages, gates::CompositeGate;
        timer = nothing, allocator = default_allocator(state), kwargs...,
    )
    return @maybe_timeit timer "apply! CompositeGate" begin
        T = real(scalartype(state))
        ϵ = zero(T)
        logλ = zero(T)
        for gate in gates.gatelist
            state, msgs, info = apply!(state, msgs, gate; timer, allocator, kwargs...)
            @debug "between gates" sites = gate.sites isempty = buffer_isempty(allocator) stats = buffer_stats(allocator)
            ϵ = max(ϵ, info.ϵ)
            logλ += info.logλ
        end
        (state, msgs, (; ϵ, logλ))
    end
end

"""
    Circuit{T, S, V, G <: AbstractGate{T, S, V}}

Ordered sequence of gates applied in turn by [`apply!`](@ref). Pure operator
content — does not include BP reconvergence.

As for [`CompositeGate`](@ref), [`apply!`](@ref) forwards its keyword arguments to every gate,
so the same circuit is reusable under any [`GateAction`](@ref).
"""
struct Circuit{T <: Number, S <: ElementarySpace, V, G <: AbstractGate{T, S, V}}
    gatelist::Vector{G}
end

Adapt.adapt_structure(to, c::Circuit) = Circuit(map(g -> adapt(to, g), c.gatelist))

function apply!(
        state::AbstractTensorNetwork, msgs::BPMessages, circuit::Circuit;
        timer = nothing, kwargs...,
    )
    return @maybe_timeit timer "apply! Circuit" begin
        T = real(scalartype(state))
        ϵ = zero(T)
        logλ = zero(T)
        for gate in circuit.gatelist
            state, msgs, info = apply!(state, msgs, gate; timer, kwargs...)
            ϵ = max(ϵ, info.ϵ)
            logλ += info.logλ
        end
        (state, msgs, (; ϵ, logλ))
    end
end
