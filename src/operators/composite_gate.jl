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

Adapt.adapt_structure(to, gates::CompositeGate) =
    CompositeGate(map(g -> adapt(to, g), gates.gatelist))

function apply!(
        state::TensorNetworkState, msgs::BPMessages, gates::CompositeGate;
        timer = nothing, pool::BufferPool = _default_pool(), kwargs...,
    )
    return @maybe_timeit timer "apply! CompositeGate" begin
        T = real(scalartype(state))
        ϵ = similar(gates.gatelist, T)
        logλ = similar(gates.gatelist, T)
        ngates = length(gates.gatelist)
        nworkers = min(Threads.nthreads(), ngates)
        counter = Threads.Atomic{Int}(1)
        Threads.@sync for _ in 1:nworkers
            Threads.@spawn withbuffer(pool) do allocator
                while true
                    i = Threads.atomic_add!(counter, 1)
                    i > ngates && break
                    state, msgs, info = apply!(state, msgs, gates.gatelist[i]; timer, allocator, kwargs...)
                    ϵ[i] = info.ϵ
                    logλ[i] = info.logλ
                end
            end
        end
        (state, msgs, (; ϵ = maximum(ϵ), logλ = sum(logλ)))
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

Adapt.adapt_structure(to, c::Circuit) = Circuit(map(g -> adapt(to, g), c.gatelist))

function apply!(
        state::TensorNetworkState, msgs::BPMessages, circuit::Circuit;
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
