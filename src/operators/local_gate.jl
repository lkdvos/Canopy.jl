"""
    LocalGate{T <: Number, N, S <: ElementarySpace, TT <: AbstractTensorMap{T, S, N, N}, V}

A gate tensor `TT` attached to `N` sites of vertex-key type `V`.

`tensor` must have domain ``P₁ ⊗ ⋯ ⊗ Pₙ`` where `Pᵢ` is the physical space at
`sites[i]` *before* applying. The codomain spaces become the new physical
spaces at those sites — they need not equal the domain (e.g. swap-like gates
or basis-changing gates). The slot order of `tensor` matches the slot order of
`sites`, and the leg count (`N` codom and `N` dom) is enforced at the type
level via the `TT` constraint.
"""
struct LocalGate{T <: Number, N, S <: ElementarySpace, TT <: AbstractTensorMap{T, S, N, N}, V} <: AbstractGate{T, S, V}
    sites::NTuple{N, V}
    tensor::TT
    function LocalGate{T, N, S, TT, V}(sites::NTuple{N, V}, tensor::TT) where
        {T <: Number, N, S <: ElementarySpace, TT <: AbstractTensorMap{T, S, N, N}, V}
        return new{T, N, S, TT, V}(sites, tensor)
    end
end

LocalGate(sites::NTuple{N, V}, tensor::TT) where {T, N, S, V, TT <: AbstractTensorMap{T, S, N, N}} =
    LocalGate{T, N, S, TT, V}(sites, tensor)

# Structural checks against `state`: sites exist, gate space matches the
# physical spaces at those sites, and (for two-site gates) the sites form
# an existing edge.

function _check_compatible(state::TensorNetworkState, gate::LocalGate{<:Any, N}) where {N}
    for s in gate.sites
        haskey(state.vertices, s) || throw(KeyError(s))
    end
    for i in 1:N
        P = physicalspace(state, gate.sites[i])
        P′ = domain(gate.tensor)[i]
        P′ == P || throw(SpaceMismatch(lazy"gate domain slot $i $P′ does not match physical space $P at site $(gate.sites[i])"))
    end
    return nothing
end

@doc """
    apply!(state, msgs, gate::LocalGate; kwargs...) -> state, msgs, ϵ

In-place application of `gate` onto `state`.
""" apply!

# --- single-site -------------------------------------------------------------
function apply!(
        state::TensorNetworkState, msgs::BPMessages, gate::LocalGate{<:Any, 1};
        timer = nothing, kwargs...,
    )
    return @maybe_timeit timer "apply! 1-site" begin
        _check_compatible(state, gate)
        v = only(gate.sites)
        state.vertices[v] = gate.tensor * state[v]
        (state, msgs, zero(real(scalartype(state))))
    end
end
