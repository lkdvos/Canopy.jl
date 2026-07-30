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

Adapt.adapt_structure(to, g::LocalGate) = LocalGate(g.sites, adapt(to, g.tensor))

# Structural checks against the network: sites exist, the gate's spaces match the physical
# spaces at those sites, and (for two-site gates) the sites form an existing edge. The
# side-dependent variants for operators are in `operators/sided_gate.jl`.

# Physical slot `slot` of `net[site]` against the gate space `Pgate` at gate slot `i`. Slot 1
# (the ket) pairs with the gate space directly; slot 2 (the bra) carries `dual`, which is what
# lets a right action contract a non-dual gate leg without any transpose.
function _check_slot(net::AbstractTensorNetwork, site, slot::Int, Pgate, i::Int)
    P = physicalspace(net, site, slot)
    expected = slot == 1 ? Pgate : dual(Pgate)
    expected == P || throw(
        SpaceMismatch(
            lazy"gate slot $i ($Pgate) does not match physical slot $slot ($P) at site $site"
        )
    )
    return nothing
end

function _check_sites(net::AbstractTensorNetwork, gate::LocalGate{<:Any, N}) where {N}
    for s in gate.sites
        has_vertex(net, s) || throw(KeyError(s))
    end
    N == 2 && !has_edge(net, gate.sites...) &&
        throw(ArgumentError(lazy"sites $(gate.sites) do not form an edge of the network"))
    return nothing
end

function _check_compatible(net::AbstractTensorNetwork, gate::LocalGate{<:Any, N}) where {N}
    _check_sites(net, gate)
    for i in 1:N
        _check_slot(net, gate.sites[i], 1, domain(gate.tensor)[i], i)
    end
    return nothing
end

@doc """
    apply!(state::TensorNetworkState, msgs, gate::LocalGate; kwargs...) -> state, msgs, info
    apply!(op::TensorNetworkOperator, msgs, gate::Union{LeftGate, RightGate, SandwichGate}; kwargs...)

In-place application of `gate` onto `state`. Returns `info = (; ϵ, logλ)`:
`ϵ` is the truncation error and `logλ` is the log of the bond-message norm
absorbed during normalization. The 2-site method renormalizes the new bond
message to unit `normp`-norm (kwarg `normp::Real = 2`; `normp = 0` disables
normalization and returns `logλ = 0`). The 1-site method does not touch any
bond message and always returns `logλ = 0`.

The 2-site method gauges with the BP messages via an eigh-based square root;
its pseudo-inverse clips message eigenvalues at or below `gauge_tol::Real`
(default [`default_gauge_tol`](@ref)`(state)`), which drops numerical-noise
directions that would otherwise be inverted and blow up the gauge. Pass
`gauge_tol = 0` to disable clipping.

On a `TensorNetworkOperator` a bare `LocalGate` has no method: wrap it in
[`LeftGate`](@ref), [`RightGate`](@ref) or [`SandwichGate`](@ref) to say which physical
leg(s) it acts on. There is deliberately no default side. Everything above applies
unchanged to the wrapped forms.
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
        T = real(scalartype(state))
        (state, msgs, (; ϵ = zero(T), logλ = zero(T)))
    end
end
