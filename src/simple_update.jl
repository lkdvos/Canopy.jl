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
struct LocalGate{T <: Number, N, S <: ElementarySpace, TT <: AbstractTensorMap{T, S, N, N}, V}
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
        state::TensorNetworkState, msgs::BPMessages, gate::LocalGate{<:Any, 1}; kwargs...
    )
    _check_compatible(state, gate)
    v = only(gate.sites)
    state.vertices[v] = gate.tensor * state[v]
    return state, msgs
end

# --- two-site BP-gauge SU ----------------------------------------------------
function apply!(
        state::TensorNetworkState, msgs::BPMessages, gate::LocalGate{<:Any, 2};
        trunc = notrunc(), gauge_tol::Real = 0.0,
    )
    _check_compatible(state, gate)
    s₁, s₂ = gate.sites
    G = gate.tensor

    # Canonical orientation: smaller vertex on the codomain side of the SVD.
    if s₁ > s₂
        G = permute(G, ((2, 1), (4, 3)))
        s₁, s₂ = s₂, s₁
    end

    k₁ = leg_index(state, DirectedEdge(s₁, s₂))
    k₂ = leg_index(state, DirectedEdge(s₂, s₁))
    Nd = numin(state[s₁])

    # Absorb square root factors and factorize
    gauge₁ = _gauge_factors(state, msgs, DirectedEdge(s₁, s₂); tol = gauge_tol)
    T₁ = _absorb_legs(state[s₁], (k => L for (k, L, _) in gauge₁))
    gauge₂ = _gauge_factors(state, msgs, DirectedEdge(s₂, s₁); tol = gauge_tol)
    T₂ = _absorb_legs(state[s₂], (k => L for (k, L, _) in gauge₂))

    legs = ntuple(identity, Nd + 1)
    Q₁, R₁ = qr_compact!(permute(T₁, (TupleTools.deleteat(legs, (1, k₁ + 1)), (1, k₁ + 1))))
    Q₂, R₂ = qr_compact!(permute(T₂, (TupleTools.deleteat(legs, (1, k₂ + 1)), (1, k₂ + 1))))

    # Apply gate and factorize
    @tensor θ[-1 -2; -3 -4] := R₁[-1; 1 2] * R₂[-3; 3 2] * G[-2 -4; 1 3]
    U, Σ, Vᴴ, ϵ = svd_trunc!(θ; trunc)

    # Split √Σ half/half; keep `Σdiag` for the bond-message update
    Σdiag = collect(scalartype(msgs), diagview(Σ))
    diagview(Σ) .= sqrt.(diagview(Σ))
    rmul!(U, Σ)
    lmul!(Σ, Vᴴ)

    # Store result in state
    outer = ntuple(identity, Nd - 1)
    T₁ = permute(
        Q₁ * permute(U, ((1,), (2, 3))),
        ((Nd,), TupleTools.insertafter(outer, k₁ - 1, (Nd + 1,))),
    )
    state.vertices[s₁] = _absorb_legs(T₁, (k => Linv for (k, _, Linv) in gauge₁))
    T₂ = permute(
        Q₂ * permute(Vᴴ, ((2,), (1, 3))),
        ((Nd + 1,), TupleTools.insertafter(outer, k₂ - 1, (Nd,))),
    )
    state.vertices[s₂] = _absorb_legs(T₂, (k => Linv for (k, _, Linv) in gauge₂))

    # Bond messages from the new Schmidt diagonal
    V′ = virtualspace(state, DirectedEdge(s₁, s₂))
    msgs.messages[DirectedEdge(s₂, s₁)] = DiagonalTensorMap(Σdiag, V′)
    V′ᵈ = virtualspace(state, DirectedEdge(s₂, s₁))
    msgs.messages[DirectedEdge(s₁, s₂)] = DiagonalTensorMap(Σdiag, V′ᵈ)

    return state, msgs, ϵ
end

# --- helpers -----------------------------------------------------------------

# Eigh-based square root and pseudo-inverse of a Hermitian PSD message,
# clipping eigenvalues below `tol`.
function _eigh_sqrt(m; tol::Real)
    @assert tol ≥ 0 "Tolerance must be positive"
    D, U = eigh_full(m)
    λ = diagview(D)
    λ⁻¹ = similar(λ)
    z = zero(eltype(λ))
    @inbounds @simd for i in eachindex(λ)
        λᵢ = λ[i]
        if λᵢ > tol
            λ[i] = sqrt(λᵢ)
            λ⁻¹[i] = inv(λ[i])
        else
            λ⁻¹[i] = λ[i] = z
        end
    end
    Λ = D * U'
    Λ⁻¹ = rmul!(U, DiagonalTensorMap(λ⁻¹))
    return Λ, Λ⁻¹
end

# `(leg_index, L, Linv)` for each non-partner neighbour of `first(edge)`.
function _gauge_factors(
        state::TensorNetworkState, msgs::BPMessages, edge::DirectedEdge; tol::Real,
    )
    M = eltype(msgs)
    factors = Vector{Tuple{Int, M, M}}()
    site = first(edge)
    for n in neighbors(state, site)
        n == last(edge) && continue
        incoming = DirectedEdge(n, site)
        k = leg_index(state, reverse(incoming))
        L, Linv = _eigh_sqrt(msgs[incoming]; tol)
        push!(factors, (k, L, Linv))
    end
    return factors
end
