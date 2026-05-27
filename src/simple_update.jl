# --- two-site BP-gauge simple update -----------------------------------------
function apply!(
        state::TensorNetworkState, msgs::BPMessages, gate::LocalGate{<:Any, 2};
        trunc = notrunc(), gauge_tol::Real = 0.0, normp::Real = 2, timer = nothing,
    )
    return @maybe_timeit timer "apply! 2-site" begin
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
        T₁, T₂, gauge₁, gauge₂ = @maybe_timeit timer "gauge in" begin
            g₁ = _gauge_factors(state, msgs, DirectedEdge(s₁, s₂); tol = gauge_tol)
            t₁ = _absorb_legs(state[s₁], (k => L for (k, L, _) in g₁))
            g₂ = _gauge_factors(state, msgs, DirectedEdge(s₂, s₁); tol = gauge_tol)
            t₂ = _absorb_legs(state[s₂], (k => L for (k, L, _) in g₂))
            (t₁, t₂, g₁, g₂)
        end

        legs = ntuple(identity, Nd + 1)
        Q₁, R₁, Q₂, R₂ = @maybe_timeit timer "QR" begin
            q₁, r₁ = qr_compact!(permute(T₁, (TupleTools.deleteat(legs, (1, k₁ + 1)), (1, k₁ + 1))))
            q₂, r₂ = qr_compact!(permute(T₂, (TupleTools.deleteat(legs, (1, k₂ + 1)), (1, k₂ + 1))))
            (q₁, r₁, q₂, r₂)
        end

        # Apply gate and factorize
        U, Σ, Vᴴ, ϵ = @maybe_timeit timer "gate+SVD" begin
            @tensor θ[-1 -2; -3 -4] := R₁[-1; 1 2] * R₂[-3; 3 2] * G[-2 -4; 1 3]
            svd_trunc!(θ; trunc)
        end

        Σdiag = collect(scalartype(msgs), diagview(Σ))
        T = real(scalartype(state))
        if normp == 0
            logλ = zero(T)
            diagview(Σ) .= sqrt.(diagview(Σ))
        else
            α = norm(Σdiag, normp)
            logλ = log(α)
            Σdiag ./= α
            diagview(Σ) .= sqrt.(diagview(Σ) ./ α)
        end
        rmul!(U, Σ)
        lmul!(Σ, Vᴴ)

        @maybe_timeit timer "reconstruct" begin
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
        end

        # Bond messages from the new Schmidt diagonal
        V′ = virtualspace(state, DirectedEdge(s₁, s₂))
        msgs.messages[DirectedEdge(s₂, s₁)] = DiagonalTensorMap(Σdiag, V′)
        V′ᵈ = virtualspace(state, DirectedEdge(s₂, s₁))
        msgs.messages[DirectedEdge(s₁, s₂)] = DiagonalTensorMap(Σdiag, V′ᵈ)

        (state, msgs, (; ϵ, logλ))
    end
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
    for incoming in incoming_edges(state, site; exclude=(last(edge),))
        k = leg_index(state, reverse(incoming))
        L, Linv = _eigh_sqrt(msgs[incoming]; tol)
        push!(factors, (k, L, Linv))
    end
    return factors
end
