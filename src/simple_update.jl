# --- two-site BP-gauge simple update -----------------------------------------
function apply!(
        state::TensorNetworkState, msgs::BPMessages, gate::LocalGate{<:Any, 2};
        trunc = notrunc(), gauge_tol::Real = default_gauge_tol(state), normp::Real = 2,
        timer = nothing, backend = DefaultBackend(), allocator = default_allocator(state),
    )
    return @maybe_timeit timer "apply! 2-site" begin
        _check_compatible(state, gate)
        s₁, s₂ = gate.sites
        G = gate.tensor

        @debug "apply! 2-site entry" sites = gate.sites isempty = buffer_isempty(allocator) stats = buffer_stats(allocator)

        # The buffer-allocating steps below are each self-contained: `_absorb_legs`
        # (gauge in / reconstruct) frees its temporaries and resets the buffer, and
        # the `@tensor` gate contraction does the same automatically. The new site
        # tensors and bond messages are heap-allocated and escape. The
        # MatrixAlgebraKit factorizations (`qr_compact!`, `svd_trunc!`, `eigh_full`)
        # do not use this allocator and always allocate on the heap.

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
            t₁ = _absorb_legs(state[s₁], (k => L for (k, L, _) in g₁), backend, allocator)
            g₂ = _gauge_factors(state, msgs, DirectedEdge(s₂, s₁); tol = gauge_tol)
            t₂ = _absorb_legs(state[s₂], (k => L for (k, L, _) in g₂), backend, allocator)
            (t₁, t₂, g₁, g₂)
        end

        legs = ntuple(identity, Nd + 1)
        Q₁, R₁, Q₂, R₂ = @maybe_timeit timer "QR" begin
            q₁, r₁ = qr_compact!(permute(T₁, (TupleTools.deleteat(legs, (1, k₁ + 1)), (1, k₁ + 1))))
            q₂, r₂ = qr_compact!(permute(T₂, (TupleTools.deleteat(legs, (1, k₂ + 1)), (1, k₂ + 1))))
            (q₁, r₁, q₂, r₂)
        end

        # Apply gate and factorize
        U, Σ, Vᴴ, logλ, ϵ = @maybe_timeit timer "gate+SVD" begin
            @tensor backend = backend allocator = allocator θ[-1 -2; -3 -4] := R₁[-1; 1 2] * R₂[-3; 3 2] * G[-2 -4; 1 3]
            U, Σ, Vᴴ, ϵ = svd_trunc!(θ; trunc)
            if iszero(normp)
                logλ = zero(scalartype(Σ))
            else
                α = norm(Σ, normp)
                logλ = log(α)
                scale!(Σ, inv(α))
            end
            sqrtΣ = sqrt(Σ)
            rmul!(U, sqrtΣ)
            lmul!(sqrtΣ, Vᴴ)
            U, Σ, Vᴴ, logλ, ϵ
        end

        @maybe_timeit timer "reconstruct" begin
            # Store result in state
            outer = ntuple(identity, Nd - 1)
            T₁ = permute(
                Q₁ * permute(U, ((1,), (2, 3))),
                ((Nd,), TupleTools.insertafter(outer, k₁ - 1, (Nd + 1,))),
            )
            state.vertices[s₁] = _absorb_legs(T₁, (k => Linv for (k, _, Linv) in gauge₁), backend, allocator)
            T₂ = permute(
                Q₂ * permute(Vᴴ, ((2,), (1, 3))),
                ((Nd + 1,), TupleTools.insertafter(outer, k₂ - 1, (Nd,))),
            )
            state.vertices[s₂] = _absorb_legs(T₂, (k => Linv for (k, _, Linv) in gauge₂), backend, allocator)
        end

        # Bond messages from the new Schmidt diagonal
        V′ = virtualspace(state, DirectedEdge(s₁, s₂))
        msgs.messages[DirectedEdge(s₂, s₁)] = DiagonalTensorMap(Σ.data, V′)
        V′ᵈ = virtualspace(state, DirectedEdge(s₂, s₁))
        msgs.messages[DirectedEdge(s₁, s₂)] = DiagonalTensorMap(Σ.data, V′ᵈ)

        @debug "apply! 2-site exit" sites = gate.sites isempty = buffer_isempty(allocator) stats = buffer_stats(allocator)
        (state, msgs, (; ϵ, logλ))
    end
end

# --- helpers -----------------------------------------------------------------

"""
    default_gauge_tol(x) -> Real

Default eigenvalue floor for the gauge pseudo-inverse in [`_eigh_sqrt`](@ref),
derived from the scalar type of `x` (a state, messages, or tensor). BP messages
are normalized, so an absolute floor at `eps^(3/4)` cleanly separates the
numerical-noise eigenvalues — which must *not* be inverted, or the gauge blows
up — from physical Schmidt weight. Pass `gauge_tol` to [`apply!`](@ref) to
override it (e.g. `0` to disable clipping).
"""
default_gauge_tol(x) = eps(real(scalartype(x)))^(3 // 4)

safe_sqrt(x, tol::Real) = x > tol ? sqrt(x) : zero(x)
safe_inv(x) = iszero(x) ? x : inv(x)

# Eigh-based square root and pseudo-inverse of a Hermitian PSD message,
# clipping eigenvalues at or below `tol` (see [`default_gauge_tol`](@ref)).
function _eigh_sqrt(m; tol::Real = default_gauge_tol(m))
    @assert tol ≥ 0 "Tolerance must be non-negative"
    D, U = eigh_full(m)
    dD = parent(diagview(D))
    dD .= safe_sqrt.(dD, tol)
    Λ = lmul!(D, copy(U'))
    dD .= safe_inv.(dD)
    Λ⁻¹ = rmul!(U, D)
    return Λ, Λ⁻¹
end

# `(leg_index, L, Linv)` for each non-partner neighbour of `first(edge)`.
function _gauge_factors(
        state::TensorNetworkState, msgs::BPMessages, edge::DirectedEdge; tol::Real,
    )
    M = eltype(msgs)
    factors = Vector{Tuple{Int, M, M}}()
    site = first(edge)
    for incoming in incoming_edges(state, site; exclude = (last(edge),))
        k = leg_index(state, reverse(incoming))
        m = msgs[incoming]
        Λ, Λ⁻¹ = _eigh_sqrt(m; tol)
        isdual(space(m, 1)) && twist!(Λ⁻¹, 1)
        push!(factors, (k, Λ, Λ⁻¹))
    end
    return factors
end
