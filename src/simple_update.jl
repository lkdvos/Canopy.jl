# --- two-site BP-gauge simple update -----------------------------------------
#
# One kernel serves a state and all three operator gate actions. The only things that vary
# are `acted` — which physical slots enter the `R` factor rather than staying in the
# environment — and the `θ` contraction.
#
# Because a leg the gate does not touch stays in `Q`, a one-sided gate on an operator costs
# exactly what the same gate costs on a state. Only `SandwichAction` pays the `d²` price.
#
# The `action` is a runtime value, but `length(acted)` is a *tuple length* the kernel derives
# every leg permutation from, and the three `θ`s differ in rank. So `_apply_2site!` union-splits
# the action by hand: each branch pins both as types — a literal slot tuple, and its own `θ`
# contraction as a `do` block, which is a distinct closure type — and `_2site_kernel!` then
# infers exactly as it would for a single fixed action.

function apply!(
        state::TensorNetworkState, msgs::BPMessages, gate::LocalGate{<:Any, 2};
        action = nothing, kwargs...,
    )
    _check_no_action(action)
    _check_compatible(state, gate)
    # a state's single physical leg is exactly the left action
    return _apply_2site!(state, msgs, gate, LeftAction; kwargs...)
end

function apply!(
        op::TensorNetworkOperator, msgs::BPMessages, gate::LocalGate{<:Any, 2};
        action::GateAction = SandwichAction, kwargs...,
    )
    _check_compatible(op, gate, action)
    return _apply_2site!(op, msgs, gate, action; kwargs...)
end

# `θ` glues the two `R` factors across the old bond and applies the gate. In all three branches
# `θ`'s codomain holds site 1's legs and its domain site 2's, so `svd_trunc!` cuts exactly
# across the bond. A contracted index sits on the gate's *domain* for a left action and on
# its *codomain* for a right action — that asymmetry is the whole content of `ρG` pairing ρ's
# column index with `G`'s row index, and the dual slot-2 leg supplies the transpose for free.
function _apply_2site!(
        net::AbstractTensorNetwork, msgs::BPMessages, gate::LocalGate{<:Any, 2},
        action::GateAction; kwargs...,
    )
    if action === LeftAction
        return _2site_kernel!(net, msgs, gate, (1,); kwargs...) do R₁, R₂, G, backend, allocator
            @tensor backend = backend allocator = allocator θ[-1 -2; -3 -4] :=
                R₁[-1; 1 2] * R₂[-3; 3 2] * G[-2 -4; 1 3]
        end
    elseif action === RightAction
        return _2site_kernel!(net, msgs, gate, (2,); kwargs...) do R₁, R₂, G, backend, allocator
            @tensor backend = backend allocator = allocator θ[-1 -2; -3 -4] :=
                R₁[-1; 1 2] * R₂[-3; 3 2] * G[1 3; -2 -4]
        end
    else
        return _2site_kernel!(net, msgs, gate, (1, 2); kwargs...) do R₁, R₂, G, backend, allocator
            Gd = G'
            @tensor backend = backend allocator = allocator θ[-1 -2 -3; -4 -5 -6] :=
                R₁[-1; 1 2 5] * R₂[-4; 3 4 5] * G[-2 -5; 1 3] * Gd[2 4; -3 -6]
        end
    end
end

function _2site_kernel!(
        theta::F, state::AbstractTensorNetwork, msgs::BPMessages, gate::LocalGate{<:Any, 2},
        acted::Tuple{Vararg{Int}};
        trunc = notrunc(), gauge_tol::Real = default_gauge_tol(state), normp::Real = 2,
        timer = nothing, backend = DefaultBackend(), allocator = default_allocator(state),
    ) where {F}
    # `apply!` has no blocked variant; unwrap a kernel selector here so the
    # `@tensor` gate contraction below sees a real TensorOperations backend.
    backend = inner_backend(backend)
    return @maybe_timeit timer "apply! 2-site" begin
        gatesites = sites(gate)
        s₁, s₂ = gatesites
        G = gate.tensor

        @debug "apply! 2-site entry" gatesites isempty = buffer_isempty(allocator) stats = buffer_stats(allocator)

        # Canonical orientation: smaller vertex on the codomain side of the SVD. Canonicalize
        # `G` *before* `theta` derives the bra-side factor from it, so that inherits the swap.
        if s₁ > s₂
            G = permute(G, ((2, 1), (4, 3)))
            s₁, s₂ = s₂, s₁
        end

        k₁ = leg_index(state, DirectedEdge(s₁, s₂))
        k₂ = leg_index(state, DirectedEdge(s₂, s₁))
        Nd = numin(state[s₁])
        np = numout(state[s₁])
        M = np + Nd
        # Legs entering `R`: the acted physical slots plus the shared bond. Everything else —
        # including any physical slot the gate does not touch — goes to `Q`.
        rlegs₁ = (acted..., np + k₁)
        rlegs₂ = (acted..., np + k₂)
        qlegs₁ = TupleTools.deleteat(ntuple(identity, M), rlegs₁)
        qlegs₂ = TupleTools.deleteat(ntuple(identity, M), rlegs₂)

        # Absorb square root factors and factorize
        T₁, T₂, gauge₁, gauge₂ = @maybe_timeit timer "gauge in" begin
            g₁ = _gauge_factors(state, msgs, DirectedEdge(s₁, s₂); tol = gauge_tol)
            t₁ = _absorb_legs(state[s₁], (k => L for (k, L, _) in g₁), backend, allocator)
            g₂ = _gauge_factors(state, msgs, DirectedEdge(s₂, s₁); tol = gauge_tol)
            t₂ = _absorb_legs(state[s₂], (k => L for (k, L, _) in g₂), backend, allocator)
            (t₁, t₂, g₁, g₂)
        end

        Q₁, R₁, Q₂, R₂ = @maybe_timeit timer "QR" begin
            q₁, r₁ = qr_compact!(permute(T₁, (qlegs₁, rlegs₁)))
            q₂, r₂ = qr_compact!(permute(T₂, (qlegs₂, rlegs₂)))
            (q₁, r₁, q₂, r₂)
        end

        # Apply gate and factorize
        U, Σ, Vᴴ, logλ, ϵ = @maybe_timeit timer "gate+SVD" begin
            θ = theta(R₁, R₂, G, backend, allocator)
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
            na = length(acted)
            X₁ = permute(U, ((1,), ntuple(i -> i + 1, na + 1)))
            p₁ = TupleTools.invperm((qlegs₁..., acted..., np + k₁))
            T₁ = permute(
                Q₁ * X₁,
                (ntuple(i -> p₁[i], np), ntuple(j -> p₁[np + j], Nd)),
            )
            state.vertices[s₁] = _absorb_legs(T₁, (k => Linv for (k, _, Linv) in gauge₁), backend, allocator)

            X₂ = permute(Vᴴ, ((2,), (1, ntuple(i -> i + 2, na)...)))
            p₂ = TupleTools.invperm((qlegs₂..., np + k₂, acted...))
            T₂ = permute(
                Q₂ * X₂,
                (ntuple(i -> p₂[i], np), ntuple(j -> p₂[np + j], Nd)),
            )
            state.vertices[s₂] = _absorb_legs(T₂, (k => Linv for (k, _, Linv) in gauge₂), backend, allocator)
        end

        # Bond messages from the new Schmidt diagonal
        V′ = virtualspace(state, DirectedEdge(s₁, s₂))
        msgs.messages[DirectedEdge(s₂, s₁)] = DiagonalTensorMap(Σ.data, V′)
        V′ᵈ = virtualspace(state, DirectedEdge(s₂, s₁))
        msgs.messages[DirectedEdge(s₁, s₂)] = DiagonalTensorMap(Σ.data, V′ᵈ)

        @debug "apply! 2-site exit" gatesites isempty = buffer_isempty(allocator) stats = buffer_stats(allocator)
        (state, msgs, (; ϵ, logλ))
    end
end

# --- helpers -----------------------------------------------------------------

"""
    default_gauge_tol(x) -> Real

Default eigenvalue floor for the gauge pseudo-inverse in `_eigh_sqrt`,
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
        state::AbstractTensorNetwork, msgs::BPMessages, edge::DirectedEdge; tol::Real,
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
