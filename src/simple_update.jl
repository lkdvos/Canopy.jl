# --- two-site BP-gauge simple update -----------------------------------------
#
# One kernel serves a state and all three sided operator gates. The only things that vary
# are `_acted_slots(gate)` — which physical slots enter the `R` factor rather than staying in
# the environment — and the `θ` contraction, `_gate_theta` below.
#
# Because a leg the gate does not touch stays in `Q`, a one-sided gate on an operator costs
# exactly what the same gate costs on a state. Only `SandwichGate` pays the `d²` price.

apply!(state::TensorNetworkState, msgs::BPMessages, gate::LocalGate{<:Any, 2}; kwargs...) =
    _apply_2site!(state, msgs, gate; kwargs...)

function apply!(
        op::TensorNetworkOperator, msgs::BPMessages,
        gate::SidedGate{<:Any, <:Any, <:Any, <:LocalGate{<:Any, 2}}; kwargs...,
    )
    _check_bosonic(op)
    return _apply_2site!(op, msgs, gate; kwargs...)
end

# the raw gate tensor behind a bare or wrapped gate
_gate_tensor(gate::LocalGate) = gate.tensor
_gate_tensor(gate::SidedGate) = _gate_tensor(gate.gate)

# `θ` glues the two `R` factors across the old bond and applies the gate. In every variant
# `θ`'s codomain holds site 1's legs and its domain site 2's, so `svd_trunc!` cuts exactly
# across the bond. A contracted index sits on the gate's *domain* for a left action and on
# its *codomain* for a right action — that asymmetry is the whole content of `ρG` pairing ρ's
# column index with `G`'s row index, and the dual slot-2 leg supplies the transpose for free.
function _gate_theta(::Union{LocalGate, LeftGate}, R₁, R₂, G, Gd, backend, allocator)
    @tensor backend = backend allocator = allocator θ[-1 -2; -3 -4] :=
        R₁[-1; 1 2] * R₂[-3; 3 2] * G[-2 -4; 1 3]
    return θ
end
function _gate_theta(::RightGate, R₁, R₂, G, Gd, backend, allocator)
    @tensor backend = backend allocator = allocator θ[-1 -2; -3 -4] :=
        R₁[-1; 1 2] * R₂[-3; 3 2] * G[1 3; -2 -4]
    return θ
end
function _gate_theta(::SandwichGate, R₁, R₂, G, Gd, backend, allocator)
    @tensor backend = backend allocator = allocator θ[-1 -2 -3; -4 -5 -6] :=
        R₁[-1; 1 2 5] * R₂[-4; 3 4 5] * G[-2 -5; 1 3] * Gd[2 4; -3 -6]
    return θ
end

function _apply_2site!(
        state::AbstractTensorNetwork, msgs::BPMessages, gate;
        trunc = notrunc(), gauge_tol::Real = default_gauge_tol(state), normp::Real = 2,
        timer = nothing, backend = DefaultBackend(), allocator = default_allocator(state),
    )
    return @maybe_timeit timer "apply! 2-site" begin
        _check_compatible(state, gate)
        gatesites = sites(gate)
        s₁, s₂ = gatesites
        G = _gate_tensor(gate)

        @debug "apply! 2-site entry" gatesites isempty = buffer_isempty(allocator) stats = buffer_stats(allocator)

        # The buffer-allocating steps below are each self-contained: `_absorb_legs`
        # (gauge in / reconstruct) frees its temporaries and resets the buffer, and
        # the `@tensor` gate contraction does the same automatically. The new site
        # tensors and bond messages are heap-allocated and escape. The
        # MatrixAlgebraKit factorizations (`qr_compact!`, `svd_trunc!`, `eigh_full`)
        # do not use this allocator and always allocate on the heap.

        # Canonical orientation: smaller vertex on the codomain side of the SVD. Canonicalize
        # `G` *before* deriving the bra-side factor, so it inherits the swap.
        if s₁ > s₂
            G = permute(G, ((2, 1), (4, 3)))
            s₁, s₂ = s₂, s₁
        end
        Gd = gate isa SandwichGate ? G' : nothing

        k₁ = leg_index(state, DirectedEdge(s₁, s₂))
        k₂ = leg_index(state, DirectedEdge(s₂, s₁))
        Nd = numin(state[s₁])
        np = numout(state[s₁])
        M = np + Nd
        # Legs entering `R`: the acted physical slots plus the shared bond. Everything else —
        # including any physical slot the gate does not touch — goes to `Q`.
        acted = _acted_slots(gate)
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
            θ = _gate_theta(gate, R₁, R₂, G, Gd, backend, allocator)
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
            # `Q * X` re-attaches the environment, leaving legs in the order
            # `(qlegs..., acted..., new bond)` for site 1 and `(qlegs..., new bond, acted...)`
            # for site 2. Inverting that permutation puts every leg back in its original slot,
            # with the new bond taking the shared bond's place.
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
