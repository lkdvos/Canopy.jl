# --- gate actions: which physical leg(s) of an operator a gate acts on --------
#
# A `TensorNetworkOperator` stores the vectorization of a linear map: slot 1 is the ket
# (row) index and slot 2 the bra (column) index, carrying `dual` of the row space. A gate
# can therefore act from the left, from the right, or from both sides, and the `action`
# keyword of `apply!` records which.
#
# The duality convention is what makes this cheap: a *dual* codomain leg contracts directly
# against a *non-dual* gate leg. So for a gate `G : P′ ← P`,
#
#   ρ → Gρ    contract slot 1 against G's DOMAIN     (leaves codomain(G)[i]        )
#   ρ → ρG    contract slot 2 against G's CODOMAIN   (leaves dual(domain(G)[i])    )
#   ρ → GρG†  both, with G† = G' on the bra side
#
# and no `transpose` or `flip` appears anywhere. Note the asymmetry: right-multiplication
# consumes the gate's *codomain* and produces its domain, because `ρG` requires ρ's column
# index to match G's row index.

"""
    GateAction

Which physical leg(s) of a [`TensorNetworkOperator`](@ref) a gate acts on, selected by the
`action` keyword of [`apply!`](@ref). For a gate `G : P′ ← P` and gate slot `i`:

| Value | Action | Slot 1 (ket) | Slot 2 (bra) | Requires |
|---|---|---|---|---|
| [`LeftAction`](@ref) | `ρ ↦ Gρ` | `G`'s **domain** | untouched | `physicalspace(op, v, 1) == domain(G)[i]` |
| [`RightAction`](@ref) | `ρ ↦ ρG` | untouched | `G`'s **codomain** | `physicalspace(op, v, 2) == dual(codomain(G)[i])` |
| [`SandwichAction`](@ref) | `ρ ↦ GρG†` | `G`'s domain | `G†`'s codomain | both, i.e. [`isvectorized`](@ref Canopy.isvectorized) |

`apply!` on an operator defaults to `SandwichAction`, the two-sided evolution used for density
matrices. The keyword threads through [`CompositeGate`](@ref) and [`Circuit`](@ref) to every
gate they contain, so a Trotter circuit is reusable as-is:

```julia
circuit = trotterize(bond_hams, dβ / 2, Strang())
apply!(ρ, msgs, circuit; trunc)     # SandwichAction by default: advances β by dβ
```

A [`TensorNetworkState`](@ref) has a single physical leg and hence no choice to make; the
keyword is rejected there rather than silently ignored.
"""
@enum GateAction LeftAction RightAction SandwichAction

@doc """
    LeftAction::GateAction

Apply the gate to the **ket** (row) legs of a [`TensorNetworkOperator`](@ref): `ρ ↦ Gρ`.

Requires `physicalspace(op, v, 1) == domain(G)[i]` at each site, and replaces it with
`codomain(G)[i]`. With a rectangular `G` this deliberately breaks
[`isvectorized`](@ref Canopy.isvectorized) — `Gρ` for rectangular `G` genuinely is not a
square operator.

See also [`GateAction`](@ref), [`RightAction`](@ref), [`SandwichAction`](@ref).
""" LeftAction

@doc """
    RightAction::GateAction

Apply the gate to the **bra** (column) legs of a [`TensorNetworkOperator`](@ref): `ρ ↦ ρG`.

Requires `physicalspace(op, v, 2) == dual(codomain(G)[i])` at each site, and replaces it with
`dual(domain(G)[i])`. Note this consumes the gate's *codomain*, the mirror of
[`LeftAction`](@ref): `ρG` pairs ρ's column index with `G`'s row index.

See also [`GateAction`](@ref), [`LeftAction`](@ref), [`SandwichAction`](@ref).
""" RightAction

@doc """
    SandwichAction::GateAction

Apply the gate to **both** physical legs of a [`TensorNetworkOperator`](@ref): `ρ ↦ G ρ G†`.
This is the two-sided evolution used for density matrices (imaginary time, Heisenberg
picture); it preserves hermiticity and positivity of `ρ` up to truncation, and is the default
action of [`apply!`](@ref) on an operator.

Requires the operator to be vectorized at the acted sites
(`physicalspace(op, v, 2) == dual(physicalspace(op, v, 1))`, see [`isvectorized`](@ref Canopy.isvectorized)),
which it then preserves — even for a rectangular `G`.

!!! note
    A sandwich step applies the gate on *both* sides, so evolving `ρ = exp(-βH)` with
    `exp(-dτ H)` advances `β` by `2dτ`. See the operator docs.

See also [`GateAction`](@ref), [`LeftAction`](@ref), [`RightAction`](@ref).
""" SandwichAction

# Which physical (codomain) slots of an on-site tensor each action touches — `(1,)` / `(2,)` /
# `(1, 2)` — is the single parameter the two-site kernel needs: slots *not* acted on stay in the
# QR environment, so a one-sided operator gate costs exactly what the same gate costs on a
# state. It is a *tuple length*, so the kernel gets it as a literal from a hand-written union
# split rather than from a function of the action; see `simple_update.jl`.

# A state has one physical leg and hence nothing to choose. Rejecting the keyword matters
# because the one-site methods swallow unknown keywords by design (so a circuit can hand
# `trunc` and friends to every gate), and silently applying a left action where a right one
# was asked for would be a correctness bug.
_check_no_action(::Nothing) = nothing
_check_no_action(action) = throw(
    ArgumentError(
        lazy"`action = $action` is only meaningful for a TensorNetworkOperator; a TensorNetworkState has a single physical leg"
    )
)

# --- compatibility checks -----------------------------------------------------
# `_check_slot` / `_check_sites` and the plain left-action `_check_compatible` live in
# `operators/local_gate.jl`; only the action-dependent variants are here.
#
# `LeftAction`: slot 1 pairs with the gate's domain, exactly as for a state.
# `RightAction`: slot 2 (space `dual(P)`) pairs with the gate's *codomain*.

function _check_compatible(op::TensorNetworkOperator, gate::LocalGate, action::GateAction)
    action === LeftAction && return _check_compatible(op, gate)
    _check_sites(op, gate)
    if action === RightAction
        for (i, v) in enumerate(gate.sites)
            _check_slot(op, v, 2, codomain(gate.tensor)[i], i)
        end
    else
        for (i, v) in enumerate(gate.sites)
            physicalspace(op, v, 2) == dual(physicalspace(op, v, 1)) || throw(
                SpaceMismatch(
                    lazy"SandwichAction requires a vectorized operator, but site $v has physical spaces $(physicalspaces(op, v))"
                )
            )
            _check_slot(op, v, 1, domain(gate.tensor)[i], i)
            _check_slot(op, v, 2, codomain(gate.tensor')[i], i)
        end
    end
    return nothing
end

# --- fermions -----------------------------------------------------------------
# The operator path needs no fermion-specific code. Its two extra braid sites relative to the
# state kernel — the second codomain leg crossing the QR/SVD partition, and the bra-side gate
# contraction — are both handled by TensorKit's automatic braiding, because every leg move here
# goes through `permute`/`repartition`/`tensorcontract!` rather than through the space-preserving
# `_mul_leg!` fast path (which carries its own twist). In particular right-multiplication on a
# dual codomain leg needs *no* compensating `twist!`: the dual leg is exactly what supplies the
# transpose, and the braiding that comes with it is the one the contraction already asks for.
# `test/test_apply_gate_operator.jl` runs the dense-equivalence sweep on `fℤ₂` and `fℤ₂ ⊠ U(1)`
# to keep this honest; see `docs/src/fermions.md`.

# --- single-site --------------------------------------------------------------
function apply!(
        op::TensorNetworkOperator, msgs::BPMessages, gate::LocalGate{<:Any, 1};
        action::GateAction = SandwichAction,
        timer = nothing, backend = DefaultBackend(), allocator = default_allocator(op), kwargs...,
    )
    return @maybe_timeit timer "apply! 1-site operator" begin
        _check_compatible(op, gate, action)
        v = only(sites(gate))
        t = op[v]
        # Absorbing into slot 1 consumes the gate's domain (its leg 2), into slot 2 its codomain
        # (leg 1). Split by hand: each branch absorbs a statically known number of tensors, of
        # statically known types — `gate.tensor'` is not the same type as `gate.tensor`.
        if action === LeftAction
            t = _absorb_slot(t, 1, gate.tensor, 1, backend, allocator)
        elseif action === RightAction
            t = _absorb_slot(t, 2, gate.tensor, 2, backend, allocator)
        else
            t = _absorb_slot(t, 1, gate.tensor, 1, backend, allocator)
            t = _absorb_slot(t, 2, gate.tensor', 2, backend, allocator)
        end
        op.vertices[v] = t
        T = real(scalartype(op))
        (op, msgs, (; ϵ = zero(T), logλ = zero(T)))
    end
end
