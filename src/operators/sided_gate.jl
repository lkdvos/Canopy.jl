# --- sided gates: which physical leg(s) of an operator a gate acts on ---------
#
# A `TensorNetworkOperator` stores the vectorization of a linear map: slot 1 is the ket
# (row) index and slot 2 the bra (column) index, carrying `dual` of the row space. A gate
# can therefore act from the left, from the right, or from both sides, and the wrapper
# types below record which.
#
# The duality convention is what makes this cheap: a *dual* codomain leg contracts directly
# against a *non-dual* gate leg. So for a wrapped gate `G : P′ ← P`,
#
#   ρ → Gρ    contract slot 1 against G's DOMAIN     (leaves codomain(G)[i]        )
#   ρ → ρG    contract slot 2 against G's CODOMAIN   (leaves dual(domain(G)[i])    )
#   ρ → GρG†  both, with G† = G' on the bra side
#
# and no `transpose` or `flip` appears anywhere. Note the asymmetry: right-multiplication
# consumes the gate's *codomain* and produces its domain, because `ρG` requires ρ's column
# index to match G's row index.

"""
    LeftGate(gate)
    LeftGate(sites, tensor)

Apply `gate` to the **ket** (row) legs of a [`TensorNetworkOperator`](@ref): `ρ ↦ Gρ`.

Requires `physicalspace(op, v, 1) == domain(G)[i]` at each site, and replaces it with
`codomain(G)[i]`. With a rectangular `G` this deliberately breaks
[`isvectorized`](@ref Canopy.isvectorized) — `Gρ` for rectangular `G` genuinely is not a square operator.

See also [`RightGate`](@ref), [`SandwichGate`](@ref).
"""
struct LeftGate{T, S, V, G <: AbstractGate{T, S, V}} <: AbstractGate{T, S, V}
    gate::G
end

"""
    RightGate(gate)
    RightGate(sites, tensor)

Apply `gate` to the **bra** (column) legs of a [`TensorNetworkOperator`](@ref): `ρ ↦ ρG`.

Requires `physicalspace(op, v, 2) == dual(codomain(G)[i])` at each site, and replaces it
with `dual(domain(G)[i])`. Note this consumes the gate's *codomain*, the mirror of
[`LeftGate`](@ref): `ρG` pairs ρ's column index with `G`'s row index.

See also [`LeftGate`](@ref), [`SandwichGate`](@ref).
"""
struct RightGate{T, S, V, G <: AbstractGate{T, S, V}} <: AbstractGate{T, S, V}
    gate::G
end

"""
    SandwichGate(gate)
    SandwichGate(sites, tensor)

Apply `gate` to **both** physical legs of a [`TensorNetworkOperator`](@ref):
`ρ ↦ G ρ G†`. This is the two-sided evolution used for density matrices (imaginary time,
Heisenberg picture); it preserves hermiticity and positivity of `ρ` up to truncation.

Requires the operator to be vectorized at the acted sites
(`physicalspace(op, v, 2) == dual(physicalspace(op, v, 1))`, see [`isvectorized`](@ref Canopy.isvectorized)),
which it then preserves — even for a rectangular `G`.

Wrapping a `CompositeGate` or `Circuit` distributes over its gate list, so
a Trotter circuit is reusable as-is:

```julia
circuit = trotterize(bond_hams, dβ / 2, Strang())
apply!(ρ, msgs, SandwichGate(circuit); trunc)   # advances β by dβ
```

!!! note
    A `SandwichGate` step applies the gate on *both* sides, so evolving
    `ρ = exp(-βH)` with `exp(-dτ H)` advances `β` by `2dτ`. See the operator docs.

See also [`LeftGate`](@ref), [`RightGate`](@ref).
"""
struct SandwichGate{T, S, V, G <: AbstractGate{T, S, V}} <: AbstractGate{T, S, V}
    gate::G
end

const SidedGate{T, S, V, G} = Union{LeftGate{T, S, V, G}, RightGate{T, S, V, G}, SandwichGate{T, S, V, G}}

for W in (:LeftGate, :RightGate, :SandwichGate)
    @eval begin
        # `$W(gate)` itself is the struct's default outer constructor, which already infers
        # `T, S, V` from `G <: AbstractGate{T, S, V}`.
        $W(sites::Tuple, tensor::AbstractTensorMap) = $W(LocalGate(sites, tensor))
        Adapt.adapt_structure(to, g::$W) = $W(adapt(to, g.gate))
        # distribute over aggregates, so `trotterize` output can be lifted wholesale
        $W(gates::CompositeGate) = CompositeGate(map($W, gates.gatelist))
        $W(circuit::Circuit) = Circuit(map($W, circuit.gatelist))
    end
end

sites(g::SidedGate) = sites(g.gate)

# see `Canopy._acted_slots` in `operators/abstract_gate.jl` for what these mean
_acted_slots(::LeftGate) = (1,)
_acted_slots(::RightGate) = (2,)
_acted_slots(::SandwichGate) = (1, 2)

# the tensor actually contracted onto each acted slot, in slot order
_side_tensors(g::LeftGate) = (g.gate.tensor,)
_side_tensors(g::RightGate) = (g.gate.tensor,)
_side_tensors(g::SandwichGate) = (g.gate.tensor, g.gate.tensor')

# --- compatibility checks -----------------------------------------------------
# `_check_slot` / `_check_sites` and the plain left-action `_check_compatible` live in
# `operators/local_gate.jl`; only the side-dependent variants are here.
#
# `LeftGate`: slot 1 pairs with the gate's domain, exactly as for a state.
# `RightGate`: slot 2 (space `dual(P)`) pairs with the gate's *codomain*.

function _check_compatible(op::TensorNetworkOperator, g::LeftGate)
    return _check_compatible(op, g.gate)
end

function _check_compatible(op::TensorNetworkOperator, g::RightGate)
    gate = g.gate
    _check_sites(op, gate)
    for i in 1:length(gate.sites)
        _check_slot(op, gate.sites[i], 2, codomain(gate.tensor)[i], i)
    end
    return nothing
end

function _check_compatible(op::TensorNetworkOperator, g::SandwichGate)
    gate = g.gate
    _check_sites(op, gate)
    for i in 1:length(gate.sites)
        v = gate.sites[i]
        physicalspace(op, v, 2) == dual(physicalspace(op, v, 1)) || throw(
            SpaceMismatch(
                lazy"SandwichGate requires a vectorized operator, but site $v has physical spaces $(physicalspaces(op, v))"
            )
        )
        _check_slot(op, v, 1, domain(gate.tensor)[i], i)
        _check_slot(op, v, 2, codomain(gate.tensor')[i], i)
    end
    return nothing
end

# --- fermionic guard ----------------------------------------------------------
# The fermionic sign structure of the operator path has not been worked through: the extra
# codomain leg is re-partitioned by the QR/SVD and contracted against a gate on the bra
# side, both new braid sites relative to the state kernel documented in
# `docs/src/fermions.md`. Refuse rather than return silently wrong signs.
function _check_bosonic(op::TensorNetworkOperator)
    I = sectortype(spacetype(op))
    BraidingStyle(I) isa Fermionic && throw(
        ArgumentError(
            lazy"gate application on a TensorNetworkOperator is not yet implemented for fermionic sectors (got $I): the operator path's braiding signs have not been validated. Bosonic and bosonic-graded sectors are supported."
        )
    )
    return nothing
end

# --- single-site --------------------------------------------------------------
function apply!(
        op::TensorNetworkOperator, msgs::BPMessages, gate::SidedGate{<:Any, <:Any, <:Any, <:LocalGate{<:Any, 1}};
        timer = nothing, backend = DefaultBackend(), allocator = default_allocator(op), kwargs...,
    )
    return @maybe_timeit timer "apply! 1-site operator" begin
        _check_bosonic(op)
        _check_compatible(op, gate)
        v = only(sites(gate))
        t = op[v]
        for (slot, L) in zip(_acted_slots(gate), _side_tensors(gate))
            # slot 1 consumes the gate's domain (leg 2), slot 2 its codomain (leg 1)
            t = _absorb_slot(t, slot, L, slot == 1 ? 1 : 2, backend, allocator)
        end
        op.vertices[v] = t
        T = real(scalartype(op))
        (op, msgs, (; ϵ = zero(T), logλ = zero(T)))
    end
end
