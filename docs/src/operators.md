# Operators and density matrices

A [`TensorNetworkOperator`](@ref) is the two-physical-leg counterpart of a
[`TensorNetworkState`](@ref): where a state carries one physical leg per site, an operator
carries two, and represents a global operator or a density matrix. Finite-temperature states
and Heisenberg-picture evolution both live here.

## The vectorization convention

Both physical legs sit in the **codomain**, as the vectorization of a linear map:

```
op[v] :: P_v ⊗ dual(P_v) ← V_1 ⊗ … ⊗ V_N
```

- slot 1 is the **ket** (row) index, carrying `P_v`;
- slot 2 is the **bra** (column) index, carrying `dual(P_v)`;
- slots `3 … N+2` are the virtual legs, exactly as for a state — same per-edge storage, same
  non-dual-on-the-smaller-vertex duality rule, same `oneunit(S)` padding.

So virtual leg `k` sits at tensor slot `k + 2` rather than `k + 1`. Code that needs the
offset reads it off the tensor with `numout`, or off the network with
[`Canopy.num_physical`](@ref).

Two consequences are worth stating explicitly.

**The duality is what removes every transpose.** A *dual* codomain leg contracts directly
against a *non-dual* gate leg, so right-multiplication needs no `transpose` and no `flip` —
see the gate table below.

**Non-square operators are legal.** The type does not require slot 2 to be `dual` of slot 1;
[`Canopy.isvectorized`](@ref) reports whether it is. A network that fails that predicate is a
perfectly good *purification* — one physical leg and one ancilla — and supports one-sided
gates freely. It just has no `ρ ↦ GρG†`.

## Construction

```julia
using Canopy, TensorKit

es = square_lattice(3, 3)
P = ComplexSpace(2)

ρ = identity_operator(ComplexF64, es, P)     # 𝟙, i.e. β = 0
op = randn_operator(ComplexF64, es, P, ComplexSpace(4))
```

[`identity_operator`](@ref) is the anchor for finite-temperature work. Its bonds are all
`oneunit(S)`, so it is a product operator that needs *no* bond-charge solve — unlike
[`product_state`](@ref), which has to route nontrivial local charges through its bonds, the
identity carries trivial total charge. One-dimensional bonds also make belief propagation
exact on it. Note it is **not** trace-normalized: `tr(𝟙) = ∏_v dim(P_v)`.

`pspaces` follows the [`product_state`](@ref) convention — either a single space for every
vertex or a `Dictionary` keyed by vertex:

```julia
identity_operator(ComplexF64, es, Dictionary(verts, myspaces))
```

[`randn_operator`](@ref) and [`rand_operator`](@ref) mirror
[`randn_state`](@ref)/[`rand_state`](@ref), minus the `total_charge` keyword: the operator
analogue of a charge bath is a charge-*shifting* operator, which needs its own semantics.
A random operator is neither Hermitian nor positive — for a physical density matrix, start
from `identity_operator` and evolve.

`TensorNetworkOperator(state)` lifts a state by inserting a trivial bra leg. That is
`|ψ⟩` viewed as an operator, *not* `|ψ⟩⟨ψ|`, which would need bond dimension `χ²`.

## Belief propagation comes for free

Fusing two adjacent codomain legs is a pure reinterpretation of the same storage, so an
operator is handed to the *existing* belief propagation as a state on the fused physical
space:

```julia
msgs = BPMessages(ρ)                                  # forwards through the fused view
msgs = belief_propagation(msgs, ρ; maxiter = 50, tol = 1e-10)
apply!(ρ, msgs, gate; trunc)                          # messages are usable on the operator
```

`TensorNetworkState(op)` is that
view. It copies nothing, and the messages it defines are *identical* to the ones the operator
defines, because the fusion map is unitary and belief propagation closes every physical leg
between the ket and the bra at the same vertex. Not one line of the BP kernels knows that
operators exist.

Two caveats:

- The view **aliases** the operator's storage, and goes stale as soon as `apply!` rebinds a
  vertex tensor. Build it where it is used; do not cache it.
- Its physical space is `fuse(P ⊗ dual(P))`, so a `reduced_density_matrix` taken on it
  is expressed in fused indices.

This is also the one place Canopy deliberately depends on TensorKit's internal block layout.
`test/test_operators.jl` guards the assumption directly, so that a layout change fails loudly
there instead of quietly corrupting BP.

## Gate application: one-sided and two-sided

The `action::`[`GateAction`](@ref) keyword of [`apply!`](@ref) says which physical legs a gate
acts on. For `G : P′ ← P` at gate slot `i`:

| `action` | Action | Slot 1 (ket) | Slot 2 (bra) | Requires |
|---|---|---|---|---|
| [`LeftAction`](@ref) | `ρ ↦ Gρ` | `G`'s **domain** | untouched | `physicalspace(op, v, 1) == domain(G)[i]` |
| [`RightAction`](@ref) | `ρ ↦ ρG` | untouched | `G`'s **codomain** | `physicalspace(op, v, 2) == dual(codomain(G)[i])` |
| [`SandwichAction`](@ref) | `ρ ↦ GρG†` | `G`'s domain | `G†`'s codomain | both, i.e. [`isvectorized`](@ref Canopy.isvectorized) |

```julia
apply!(ρ, msgs, LocalGate((u, v), G); trunc = truncrank(χ))                     # SandwichAction
apply!(ρ, msgs, LocalGate((u, v), G); action = LeftAction, trunc)
```

`SandwichAction` is the default, since two-sided evolution is what a density matrix wants. On a
[`TensorNetworkState`](@ref) the keyword is an error rather than a silent no-op: a state has a
single physical leg and hence no choice to make.

Note the asymmetry in the table: right-multiplication consumes the gate's **codomain** and
produces its domain, the mirror of the left action. That is not a quirk of the implementation
— `ρG` pairs `ρ`'s column index with `G`'s row index, and the dual slot-2 leg supplies the
transpose for free.

Because the action is a keyword rather than part of the gate, gates carry no side information
and the aggregates need no special casing: `apply!` forwards its keywords to every gate in a
`CompositeGate` or `Circuit`, so a Trotter circuit is reusable verbatim under any action:

```julia
circuit = trotterize(bond_hams, dτ, Strang())
apply!(ρ, msgs, circuit; trunc = truncrank(χ))                  # two-sided
apply!(X, msgs, circuit; action = LeftAction, trunc)            # one-sided, e.g. on a purification
```

### Cost

A one-sided gate costs exactly what the same gate costs on a state: the physical leg the gate
does *not* touch stays in the environment `Q` rather than entering the `R` factor, so `R`
remains `d·χ`. Only [`SandwichAction`](@ref) puts both legs in `R`, paying `d²` there and up to
`d²χ` in the new bond dimension. That is inherent to two-sided evolution, and is the honest
argument for the one-sided/purification workflow below.

## Temperature bookkeeping: the factor of two

Belief propagation on a two-physical-leg network converges the environment of `tr(ρ†ρ)`, not
`tr(ρ)` — `compute_message!` closes ket against bra over *all* physical legs. This is the
right and the only free choice, and it has one consequence that is very easy to get wrong:

- Evolving `ρ = exp(-βH)` two-sided with `exp(-dτ H)` advances `β` by **`2dτ`**
  per step, because the gate is applied on both sides.
- Equivalently, and often more convenient: treat the network as `X = exp(-βH/2)` and evolve
  it **one-sided**. Then `⟨O⟩ = tr(X†OX) / tr(X†X)` is the thermal average at `β`, and it is
  computable with the existing double-layer machinery — `X` is exactly a purification.

A true `trace(ρ)` is *not* provided. Closing slot 1 against slot 2 at every vertex leaves a
**single-layer** network, whose Bethe approximation needs vector-valued messages and a
different fixed-point iteration. That is a new algorithm, not a missing accessor.

## Fermions are not yet supported

Gate application on a `TensorNetworkOperator` throws for fermionic sectortypes rather than
returning silently wrong signs. The operator path adds braid sites the state analysis in
[Fermionic correctness](@ref) does not cover: the extra codomain leg is re-partitioned by the
QR/SVD, and it is contracted against a gate on the bra side. Bosonic and bosonic-graded
sectors (`Trivial`, `U₁`, `SU₂`, …) are supported.

The operator *type*, its constructors and the fused view are fermion-safe — only `apply!` is
gated.

## Operator API reference

```@docs
AbstractTensorNetwork
TensorNetworkOperator
Canopy.num_physical
Canopy.isvectorized
identity_operator
randn_operator
rand_operator
Canopy.physicalspaces
GateAction
LeftAction
RightAction
SandwichAction
```
