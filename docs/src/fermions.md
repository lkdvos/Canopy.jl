# Fermionic correctness

Canopy represents fermionic states on `fℤ₂`-graded (or richer graded) spaces via
TensorKit.jl. On such spaces, re-partitioning tensor legs between the codomain
and the domain — every `permute`, `braid`, QR, SVD, or eigendecomposition —
inserts Koszul/braiding signs. Getting a fermionic PEPS algorithm right is
therefore a matter of (a) fixing *one* consistent leg order everywhere, (b)
committing to explicit decomposition directions, and (c) inserting `twist`
factors at the handful of places where the first two would otherwise disagree.

This page maps out every mechanism that keeps the fermionic code correct. None of
it is fermion-specific *code paths*: the same generic routines run for bosonic
and fermionic spaces, and the fermionic signs fall out of the conventions below
plus TensorKit's automatic braiding.

## 1. A global vertex order fixes the fermion order

Everything rests on one idea: a single global ordering of the vertex keys
(`isless`/`<`) together with canonically oriented edges pins a consistent
fermionic leg order across the whole network. The base conventions are described
under [Conventions](@ref) in the design notes; here is the fermionic consequence.

- **Leg layout.** Every on-site tensor is a `TensorMap{T,S,1,N}`: leg 1 (the
  codomain) is the *physical* leg, legs `2 … N+1` (the domain) are the *virtual*
  bonds, one per neighbor (unused legs padded with `oneunit(S)`). A bond `k`
  returned by `leg_index` therefore sits at tensor slot `k+1`.

- **Bond duality.** Input virtual spaces are required non-dual, and for an edge
  `(u, v)` the space is used as-is on the *smaller* vertex and dualised on the
  larger:

  ```julia
  vertex < neighbors[i] ? Vspace : dual(Vspace)
  ```

  Because `UndirectedEdge` always stores `src < dst` and both endpoints look up
  the *same* per-edge space, the two sides of any bond are guaranteed to be `V`
  and `dual(V)` of one underlying space — a matched contraction pair. The
  invariant is re-verified by `check_consistency`:

  ```julia
  virtualspace(state, edge) == dual(virtualspace(state, reverse(edge)))
  ```

  with the non-dual side always the smaller vertex.

- **Charge convention.** For product states the same rule is stated at the level
  of symmetry charges: `c_v = ⊗_w σ(v, w) q_e` with `σ = identity if v < w else
  dual`, and `q_e` stored on the smaller-vertex side of each edge.

State construction inserts **no** explicit twist. The `<` ordering *is* the
fermion-ordering convention; the actual braiding signs are produced by
TensorKit's `ncon` whenever tensors are contracted (for example in the dense
materialization `TensorMap(state)`).

## 2. Belief-propagation message conventions

BP messages live on the directed graph. For a directed edge `e = (s, r)`
(sender → receiver), the message `msgs[e]` lives on the *receiver's* side of the
bond and has space `V_r ← V_r`, where the **codomain is the bra leg** and the
**domain is the ket leg**. Messages are identity-initialized (exact on trees).

The bra layer is always the TensorKit adjoint `'`: it enters either through the
conjugation flag in `tensorcontract!` or as adjoint messages `incoming[k+1]'` in
the batched (vertex-centric) kernel.

Messages are absorbed into a site tensor by `_absorb_legs`, which distinguishes
two cases per leg:

```julia
if space(L, 1) == space(out, d)
    _mul_leg!(new_out, out, L, k, backend, allocator)   # space-preserving (BP message)
else
    tensorcontract!(...)                                # space-flipping (gauge √)
end
```

A BP message preserves the bond space, so it takes the `_mul_leg!` path, which
carries the **per-sector fermionic sign**:

```julia
dualleg = isdual(space(src, d))
σ = f₂.uncoupled[k]
α = dualleg ? convert(eltype(dst), twist(σ)) : One()
```

When a message is absorbed onto a **dual** virtual leg, each fusion-tree block is
scaled by `twist(σ)` (the ±1 parity of sector `σ`); on a non-dual leg the factor
is `One()`. Gauge square-root factors instead *flip* the bond space and take the
plain `tensorcontract!` path — their fermionic correction is handled separately
(see [The gauge square root and the fermionic fix](@ref)).

A message itself is the partial trace of the sender environment onto the bond:
the message-dressed ket is contracted against the bra over the physical leg and
every virtual leg *except* the target bond (and except the incoming message from
the receiver — leave-one-out), leaving the `V_r ← V_r` map with codomain = bra
and domain = ket.

## 3. Decomposition directions in simple update

The two-site `apply!` pipeline is direction-committed at every step, because on
`fℤ₂` the choice of which legs go to the codomain versus the domain fixes the
braiding signs. The stages are:

1. **Canonical site order.** If `s₁ > s₂`, the sites are swapped and the gate
   permuted so the smaller vertex is always on the SVD codomain side:

   ```julia
   if s₁ > s₂
       G = permute(G, ((2, 1), (4, 3)))
       s₁, s₂ = s₂, s₁
   end
   ```

   This fixes a single fermionic leg order regardless of the caller's site order.

2. **Gauge in.** `_gauge_factors` builds the message square root `Λ` for every
   neighbor bond *except* the shared one, and `_absorb_legs` contracts each `Λ`
   into the site (a Vidal-like BP gauge). The shared bond is deliberately skipped
   — it is about to be recomputed.

3. **QR.** Each site tensor is permuted so its **environment/spectator legs go to
   the codomain (→ Q, an isometry)** while the **physical leg and the shared-bond
   leg go to the domain (→ R)**. `R₁` and `R₂` are the minimal blocks that enter
   the gate.

4. **Gate + SVD.** The two `R` factors are contracted with each other across the
   old bond and with the gate across the physical legs, then the bond is cut:

   ```julia
   @tensor θ[-1 -2; -3 -4] := R₁[-1; 1 2] * R₂[-3; 3 2] * G[-2 -4; 1 3]
   U, Σ, Vᴴ, ϵ = svd_trunc!(θ; trunc)
   ```

   `Σ` is normalized to unit `normp`-norm (the absorbed norm is returned as
   `logλ`), and `√Σ` is split symmetrically: `U·√Σ` goes to `s₁`, `√Σ·Vᴴ` to
   `s₂`. Because `θ` places `s₁`'s legs in the codomain and `s₂`'s in the domain,
   the cut is exactly the canonical Schmidt cut across the bond.

5. **Reconstruct and gauge out.** `Q₁*U` and `Q₂*Vᴴ` re-attach the environments,
   `permute` restores the `(physical, bonds…)` leg order, and `_absorb_legs` with
   the stored inverse factors `Λ⁻¹` undoes the gauge-in on the non-shared bonds.

6. **New messages.** The fresh Schmidt spectrum `Σ` becomes both directed bond
   messages, as `DiagonalTensorMap`s on the two (dual-paired) virtual spaces.

### The gauge square root and the fermionic fix

The gauge factors come from an eigen-decomposition of the Hermitian PSD message
`m = U D U'`:

```julia
D, U = eigh_full(m)
dD .= safe_sqrt.(dD, tol)
Λ = lmul!(D, copy(U'))     # √D · U'   — absorbed on gauge-in
dD .= safe_inv.(dD)
Λ⁻¹ = rmul!(U, D)          # U · (√D)⁻¹ — absorbed on gauge-out
```

Eigenvalues at or below `tol` are floored to zero (`safe_sqrt`/`safe_inv`) rather
than inverted, which prevents numerical-noise directions from blowing up the
gauge. The floor defaults to `default_gauge_tol(x) = eps(real(scalartype(x)))^(3//4)`
and is settable via `apply!(...; gauge_tol = …)` (`0` disables clipping).

`Λ` (gauge-in) and `Λ⁻¹` (gauge-out) come from the same eigenbasis, so
numerically `Λ⁻¹ Λ = 1`. But they are absorbed on **opposite sides** of the QR/SVD
leg re-partitioning. For a dual (bra-oriented) fermionic leg the braiding path
taken on the way in is not the mirror of the way out, leaving an uncancelled
parity sign on odd-parity sectors. A single `twist!` repairs it:

```julia
Λ, Λ⁻¹ = _eigh_sqrt(m; tol)
isdual(space(m, 1)) && twist!(Λ⁻¹, 1)
```

This re-inserts exactly the missing parity factor so that gauge-in ∘ gauge-out is
the identity on `fℤ₂` too. The guard restricts it to dual legs (the non-dual case
already round-trips), and it mirrors the `isdual → twist` rule in `_mul_leg!`.

## 4. Expectation values

`reduced_density_matrix` builds a ket–bra double layer — the ket is the
message-dressed site tensor (`attach_all_messages` / `attach_messages`), the bra
is the adjoint `state[v]'` — contracts it with `ncon`, and then applies a
`twist!` on the physical **ket** legs:

```julia
# single site
ρ = twist!(repartition(ncon(tensors, indices), 1, 1), 1)
# two sites (physical ket legs 1 and 2)
ρ = twist!(repartition(ncon(tensors, indices), 2, 2), (1, 2))
```

For the two-site case the shared virtual bond between the sites is stitched by
overwriting the `ncon` index slots via `leg_index` in both directions. The result
is trace-normalized (`tr(ρ) ≈ 1`, positive-definite), so an expectation value is
just

```julia
expect(state, msgs, op, sites) = tr(op * reduced_density_matrix(sites, state, msgs))
```

All fermionic signs are already inside `ρ`; `expect` needs none of its own.

## 5. Tests that lock it in

The fermionic correctness gate is the *dense-equivalence diagnostic* in
`test/test_apply_gate.jl`. It runs a Cartesian product of five geometries —
chain, degree-4 star, 4-cycle, `K₄`, and a 3×3 periodic grid, spanning tree →
single loop → multi-loop — against two spaces:

```julia
("bosonic",   ComplexSpace(2),        ComplexSpace(2)),
("fermionic", fermion_space(Trivial), Vect[fℤ₂](0 => 1, 1 => 1)),
```

The fermionic row uses `fℤ₂`-graded virtual legs, exercising every twist above.
Each case compares `apply!` against a direct dense gate contraction
(`_dense_apply_2site`, via `ncon` on the materialized wavefunction), with
`rtol = 1e-10`, no truncation, and `normp = 0` so nothing is lost. The checks:

- single-site and two-site identity gates are no-ops;
- an untruncated random unitary reproduces the dense reference;
- the **reverse-orientation** gate `LocalGate((v, u), permute(G, ((2,1),(4,3))))`
  gives the *same* result — validating the site-order canonicalization of step 1;
- a `g ∘ g†` round-trip restores the initial state — the gauge in/out inverse
  property that the `twist!(Λ⁻¹, 1)` fix makes hold on `fℤ₂`.

The bosonic round-trip tests additionally hit the `k₁ = 1` and `k₁ = N` leg-
permutation boundary cases.

## Quick reference: every fermion-sign site

| Concern | Location | Mechanism |
|---|---|---|
| Fermion leg order | global `<` + `UndirectedEdge` | smaller vertex = non-dual side |
| Bond duality | `TensorNetworkState` constructor + `check_consistency` | `v < w ? V : dual(V)`, asserted and re-checked |
| Site-order canonicalization | `apply!` (2-site) | `permute(G, ((2,1),(4,3)))` |
| QR / SVD direction | `apply!` (2-site) | env → Q, physical + bond → R; SVD across the bond |
| Gauge sqrt + clipped pseudo-inverse | `_eigh_sqrt`, `default_gauge_tol` | floor noise eigenvalues instead of inverting |
| **Gauge twist fix** | `_gauge_factors` | `isdual(space(m,1)) && twist!(Λ⁻¹, 1)` |
| Message-absorption twist | `_mul_leg!` | `dualleg ? twist(σ) : One()` per fusion tree |
| Absorb-path selection | `_absorb_legs` | space-preserving → twist; space-flipping → none |
| Bra layer (conjugation) | `compute_message!`, `reduced_density_matrix` | adjoint `'` / conjugation flag |
| RDM physical-leg twist | `reduced_density_matrix` | `twist!(ρ, 1)` / `twist!(ρ, (1, 2))` |
| Correctness gate | `test/test_apply_gate.jl` | dense-equivalence diagnostic, fermionic row |
