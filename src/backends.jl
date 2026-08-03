# --- kernel selection ---------------------------------------------------------
#
# The vertex-batched BP message kernel has two formulations (see `docs/src/design.md`):
# the `Layout(k)` *blocked* one, which is what runs, and a *pairwise* one kept as the
# differential-test oracle. There is no longer a selection **rule** — blocked runs
# unconditionally — so the only selector left is the one that forces the oracle.
#
# `PairwiseBackend` is a **selector, not a backend**: no `TO.tensoradd!` /
# `TO.tensorcontract!` method accepts it, deliberately. Adding forwarding methods would
# risk ambiguity with TensorOperations' own `(::AbstractArray, …, ::StridedBackend, …)`
# methods, and would make a kernel that forgot to unwrap silently take the wrong path.
# Instead every kernel entry point unwraps with [`inner_backend`](@ref) on its first
# line, so a missed site fails loudly with a `MethodError`.

"""
    PairwiseBackend(inner = DefaultBackend())

Force the **pairwise** formulation of the vertex-batched BP message kernel
([`compute_message!`](@ref)), with `inner` as the underlying TensorOperations backend.

**A testing and benchmarking handle, not a production one.** An ordinary backend runs the
`Layout(k)` blocked formulation unconditionally. The pairwise formulation is retained
because it is the oracle the blocked kernel is differentially tested against, and the arm
the A/B in `benchmark/reports/backend_ab.md` measures.

Only the vertex-batched message kernel is specialized; every other kernel simply uses
`inner`. The single-edge [`compute_message!`](@ref) and the residual-driven schedules that
use it are unaffected.

## There used to be a selection rule, and every condition in it was wrong

`uses_blocked_kernel(spacetype, storagetype)` gated the blocked kernel on four conditions.
All four are gone, and the history is recorded because three were removed by *measuring* an
argument that had only been asserted:

- **abelian fusion** (`UniqueFusion`) — never needed. Every step of the blocked chain is a
  `TensorMap`-level operation that handles a general fusion style itself, and no step has
  to recover all uncoupled sectors from one coupled label. Non-abelian is in fact where the
  blocked kernel wins *most* (`fz2_su2` 2.30×, `su2` 1.58× at χ = 64).
- **`I !== Trivial`** — justified by the claim that the pairwise path short-circuits via
  `has_array_view` to plain-array TensorOperations and could not be beaten. Never measured,
  and false: blocked is 1.10-1.17× faster there over χ ∈ 32…128. The short-circuit does
  save pairwise the fusion-tree overhead, but it does not make index permutations free, so
  blocked's `2d` copy passes against pairwise's `≈4d - 2` still decide it.
- **`numout(t) == 1`** — the chain now threads `numout(T)` through its slot arithmetic.
  That removed an assumption rather than enabling a feature: nothing can currently supply a
  two-physical-leg tensor (`docs/src/design.md`, "Physical arity").
- **`A <: Array`** — removed deliberately, and this is the one to know about:
  **GPU storage now takes the blocked kernel and there is no GPU test anywhere.** The
  kernel is written to be storage-generic and `default_allocator` already routes
  non-`Array` storage away from the CPU Bumper buffer, so there is no *known* defect — but
  neither formulation has ever been run on a GPU. If GPU BP matters, the first thing to add
  is a differential test, not a predicate. And the CPU A/B does not transfer automatically:
  blocked issues two operations per chain step where pairwise issues one fused contraction,
  and each `mul!` is one gemm *per coupled sector*, so on a device where launch overhead
  dominates, the graded fixtures could invert.

## What the A/B actually says, now that nothing is gated on it

`benchmark/reports/backend_ab.md`, seven symmetries × χ ∈ {64, 128}, each row carrying its
own control arm (a second blocked arm, so `control` is byte-identical code against itself;
they land at 0.962-1.014, i.e. a ±4 % floor):

- ratios run **1.04 to 2.48**, and the wins concentrate exactly where the formulation was
  designed to win — smallest mean subblock first: `fz2_su2` 2.48/1.89, `su2` 1.68/1.50,
  `fz2_u1` 1.74/1.23;
- `:trivial` is 1.17/1.12 with `maxdiff = 0`, i.e. the two formulations agree *bitwise*
  there;
- `:z2` at χ = 64 is the weakest row and the only one that has not cleanly cleared its
  control: 1.036 here against 0.987 in an immediately preceding run, i.e. somewhere around
  parity-to-slightly-ahead with a per-arm spread near 10 %. **Quote the supportable claim,
  "never meaningfully slower", rather than "always faster"** — and note that with the
  selection rule gone there is no longer anything a threshold could have protected there.

Non-symmetric braiding is *not* part of the removed list: it is rejected outright by
`Canopy._require_symmetric_braiding`, because belief propagation cannot support it in either
formulation — every blocked primitive succeeds under anyonic braiding, but the sign
derivation needs `twist(σ)² = 1`, so a silent fallback would have been the wrong shape.
"""
struct PairwiseBackend{B <: TensorKit.TO.AbstractBackend} <: TensorKit.TO.AbstractBackend
    inner::B
end
PairwiseBackend() = PairwiseBackend(DefaultBackend())

"""
    inner_backend(backend) -> TO.AbstractBackend

The real TensorOperations backend behind a Canopy kernel selector; the identity on anything
else. Called on the first line of every Canopy kernel that forwards its `backend` to
TensorOperations.
"""
inner_backend(backend) = backend
inner_backend(backend::PairwiseBackend) = backend.inner
