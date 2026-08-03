# Blocked message kernel — follow-up work and design notes

Handoff notes for continuing the symmetric-BP performance work in a fresh session.
Written after the `Layout(k)` kernel landed and after `phase4a.md` returned a no-go on
one particular threading decomposition. **`phase4a.md`'s "no" is narrower than it
reads — see §2.**

State at time of writing: `symmetry-kernel`, PR #27 (draft), rebased onto `#26`.
`Pkg.test()` 9576/9576, `scripts/hubbard_quench` 249/249, all six examples rendered.

---

## 1. The `UniqueFusion` gate is probably unnecessary — drop it after testing

`uses_blocked_kernel` (`src/backends.jl`) currently requires
`FusionStyle(sectortype(S)) === UniqueFusion()`, so `SU2Irrep` and
`fℤ₂ ⊠ SU2Irrep` fall back to the pairwise kernel.

**That restriction is very likely wrong.** The blocked kernel is written entirely in
`TensorMap`-level operations, every one of which handles non-abelian fusion correctly
on its own:

| step | operation | non-abelian? |
|---|---|---|
| entry braids, chain relayouts | `tensoradd!` with a permutation | fine — TensorKit routes through `GenericTreeTransformer`, which is the correct basis change |
| absorption, closing | `mul!` / `adjoint` | fine — composition is block-wise in the *coupled* sector and never mixes fusion trees, for any fusion style |
| physical-leg twist | `twist!(ket1, (1,))` | fine — TensorKit's own, walks fusion trees |
| message twist | `_twist_message!` scaling `blocks(mt)` | fine — a `1 ← 1` tensor has one tree pair per coupled sector and a single-leg tree's uncoupled sector *is* its coupled one, so this equals `twist!(mt, (1,))` for any fusion style |

The sign derivation in `src/messages.jl` uses `twist(σ)² = 1`, which needs
**`SymmetricBraiding`**, not `UniqueFusion`. And critically, no step ever needs to
recover all uncoupled sectors from one coupled label: `twist(σⱼ)` is folded onto
message `j`, where leg `j`'s sector is manifest, and the physical factor onto the ket
entry. That was the only place abelian-ness looked load-bearing, and it isn't.

**Action.** Drop `FusionStyle(I) === UniqueFusion()`; keep `BraidingStyle(I) isa
SymmetricBraiding`, keep `A <: Array`, keep the `Trivial` exclusion (that one is a
*performance* choice — the pairwise path short-circuits via `has_array_view` to plain
arrays and one large BLAS call), and keep `numout(t) == 1` (§3).

**Verification before trusting it** — extend the existing harness, do not write a new one:
- add `Vect[SU2Irrep]` and `Vect[fℤ₂ ⊠ SU2Irrep]` rows to `_MSG_SPACES` and run the
  full `test/test_messages_blocked.jl` sweep against the per-edge oracle **and** the
  pairwise kernel;
- the **non-Hermitian-messages** testset is the decisive one, as always: Hermitian
  messages mask a conjugation error in the bra chain;
- `test/test_messages_blocked.jl` currently *asserts the fallback fires* for
  `SU2Irrep`. That assertion has to be inverted, not deleted — keep an asserted-path
  test so a silently-wrong selection can't pass;
- measure. Non-abelian permutes pay a genuine basis change (`GenericTreeTransformer`
  rather than the cached `AbelianTreeTransformer`), but the *pairwise* path pays it too,
  so blocked should still win or tie. Use `benchmark/bench_backend_ab.jl`, not a `SUITE`
  group (see §5).

---

## 2. The threading design that phase 4a did **not** measure

`phase4a.md` measured **threading the coupled-sector loop inside one `mul!`** and
returned no-go, for three reasons: the chain is sequential (`ket[k+1]` comes from
`ket[k]`) so only one `mul!`'s blocks are concurrent; there are only 2 / 4 / 7-15 of
them; and they are unbalanced by up to 164× in flops.

**The intended design is different and those three objections do not apply to it.**

> Use the current kernel as the *inner* part of a multithreaded **outer** loop that
> unpacks each symmetry subblock and hands it to the same kernel. Accept a **per-thread
> set of output messages**, summed at the end.

Why this is a different proposition:

- **Concurrency is the number of subblocks, not coupled sectors.** At the production
  point that is 73 / 121 / 181 / 253 / 337 fusion trees for `fℤ₂ ⊠ U1Irrep` at
  χ = 8 / 16 / 32 / 64 / 128 (`benchmark/reports/structure.csv`), against the 7-15
  coupled sectors phase 4a was limited by. An order of magnitude more parallelism.
- **The sequential chain stops mattering.** Each thread runs a *complete, independent*
  chain for its own block, so the `ket[k] → ket[k+1]` dependency is sequencing *within*
  a thread rather than a barrier across threads.
- **Load balance becomes tractable.** With 73-337 units instead of 7-15, dynamic
  scheduling can absorb the flop imbalance that sank the coarse version.
- **The output collision is solved by the reduction, not by partitioning.** Partitioning
  by output block is impossible — each outer iteration writes a *different* block for
  each of the `d` targets — which is exactly why per-thread outputs are the right answer.
  Cost is `nthreads × d` message-sized (χ²) buffers, small against the `≈2d` chain links
  of `dim(P)·χ^d` the kernel already holds live.

### What to measure first

1. **Achievable concurrency and per-unit work** at production `(symmetry, χ)`: number of
   independent units, and median work per unit against the measured `@spawn` round trip
   (`benchmark/reports/blockloop_probe_t*.md` has the calibration).
2. **Reduction cost** as a fraction of the whole — `nthreads × d × χ²` adds and the
   allocation of those buffers.
3. **The serial `t_block_loop / t_kernel` ceiling still bounds you.** Measured 0.57-0.60
   at the production point, so even perfect scaling of the parallel part caps the kernel
   speedup around 2.3×. Do not expect more, and note this is against a kernel that is
   already 2.6-2.9× faster than where the project started.

### Hazards, each with its known mitigation

| hazard | mitigation |
|---|---|
| `Bumper.ResizeBuffer` is task-local and **not** thread-safe (non-atomic RMW on `offset`) — sharing one silently corrupts temporaries, wrong numbers rather than a crash | restore the `BufferPool` from `git show 1169855 -- src/bumper.jl` (`acquire!`/`release!`/`withbuffer`, task-local `_default_pool()`). Its docstring records why naive per-task buffers OOM: malloc-backed, reclaimed only by finalizers the GC rarely runs, so off-heap bytes accumulate invisibly |
| the allocator is captured in `BeliefPropagation` at construction, on the constructing task | store a pool or factory, resolve per task inside the parallel region. Fix the adjacent latent bug at the same time: the constructor defaults to unconditional `_default_allocator()` while `belief_propagation` passes the storage-aware `default_allocator(state)` |
| shared `TimerOutput` via `@maybe_timeit` is not concurrency-safe | keep `@maybe_timeit` strictly outside the parallel region — which is where the existing ones already are. Add none inside |
| global LRU space-structure cache takes a `SpinLock` on **every** lookup including hits; 70 lookups per vertex call | hoist all structure queries above the parallel region and warm the caches with one serial call first. This is what the plan-cache work in §4 would also address |
| non-determinism from reduction order | reduce in **fixed** order. Then demand bitwise reproducibility: `nthreads == 1` bitwise equal to serial, and two runs at the same `nthreads` bitwise equal to each other |
| `ParallelTestRunner` hard-codes `JULIA_NUM_THREADS=1` in every worker, so threaded tests silently get one thread | pass `exeflags = ["--threads=N"]` from `test/runtests.jl` (a `-t` CLI flag beats the env var); gate on an env var such as `CANOPY_TEST_THREADS`. A threaded test file must assert it actually got threads |

---

## 3. What *is* a genuine restriction: one physical leg

`uses_blocked_kernel` requires `numout(t) == 1`. This one is real. `layout(k)`
addresses virtual leg `k` at tensor slot `k + 1`, and `dual_phys` reads slot 1, so a
`TensorNetworkOperator` site tensor (`numout == 2`, after `#26`) would be contracted on
the wrong slots and return wrong numbers **silently**.

Generalizing means treating the physical legs as a group rather than as slot 1:
`layout`, `dual_phys` and both entry braids need `np = numout(T)` threaded through, and
the physical twist becomes a product over the physical legs. The pairwise kernel is
already generic in codomain arity (`#26` did that), so operators are correct today —
just not accelerated.

---

## 4. Other levers, ranked

1. **BLAS threads — unexplored, and plausibly the largest.** Every measurement in this
   project pinned `BLAS.set_num_threads(1)` for reproducibility. Direct timing puts the
   block loop at **57-74%** of the kernel at χ=128, so on a many-core box simply letting
   BLAS use cores is free parallelism nobody has tested. Check this *before* the §2 work
   — it may compete with it for the same cores, and it costs nothing to try.
2. **The residual and synchronous schedules get none of this.** The blocked kernel serves
   only the *batch* path. `SynchronousSchedule` and `ResidualSchedule` use the single-edge
   `compute_message!`, which has no blocked variant. The default schedule is batch-based
   so it benefits; those two do not. Widening who benefits may be worth more than shaving
   the fast path.
3. **Plan cache — small-χ only.** 70 global-LRU lookups per vertex call. A *perfect* cache
   (measured, bitwise-identical prototype) is worth 1.33-1.93× at χ=8, ≤1.08× at χ≥32, and
   is **negative** at χ≥64, where it loses the real kernel's single-arena locality. Only
   pursue if small-χ workloads matter. `phase4a.md` §B has the numbers.
4. **The remaining `2d` copy passes.** `tensoradd!`/`add_transform` is 18-38% depending on
   regime, the largest non-BLAS item. Fusing the relayout into the composition was **tried
   and failed**: `TO.isblasdestination` is false for that swap, so `blas_contract!` takes
   `copyC` and allocates the same intermediate anyway — 0.3-9.8% *slower* at all 12
   measured points. Beating it needs a formulation where consecutive absorptions don't
   need the axis swap.

---

## 5. Measurement rules that cost this project real time

- **Never gate a ratio on a `SUITE` group.** `BenchmarkGroup` does not execute keys in
  loop order, so A/B pairs are not adjacent in time; at χ=64 a pair takes ~10 s and drift
  swamps a 20% effect. The committed `SUITE` once misread a 1.199 ratio as 0.970. Use
  `benchmark/bench_backend_ab.jl`: arms alternated inside each repetition, min-of-inner
  then min-over-reps, both arms in one process on one fixture.
- **Always carry a control arm** whose two sides run byte-identical code, and quote no
  result that does not clear its deviation. `:trivial` serves for kernel A/B (it falls
  back, so both arms are the same code) — its floor is ~0.4-2.2%.
- **Record `/proc/loadavg`** at start and end. This box has co-tenants and load ranged
  2.5-20 during the project; several apparent results were load.
- **`nthreads > 1` at χ ≤ 64 is untrustworthy** in the existing probe: control deviation
  up to **82%**, an arm-ordering artefact (the control runs after four threaded arms and
  Julia leaves workers spinning). Fix by randomising arm order or inserting a quiescence
  gap before quoting anything there.
- **Flame-graph gemm shares are lower bounds.** `SIGPROF` cannot unwind OpenBLAS's
  assembly kernels, so those samples carry no Julia frame and match no category, while
  every other category keeps its true share. `benchmark/reports/kernel_profile.md` is
  affected. Cross-check with direct interleaved timing; a bare BLAS call of the same
  shape reproduced the direct figure to 1%.
- **`buffer_stats(...).peak` is a process-lifetime high-water mark.** `Bumper.reset_buffer!`
  first or you report a previous fixture's peak.
- **`@allocated` cannot see Bumper's off-heap temporaries.** Pair every allocation claim
  with `buffer_stats` / `buffer_isempty` / `noverflow`.

---

## 6. Loose ends unrelated to the kernel

- Rendered examples embed the **absolute path of the worktree that generated them**, so
  every regeneration diff carries path noise and no two checkouts can produce identical
  output. Worth making the path relative.
- `examples/free_fermion_ring` and `free_fermion_honeycomb` are sensitive at D=4-8: a
  ~1e-16 kernel difference flips a near-degenerate truncation decision and amplifies to
  ~1e-3 over ~80 Trotter steps. Expect their numbers to move on any numerical change; it
  is not a regression signal. (An earlier attribution of this to `edge_coloring(keys(::Dict))`
  giving build-dependent orderings was **wrong** — `Dict` order, colouring and hashes are
  bit-identical across processes for `UndirectedEdge{Int}`, and `run_one` is bitwise
  reproducible in-process.)
- The default `SpanningTreeSchedule` is ~1.3× slower to converge on degree-2 rings and
  chains, where iteration count had no headroom left. Documented in its docstring; accepted
  deliberately for the 2.6-4× on production honeycomb.
