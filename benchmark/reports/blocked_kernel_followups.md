# Blocked message kernel — follow-up work and design notes

Handoff notes for continuing the symmetric-BP performance work in a fresh session.
Written after the `Layout(k)` kernel landed and after `phase4a.md` returned a no-go on
one particular threading decomposition. **`phase4a.md`'s "no" is narrower than it
reads — see §2.**

State at time of writing: `symmetry-kernel`, PR #27 (draft), rebased onto `#26`.
`Pkg.test()` 9576/9576, `scripts/hubbard_quench` 249/249, all six examples rendered.

---

> **STATUS (updated after acting on §1 and §2).**
> **§1 is DONE** — the gate is dropped, non-abelian is tested and passes. **§2 is
> MEASURED and it is a GO above χ = 64**: 5.46× on the message kernel at the production
> point, against §2's own predicted 2.3× cap, which this design beats because it threads
> the copies as well as the gemms. See `subblock_probe.md` for the verdict and
> `subblock_probe_t{1,8}.md` for raw tables. Three claims in §2 below are **wrong** and
> are corrected inline: the 2.3× ceiling, the "dynamic scheduling" recommendation, and
> the bitwise-reproducibility requirement. §2's implementation has **not** been written.

## 1. The `UniqueFusion` gate is unnecessary — DONE, dropped and tested

> **DONE.** `FusionStyle(I) === UniqueFusion()` is gone from `uses_blocked_kernel`;
> `SymmetricBraiding`, `A <: Array`, `I !== Trivial` and `numout(t) == 1` remain. The
> analysis below was correct in every particular.
>
> `test_messages_blocked.jl` passes 7697/7697 with `Vect[SU2Irrep]` and
> `hubbard_space(Trivial, SU2Irrep)` added to `_MSG_SPACES`, including the decisive
> non-Hermitian testset, the dual-physical-space and padded-leg testsets, and the
> `tensoralloc` fingerprint test that proves the blocked path is the one that ran. The
> SU(2) fallback assertions were **inverted, not deleted**, and the selection table is
> now keyed by name so reordering a row cannot silently realign expectations. A missing
> guard for the one *real* restriction was added: `!uses_blocked_kernel` on a
> `numout == 2` tensor.
>
> **How much this is worth, stated carefully.** The A/B ratios are the *best in the
> table* — `fz2_su2` 2.415× (χ=64) / 1.896× (χ=128) and `su2` 1.542× / 1.408×, against
> `fz2_u1`'s 1.232× at χ=128 and a `:trivial` control of 0.985-1.024
> (`benchmark/reports/backend_ab.md`). The mechanism is clear: these have the smallest
> mean subblocks in the suite (91-122 elements against `fz2_u1`'s 1365), which is the
> regime the blocked formulation targets, and the *pairwise* arm pays the
> `GenericTreeTransformer` tax too, so non-abelian does not hand it back.
>
> **But no production path in this repo reaches it yet.** `scripts/hubbard_quench`
> *rejects* SU(2) outright — `sectortypes` throws, because `product_state` only handles
> abelian sectors and an AFM/CDW product state breaks spin rotation regardless — and no
> `examples/` fixture is non-abelian (they are all `fermion_space(Trivial)`, i.e. `fℤ₂`).
> So this is a capability improvement plus a large *latent* win, not an active speedup of
> anything currently run. Do not cite it as "speeds up production". The win becomes real
> the moment a non-abelian initial state is reachable, which is a `product_state`
> limitation and not a BP one.
>
> **Cost, and the trim it forced. MEASURED, three runs:** 11m39s baseline (HEAD's test
> file against the new source) → **18m05s** with the two non-abelian rows on all six
> geometries → **15m44s** with `K6` trimmed from them (6857/6857 passing). So +4m05s
> (+35 %) net, and the trim recovers 37 % of the addition.
>
> **The added time does not extend the suite, and the trim is what keeps it that way.**
> Full `Pkg.test()` is **11260/11260 in 26m30s** on 15 parallel workers, and the critical
> path is `test_apply_gate_operator` at 1580 s — `test_messages_blocked` runs 1351 s
> under that load, i.e. inside the slack. Untrimmed it would have been the critical path
> instead.
>
> Essentially all of it is *compilation*: the warm sweep over all six geometries × three
> graded symmetries runs in ~20 s (`fZ2xSU2`/`K6`, the worst single fixture, is 6.2 s of
> BP plus 2.1 s of sweep). Compilation is driven by distinct `(sector type, numind)`
> pairs, which is why geometry count is almost irrelevant and the only trim with leverage
> is dropping the one geometry contributing an otherwise-unused arity — `K6`, the sole
> degree-5 (`M = 6`) fixture. That is what `_msg_geometries` does. Dropping
> `cycle`/`grid`/`K5` instead would have saved seconds, not minutes, since they share
> `M = 5`/`M = 3` with fixtures that stay. Degree 5 is still swept for all four abelian
> symmetries.
>
> Still **not** covered: `BraidingStyle` is the remaining gate and no anyonic fixture
> tests that the fallback fires for it. And the §2 threading probe is abelian-only, so
> no threading claim covers these new symmetries.

### Original analysis (retained — it was right)

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

## 2. The threading design that phase 4a did **not** measure — MEASURED, GO above χ = 64

> **MEASURED — see `subblock_probe.md`.** The design works: **4.4-5.5×** on the message
> kernel at `fz2_u1` / honeycomb / χ = 128 / 8 threads (range over two independent runs —
> quote the range), 1.25-1.42× at χ = 64, and a **0.48-0.66× regression at χ = 32** (so a
> size guard is required). Correct to 1e-16 against the real kernel. All three of phase
> 4a's objections fail to transfer, exactly as argued below — 337 units, intra-thread
> sequencing, and an imbalance of 16.6× against a 42× per-thread share.
>
> **Scheduler: an atomic-counter pull loop over units, and no work model.** Measured
> equal to a deterministic longest-processing-time-first assignment within the control
> deviation at χ = 128 (one row each way), so the plan-time work estimate LPT would need —
> whose copy-vs-flop weighting was never calibrated — should not be written.
>
> **The implementation brief lives at the end of `subblock_probe.md`** — decisions already
> taken, the TensorKit-internals dependency and why the two internals-free alternatives do
> not reach the win, and ten open items in the order they bite. Start there, not here.
>
> **Three things below are wrong.** They are left in place with corrections rather than
> edited away, because each one would have misdirected the implementation:
>
> 1. **"the serial `t_block_loop / t_kernel` ceiling still bounds you … caps the kernel
>    speedup around 2.3×" — WRONG for this design.** That ceiling assumes only the gemms
>    thread and the `tensoradd!` transports stay serial, which is phase 4a's
>    decomposition. Here the transports are per-subblock too (TensorKit's
>    `AbelianTreeTransformer` is already a per-tree loop), so they thread with everything
>    else and the serial fraction collapses to the `d` message transposes plus the
>    reduction. Measured 5.46× against the predicted 2.3× cap. **Do not use this ceiling
>    to scope the work.**
> 2. **"dynamic scheduling can absorb the flop imbalance" — the imbalance needs
>    absorbing, but not by *dynamic* scheduling specifically.** Deterministic
>    longest-processing-time-first assignment hits the theoretical bound exactly at every
>    fixture and thread count measured. What matters, and is *not* mentioned below, is
>    that the obvious *contiguous* chunking is a strawman: heavy units cluster in
>    enumeration order, capping it at 2.66× on 8 threads against a bound of 8.00×. LPT vs
>    an atomic-counter pull loop is settled in `subblock_probe.md` — the pull loop needs
>    no work model, so it wins unless it strands the heaviest unit.
> 3. **The whole "reproducibility" row of the hazard table is MOOT — the user does not
>    require bitwise equality (2026-08-03).** Note also that as *stated* it was
>    impossible: partitioning a sum across `t` buffers regroups the additions, so no
>    implementation can make a `t`-thread result bitwise equal to a 1-thread one short of
>    one unit per thread. What determinism was buying, and is now being spent, is §6's
>    "`run_one` is bitwise reproducible in-process" — with a nondeterministic reduction
>    order the two `free_fermion` examples become irreproducible run to run, since §6
>    records them amplifying 1e-16 to 1e-3 over ~80 Trotter steps. Fine while they are
>    illustrative; a problem if one becomes a regression gate.
>
> **A hazard this section omits, which turned out to be benign at scale.** Splitting the
> outer loop by subblock splits `mul!`'s merged per-coupled-sector gemms along their `m`
> axis — 105 → 2359 calls per vertex call at χ = 128, identical flops. It costs *nothing*
> there (the unit loop is 7 % faster than the real kernel serially, from better
> locality) because those gemms are ≈0.13 flop/byte and memory-bound, not
> compute-bound. It costs 1.72× at χ = 32, which is part of why that point regresses.
>
> **§4.1 was checked first, as §4 asks.** BLAS threads give **1.27×** at the production
> point for zero code, and are a *pessimization* at χ = 64 (0.81-0.91×). Not a substitute
> for this work, and the same magnitude as phase 4a's `TRANSFORMER_THREADS` finding.
>
> **Scope gap:** the probe is **abelian-only**. `SU2Irrep` and `fℤ₂ ⊠ SU2Irrep` now take
> the blocked kernel (§1) and no threading claim covers them.

### Original design argument (retained — it was right)

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

1. ~~**BLAS threads — unexplored, and plausibly the largest.**~~ **MEASURED — 1.27× at
   the production point, and negative at χ = 64.** See `subblock_probe.md` §4.1. It is
   free, so take it behind a size-and-symmetry guard, but it is not the largest: the §2
   design is 5.46× at the same point. Original text follows.

   **BLAS threads — unexplored, and plausibly the largest.** Every measurement in this
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
