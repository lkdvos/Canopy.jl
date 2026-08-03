# §2 — subblock-outer-loop threading: feasibility

**Verdict: GO above χ = 64, with a size guard. 4.4-5.5× on the message kernel** at the
production point (`fz2_u1`, honeycomb degree 3, χ = 128, 8 threads), against a *free*
alternative worth 1.27× and a design phase 4a's numbers appeared to cap at 1.33×.

The range is two independent t8 runs (5.46× and 4.53×), not one measurement's error bar;
the second run had three threaded arms competing for cores. **Quote the range, not the
best reading.**

Generated from `benchmark/subblock_probe.jl`. Raw tables in
`subblock_probe_t1.md` (serial arms) and `subblock_probe_t8.md` (all arms). No `src/`
changes — the probe is a hand-rolled kernel measured against the real one.

- host `ccqlin038.flatironinstitute.org` · cascadelake · 32 cores · julia 1.12.6
- git sha `b725220` · `OPENBLAS_NUM_THREADS=1`, `BLAS.set_num_threads(1)` except in the
  `--blas` arm which sweeps it deliberately
- loadavg (1/5/15): t1 `15.19 15.04 10.77` → `16.94 15.98 11.65`;
  t8 `5.39 9.15 10.03` → `6.80 7.81 9.30`
- 9 repetitions × 5 inner calls, arms alternated inside each repetition, min-of-inner
  then min-over-reps, every arm in one process on one fixture
- **abelian fixtures only** — see "Scope" at the end, this is a real limitation

---

## The design, and why phase 4a's no-go does not apply

`blocked_kernel_followups.md` §2: use the existing kernel as the *inner* part of a
threaded **outer** loop over symmetry subblocks, with per-thread output messages
summed at the end. `phase4a.md` measured something else — threading the
coupled-sector loop *inside* one `mul!` — and returned no-go on three grounds, none of
which survive here:

| phase 4a's objection | here |
|---|---|
| only 7-15 concurrent blocks | **337 units** at χ = 128 (`fz2_u1`), measured |
| the chain is sequential, so only one `mul!`'s blocks are concurrent | each unit runs a *complete* chain, so the dependency is intra-thread sequencing |
| blocks unbalanced 164× in flops | heaviest unit is 16.6× the mean, against a 42× per-thread share at 8 threads — not binding |

### What an independent unit is

Label every subblock by the map `leg id → sector on that leg`. That label is

* **invariant along both chains** — messages are sector-diagonal so absorbing message
  `j` cannot change `σ_j`, and a relayout is a permutation, which reorders legs and
  recouples *internal* tree structure but never moves a sector off its leg. This holds
  for a general fusion style: F-moves and braids both preserve external labels;
* **exactly what the closing sums over** — `out[k]_c = Σ_{f₁} bra[k][f₁,c]† ket[k][f₁,c]`,
  independent over units and *additive* into `out`. Which is why per-thread outputs plus
  a reduction is the answer, and partitioning the output is impossible.

The probe asserts the partition rather than assuming it: every layout must carry the
same *set* of labels as the source space, one subblock each, checked against
`Wl[a] == Wref[legids[a]]` and `hassector(Wl[a], s)` at plan time.

---

## The tax §2 does not mention, and why it turned out not to matter

`mul!(::AbstractTensorMap, …)` issues **one gemm per coupled sector**, on the block
that *merges* every fusion tree sharing that sector. Splitting the outer loop by unit
splits those merged gemms along their `m` axis. Measured, per vertex call:

| fixture | gemm calls, per-sector → per-unit |
|---|---|
| `fz2` (any χ) | 14 → **56** |
| `fz2_u1` χ = 128 | 105 → **2359** (22.5×) |
| `fz2_u1_flat` | 28 → **112** |

Identical total flops, 22.5× the calls, on gemms that are already `(2035, 3, 3)`-shaped.
This looked like the thing that would sink the design. **It does not, at scale**
(`real / unit`, > 1 means the unit loop is faster; control deviation ≤ 1.8 % on every
row):

| fixture | χ = 32 | χ = 64 | χ = 128 |
|---|---|---|---|
| `fz2` | 1.032 | 0.974 | **0.876** |
| **`fz2_u1`** | 1.723 | 1.347 | **0.936** |
| `fz2_u1_flat` | 1.033 | 1.055 | **0.831** |

At χ = 128 the unit loop is *faster than the real kernel serially*, by 7-17 %. The
reason the call-count multiplication is free is that these gemms were never
compute-bound — ≈0.13 flop/byte, and the measured block loop runs at ~6 GFlop/s where
a fat gemm on this box reaches 55 — so splitting `m` moves the same bytes in more
calls. The gain on top comes from locality: the unit loop walks one unit's slab through
the whole chain while it is cache-hot, where the real kernel streams a 33 MB link per
step.

The tax is real at χ = 32 (1.72× for `fz2_u1`), which is the same small-block regime in
which every other lever in this project also fails.

---

## Threaded results

8 threads, χ = 128 rows only — see "Measurement caveats". `speedup` is against the same
call at `nthreads = 1`, `vs real` against the real kernel.

| fixture | units | serial µs | threaded µs | speedup | **vs real** | control |
|---|---|---|---|---|---|---|
| `fz2` | 8 | 191100 | 52389 | 3.648 | **4.252** | 0.4 % |
| **`fz2_u1`** | 337 | 35488 | 6913 | 5.133 | **5.458** | 5.3 % |
| `fz2_u1_flat` | 16 | 39050 | 13965 | 2.796 | **3.213** | 5.7 % |

Second run, three threaded arms interleaved so each gets less of the machine — this is the
low end of the quoted range and the more conservative one to plan against:

| fixture | serial µs | LPT µs | **vs real** | control |
|---|---|---|---|---|
| `fz2` | 197667 | 52738 | **4.17** | 1.7 % |
| **`fz2_u1`** | 34303 | 7831 | **4.53** | 1.8 % |
| `fz2_u1_flat` | 38186 | 13727 | **2.99** | 5.3 % |

At χ = 64, `fz2_u1` is **1.25-1.42×** and at χ = 32 it is **0.48-0.66×** — a regression,
hence the size guard. The reduction costs 2-109 µs, ≤ 0.3 % of the call.

### §2's own 2.3× ceiling is wrong for this design

§2 warns that "the serial `t_block_loop / t_kernel` ceiling still bounds you … caps the
kernel speedup around 2.3×". That ceiling assumes only the gemms are threaded and the
`tensoradd!` transports stay serial, which is true of phase 4a's decomposition. Here
the transports are *per-subblock too* — TensorKit's `AbelianTreeTransformer` is already
a per-tree loop — so they thread with everything else, and the serial fraction collapses
to the `d` message transposes plus the reduction. 5.46× measured against a predicted
2.3× cap.

### Scheduling: balance matters, determinism does not

**Bitwise reproducibility is not a requirement** — the user's call, 2026-08-03, which
overrides §2's hazard table on this point. Three consequences, the middle one reversing a
conclusion drawn before that was known:

1. **Contiguous chunking is a strawman, and this stands either way.** Heavy units cluster
   in enumeration order (neighbouring units share coupled-sector blocks), so contiguous
   chunks cap at **2.66×** on 8 threads against a bound of 8.00× for `fz2_u1` at χ = 128.
   Measuring that would have repeated phase 4a's mistake one level down. Both schedulers
   below beat it.
2. **The work model is dead code — use the atomic-counter pull loop. MEASURED.**
   Longest-processing-time-first hits the theoretical bound exactly (`lpt` equals `dyn` at
   every fixture and thread count in the census), but it needs a plan-time work estimate
   whose 1:1 copy-vs-flop weighting nobody has calibrated (direct timing puts the real
   split nearer 40:58 at χ = 128). A pull loop needs no estimate. Interleaved in the same
   repetition, at χ = 128:

   | fixture | LPT | dynamic | dyn / lpt | control |
   |---|---|---|---|---|
   | `fz2` | 3.748 | 3.680 | 1.018 | 1.7 % |
   | `fz2_u1` | 4.380 | 4.236 | 1.034 | 1.8 % |
   | `fz2_u1_flat` | 2.782 | 2.985 | 0.932 | 5.3 % |

   Within or near the control deviation in all three, one in each direction — so the
   feared straggler penalty does not materialise, and at χ = 64 dynamic is clearly ahead
   (5.97 vs 4.20 for `fz2`, though those controls are 12-32 % and not quotable). **Do not
   write the work estimate.** Pulling in index order gives the heaviest unit no priority,
   and at χ = 128 the heaviest `fz2_u1` unit is 16.6× the mean against a 42× per-thread
   share, so this was a real risk — it just did not bite, because 337 units over 8 threads
   leaves ~42 units per thread and one heavy unit is a fraction of that span.
3. **§2's hazard table asks for `nthreads == 1` bitwise equal to `t`-thread, which no
   implementation can deliver** even with a fully deterministic schedule: partitioning a
   sum across `t` buffers regroups the additions. The most that was ever on offer is
   run-to-run reproducibility at fixed `nthreads`, which the LPT arm does achieve
   (`repro` true on all nine rows).

For the record on what determinism was buying, since it is now being spent:
`blocked_kernel_followups.md` §6 documents `examples/free_fermion_ring` and
`free_fermion_honeycomb` amplifying a ~1e-16 kernel difference into ~1e-3 over ~80 Trotter
steps, and records `run_one` as bitwise reproducible in-process. A nondeterministic
reduction order ends that: those examples become irreproducible run to run *in the same
process*, not merely different from today. Harmless if they stay illustrative; a problem
if one is ever used as a regression gate.

Correctness against the real kernel is **1.0e-16 to 3.5e-16** on every threaded row, for
both schedulers. That is the race check, and the probe hard-errors above 1e-12 rather than
reporting a speedup — disjoint bands are only disjoint if the plan says so.

---

## §4.1 — BLAS threads, the free alternative, checked first

`blocked_kernel_followups.md` §4.1 ranked this "unexplored, and plausibly the largest",
and it had to be measured before the §2 work because they compete for the same cores.

| fixture | χ | BLAS=1 | BLAS=2 | BLAS=4 | BLAS=8 | best | control |
|---|---|---|---|---|---|---|---|
| `fz2` | 128 | 202408 | 140303 | 114561 | 108897 | **1.86×** | 7.1 % |
| **`fz2_u1`** | 128 | 42223 | 34529 | 33912 | 33139 | **1.27×** | 1.5 % |
| `fz2_u1_flat` | 128 | 47438 | 35033 | 30636 | 27493 | **1.73×** | 0.4 % |
| **`fz2_u1`** | 64 | 3445 | 4245 | 3792 | 3879 | **1.00×** | 6.3 % |
| `fz2_u1_flat` | 64 | 2914 | 3744 | 3299 | 3429 | **1.00×** | 2.4 % |

**1.27× at the production point for zero code**, and a *pessimization* at χ = 64 for
both graded fixtures (0.81-0.91× — OpenBLAS threading these tiny gemms costs more than
it buys). It is not a substitute: the §2 design is worth 5.46× at the same point. It is
also the same magnitude as the 1.26× `TensorKit.set_num_transformer_threads(8)` gave in
phase 4a M4, which is unsurprising — both thread one half of the kernel while the
subblock loop threads both.

---

## Measurement caveats

- **`nthreads` > 1 at χ ≤ 64 is not trustworthy**, exactly as `phase4a.md` records:
  control deviations of 16.9 % (`fz2_u1_flat` χ = 32) and 29.4 % (`fz2` χ = 64) in the
  first t8 run. The χ = 128 rows have controls of 0.4-5.7 % and are the only ones
  quoted as results. The χ ≤ 64 rows are reported because the *sign* of the effect
  (regression at 32, marginal at 64) is consistent across two independent t8 runs, not
  because their magnitudes are reliable.
- **The unit arm is a hand-rolled kernel, not a `src/` implementation.** It resolves
  every transformer and block offset at plan time, so it pays no `HomSpace` lookups. At
  χ = 128 that advantage is ~0 (the `--serial` arm shows real ≈ sector ≈ unit), but at
  χ = 32 it is worth 1.33-1.93× on its own (`phase4a.md` Part B), so **the χ = 32
  regression is if anything understated** for a real implementation.
- The work estimate driving LPT weights copy elements and gemm flops 1:1. Direct timing
  puts the real split nearer 40:58 at χ = 128. The bounds are insensitive to this
  (LPT hits the `dyn` bound, which is set by `total/t`, not by the weighting) but a
  real implementation should not treat the proxy as calibrated.
- Degree 4 is not measured, only degree 3. Phase 4a censused degree 4 and found it
  changes `n_mul!` without changing blocks per `mul!`; here it would raise the unit
  count further (455/1771/2925/4495 fusion trees for `fz2_u1`), i.e. move in the
  favourable direction.

## Scope: abelian only

`uses_blocked_kernel` accepts **non-abelian** fusion (the `UniqueFusion` gate was
dropped — see `src/backends.jl`), so the non-abelian path is genuinely uncovered by
this probe. The reason is mechanical: the per-sector cheat kernel this probe extends
writes each destination subblock with `β = 0`, assuming one destination per source,
which fails under non-abelian fusion where a relayout mixes trees. The probe *detects*
this rather than returning wrong numbers — the "one subblock per label per layout"
assertion fires.

A non-abelian unit is still well defined (the label argument above is fusion-style
agnostic); it holds the several trees sharing one label, and a relayout mixes *within*
the unit, so the unit stays independent. Extending the probe means giving each unit a
trailing multiplicity axis and a small dense mixing per relayout. Until that is done,
**no threading claim covers `SU2Irrep` or `fℤ₂ ⊠ SU2Irrep`**, which now take the blocked
kernel and are its *best* serial rows (1.4-2.4×, `backend_ab.md`).

Deferring them is the user's call (2026-08-03) and the cost is bounded: no production path
in this repo reaches non-abelian BP today — `scripts/hubbard_quench` rejects SU(2) in
`sectortypes` (`product_state` handles abelian sectors only) and every `examples/` fixture
is `fℤ₂`. So the deferral loses a latent win, not a live one.

---

# Brief for the follow-up: implementing this in `src/`

**Decisions already taken** (user, 2026-08-03), so a fresh session does not relitigate them:

| decision | value |
|---|---|
| target regime | **large bond dimension**; a small-χ gate is acceptable |
| bitwise reproducibility | **not required** |
| scheduler | **atomic-counter pull loop, no work model** (measured equal to LPT) |
| non-abelian | **deferred** out of the threading PR |
| TensorKit internals | **on the table** — this is the enabler, see below |

## The internals dependency, and why there is no way around it

There is no `TensorMap`-level API for "do this `tensoradd!` / `mul!` for one subblock
only", so unit-granular threading has to reach for `treebraider`, `subblockstructure`,
`blockstructure` and `TreeTransformer.data`. That is the `MessagePlan` phase 3 declined and
`phase4a.md` Part B advised against — **and both were right about the question they asked**:
they priced it as a *serial* optimization, worth 1.07× at χ = 32 and 0.98× at χ = 128. Here
it is not an optimization, it is the mechanism that makes 4.4-5.5× reachable. Do not read
phase 4a's "do not implement it" as applying to this.

Two internals-free alternatives were checked and neither reaches the win:

- **Thread upstream only** — `TensorKit.set_num_transformer_threads` plus the coupled-sector
  loop in `mul!`. That is phase 4a's ground, 1.26-1.33×.
- **Unpack each subblock into a restricted `TensorMap` and call the existing kernel
  verbatim** — the most literal reading of §2, and it fails on a structural point: a
  restricted `TensorMap`'s content is a *product* of per-leg sector subsets, and an
  arbitrary set of subblocks is not a product set. Forcing it to be one means partitioning
  by a single leg's sector, i.e. back to 7-15-way concurrency with the imbalance that
  sank phase 4a. Per-unit restricted `TensorMap`s do work but pay space construction and
  LRU cost 337× per call.

`benchmark/subblock_probe.jl` is the working reference implementation of the plan; port it,
do not redesign it. Note its plan partitions each transport's `data` into one vector per
unit, which is fine for a probe but allocates `nunits × ntransports` small vectors per
build — in `src/` prefer a flat `data` plus a `unit → index` permutation, which is exact
for abelian (one subblock per unit per layout) and needs one `Vector{Int}` per layout.

## Open items, in the order they bite

1. **The size gate. Gate on mean subblock size, not χ.** `dim(space(T)) / nunits` separates
   all nine measured points: `fz2` χ=32 (meanblk 4096) wins 2.10× and `fz2_u1_flat` χ=32
   (480) wins 1.28×, while `fz2_u1` χ=32 (42) loses at 0.66×. A χ gate would wrongly
   exclude the first two. `≥ 128` is the proposed threshold but it is **fitted to nine
   points** — make it a tunable constant with the data recorded next to it, not a magic
   number. Also note the probe *understates* the small-χ regression, because its plan-based
   kernel pays no `HomSpace` lookups and those are worth 1.33-1.93× at χ = 8 on their own.
2. **Allocation in the parallel region — check this first, it may delete the next item.**
   The per-unit chain does only strided copies and BLAS on views. If every buffer is
   allocated *before* the region, the region should allocate nothing. Verify with
   `@allocated` **and** `buffer_stats` / `buffer_isempty` / `noverflow` (`@allocated` cannot
   see Bumper's off-heap temporaries). If it holds, the `BufferPool` work below is
   unnecessary.
3. **`Bumper.ResizeBuffer` is task-local and not thread-safe** (non-atomic RMW on
   `offset`); sharing one silently corrupts temporaries — wrong numbers, not a crash. If
   item 2 does not hold, restore the `BufferPool` from `git show 1169855 -- src/bumper.jl`;
   its docstring records why naive per-task buffers OOM (malloc-backed, reclaimed only by
   finalizers the GC rarely runs). The probe used `TO.DefaultAllocator()` throughout and so
   never exercised any of this.
4. **Memory.** Buffers are `2d` full-size layout links (shared, written at disjoint bands)
   plus **one full-size `tmp` per worker**. `tmp` cannot be shrunk to a unit's slab because
   the transport's strides are the *block's* strides, so a subblock spans nearly the whole
   block. At χ = 128 that is 7.4 MB per buffer for `fz2_u1` (~100 MB at 8 threads) but
   33.5 MB for `fz2` (~470 MB). Cap the worker count by available memory, not just by
   `nthreads()`.
5. **The allocator is captured in `BeliefPropagation` at construction, on the constructing
   task**; resolve per task inside the region. Fix the adjacent latent bug at the same
   time: `src/beliefpropagation.jl:40` defaults to `_default_allocator()` unconditionally
   while `belief_propagation` passes the storage-aware `default_allocator(state)`, so a GPU
   state constructed through the former gets a CPU Bumper buffer.
6. **Hoist every space/structure query above the region** — the global LRU takes a
   `SpinLock` on *every* lookup including hits, 70 per vertex call. Plan construction is
   the natural place; warm the caches with one serial call first.
7. **`@maybe_timeit` must stay outside the region** (where the existing ones already are).
   Add none inside.
8. **Testing under threads.** `ParallelTestRunner` hard-codes `JULIA_NUM_THREADS=1` in
   every worker, so a threaded test silently gets one thread and passes vacuously. Pass
   `exeflags = ["--threads=N"]` from `test/runtests.jl` (a `-t` CLI flag beats the env
   var), gate on something like `CANOPY_TEST_THREADS`, and **have the test assert it
   actually got threads**. The correctness assertion is `≈` against the pairwise oracle
   plus run-to-run agreement to rounding — *not* bitwise, which is neither required nor
   achievable across thread counts.
9. **Selector plumbing.** Follow the existing pattern: a selector backend to force the path
   for tests/benchmarks (as `BlockedBackend` / `PairwiseBackend` do), auto-selection via a
   predicate, and a `tensoralloc` fingerprint test proving *which* path ran — result
   comparison cannot discriminate, since the kernels agree and often agree bitwise.
10. **Measure with `bench_backend_ab.jl`'s method, never a `SUITE` group**, and remember
    `nthreads > 1` at χ ≤ 64 is untrustworthy in this harness (control deviations up to
    32 % observed here, 82 % in phase 4a) — it is an arm-ordering artefact.
