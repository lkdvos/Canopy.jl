# Phase 4a — threading feasibility and plan-cache feasibility

>  **⚠ Scope of this no-go.** Part A measured exactly one decomposition: threading the
>  coupled-sector loop *inside* one `mul!`. A different design — an outer loop over
>  symmetry **subblocks**, each handed to the whole kernel, with per-thread output
>  messages summed at the end — is **not** covered here, and none of Part A's three
>  objections apply to it (concurrency becomes 73-337 units rather than 7-15; the
>  sequential chain becomes intra-thread sequencing; the reduction replaces the
>  impossible output partitioning). See `blocked_kernel_followups.md` §2 before
>  concluding that threading was ruled out.

**Verdict: no-go on both.** Threading the block loop fails all three pre-committed
criteria at the production symmetry, and a perfect plan cache is worth ≤1.08× at
χ ≥ 32 and nothing at χ ≥ 64.

Generated from `benchmark/blockloop_probe.jl` (Part A) and
`benchmark/plancache_probe.jl` (Part B). Raw per-thread tables are in
`blockloop_probe_t{1,2,4,8}.md` and `plancache_probe.md`. No `src/` changes.

- host `ccqlin038.flatironinstitute.org` · cascadelake · 32 cores · julia 1.12.6
- git sha `31494ee` · `BLAS.set_num_threads(1)` · `OPENBLAS_NUM_THREADS=1`
- loadavg (1/5/15) per run, start → end:
  - probe `--threads=1` `3.24 3.73 3.63` → `4.47 4.23 3.85`;
    `=2` `4.26 4.16 3.45` → `4.01 4.06 3.56`;
    `=4` and `=8` recorded in the per-thread files, all in `3.0`–`4.6`
  - plan-cache probe `3.06 3.62 3.68` → `3.41 3.79 3.76`
- 9 repetitions × 5 inner calls, arms alternated inside each repetition,
  min-of-inner then min-over-reps, both arms in one process on one fixture.

## Noise floor

Every table carries its own control arm — a second copy of one of the arms,
running byte-identical code — and no claim below is quoted unless it clears that
row's control deviation.

| where | control deviation |
|---|---|
| plan-cache A/B, all 12 rows | **0.0 – 2.0 %** |
| threading probe, χ = 128 rows (all thread counts) | **0.1 – 1.6 %** |
| threading probe, χ = 32/64 rows at `nthreads` = 1 | 0.6 – 5.6 % |
| threading probe, χ = 32/64 rows at `nthreads` > 1 | **up to 82 %** |

The last row is not random noise, it is an ordering artefact and it is worth
recording: the control arm runs *after* four threaded arms inside each
repetition, and Julia's task scheduler leaves worker threads spinning, so a
short serial arm measured immediately afterwards is contaminated. It only bites
where the arm is short (150 µs – 3 ms). At χ = 128 every arm is ≥ 20 ms and the
floor is under 2 %, which is where the decisive numbers live. **Nothing at
χ ≤ 64 with `nthreads` > 1 is quoted as a result below.**

---

# Part A — threading feasibility

## The loop in question

`_blocked_message!` (`src/messages.jl:641`) issues, per degree-`d` vertex call:
`d` message transposes, 2 entry braids, `2d-2` chain re-permutes (all
`tensoradd!`), `2d-2` absorptions and `d` closings (all `mul!`). Every `mul!` is
`LinearAlgebra.mul!(::AbstractTensorMap, …)`
(`TensorKit/src/tensors/linalg.jl:330`), a serial merge-join over coupled
sectors that literally carries the comment `# TODO: consider spawning threads for
different blocks`. That per-coupled-sector loop is what 4b would thread.

**The chain is sequential.** `ket[k+1]` is computed from `ket[k]`, so the blocks
of *different* `mul!`s cannot run concurrently; only the `nblk/mul!` blocks
inside one `mul!` are independent (plus the `d` closings, which are mutually
independent but are the last step). The probe therefore reports two families:

- **grouped** — one parallel region per `mul!`. This is what 4b could build.
- **flat** — one region over every block of the call. Not achievable; reported
  only as an optimistic bound, and the gap between the two is the price of the
  dependency structure.

## M1 — block census

Structure only, so it is machine-independent and covers χ = 128 at degree 4
where the tensors would be 8 GB. Cross-checked inside `--probe`: the predicted
multiset of `(m,n,k)` is asserted equal to what `blocks(...)` actually yields on
the real fixtures, for every timed row.

| geom | sym | χ | d | nsectors | ntrees | dim | n_mul! | **nblk/mul!** | (m,n,k) min | (m,n,k) med | (m,n,k) max | 2mnk min/med/max | Σ2mnk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hex | fz2 | 8 | 3 | 2 | 8 | 512 | 7 | **2** | (64,4,4) | (64,4,4) | (4,4,64) | 2048/2048/2048 | 2.9e4 |
| hex | fz2 | 32 | 3 | 2 | 8 | 32768 | 7 | **2** | (1024,16,16) | (1024,16,16) | (16,16,1024) | 5.2e5/5.2e5/5.2e5 | 7.3e6 |
| hex | fz2 | 64 | 3 | 2 | 8 | 262144 | 7 | **2** | (4096,32,32) | (4096,32,32) | (32,32,4096) | 8.4e6/8.4e6/8.4e6 | 1.2e8 |
| hex | fz2 | 128 | 3 | 2 | 8 | 2097152 | 7 | **2** | (16384,64,64) | (16384,64,64) | (64,64,16384) | 1.3e8/1.3e8/1.3e8 | 1.9e9 |
| hex | **fz2_u1** | 8 | 3 | 7 | 73 | 119 | 7 | **7** | (9,1,1) | (15,1,1) | (2,2,18) | 18/30/144 | 2.2e3 |
| hex | **fz2_u1** | 32 | 3 | 11 | 181 | 7595 | 7 | **11** | (53,1,1) | (174,1,1) | (8,8,302) | 106/348/38656 | 6.1e5 |
| hex | **fz2_u1** | 64 | 3 | 13 | 253 | 59369 | 7 | **13** | (119,1,1) | (483,3,3) | (14,14,1162) | 238/8694/455504 | 8.9e6 |
| hex | **fz2_u1** | 128 | 3 | 15 | 337 | 459893 | 7 | **15** | (269,1,1) | (2035,3,3) | (26,26,4454) | 538/36630/6.0e6 | 1.3e8 |
| hex | fz2_u1_flat | 8 | 3 | 4 | 16 | 120 | 7 | **4** | (1,1,1) | (1,1,35) | (3,3,21) | 2/70/378 | 4.0e3 |
| hex | fz2_u1_flat | 32 | 3 | 4 | 16 | 7680 | 7 | **4** | (16,4,4) | (4,4,560) | (12,12,336) | 512/17920/96768 | 1.0e6 |
| hex | fz2_u1_flat | 64 | 3 | 4 | 16 | 61440 | 7 | **4** | (64,8,8) | (8,8,2240) | (24,24,1344) | 8192/2.9e5/1.5e6 | 1.7e7 |
| hex | fz2_u1_flat | 128 | 3 | 4 | 16 | 491520 | 7 | **4** | (256,16,16) | (16,16,8960) | (48,48,5376) | 1.3e5/4.6e6/2.5e7 | 2.6e8 |
| square | fz2 | 8 | 4 | 2 | 16 | 4096 | 10 | **2** | (512,4,4) | (512,4,4) | (4,4,512) | 1.6e4/1.6e4/1.6e4 | 3.3e5 |
| square | fz2 | 32 | 4 | 2 | 16 | 1048576 | 10 | **2** | (32768,16,16) | (32768,16,16) | (16,16,32768) | 1.7e7/1.7e7/1.7e7 | 3.4e8 |
| square | fz2 | 64 | 4 | 2 | 16 | 16777216 | 10 | **2** | (262144,32,32) | — | (32,32,262144) | 5.4e8 | 1.1e10 |
| square | fz2 | 128 | 4 | 2 | 16 | 268435456 | 10 | **2** | (2097152,64,64) | — | (64,64,2097152) | 1.7e10 | 3.4e11 |
| square | **fz2_u1** | 8 | 4 | 7 | 455 | 834 | 10 | **7** | (73,1,1) | (108,1,1) | (2,2,119) | 146/216/952 | 2.1e4 |
| square | **fz2_u1** | 32 | 4 | 11 | 1771 | 206740 | 10 | **11** | (2205,1,1) | (5550,1,1) | (8,8,7595) | 4410/11100/9.7e5 | 2.3e7 |
| square | **fz2_u1** | 64 | 4 | 13 | 2925 | 3246190 | 10 | **13** | (11473,1,1) | (34293,3,3) | (14,14,59369) | 2.3e4/6.2e5/2.3e7 | 6.7e8 |
| square | **fz2_u1** | 128 | 4 | 15 | 4495 | 50521698 | 10 | **15** | (59853,1,1) | (278180,3,3) | (26,26,459893) | 1.2e5/5.0e6/6.2e8 | 2.0e10 |
| square | fz2_u1_flat | 8 | 4 | 4 | 84 | 1716 | 10 | **4** | (120,1,1) | (1,1,210) | (3,3,252) | 240/420/4536 | 9.0e4 |
| square | fz2_u1_flat | 32 | 4 | 4 | 84 | 439296 | 10 | **4** | (7680,4,4) | (4,4,13440) | (12,12,16128) | 2.5e5/4.3e5/4.6e6 | 9.2e7 |
| square | fz2_u1_flat | 64 | 4 | 4 | 84 | 7028736 | 10 | **4** | (61440,8,8) | (8,8,107520) | (24,24,129024) | 7.9e6/1.4e7/1.5e8 | 2.9e9 |
| square | fz2_u1_flat | 128 | 4 | 4 | 84 | 112459776 | 10 | **4** | (491520,16,16) | (16,16,860160) | (48,48,1032192) | 2.5e8/4.4e8/4.8e9 | 9.4e10 |

**`nblocks == nsectors(V(χ))` in every row, confirming phase 0** — and extending
it: the identity holds not only for the closing but for *every* `mul!` in the
call, because `blocksectors(Layout(k))` is exactly the sector set of the target
leg. For `fz2_u1` that is **7 / 11 / 13 / 15** at χ = 8 / 32 / 64 / 128 — phase 0's
"7-13" was measured over χ ≤ 64 and is correct; χ = 128 adds one more. It is
*not* the 73-337 fusion-tree count. No correction to phase 0 was needed; my own
first census run undercounted `:fz2_u1_flat` by one sector and was fixed
(`_msgspace` had built `V' ← V'` instead of `V ← V`, which drops any sector whose
conjugate is absent — `:fz2_u1_flat` has charges `-2:1`, deliberately not
symmetric about zero). That is why the census is now cross-checked against the
real `blocks(...)` triples on every timed fixture.

Two structural facts that decide the rest of Part A:

1. **`nblk/mul!` is small and grows only like `log χ`** — 2 for `fz2`, 4 for the
   flat fixture, 7-15 for the production `fz2_u1`. Degree 4 does not help: it
   adds `mul!`s, not blocks per `mul!`.
2. **Blocks inside one `mul!` are wildly unbalanced.** For `fz2_u1` at χ = 128 the
   largest block is `2mnk = 6.0e6` against a median of `3.7e4` — a factor **164**.
   One block is roughly half the work of its `mul!`, so a per-`mul!` parallel
   region is load-bound at ~2× no matter how many threads it gets.

## M2 — probe timings

Interleaved arms, `nthreads` ∈ {1,2,4,8}. Full tables in
`blockloop_probe_t{1,2,4,8}.md`; summary here.

### Amdahl ceiling `t_block_loop / t_kernel`

| sym | χ | t=1 | t=2 | t=4 | t=8 | max |
|---|---|---|---|---|---|---|
| fz2 | 32 | 0.543 | 0.529 | 0.485 | 0.510 | 0.543 |
| fz2 | 64 | 0.653 | 0.654 | 0.655 | 0.655 | 0.655 |
| fz2 | 128 | 0.603 | 0.669 | 0.725 | 0.740 | **0.740** |
| **fz2_u1** | 32 | 0.159 | 0.188 | 0.195 | 0.179 | **0.195** |
| **fz2_u1** | 64 | 0.409 | 0.413 | 0.415 | 0.395 | **0.415** |
| **fz2_u1** | 128 | 0.573 | 0.583 | 0.580 | 0.597 | **0.597** |
| fz2_u1_flat | 32 | 0.398 | 0.438 | 0.402 | 0.407 | 0.438 |
| fz2_u1_flat | 64 | 0.576 | 0.560 | 0.552 | 0.542 | 0.576 |
| fz2_u1_flat | 128 | 0.621 | 0.626 | 0.654 | 0.609 | 0.654 |

The `fz2` χ = 128 row drifts 0.603 → 0.740 across the four runs. The block loop
itself is stable (149.4-152.0 ms in all four) while the *kernel* speeds up
251 → 203 ms as the machine's load fell from 4.3 to 3.2. The non-gemm half of the
kernel is strided-copy work and therefore memory-bandwidth bound, which is what
co-tenants contend for; the gemms are compute bound and are not affected. Read
the Amdahl column as ±0.07 at that row and ±0.02 elsewhere.

### Block-loop speedup, achievable vs optimistic

`serial / best threaded arm`, > 1 is faster.

| sym | χ | grouped t=2 | t=4 | t=8 | | flat t=2 | t=4 | t=8 |
|---|---|---|---|---|---|---|---|---|
| fz2 | 32 | 1.19 | 0.87 | 0.61 | | 0.99 | 1.39 | 2.04 |
| fz2 | 64 | 1.99 | 1.99 | 1.47 | | 2.01 | 3.49 | 3.87 |
| fz2 | 128 | 1.86 | 1.86 | **1.83** | | 1.93 | 3.14 | 4.56 |
| **fz2_u1** | 32 | 0.35 | 0.48 | 0.51 | | 0.74 | 1.00 | 0.96 |
| **fz2_u1** | 64 | 0.94 | 0.98 | 0.80 | | 1.01 | 1.42 | 1.63 |
| **fz2_u1** | 128 | 1.01 | 1.37 | **1.72** | | 1.78 | 3.16 | 4.58 |
| fz2_u1_flat | 32 | 0.40 | 0.42 | 0.44 | | 0.70 | 1.03 | 1.53 |
| fz2_u1_flat | 64 | 1.14 | 0.98 | 0.80 | | 1.16 | 1.25 | 1.72 |
| fz2_u1_flat | 128 | 1.22 | 1.64 | **1.64** | | 1.81 | 2.95 | 4.09 |

The **flat** arm scales properly (4.1-4.6× at 8 threads at χ = 128) — the hardware
and the task API are not the problem. The **grouped** arm, which is the only one
the dependency structure permits, saturates at **1.6-1.8× on 8 threads**: 21 %
parallel efficiency, exactly as predicted by the 164× intra-`mul!` block
imbalance. At χ ≤ 64 grouped threading is a *slowdown* at every thread count for
the production symmetry.

### Implied whole-kernel speedup (Amdahl, grouped arm)

| sym | χ | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| fz2 | 128 | 1.45 | 1.51 | 1.51 |
| **fz2_u1** | 32 | 0.74 | 0.82 | 0.85 |
| **fz2_u1** | 64 | 0.97 | 0.99 | 0.91 |
| **fz2_u1** | 128 | 1.01 | 1.19 | **1.33** |
| fz2_u1_flat | 128 | 1.12 | 1.34 | 1.31 |

**The best case for phase 4b, anywhere in the grid, is 1.33× on 8 threads at the
production symmetry and χ = 128; below χ = 128 it is a regression.**

### Spawn round trip

`Threads.@spawn`-and-`wait`, one task: **0.93-1.10 µs** at `nthreads = 1`, and
**1.1-9.4 µs** at `nthreads` > 1 (waking a parked worker). `nthreads` tasks:
0.88-19.0 µs.

## M3 — a correction: the flame graph undercounts gemm

The profile puts `mul!`/`gemm_wrapper!`/`gemm!` at **24.7-28.4 %** of the kernel
at χ = 128, while the direct interleaved timing of exactly those calls gives
**57-74 %**. The direct number is right, and this is checked independently: the
probe times the median-flops block on its own and gets **9868 µs** for the
`(16384, 64, 64)` complex gemm of `fz2` χ = 128; a bare
`mul!(::Matrix{ComplexF64}, ::Matrix, ::Matrix)` of the same shape on this box
takes **9760 µs** (55 GFlop/s, near single-core peak). The two agree to 1 %.

The mechanism is almost certainly that `SIGPROF` cannot unwind through
OpenBLAS's assembly kernels, so those samples carry no Julia frame and match no
category, while every non-BLAS category keeps its true share. **The committed
gemm fractions in `benchmark/reports/kernel_profile.md`, and the plan's
"gemm share is now 20.9-51.9 %", are therefore lower bounds.** Using the larger
direct figure is also the choice generous to a "go" verdict here, and the rule
still fails.

## M4 — the zero-Canopy-code baseline

Both upstream knobs were verified against the installed TensorKit 0.17.1 source
before being relied on:

- `TRANSFORMER_THREADS = Ref(1)` (`TensorKit.jl:226`) — confirmed, default 1. It
  gates `use_threaded_transform`, which **additionally** requires
  `length(t.data) > Strided.MINTHREADLENGTH = 32768`
  (`indexmanipulations.jl:579-584`). Every χ = 32 fixture here is below that, so
  the knob is inert there no matter what it is set to — the probe records the
  eligibility flag per row.
- `blockscheduler` / `with_blockscheduler` (`tensors/backends.jl:12-33`) —
  confirmed **read by nothing**: `default_blockscheduler` has zero callers in
  0.17.1 (grep over `src/` and `ext/`), and `foreachblock`'s `scheduler` kwarg is
  accepted and ignored (`blockiterator.jl:17,33-45`, `# TODO: implement
  scheduler`). Measured accordingly: 12 rows × 4 thread counts, every one within
  the row's noise of the unscoped call. **It is a no-op.**

`compute_message!` walltime, µs, at `Threads.nthreads() = 8`:

| sym | χ | data > MINTHREADLENGTH | TT=1 | TT=2 | TT=4 | TT=8 | gain at TT=8 |
|---|---|---|---|---|---|---|---|
| fz2_u1 | 32 | no | 798 | 790 | 801 | 849 | 0.94× |
| fz2_u1_flat | 32 | no | 373 | 374 | 373 | 371 | 1.01× |
| fz2 | 32 | no | 1567 | 1563 | 1573 | 1568 | 1.00× |
| fz2_u1 | 64 | yes | 3482 | 4935 | 4868 | 3662 | 0.95× |
| fz2_u1_flat | 64 | yes | 2712 | 3198 | 3323 | 3452 | 0.79× |
| fz2 | 64 | yes | 18314 | 22904 | 18594 | 17082 | 1.07× |
| **fz2_u1** | 128 | yes | 34478 | 39326 | 32236 | **27471** | **1.26×** |
| **fz2_u1_flat** | 128 | yes | 42007 | 38190 | 34499 | **33864** | **1.24×** |
| fz2 | 128 | yes | 191106 | 188192 | 182369 | 182802 | 1.05× |

At 2 and 4 threads the knob is a **pessimization** on the graded fixtures at
χ = 64 (up to 1.4× slower at TT = 2 — see `blockloop_probe_t2.md`); it only pays
when `TT == nthreads` and the tensor is large.

**Audit of the single-buffer assumption the plan flagged.** For `UniqueFusion` —
every symmetry Canopy's blocked kernel accepts — the threaded region is
`add_transform_kernel!(data_dst, data_src, p, ::AbelianTreeTransformer, …)`
(`indexmanipulations.jl:655-668`), which is `tforeach(transformer.data)` over
`TO.tensoradd!` on `StridedView`s and **allocates nothing inside**: the
`@lock buffer_lock TO.tensoralloc(…)` the plan worried about lives only in the
`GenericTreeTransformer` branch, i.e. the non-abelian path, which
`uses_blocked_kernel` already excludes. Destinations are disjoint because the
tree transform is a bijection under `UniqueFusion`. **So the knob is safe with a
Bumper `ResizeBuffer` on Canopy's path** — but it is a process-global `Ref`, not
a scoped value, so switching it on inside Canopy would change behaviour for every
other TensorKit user in the process, and it needs a `dim`-and-`nthreads` guard or
it loses 5-21 % at χ ≤ 64.

## Go/no-go against the pre-committed rule

*Go iff all three hold at the production `(symmetry, χ)` — `fz2_u1`, honeycomb,
degree 3.*

| criterion | bar | χ = 32 | χ = 64 | χ = 128 | |
|---|---|---|---|---|---|
| `t_block_loop / t_kernel` | > 0.7 | 0.16-0.20 | 0.40-0.42 | 0.57-0.60 | **fail** |
| median per-block time vs 20 × spawn round trip | > | 0.45 µs vs 19-161 µs | 3.5-4.7 µs vs 20-176 µs | 14.6-15.1 µs vs 19-149 µs | **fail** |
| `nblocks ≥ 4 × nthreads_target` | ≥ | 11 → only `nthreads ≤ 2` | 13 → `≤ 3` | 15 → `≤ 3` | **fail for any target ≥ 4** |

### **NO GO. Do not start phase 4b.**

All three criteria fail at every production χ, and they fail in a way that is
structural rather than marginal:

- The Amdahl ceiling never reaches 0.7 for `fz2_u1`. Its maximum anywhere in the
  grid is 0.740, at `fz2` χ = 128, where `nblk/mul! = 2` makes criterion 3
  hopeless.
- Criteria 2 and 3 are **anti-correlated by construction**. Blocks are numerous
  exactly when the symmetry is fine-grained, and fine-grained symmetry is exactly
  what makes each block small. There is no `(symmetry, χ)` in the reachable range
  where both hold.
- The measured achievable win agrees: **1.33× kernel speedup for 8 threads** at
  the single best production point, a *regression* everywhere else, against an
  implementation carrying the entire phase-4b hazard table (`BufferPool`
  restoration, allocator-per-task, `maxrss` plateau tests, a CI thread matrix).
- `TensorKit.set_num_transformer_threads(8)` delivers **1.26×** at that same
  point for **zero Canopy code** — 95 % of the win. It threads the *other* half
  of the kernel (the transports), which is the half the direct timing shows is
  40 % of the time at χ = 128 and 80 % at χ = 32.

If threading is revisited, the lever to pull is upstream's transport threading
behind a size guard, not a Canopy block-task API.

---

# Part B — plan-cache feasibility

## B1 — how many lookups, and what they cost

Counted from `LRUCache.cache_info` deltas on `TensorKit.GLOBAL_CACHES` over 20
warm calls. **Identical for every `(sym, χ)`** — it is a function of `d` and the
call shape only, as it should be.

| cache | lookups per vertex call (d = 3, 3 targets) |
|---|---|
| `GLOBAL_SECTORSTRUCTURE_CACHE` | **24** |
| `GLOBAL_DEGENERACYSTRUCTURE_CACHE` | **37** |
| `GLOBAL_TREEBRAIDER_CACHE` | **9** |
| `GLOBAL_TREETRANSPOSER_CACHE` | 0 |
| `GLOBAL_FSBRAID_CACHE` / `GLOBAL_FSTRANSPOSE_CACHE` | 0 |
| **total** | **70** |

**The plan's estimate of `~3d` (= 9) is wrong by 8×; the true figure is ~23`d`.**
The count decomposes exactly:

- 24 `blockstructure`-equivalents, each costing one `sectorstructure` +
  one `degeneracystructure`: 21 from the seven `mul!`s (`blocks` is called on all
  three operands, `linalg.jl:336-338`) + 3 from `_twist_message!`'s `blocks(mt)`;
- 13 further `degeneracystructure` from `tensoralloc_add` → `dim(::HomSpace)`:
  3 messages + 2 entries + 4 chain steps × 2 (`tmp` and `result`);
- 9 `treebraider`, one per transport (3 message transposes + 2 entry braids +
  4 chain re-permutes).

Cost of a single **hit**, same fixture (µs) — this is what a plan cache saves per
lookup:

| sym | χ | sectorstructure | degeneracystructure | treebraider | `hash(::HomSpace)` |
|---|---|---|---|---|---|
| fz2 | 8 → 128 | 0.166 → 0.215 | 0.131 → 0.131 | 0.275 → 0.274 | 0.016 → 0.017 |
| **fz2_u1** | 8 | 0.609 | 0.627 | 1.312 | 0.176 |
| **fz2_u1** | 32 | 0.728 | 0.798 | 1.681 | 0.267 |
| **fz2_u1** | 64 | 0.778 | 0.882 | 1.837 | 0.281 |
| **fz2_u1** | 128 | 0.838 | 0.967 | 2.071 | 0.312 |
| fz2_u1_flat | 8 → 128 | 0.519 → 0.516 | 0.489 → 0.490 | 1.011 → 1.017 | 0.136 → 0.138 |

Two things follow. A hit costs **0.13-2.1 µs**, dominated by hashing the
`HomSpace` key, so it scales with the *sector count* and is essentially
**independent of χ** at fixed sector count (`fz2_u1_flat` is flat to 1 % over a
4096× range in `dim`). And the total per-call overhead is therefore a roughly
**constant 11-75 µs**, which is a large share of a small kernel and a negligible
share of a large one. Predicted totals: `fz2` 11.3 µs, `fz2_u1_flat` 39.7 µs,
`fz2_u1` 49.7 µs (χ = 8) rising to 74.5 µs (χ = 128).

## B2 — profile shares

Inclusive, overlapping, from `Profile.@profile` over the current default kernel.
Read alongside the M3 caveat: the BLAS row is a lower bound, everything else is
sound.

| sym | χ | µs/call | mul!/gemm | tensoradd!/add_transform | Strided | sectorstructure | degeneracystructure | blockstructure | subblockstructure | treebraider | hash | **LRU `get!`** | spacecheck | tensoralloc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fz2 | 8 | 103 | 14.2 | 30.5 | 20.6 | 4.3 | 4.8 | 7.2 | 0.0 | 2.2 | 2.7 | **10.7** | 0.1 | 2.1 |
| **fz2_u1** | 8 | 280 | 7.4 | 38.5 | 21.4 | 2.5 | 6.0 | 5.9 | 0.0 | 2.2 | 4.2 | **10.5** | 0.2 | 2.7 |
| fz2_u1_flat | 8 | 116 | 15.2 | 28.7 | 15.7 | 6.5 | 9.5 | 12.6 | 0.0 | 4.3 | 7.3 | **19.8** | 0.4 | 3.6 |
| fz2 | 32 | 1721 | 26.0 | 20.9 | 20.2 | 0.4 | 0.4 | 0.6 | 0.0 | 0.2 | 0.1 | **0.9** | 0.0 | 0.1 |
| **fz2_u1** | 32 | 859 | 9.9 | 37.6 | 24.6 | 1.1 | 1.8 | 2.2 | 0.0 | 1.0 | 1.9 | **3.8** | 0.1 | 0.7 |
| fz2_u1_flat | 32 | 417 | 19.6 | 25.8 | 21.5 | 1.9 | 2.8 | 3.6 | 0.0 | 1.6 | 2.1 | **6.1** | 0.2 | 1.2 |
| fz2 | 64 | 20497 | 28.0 | 18.4 | 18.2 | 0.1 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | **0.2** | 0.0 | 0.0 |
| **fz2_u1** | 64 | 3315 | 17.9 | 29.2 | 24.3 | 0.4 | 0.7 | 0.8 | 0.0 | 0.4 | 0.6 | **1.4** | 0.1 | 0.3 |
| fz2_u1_flat | 64 | 3389 | 26.3 | 19.7 | 18.9 | 0.5 | 0.6 | 0.8 | 0.0 | 0.3 | 0.3 | **1.3** | 0.0 | 0.3 |
| fz2 | 128 | 249580 | 28.4 | 18.8 | 18.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 | 0.0 |
| **fz2_u1** | 128 | 40561 | 24.7 | 22.0 | 21.0 | 0.1 | 0.1 | 0.2 | 0.0 | 0.1 | 0.1 | **0.3** | 0.0 | 0.1 |
| fz2_u1_flat | 128 | 43286 | 27.6 | 19.1 | 18.9 | 0.1 | 0.1 | 0.1 | 0.0 | 0.1 | 0.1 | **0.2** | 0.0 | 0.0 |

`subblockstructure` is 0.0 everywhere: it is consumed only when a transformer is
*built*, and the transformer is itself cached, so it never runs on the hot path.
The whole structure layer collapses from ~20 % at χ = 8 to ~0.3 % at χ = 128.

## B3 — the cheating prototype (upper bound)

`benchmark/plancache_probe.jl` builds a hand-rolled kernel that resolves every
transformer (via the very `treebraider` objects TensorKit would have looked up)
and every block offset once, at plan time, and then does nothing but
`TO.tensoradd!` on `StridedView`s and `mul!` on `reshape(view(data, rng))`. Same
subblock copies, same gemms, zero `HomSpace` queries.

**Correctness: `maxdiff == 0.0e+00` — bitwise identical to the real blocked
kernel — on all 12 fixtures.** Not `≈`, exactly equal.

| sym | χ | dim | ntrees | real µs | cheat µs | control µs | **real / cheat** | control dev |
|---|---|---|---|---|---|---|---|---|
| fz2 | 8 | 512 | 8 | 54.6 | 41.2 | 41.8 | **1.325** | 1.4 % |
| **fz2_u1** | 8 | 119 | 73 | 257.4 | 193.9 | 192.6 | **1.327** | 0.7 % |
| fz2_u1_flat | 8 | 120 | 16 | 97.8 | 50.8 | 50.3 | **1.926** | 1.0 % |
| fz2 | 32 | 32768 | 8 | 1663.1 | 1755.1 | 1760.7 | **0.948** | 0.3 % |
| **fz2_u1** | 32 | 7595 | 181 | 815.1 | 761.8 | 763.9 | **1.070** | 0.3 % |
| fz2_u1_flat | 32 | 7680 | 16 | 403.0 | 371.7 | 364.3 | **1.084** | 2.0 % |
| fz2 | 64 | 262144 | 8 | 19702.3 | 19715.5 | 19584.5 | **0.999** | 0.7 % |
| **fz2_u1** | 64 | 59369 | 253 | 3299.6 | 3515.9 | 3467.6 | **0.938** | 1.4 % |
| fz2_u1_flat | 64 | 61440 | 16 | 2819.1 | 3010.9 | 2975.1 | **0.936** | 1.2 % |
| fz2 | 128 | 2097152 | 8 | 245942.4 | 266236.5 | 264507.7 | **0.924** | 0.6 % |
| **fz2_u1** | 128 | 459893 | 337 | 35450.8 | 36243.4 | 36024.9 | **0.978** | 0.6 % |
| fz2_u1_flat | 128 | 491520 | 16 | 43387.4 | 42108.2 | 42098.1 | **1.030** | 0.0 % |

**This is an upper bound and nothing more.** A real `MessagePlan` pays a cache
lookup, a validation of the plan against the actual spaces, and generality over
`(d, target set, layout)` that the cheat hard-codes into a single prebuilt
object. The honest reading is "a plan cache buys *at most* this".

Read it with the χ = 8 rows first: **1.33× / 1.33× / 1.93×**, saving 13.4 / 63.5 /
47.0 µs per call. Those savings sit within 25 % of the 11.3 / 49.7 / 39.7 µs the
lookup accounting in B1 predicts from an entirely independent measurement, which
is the strongest evidence in this report that both numbers are right.

At χ ≥ 32 the ceiling collapses to **1.07-1.08×** and at χ ≥ 64 it is gone —
indeed the cheat is 6-8 % *slower* at the largest fixtures. That is not a
measurement error (control deviation ≤ 1.4 %); it is the cheat's buffer layout.
It holds `2d + 1` full-size chain buffers live and separate, where the real
kernel bump-allocates from one contiguous arena and recycles two temporaries, so
at 33 MB per link the real kernel has strictly better locality. A real plan cache
would inherit the same problem if it preallocated per-slot buffers, which the
phase-3 sketch's "one fused bump allocation with integer slot offsets" was
designed to avoid — but the measurement says the thing being optimised is worth
2 % at that size anyway.

## Recommendation on the plan cache

**Do not implement it. The phase-3 judgement that the full `MessagePlan` was too
complex for its expected return still holds, and the number now behind it is
1.07× at χ = 32 and 0.98× at χ = 128.**

Reasoning, in the order the evidence supports it:

1. The measured ceiling at every production point (`fz2_u1`, χ ∈ 32…128) is
   **1.07× / 0.94× / 0.98×**. Only χ = 8 shows a real win (1.33×), and χ = 8 is
   the regime where the *unsymmetric* kernel is 12× faster anyway — nobody runs
   production there.
2. The overhead is an approximately **constant 11-75 µs per vertex call**,
   because it is dominated by hashing a `HomSpace` and that cost tracks the
   sector count, not χ. It cannot grow into relevance: `dim(space(T)) = |P| χ^d`
   grows as χ³ while the lookup bill stays flat.
3. The complexity is exactly what phase 3 declined: a plan struct, a builder that
   must reproduce `treebraider` and `blockstructure` semantics, a keyed and
   validated module-level cache, and a `treebraider`-equivalence test written
   before the kernel. Phase 3's *reduced* scope shipped 1.08-1.12× for one
   ~40-line hoist; this would be an order of magnitude more code for less.
4. Where the time actually goes at production χ is now unambiguous:
   **~57-74 % gemm and ~20-40 % strided subblock copies**, with the structure
   layer at 0.3 %. Both remaining halves are already near their hardware limits —
   the gemms measured at 55 GFlop/s, within 1 % of a bare BLAS call of the same
   shape.

The only lever the data still supports is **making the individual gemms less
skinny**, i.e. changing the block shapes rather than the bookkeeping around them —
and the census says the shapes are fixed by the symmetry, not by the kernel.

---

# Caveats and things not done

- **The flame-graph gemm share is a lower bound** (M3). Any phase that quotes
  `benchmark/reports/kernel_profile.md`'s gemm column should say so. I did not
  fix `profile_kernel.jl`; that is a `benchmark/` change worth making, but it
  changes a committed report and belongs in its own step.
- **`nthreads` > 1 measurements at χ ≤ 64 are not trustworthy** (noise floor up to
  82 %, an arm-ordering artefact described above). No conclusion rests on them.
  Fixing it means randomising arm order per repetition or inserting a quiescence
  gap; both cost run time and neither would change the verdict.
- **The Amdahl ratio for `fz2` at χ = 128 drifts ±0.07 with machine load.** It is
  the only row that gets near 0.7 and it is the least reliable one.
- The threading probe's `t_block_loop` is measured on prebuilt links with
  dedicated destination buffers, so its cache behaviour differs slightly from the
  in-kernel loop's. The median-block cross-check against a bare BLAS call
  (agreement to 1 %) bounds that effect as small.
- Degree 4 is censused but **not timed**: a `:trivial`/`fz2` degree-4 vertex at
  χ = 128 is 4 GB and `fz2_u1` is 0.8 GB, and the census already shows degree 4
  changes `n_mul!` (7 → 10) without changing `nblk/mul!`, which is the quantity
  the decision rule reads.
- I did not measure `belief_propagation` end to end under any threading
  configuration; with a per-kernel ceiling of 1.33× at one point and regressions
  elsewhere there was nothing to carry into an end-to-end measurement.
