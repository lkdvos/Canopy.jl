# Benchmark baselines

Saved `BenchmarkTools` results for `benchmark/benchmarks.jl`, one file per
(code state, machine, thread count). Committed deliberately — the same precedent
as `benchmark/realtime_timing/data/*.csv`.

## ⚠ A pass/fail comparison is only valid between two interleaved runs on the same machine

These files exist so a later phase can *see* what a baseline looked like, not so
it can be `judge`d against from another host. Wall-clock timings on a shared HPC
system depend on the CPU model, the memory-bandwidth contention from other jobs
on the node, the OpenBLAS build, the Julia version, and the code layout the JIT
happened to produce. Judging a run on machine A against a baseline recorded on
machine B produces confident-looking nonsense.

The only comparison that carries a pass/fail verdict is:

1. same host, same `Threads.nthreads()`, same `OPENBLAS_NUM_THREADS`,
2. same `benchmark/params.json` loaded via `loadparams!` on **both** sides
   (otherwise `run` re-tunes `evals` per key and injects variance between the two
   runs you are comparing),
3. the two runs taken **back-to-back**, ideally alternating baseline/candidate,
   with nothing else scheduled on the node.

### Workstation contention is a first-order effect, not a rounding error

Point 3 is the one that actually bites. These are shared workstations: another
job at 100% CPU on a different core still competes for memory bandwidth and
last-level cache, and the symmetric fixtures here are memory-bound (many small
blocks, little arithmetic per byte). Pinning BLAS to one thread does nothing about
that.

So: `benchmarks.jl` prints `/proc/loadavg` on every run. **Check it on both sides
of a comparison.** If the loads differ materially, or if either run was taken
while something else was hammering the box, re-run — do not reason about the
deltas. And when a group the change cannot possibly have touched (`:sync` keys,
`:trivial` keys) comes back flagged, contention is the most likely explanation and
the comparison is void.

The floors recorded in `benchmark/reports/noise_floor.md` were measured **with
co-tenant load on the machine** (two unrelated Julia jobs at ~99% CPU). They are
therefore an upper bound on this host, not a property of the code. Re-measure the
floor on the machine, and under the load conditions, where you intend to judge.

`judge` also needs a noise control. Every phase of this project changes only part
of the code, so some `SUITE` keys must come back `:invariant`. If a key that
*cannot* have been affected is flagged as a regression or an improvement, the
whole comparison is void — re-run it.

## Naming

    <label>_<gitsha7>_t<nthreads>.json

`label` is the code state (`main`, `noise1`, `phase1`, …), `gitsha7` is
`git rev-parse --short=7 HEAD` at the time of the run, `nthreads` is
`Threads.nthreads()`.

Always save the full `results`, never `median(results)`: medians can be taken
later, samples cannot be recovered.

## Recording a baseline

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=benchmark -e '
  include("benchmark/benchmarks.jl"); tune!(SUITE)
  BenchmarkTools.save("benchmark/params.json", params(SUITE))
  sha = readchomp(`git rev-parse --short=7 HEAD`)
  BenchmarkTools.save("benchmark/baselines/main_$(sha)_t1.json", run(SUITE; verbose = true))'
```

## Files

All files below: host `ccqlin038.flatironinstitute.org`, CPU `cascadelake`
(32 logical CPUs), julia 1.12.6, `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
default `SUITE` (no `CANOPY_BENCH_FULL`), 93 keys.

| file | code state | notes |
|---|---|---|
| `noise1_e79a384_t1.json` | phase-0 harness on `symmetry-kernel`; `src/` byte-identical to `main` (`e79a384`) | run 1 of the noise-floor pair |
| `noise2_e79a384_t1.json` | identical — same commit, same `params.json`, taken immediately after run 1 | run 2 of the noise-floor pair |

Note the `_e79a384_` in both names is `main`'s sha, and that is the point: phase 0
adds no `src/` diff, so both runs execute exactly the code `main` executes.

### ⚠ Both files predate the `:fz2_u1_flat` addition — expect missing keys, not regressions

`noise1` / `noise2` were recorded when `BENCH_SPACES` held four symmetries. The
phase-0 addendum promoted `:fz2_u1_flat` from census-only into `BENCH_SPACES`, so
the current default `SUITE` has **96** keys where these files have 93. The three
extra keys are

    SUITE["message"]["hex_vertex", :fz2_u1_flat, χ]   for χ ∈ {8, 16, 32}
    (plus χ = 64 under CANOPY_BENCH_FULL=1)

and they are the only difference — no existing key changed, since `bench_sweep.jl`,
`bench_schedule.jl` and `bench_allocator.jl` name their symmetries explicitly
rather than iterating `BENCH_SPACES`.

Consequence for a future `judge` against these files: **BenchmarkTools reports keys
present on only one side as missing from the comparison, not as regressions.** The
new keys will simply be absent from the `judge` output. That is expected and is not
evidence of anything. The recorded noise floors in
`benchmark/reports/noise_floor.md` remain valid for the 93 shared keys — the
`message` group's floor in particular was measured over the other symmetries at the
same fixture and geometry, so applying it to the new keys is reasonable but is an
extrapolation, not a measurement.

Re-taking the pair (≈45 min for two full runs) was deliberately deferred: it buys
one group's floor for three new keys and nothing else. Do re-take it before any
phase gates on `:fz2_u1_flat` timings quantitatively.

`benchmark/params.json` covers the new keys. Only the three new entries were
tuned and appended; the 93 pre-existing entries were left **byte-identical**, so
`loadparams!` still reproduces exactly the `evals`/`samples`/`seconds` that
`noise1`/`noise2` were recorded with. A blanket `tune!(SUITE)` would *not* have
that property — it would re-derive `evals` for every key and quietly break the
one condition point 2 above insists on. If you ever do need a full re-tune,
re-take the baselines in the same session.

### The noise floor

`noise1` and `noise2` were produced by running the harness twice over *identical*
code — phase 0 changes nothing under `src/`, which is what makes the measurement
possible at all — with `params.json` loaded via `loadparams!` on both sides and no
`tune!` in either (tuning first would leave the two processes in different states).
`judge(median(run2), median(run1))` therefore reports noise and nothing else, and
the maximum absolute delta per group is the floor below which no later phase's
result is a result:

    threshold(group) = max(5%, 2 × noise_floor(group))

The measured floors and thresholds are in `benchmark/reports/noise_floor.md`.
Regenerate with:

```bash
julia --project=benchmark benchmark/report_noise.jl \
    benchmark/baselines/noise1_e79a384_t1.json \
    benchmark/baselines/noise2_e79a384_t1.json --label noise_floor
```

Two things to know before quoting a floor:

- **It includes workstation contention** (see above), so it is an upper bound for
  this host under the load these runs saw, not a property of the code.
- **The groups differ by an order of magnitude.** `message` is looped thousands of
  times per sample and is tight; the `evals = 1` groups (`sweep`, `schedule`,
  `convergence`) take few samples of an expensive operation and are loose. A single
  global threshold would be simultaneously too strict for one and too lax for
  another — use the per-group numbers.
- For `SUITE["schedule"]` specifically, prefer `benchmark/reports/schedules.md`'s
  `iters` column as the primary signal: it is exactly reproducible and
  machine-independent, whereas that group's *timing* floor is the widest in the
  suite.
