# Real-time Hubbard quench on Canopy.jl

A cluster-ready CLI driver for real-time dynamics of the spinful Hubbard model on a 2D
lattice, evolved by belief propagation + bond truncation (simple update).

```
H = t Σ_<ij>,σ (c†_iσ c_jσ + h.c.)  +  U Σ_i n_i↑ n_i↓
```

Starting from a product state (a global AFM/CDW, or a doublon+hole in an AFM background),
this tracks the staggered order parameter against the exact free-fermion result at `U = 0`
and against energy conservation at `U ≠ 0`.

It is a port of a TensorNetworkQuantumSimulator script (not included here — it depends on a
different library), with symmetry treated as a first-class parameter.

## What this found

Four results that came out of using it, each documented in full under
[Things that will bite you](#things-that-will-bite-you):

1. **The truncation cutoff is a correctness parameter, not a tuning knob.** Below Canopy's gauge
   tolerance (~1.8e-12) the gauge pseudo-inverse amplifies numerically-null Schmidt directions.
   The failure is silent and *anti-convergent* — reported truncation error shrinks while
   observables go wrong, worse at larger χ. The source script's `cutoff = 1e-14` is below the
   floor; the default here is 100× the gauge tolerance and `truncation` warns below it.
2. **χ-convergence is not accuracy.** On hex 4×4 the error against the exact free-fermion result
   saturates at 1.0e-1 while discarded weight falls 100×: the state converges in χ to a
   well-defined *wrong* answer, because BP's Bethe approximation is uncontrolled on a loopy
   graph. Establish the horizon with a `U = 0` run before trusting `U ≠ 0` at the same times.
3. **Symmetry buys walltime, not accuracy at fixed χ** — and it only pays above χ ≈ 32–64. Both
   U(1)s together is the right choice at large χ (3.96× at χ=128, 3.3× faster than either alone
   at χ=256), but it is the *slowest* option below χ ≈ 48.
4. **Belief propagation, not the simple update, is the bottleneck for symmetric runs.** Symmetry
   makes the gates 15.7× cheaper at χ=128 while BP stays flat, leaving BP at 57–82% of a step.
   A block-sparse message kernel is what would unlock the rest.

## Layout

| file | role |
|---|---|
| `HubbardQuench.jl` | physics: symmetries, local states, lattices, gates, observables, references |
| `run.jl` | CLI driver — one parameter point per invocation |
| `make_sweep.jl` | emits a disBatch task file for a parameter sweep |
| `slurm/disbatch.sbatch` | runs a task file under disBatch in one allocation |
| `aggregate.jl` | result folders → `summary.csv` + per-run figures + timing/crossover report |
| `plot_benchmark.jl` | symmetry-benchmark figures (cost, speedup, gate/BP split) |
| `test/runtests.jl` | unit tests + the end-to-end `U = 0` correctness gate |

Self-contained environment with a committed `Manifest.toml`, so numbers stay comparable
across weeks. Note it mirrors the root project's `[sources]` entry for `TensorKitTensors`
(0.3 is unregistered and comes from git; `[sources]` is only honored in an environment's
top-level project).

## Quick start

```bash
# one run
julia --project=scripts/hubbard_quench scripts/hubbard_quench/run.jl \
      --lattice hex --size 6 6 --U 4 --chi 16 --tfinal 1.0 \
      --particle-symmetry u1 --spin-symmetry u1

julia --project=scripts/hubbard_quench scripts/hubbard_quench/run.jl --help

# tests (~8 min)
julia --project=scripts/hubbard_quench scripts/hubbard_quench/test/runtests.jl
```

## Output

Each run writes a self-describing folder whose name encodes the physics:

```
results/hex6x6_cdw_U4_t-1_dt0.01_T1_chi16_sym-u1u1_bp30-1e-10/
  params.toml            full parameters + derived quantities + environment (Julia,
                         hostname, Canopy git SHA, every direct dependency version)
  observables.csv        one row per measurement
  timings.csv            one row per Trotter step, per-phase wall time and heap bytes
  site_observables.csv   per-site occupations (with --site-resolved)
  run.log                full structured log
  timer.txt              TimerOutputs breakdown
```

Rows are appended and flushed as they are produced, so a job killed at the wall clock keeps
everything computed up to that point.

Every log a cluster job produces is collected under the results root, one folder per job:

```
results/<sweep>/_run/<jobid>/
  slurm.log        the batch script's own output
  disbatch_*       disBatch driver, engine, kvsinfo and status files
  tasks/task_N.log one per task
```

`slurm/disbatch.sbatch` does this by reading the `#HQ OUTROOT` header that `make_sweep.jl`
writes into the task file. Two wrinkles worth knowing:

- **Slurm's `--output` cannot point here on its own** — Slurm expands it at submit time, before
  the script runs, so it cannot depend on the task file. `make_sweep.jl` prints a recommended
  `sbatch` line with `--output=<outroot>/_run/slurm-%j.log`; without it the job's top-level log
  lands in whatever directory you submitted from (the script still redirects everything after
  the banner into the run folder, so only a one-line pointer is left behind).
- Per-task logs follow `HQ_TASKLOG_DIR`, which the batch script exports. The task file falls
  back to its own `--logdir` when that is unset, so running a task file by hand still works.

`observables.csv` columns: `step, time, m_s_all, m_s_bulk, n_mean, docc_mean, E_kin, E_int,
E_tot, ref_circuit, ref_cont, maxdim, nsectors_max, maxsecdim, eps_max, eps_sum, eps_cum,
logl_cum, bp_resid`.

## Sweeps

`make_sweep.jl` writes a disBatch task file and prints the exact `sbatch` line to use — it
never runs a Slurm command itself. See *Running on Rusty* below for the full three-step flow.

Canopy's `apply!` is **single-threaded** — there is no `Threads` anywhere in `src/`, and the
comment in `examples/realtime/main.jl` claiming otherwise is stale. So the efficient shape is
one process per parameter point with `BLAS.set_num_threads(1)` (the default), packed densely
into one allocation by disBatch. Do not ask for many threads per task.

**Concurrency has to shrink as χ grows, or the job gets OOM-killed.** A site tensor is
`4·χ³` complex on a coordination-3 lattice — 34 GB of state across 32 sites at χ=256 — and on
top of that every concurrent task holds its own Julia process, simple-update temporaries, and a
Bumper buffer that does not shrink back. A 40-task sweep reaching χ=256 peaked at **698 GB** and
lost its three largest tasks to the OOM killer. Split by cost instead of running one wide sweep:

| χ range | suggested `-n` |
|---|---|
| ≤ 64 | 40+ — cheap, pack them in |
| 96–128 | ~8 |
| ≥ 192 | ~4, with `--mem=0` |

Partial data survives a kill, because rows are flushed as they are produced — but a task killed
before its bonds saturate contributes nothing usable, so prefer a right-sized allocation over
relying on that.

### Running on Rusty

`slurm/disbatch.sbatch` takes a task file and runs it under disBatch:

```bash
# 1. generate the task file
julia --project=scripts/hubbard_quench scripts/hubbard_quench/make_sweep.jl \
      --chi 8 16 --U 0 4 --symmetry trivialtrivial u1u1 \
      --outfile "$PWD/scripts/hubbard_quench/tryout.disbatch" \
      --outroot "$PWD/scripts/hubbard_quench/results/tryout" \
      --logdir  "$PWD/scripts/hubbard_quench/results/tryout/logs" \
      -- --lattice hex --size 4 4 --dt 0.02 --tfinal 1.0 --measure-every 5 --site-resolved

# 2. submit — make_sweep.jl prints this line for you, with paths filled in.
#    -n = concurrent tasks; -t must cover the SLOWEST SINGLE task, not the sum.
sbatch -n 8 -t 30:00 -p ccq -C icelake \
       --output=scripts/hubbard_quench/results/tryout/_run/slurm-%j.log \
       scripts/hubbard_quench/slurm/disbatch.sbatch \
       scripts/hubbard_quench/tryout.disbatch

# 3. aggregate — writes summary.csv, per-run figures, and the timing/crossover report
julia --project=scripts/hubbard_quench scripts/hubbard_quench/aggregate.jl \
      --outroot scripts/hubbard_quench/results/tryout

# 4. for a symmetry benchmark, also render the comparison figures
julia --project=scripts/hubbard_quench scripts/hubbard_quench/plot_benchmark.jl \
      --outroot scripts/hubbard_quench/results/tryout
```

Use **absolute** paths for `--outfile/--outroot/--logdir`: the task file's `#DISBATCH PREFIX`
cds to the repo root, so relative paths happen to work, but only by coincidence.

Pin the microarchitecture with `-C icelake` (or whatever you standardise on) for any run whose
*timings* you intend to compare — step times measured across different CPUs are not comparable,
and pinning also keeps the precompilation cache warm between submissions.

Three things the sbatch script does deliberately:

- **Julia is resolved by absolute path, not from `PATH`.** juliaup may live on a workstation's
  local NVMe (`/home/$USER/.juliaup`), which compute nodes cannot see — the bare name then fails
  to resolve at all, even from a login shell. The real toolchains are under `~/.julia/juliaup`
  on shared home. `make_sweep.jl` bakes `Sys.BINDIR`'s julia into the task file, and the batch
  script resolves its own the same way, preferring 1.12.x (the site `julia` module is 1.11.x,
  below this project's compat).
- **`bash -l`** — needed for `module`, which is a shell function set up by the login profile.
  It is *not* how Julia is found.
- **A serial `Pkg.precompile()` before disBatch fans out.** Without it every task precompiles
  the same ~50 packages simultaneously, contending for the same cache files under
  `~/.julia/compiled` on GPFS and wasting minutes of allocation each. It also absorbs the case
  where the compute nodes' CPU differs from whatever last precompiled, which silently
  invalidates the cache.

Sizing: each task pays ~90 s of first-call JIT plus ~30 s for its first measurement before real
work starts, which is why it is not worth sweeping very many very short runs — a sysimage would
pay for itself there. Per-step costs at scale are in the benchmark table below.

Results are small CSVs, so `/mnt/home` is the right place for them. Move `--outroot` to
`/mnt/ceph/users/$USER` only if you start writing site-resolved data for large lattices over
long trajectories.

## Things that will bite you

### The truncation cutoff must stay above Canopy's gauge tolerance

**This is the one that silently corrupts results.** `apply!` gauges each bond through the
pseudo-inverse of the BP message spectrum, flooring eigenvalues at
`default_gauge_tol ≈ 1.8e-12`. A rank-only truncation keeps numerically-null Schmidt
directions once `χ` exceeds the rank the state actually needs, and the gauge inverse then
amplifies noise in them catastrophically.

The failure is *anti-convergent*: the reported truncation error keeps shrinking while the
observables go wrong, and it gets worse with larger `χ`. Measured on a 4-site ring, error in
the order parameter against the exact result:

| cutoff | χ=8 | χ=16 | χ=32 | χ=64 |
|---|---|---|---|---|
| `0` | 1.0e-02 | 1.1e-02 | 6.6e-03 | **3.5e-01** |
| `1e-12` | 1.0e-02 | 1.1e-02 | 1.0e-02 | **3.6e-01** |
| `1e-10` | 1.0e-02 | 1.1e-02 | 1.0e-02 | 1.0e-02 |

(The residual 1.0e-2 is genuine BP loop error on that ring, not truncation.) The default
`--cutoff` is `1.8e-10` and the driver warns if you go below it. **The TNQS script's
`cutoff = 1e-14` is below the floor and removes nothing** — do not carry it over.

### χ-convergence is not accuracy: BP loop error sets the real floor

Measured on hex 4×4 (32 sites, 40 edges → 9 independent loops), CDW quench at `U = 0`,
`dt = 0.02`, `T = 1.0`, against the exact free-fermion result:

| χ | `eps_cum` (discarded weight) | m_s(T=1) | \|err\| |
|---|---|---|---|
| 8 | 9.7e-1 | −0.2509 | 1.07e-2 |
| 16 | 3.4e-1 | −0.1596 | 1.02e-1 |
| 32 | 9.0e-2 | −0.1597 | 1.020e-1 |
| 64 | 1.1e-2 | −0.1601 | 1.015e-1 |

Truncation error falls by ~100× while the error against the exact result **saturates at
1.0e-1**: the state converges in χ to a well-defined answer of −0.160 that is simply wrong
(exact: −0.262). That residual is the Bethe/BP approximation to the reduced density matrix,
which is uncontrolled on a loopy graph and does not improve with χ. `bp_resid` is ~1e-11
throughout, so this is BP's *fixed point* being wrong, not BP failing to converge.

Two practical consequences:

- **A χ-scan alone will mislead you.** Watching m_s stabilize between χ=16 and χ=64 reads as
  "converged" when it is off by 39%. And note χ=8 has the *smallest* error here purely by
  cancellation — heavier truncation happens to suppress the correlations BP mishandles. Never
  rank χ values by agreement with the reference without also checking `eps_cum`.
- **Past the loop-error onset, χ is not the bottleneck, so symmetry does not buy accuracy.**
  Symmetry buys reachable χ; if χ is already past the point where it matters, that is not the
  constraint. To go further you need a better environment (loop-corrected BP or boundary-MPS
  environments, both listed as planned in Canopy's `docs/src/design.md`) or a less loopy
  geometry.

On this geometry BP tracks the exact result to ~2e-5 at T=0.2 and ~1e-3 at T=0.4, with the
loop error taking over from roughly T≈0.4. Establish that horizon for your lattice with a
`U = 0` run before trusting `U ≠ 0` results at the same times.

### The charge bath is a real extra vertex

Particle-U(1) at half filling has total charge `Q = N ≠ 0`, so `product_state` attaches an
auxiliary charge-bath vertex that appears in `vertices(state)`, `length(state)` and
`edges(state)`. Spin-U(1) with a balanced AFM has `Sz = 0` and needs none — the two U(1)s
behave differently. Everything here iterates the module's own `lat.verts` / `lat.edges`; if
you extend it, do the same. `params.toml` records `charge_bath`.

### Where symmetry starts winning: χ ≈ 32–64, and BP is what holds it back

Time per Trotter step on hex 4×4 (32 sites), `U = 4`, `dt = 0.05`, `T = 1.6`, icelake, one core
per run. The median is taken over steps whose bonds had actually reached χ; `—` means that run
never accumulated enough such steps to be trustworthy (see the caveats).

| χ | none | U(1) charge | U(1) spin | U(1)×U(1) |
|---|---|---|---|---|
| 8 | **0.34** | 1.21 | 1.21 | 3.90 |
| 16 | **0.71** | 1.98 | 1.65 | 6.35 |
| 24 | **1.76** | 2.48 | 2.56 | 10.49 |
| 32 | 3.97 | **3.06** | 3.25 | 11.78 |
| 48 | 11.85 | 6.53 | **6.55** | 17.16 |
| 64 | 29.11 | 11.76 | **11.46** | 25.90 |
| 96 | 92.75 | 34.32 | **31.80** | 43.66 |
| 128 | 228.23 | 83.76 | 76.14 | **57.59** |
| 192 | — | 247.65 | 236.11 | **100.07** |
| 256 | — | 617.02 | — | **189.46** |

Crossover against no symmetry: **either single U(1) at χ ≈ 32, both together at χ ≈ 64.** At
χ=128 U(1)×U(1) reaches 3.96× and overtakes the single U(1)s, which then flatten out and end up
3.3× *slower* than it by χ=256. No-symmetry runs cleanly as χ³ (measured exponents 2.86 and
3.13); U(1)×U(1) runs χ^1.0–2.2.

**Use both U(1)s** unless you are working below χ ≈ 64, where no symmetry is simply faster.

Splitting the cost shows why — and where to optimise:

| χ=128 | gates | BP | BP share |
|---|---|---|---|
| none | 161.6 s | 66.7 s | 29% |
| U(1) charge | 35.9 s | 47.9 s | 57% |
| U(1)×U(1) | **10.3 s** | 47.3 s | **82%** |

Symmetry makes the *gates* dramatically cheaper — 15.7× at χ=128, and the ratio keeps growing —
while BP stays flat or worsens, so BP ends up consuming 57–82% of a symmetric step. **Belief
propagation is the bottleneck, not the simple update.** Messages are `χ × χ` per bond and the
vertex-centric kernel contracts many small blocks; a block-sparse optimisation there is what
would let the total speedup approach the gate ratio.

Three caveats, each of which changed the numbers above when found:

- **Bond dimension is entanglement-limited, not rank-limited.** From a product state, with a
  discard threshold in play, it climbs gradually rather than saturating in a step or two — at
  χ=128 with `dt = 0.02` it went 22, 29, 39, 44, 53, 63, 77, 85, 99, 107, 118, 128 over twelve
  steps, reaching χ only at the end. Timing such a run measures the cost at some intermediate
  bond dimension: it read dense χ=64 as 9.5 s/step instead of 29.1 s and flattened the cost
  curve enough to hide every crossover. `aggregate.jl` now medians only over saturated steps
  and reports the count as `nsat`; `plot_benchmark.jl` drops thin points entirely, because they
  are biased *low* (the first steps after saturation are the cheapest). **Budget enough
  `--nsteps`, and raise `dt`** — accuracy is irrelevant to a timing run, and a larger step
  reaches high χ in fewer expensive steps.
- **`bp/s` folds in the sweep count, not just the cost per sweep.** BP is warm-started, so how
  many sweeps it needs depends on how much each step perturbs the state: the same dense χ=128
  point measured 3.1 s of BP at `dt = 0.02` and 66.7 s at `dt = 0.05`. These are therefore
  realistic adaptive-BP numbers, not clean per-sweep scaling. For the latter, run with
  `--bp-tol 0 --bp-maxiter N` so the work per step is fixed (as Canopy's own
  `benchmark/realtime_timing/run_timings.jl` does). Gate timings are unaffected.
- **The dense χ=192 and χ=256 baselines are missing** because those tasks were OOM-killed, so
  the ~10× that extrapolating χ³ implies at χ=256 is an extrapolation, not a measurement. See
  the memory note under *Sweeps*.

### Symmetry buys walltime, not accuracy at fixed χ

Worth knowing before you sweep, because the naive expectation is backwards. All symmetry
choices converge to the *same* answer, but **spin-U(1) needs a larger χ to get there.**
Staggered-order error vs the exact free-fermion result, hex 2×2, `dt = 0.05`, `T = 0.2`:

| | χ=4 | χ=8 | χ=16 | χ=32 | χ=64 |
|---|---|---|---|---|---|
| spin-trivial (either particle sym) | 1.6e-5 | 8.6e-6 | 8.3e-6 | 8.3e-6 | 8.3e-6 |
| spin-U(1) (either particle sym) | 1.9e-3 | 1.8e-3 | 3.7e-5 | 9.9e-6 | 8.4e-6 |

The common 8.3e-6 floor is BP loop error, not truncation. The gap at small χ is structural: a
symmetric bond must allocate its χ states across Sz sectors in whole numbers, so at equal
*total* χ the block structure fragments the rank and represents the state less efficiently
than an unconstrained bond. Particle-U(1) costs nothing in accuracy here; spin-U(1) does.

So the case for symmetry is **cost per χ and the χ you can afford at all**, not accuracy per
χ — compare at matched accuracy, not matched χ. Which way that lands depends strongly on
per-sector block size: on 8 sites at χ≤8 the symmetric runs are ~10× *slower* (block
bookkeeping dominates tiny blocks), while on the same lattice at χ=32 particle-U(1) ran ~10×
*faster* than no symmetry. `nsectors_max` and `maxsecdim` are in `observables.csv` precisely to
locate that crossover for your problem size.

### Symmetry options

Only abelian symmetries are usable: `{trivial, u1}` for particle number × `{trivial, u1}` for
Sz. SU(2) is rejected with an explicit error — `product_state` throws for non-abelian sectors,
and an AFM product state breaks spin rotation anyway, so nothing is lost.

The (sector, degeneracy index) → physical state map is undocumented upstream and differs per
symmetry. It is tabulated in `HubbardQuench.localstate`, re-derived from the number operators'
diagonal blocks in the tests, and re-asserted at every run by `check_initial_state` — a wrong
degeneracy index produces a perfectly valid state of the *wrong* configuration, which nothing
downstream would reveal.

### Other

- **Only nearest-neighbour observables exist.** `expect` handles one site or one bond;
  `Canopy/src/expect.jl` hard-errors otherwise. Structure factors and `⟨n₀ n_r⟩` vs `r` need a
  Canopy feature, not a script change.
- **`triangular` is rejected** — not bipartite, so the staggered order parameter is undefined.
  Periodic extents that close an odd cycle are rejected for the same reason.
- **BP reports no convergence info.** `bp_resid` is computed here with one extra sweep
  (~3% overhead) on the same scale as the `--bp-tol` you pass. Watch it: silently saturating
  `--bp-maxiter` every step is the likeliest source of unexplained error.
- **`--bp-schedule residual` uses a different tol scale** (its residual is an input-change
  surrogate); do not compare thresholds across schedules.
- **Step 1 carries JIT cost**, so it is excluded from all reported medians.
- **`--log-level debug` is very noisy** — it enables Canopy's per-message allocator tracing,
  thousands of lines per step.
- **Site counts differ from NamedGraphs.** `hexagonal_lattice(m,n)` gives `2mn` sites, while
  TNQS's `named_hexagonal_lattice_graph(5,5)` gives ~70. Pick `m,n` by site count — `(6,6)`
  is 72.

## Validation

`test/runtests.jl` (249 tests) covers the local-state table across all four symmetries, the
charge bath, regions, the Trotter layer structure, the cutoff guard, and:

- **`U = 0` on a tree** — 2 sites, where BP and the Bethe RDM are exact: all four symmetries
  reproduce the exact free-fermion trajectory to `1e-9`. This certifies the local-state table,
  the charge bath, the hopping convention, the Trotter sandwich and `measure` together.
- **`U = 0` on a loop** — hex 2×2: all four symmetries track the exact result to `1e-4` and
  agree with each other to `1e-4`.
- **Energy conservation at `U ≠ 0`** — `E_tot ≡ 0` exactly for this quench, so the measured
  drift is pure Trotter error; the test asserts it shrinks with `dt` rather than fitting a
  magic constant.

## Trotter splitting

Second-order symmetric Strang over the interaction term and the `K` edge-colour classes:

```
single(dt/2) · hop_1(dt/2) … hop_{K-1}(dt/2) · hop_K(dt) · hop_{K-1}(dt/2) … hop_1(dt/2) · single(dt/2)
```

`2K-1` hopping layers per step, mirroring Canopy's `Strang` structure (which cannot be reused
directly — `trotterize` only builds imaginary-time gates). Note this is **not** step-for-step
comparable to the TNQS script at equal `dt`: that script is first order across the colour
classes. `ref_circuit` (the identical circuit, so its residual is pure truncation error) and
`ref_cont` (`exp(-iHT)` in one shot) together quantify the difference.
