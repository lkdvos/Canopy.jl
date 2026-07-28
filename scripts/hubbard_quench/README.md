# Real-time Hubbard quench on Canopy.jl

A cluster-ready CLI driver for real-time dynamics of the spinful Hubbard model on a 2D
lattice, evolved by belief propagation + bond truncation (simple update).

```
H = t Σ_<ij>,σ (c†_iσ c_jσ + h.c.)  +  U Σ_i n_i↑ n_i↓
```

Starting from a product state (a global AFM/CDW, or a doublon+hole in an AFM background),
this tracks the staggered order parameter against the exact free-fermion result at `U = 0`
and against energy conservation at `U ≠ 0`.

This is a Canopy port of a TensorNetworkQuantumSimulator script (`../../hubbard_tnqs.jl`).

## Layout

| file | role |
|---|---|
| `HubbardQuench.jl` | physics: symmetries, local states, lattices, gates, observables, references |
| `run.jl` | CLI driver — one parameter point per invocation |
| `make_sweep.jl` | emits a disBatch task file for a parameter sweep |
| `aggregate.jl` | result folders → `summary.csv` + figures |
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

`observables.csv` columns: `step, time, m_s_all, m_s_bulk, n_mean, docc_mean, E_kin, E_int,
E_tot, ref_circuit, ref_cont, maxdim, nsectors_max, maxsecdim, eps_max, eps_sum, eps_cum,
logl_cum, bp_resid`.

## Sweeps

```bash
julia --project=scripts/hubbard_quench scripts/hubbard_quench/make_sweep.jl \
      --chi 4 8 16 32 --U 0 2 4 --symmetry trivialtrivial u1u1 \
      -- --lattice hex --size 6 6 --dt 0.01 --tfinal 1.0

# then submit it yourself (make_sweep.jl never runs a Slurm command):
sbatch -n 24 -c 1 -t 12:00:00 --wrap 'disBatch scripts/hubbard_quench/sweep.disbatch'

julia --project=scripts/hubbard_quench scripts/hubbard_quench/aggregate.jl --outroot results
```

Canopy's `apply!` is **single-threaded** — there is no `Threads` anywhere in `src/`, and the
comment in `examples/realtime/main.jl` claiming otherwise is stale. So the efficient shape is
one process per parameter point with `BLAS.set_num_threads(1)` (the default), packed densely
into one allocation by disBatch. Do not ask for many threads per task.

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

# 2. submit — `-n` is the concurrent task count, `-t` must cover the SLOWEST SINGLE task
sbatch -n 8 -t 30:00 -p ccq scripts/hubbard_quench/slurm/disbatch.sbatch \
       scripts/hubbard_quench/tryout.disbatch

# 3. aggregate
julia --project=scripts/hubbard_quench scripts/hubbard_quench/aggregate.jl \
      --outroot scripts/hubbard_quench/results/tryout
```

Use **absolute** paths for `--outfile/--outroot/--logdir`: the task file's `#DISBATCH PREFIX`
cds to the repo root, so relative paths happen to work, but only by coincidence.

Two things the sbatch script does deliberately:

- **`bash -l`** — a login shell, so juliaup's `julia` is on `PATH` and `module` works on the
  compute node.
- **A serial `Pkg.precompile()` before disBatch fans out.** Without it every task precompiles
  the same ~50 packages simultaneously, contending for the same cache files under
  `~/.julia/compiled` on GPFS and wasting minutes of allocation each. It also absorbs the case
  where the compute nodes' CPU differs from whatever last precompiled, which silently
  invalidates the cache. Consider `-C <arch>` to keep the cache warm across submissions.

Sizing, measured on hex 4×4 (32 sites, χ=16, U(1)×U(1)) on a busy workstation: **2.8 s/step**,
of which BP is 2.1 s. Add ~90 s of first-call JIT plus ~40 s for the first measurement, so a
50-step task is ~5 min. Per-task JIT is the reason not to sweep very many very short runs — a
sysimage would pay for itself there.

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

### The charge bath is a real extra vertex

Particle-U(1) at half filling has total charge `Q = N ≠ 0`, so `product_state` attaches an
auxiliary charge-bath vertex that appears in `vertices(state)`, `length(state)` and
`edges(state)`. Spin-U(1) with a balanced AFM has `Sz = 0` and needs none — the two U(1)s
behave differently. Everything here iterates the module's own `lat.verts` / `lat.edges`; if
you extend it, do the same. `params.toml` records `charge_bath`.

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
