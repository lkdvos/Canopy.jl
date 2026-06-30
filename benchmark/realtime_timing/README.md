# Real-time evolution timing benchmark

Per-step timing traces of the real-time free-fermion quench (belief propagation + bond
truncation), for comparison against other libraries. The circuit mirrors
[`examples/realtime/main.jl`](../../examples/realtime/main.jl): a 48-site honeycomb quench with
a 3-layer Trotter step (single-site → hopping → single-site) followed by a BP reconvergence,
run for a *fixed* number of BP sweeps (`tol = 0`) so the work per step is constant and
comparable across libraries.

## Layout

- `run_timings.jl` — instrumented runner; writes one CSV per χ.
- `plot_timings.jl` — reads a directory of CSVs and renders comparison figures.
- `profile_runs.jl` — sampling profiler; writes pprof artifacts per χ (see *Profiling* below).
- `data/` — output CSVs. `figs/` — output figures. `profiles/` — output pprof profiles.

Both scripts are self-contained executables (shebang + `Pkg.activate(@__DIR__)`), so the first
run precompiles this folder's environment (`Project.toml`).

## Usage

Set a fixed thread count — `apply!` threads over gates and the count is recorded in every CSV:

```bash
JULIA_NUM_THREADS=8 ./run_timings.jl --prefix canopy --nsteps 50 --chi 4 8 16 32
./plot_timings.jl
```

`--help` lists all options for either script. Equivalently, run under an explicit project:

```bash
julia --project=benchmark/realtime_timing benchmark/realtime_timing/run_timings.jl --help
```

### Model selection

`--model free-fermion` (default) runs the quench described above. `--model free-fermion-u1` runs
the *same* free-fermion quench but with conserved U(1) particle number (sector `fℤ₂ ⊠ U1Irrep`),
to measure the cost of symmetry-block bookkeeping. Since only the trivial total charge is
representable, a single charge-bath "dummy" site carrying the compensating charge is anchored to
the lattice by one 1-dimensional bond; it stays idle (no gate) so the timing reflects the U(1)
overhead on the real lattice bonds. These runs are written to `<prefix>_u1_chi<χ>.csv`.

`--model tfim` instead runs a transverse-field Ising model with a *staggered* transverse field
(`exp(-i·(±h)·dt·σˣ)`, sign alternating by sublattice) and a uniform `σᶻσᶻ` coupling, starting
from a Néel product state. It keeps the same 3-layer step structure, so the CSV schema and
`plot_timings.jl` are unchanged — the `hop` column then times the `σᶻσᶻ` coupling layer. TFIM runs
are written to a separate file, `<prefix>_tfim_chi<χ>.csv`, so they never collide with the
free-fermion CSVs.

## Profiling

The CSVs say *which phase* is slow; `profile_runs.jl` says *which lines*. It warms up to the
bond-dimension plateau and then samples the profiler over each phase separately, writing
interactive [pprof](https://github.com/google/pprof) artifacts (PProf.jl bundles the viewer, so
nothing external is needed):

```bash
JULIA_NUM_THREADS=8 ./profile_runs.jl --chi 16 32
```

Per χ this writes to `profiles/`:

- `chi<χ>_cpu_bp.pb.gz` — CPU profile of `belief_propagation` only (the dominant phase).
- `chi<χ>_cpu_hop.pb.gz` — CPU profile of the hopping layer only.
- `chi<χ>_cpu_step.pb.gz` — CPU profile of the full Trotter step (single → hop → single → bp).
- `chi<χ>_allocs.pb.gz` — allocation profile of the full step (heap hotspots only; Canopy's
  off-heap Bumper temporaries are invisible to the allocation profiler, as with the CSV `*_bytes`).
- `chi<χ>_meta.txt` — χ, threads, `bp_iters`, repeats, reached `maxdim`.

BP is sampled with `tol = 0` for a fixed `--bp-iters` sweeps, so repeated calls on a converged
state do identical, representative work. View a saved profile in a browser flamegraph with:

```bash
julia --project=benchmark/realtime_timing -e \
  'using PProf; PProf.refresh(file="benchmark/realtime_timing/profiles/chi32_cpu_bp.pb.gz")'
```

`--help` lists all options (`--warmup`, `--repeats`, `--alloc-rate`, …).

### Comparing libraries

Each run is identified by its filename prefix (`<prefix>_chi<χ>.csv`); the plot script recovers
that prefix as the run label and overlays all runs found in `--datadir`. To compare, have the
other library emit CSVs with the **same schema** and a different `--prefix` into the same
directory, then re-run `plot_timings.jl`.

## CSV schema (wide, one row per Trotter step)

```
chi, nthreads, nsites, dt, step,
single1, hop, single2, bp,
single1_bytes, hop_bytes, single2_bytes, bp_bytes,
maxdim
```

- `step` — Trotter step index; row `step = 0` is the initial BP convergence (gate columns `0.0`,
  `bp` = initial-convergence time). Rows `step ≥ 1` are the evolution steps, and `step = 1`
  carries Julia's JIT-compilation cost (no warmup is done — that cost is itself informative).
- `single1, hop, single2, bp` — wall-clock seconds for each phase of the step.
- `single1_bytes, hop_bytes, single2_bytes, bp_bytes` — **heap** bytes allocated per phase
  (from `@timed`). See the caveat below.
- `maxdim` — maximum virtual bond dimension after the step.

### Allocation caveat

Canopy bump-allocates contraction temporaries off-heap (a Bumper `ResizeBuffer`; see
[`src/bumper.jl`](../../src/bumper.jl)), and those bytes are invisible to the GC. The
`*_bytes` columns therefore measure only the **persistent heap allocation** that reaches the
GC each step (new state/message tensors that escape) — *not* the full temporary working set.
Treat them as a GC-pressure metric, not total memory traffic. (On `step = 1` they also include
one-time compilation allocations, like the timings.)

## Figures

- `scaling.svg` — median step time vs χ (log–log), one line per run.
- `phase_breakdown.svg` — stacked phase contributions (single / hop / bp) per χ, dodged by run.
- `timeseries_chi<χ>.svg` — per-step time trace at the largest χ; exposes the bond-growth ramp,
  the JIT cost of step 1, and GC spikes.
- `allocations.svg` — median heap allocation per step (MiB) vs χ, one line per run.

Headline numbers use the **median over the saturated steps** (where `maxdim` has reached its
plateau), excluding the bond-growth ramp and GC spikes.
