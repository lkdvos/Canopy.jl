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
- `data/` — output CSVs. `figs/` — output figures.

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
