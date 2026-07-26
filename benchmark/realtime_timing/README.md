# Real-time evolution timing benchmark

Per-step timing traces of a real-time quench (belief propagation + bond truncation), for
comparison against other libraries. The circuit mirrors
[`examples/realtime/main.jl`](../../examples/realtime/main.jl): a 48-site honeycomb quench with
a 3-layer Trotter step (single-site → two-site → single-site) followed by a BP reconvergence,
run for a *fixed* number of BP sweeps (`tol = 0`) so the work per step is constant and
comparable across libraries. Four model/symmetry runs are swept (see *Model selection*).

## Layout

- `run_timings.jl` — instrumented runner; writes one CSV per run (`<prefix>_<model>.csv`).
- `run_all.sh` — driver: runs all four models single-threaded, then `plot_timings.jl`.
- `plot_timings.jl` — concatenates the per-run CSVs into `data/combined.csv` and renders figures.
- `profile_runs.jl` — sampling profiler; writes pprof artifacts per χ (see *Profiling* below).
- `INSTRUCTIONS_other_library.md` — spec for reproducing these runs in another library.
- `data/` — output CSVs (per-run + `combined.csv`). `figs/` — figures. `profiles/` — pprof profiles.

Both scripts are self-contained executables (shebang + `Pkg.activate(@__DIR__)`), so the first
run precompiles this folder's environment (`Project.toml`).

## Usage

The full sweep is single-threaded (one Julia thread, one BLAS thread). Run everything with:

```bash
./run_all.sh                                 # canopy, χ = 4 8 16 32 64, 15 Trotter steps
PREFIX=canopy CHIS="4 8 16 32 64 128" NSTEPS=15 ./run_all.sh   # add χ values / change length
```

Or run one model at a time (the count of threads is recorded in every CSV):

```bash
JULIA_NUM_THREADS=1 ./run_timings.jl --prefix canopy --model tfim-z2 --nsteps 15 --chi 4 8 16 32 64
./plot_timings.jl
```

`--help` lists all options for either script. Equivalently, run under an explicit project:

```bash
julia --project=benchmark/realtime_timing benchmark/realtime_timing/run_timings.jl --help
```

### Model selection

Four `--model` values, each a separate run written to `<prefix>_<model>.csv` (with `-` → `_`):

- `free-fermion` (default) — spinless free fermions (fermion-parity `fℤ₂`).
- `free-fermion-u1` — the *same* quench with conserved U(1) particle number (`fℤ₂ ⊠ U1Irrep`), to
  measure symmetry-block bookkeeping. The lattice carries a nontrivial total charge, so
  `product_state(...; total_charge = ...)` anchors a charge-bath site carrying the compensating
  charge by one 1-dimensional bond; it stays idle (no gate) so the timing reflects the U(1)
  overhead on the real lattice bonds.
- `tfim` — a transverse-field Ising model with a *staggered* transverse field
  (`exp(-i·(±h)·dt·σˣ)`, sign alternating by sublattice) and a `σᶻσᶻ` coupling.
- `tfim-z2` — the *same* TFIM with the spin-flip ℤ₂ symmetry (`Z2Irrep`), the spin analogue of the
  free-fermion U(1) run.

Both TFIM variants start from the **same** σˣ-eigenstate product state (`|+⟩` on sublattice A, `|−⟩`
on B) — a ℤ₂ eigenstate, so it is representable under the symmetry and the no-symmetry vs ℤ₂ runs
share identical physics. All four models keep the same 3-layer step structure and CSV schema, so the
`hop` column times the two-site coupling layer (hopping or `σᶻσᶻ`).

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

Each CSV carries `library` and `model` columns; `plot_timings.jl` overlays every run found in
`--datadir`, faceting by `model` with one line per `library`. To compare, have the other library
emit CSVs with the **same schema** and a distinct `library`/`--prefix` into the same directory,
then re-run `plot_timings.jl`. See
[`INSTRUCTIONS_other_library.md`](INSTRUCTIONS_other_library.md) for the full reproduction spec.

## CSV schema (wide, one row per Trotter step, all χ of a run stacked in one file)

```
library, model, chi, nthreads, nsites, dt, step,
single1, hop, single2, bp,
single1_bytes, hop_bytes, single2_bytes, bp_bytes,
maxdim
```

- `library` — run/library label (the `--prefix`). `model` — one of the four model names.
- `chi` — the run's bond-dimension cap; one run sweeps all χ into a single `<prefix>_<model>.csv`.
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

Comparison figures put every model/symmetry on a **shared axis** (colour = model, linestyle =
library, so models overlay directly); the phase breakdown is faceted by model:

- `scaling.svg` — median BP time vs χ (log–log), one line per model/symmetry.
- `allocations.svg` — median heap allocation per step (MiB) vs χ, one line per model/symmetry.
- `timeseries.svg` — per-step time trace at the largest χ **common to all models** (so no series
  is empty for a model that crashed at a larger χ); exposes the bond-growth ramp, the JIT cost of
  step 1, and GC spikes.
- `phase_breakdown.svg` — stacked phase contributions (single / hop / bp) per χ, dodged by library,
  faceted by model.

Headline numbers use the **median over the saturated steps** (where `maxdim` has reached its
plateau), excluding the bond-growth ramp and GC spikes.
