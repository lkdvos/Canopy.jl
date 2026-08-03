# Per-schedule convergence report: (iters, converged, time_to_tol, time_per_sweep).
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=benchmark benchmark/report_schedules.jl
#
# BenchmarkTools cannot record scalars, so the iteration count — the other half of
# the schedule story — needs its own script. It never runs as part of
# `run(SUITE)`. Output goes to `benchmark/reports/schedules.{csv,md}`, committed.
#
# HOW TO READ IT
# -------------
# `SUITE["schedule"]` measures time-to-tol only. That number alone cannot
# distinguish "this schedule converges in fewer sweeps" from "this schedule made
# each sweep cheaper by doing less useful work", and the second is a regression
# dressed as a win. Read the two columns together:
#
#   time_to_tol ↓ and iters ↓   — a genuine convergence win
#   time_to_tol ↓ and iters ↑↑  — red flag: cheaper sweeps, weaker schedule
#   time_to_tol ↑ and iters ↓   — the per-sweep cost went up; check whether the
#                                 iteration saving pays for it (that is the whole
#                                 argument for a vertex-batched tree schedule)
#
# `converged = false` means the schedule did not reach `SCHED_TOL` within
# `REPORT_MAXITER` iterations. For those rows `iters` is a cap artefact and no
# timing is taken at all — it would just measure the cap — so they print as `—`.
# Note the pre-phase-0 `iterations_to_tol` returned `maxiter` on failure, which was
# indistinguishable from converging on exactly the last iteration; hence the
# `(iters, converged)` pair (see `setup.jl`).
#
# `REPORT_MAXITER` tracks `SCHED_MAXITER` so a `converged = NO` here means exactly
# that the `SUITE["schedule"]` key of the same name is a `maxiter × sweep` cap
# artefact. Note the cap is what the deliberately non-convergent rows cost: the
# periodic-`:hex` / `:fz2_u1` rows each burn the full cap by design.
#
# Timings are wall-clock minima over `NREPS` repetitions, not BenchmarkTools
# samples: a full solve is far too long to sample properly, and the point of this
# table is the iteration counts.

using Canopy
import AlgorithmsInterface as AI
using TensorKit
using Printf

include("setup.jl")

const REPORT_DIR = joinpath(@__DIR__, "reports")
mkpath(REPORT_DIR)

const NREPS = 2
const REPORT_MAXITER = SCHED_MAXITER

# Minimum wall-clock over `n` repetitions. `warmup` runs `f` once, discarded, so
# JIT compilation stays out of the reported number; skip it when a previous call
# has already compiled the same path.
function besttime(f; n::Int = NREPS, warmup::Bool = true)
    warmup && f()
    return minimum(_ -> (@elapsed f()), 1:n)
end

# `(geometry, symmetry, χ grid)`. Geometries come from `BENCH_GEOMETRIES`, so this
# table lines up row-for-row with `report_structure.jl`'s census — the census is
# what explains the timings. `:square` is degree 4, where a χ = 32 on-site tensor
# is 33 MB and a to-tolerance solve is minutes; capped at χ = 8.
#
# Both honeycomb cells are reported on purpose. `:hex` is fully periodic and is
# the fixture `SUITE["message"]` / `SUITE["sweep"]` use; its `:fz2_u1` rows are
# *expected* to come back `converged = NO`, and having that recorded is what
# justifies `SUITE["schedule"]` using `:hex_open` instead.
const REPORT_CASES = (
    (:ring, :trivial, (8, 32)),
    (:ring, :fz2_u1, (8, 32)),
    (:hex, :trivial, (8, 32)),
    (:hex, :fz2_u1, (8, 32)),        # periodic: expected NOT to converge — that is the point
    (:hex_open, :trivial, (8, 32)),
    (:hex_open, :fz2_u1, (8, 32)),   # the fixture `SUITE["schedule"]` actually uses
    (:square, :trivial, (8,)),
    (:square, :fz2_u1, (8,)),
)

struct SchedRow
    geom::Symbol
    sym::Symbol
    chi::Int
    sched::Symbol
    iters::Int
    converged::Bool
    time_to_tol::Float64      # NaN when !converged
    time_per_sweep::Float64
end

# `schedfn` is a *factory*: `SpanningTreeSchedule` carries a mutable RNG, so every
# measurement below builds its own instance and therefore starts from the same RNG
# state. Reusing one instance would make `iters` depend on how many times the
# schedule had been run before. See `BENCH_SCHEDULES` in `setup.jl`.
function sched_row(geom::Symbol, sym::Symbol, χ::Int, tag::Symbol, schedfn)
    # Seeded, and seeded with the *same* `BENCH_SEED` the `SUITE` setup blocks use.
    # `randn_state` draws from the global RNG, so without this each of the three
    # measurements below would solve a different instance, and the `iters` reported
    # here would not correspond to the `SUITE["schedule"]` key of the same name.
    statefn() = (Random.seed!(BENCH_SEED); first(bench_state(geom, sym, χ)))

    r = iterations_to_tol(statefn(), schedfn(); maxiter = REPORT_MAXITER)

    # Per-sweep cost, on a state warmed one step so the residual-driven schedules
    # are past their seeding sweep. Measured even when the solve did not converge:
    # it is a per-iteration cost, not a cap artefact.
    problem, alg, bp_state = bp_kernel_setup(statefn(); schedule = schedfn())
    AI.step!(problem, alg, bp_state)
    tsweep = besttime(; warmup = false) do
        AI.step!(problem, alg, bp_state)
    end

    ttol = if r.converged
        besttime(; warmup = false) do
            state = statefn()
            belief_propagation(
                BPMessages(state), state;
                maxiter = REPORT_MAXITER, tol = SCHED_TOL, schedule = schedfn(),
            )
        end
    else
        NaN
    end

    return SchedRow(geom, sym, χ, tag, r.iters, r.converged, ttol, tsweep)
end

const HEADER = (
    "geom", "sym", "chi", "schedule", "iters", "converged",
    "time_to_tol_s", "time_per_sweep_s",
)

_csv_cells(r::SchedRow) = (
    string(r.geom), string(r.sym), string(r.chi), string(r.sched),
    string(r.iters), string(r.converged),
    isnan(r.time_to_tol) ? "" : @sprintf("%.6f", r.time_to_tol),
    @sprintf("%.6f", r.time_per_sweep),
)

_md_cells(r::SchedRow) = (
    string(r.geom), string(r.sym), string(r.chi), string(r.sched),
    r.converged ? string(r.iters) : "—",
    r.converged ? "yes" : "**NO**",
    isnan(r.time_to_tol) ? "—" : @sprintf("%.3f", r.time_to_tol),
    @sprintf("%.4f", r.time_per_sweep),
)

function markdown_table(rows)
    cells = map(_md_cells, rows)
    w = [maximum(length, (HEADER[i], (c[i] for c in cells)...)) for i in eachindex(HEADER)]
    io = IOBuffer()
    println(io, "| ", join((rpad(HEADER[i], w[i]) for i in eachindex(HEADER)), " | "), " |")
    println(io, "|", join(("-"^(w[i] + 2) for i in eachindex(HEADER)), "|"), "|")
    for c in cells
        println(io, "| ", join((rpad(c[i], w[i]) for i in eachindex(c)), " | "), " |")
    end
    return String(take!(io))
end

function _bench_gitsha()
    try
        return readchomp(Cmd(`git rev-parse --short=7 HEAD`; dir = dirname(@__DIR__)))
    catch
        return "unknown"
    end
end

function main()
    rows = SchedRow[]
    for (geom, sym, chis) in REPORT_CASES, χ in chis, (tag, schedfn) in BENCH_SCHEDULES
        @info "report_schedules: $geom / $sym / χ=$χ / $tag"
        push!(rows, sched_row(geom, sym, χ, tag, schedfn))
    end

    csv = joinpath(REPORT_DIR, "schedules.csv")
    open(csv, "w") do io
        println(io, join(HEADER, ","))
        for r in rows
            println(io, join(_csv_cells(r), ","))
        end
    end

    failed = filter(r -> !r.converged, rows)

    md = joinpath(REPORT_DIR, "schedules.md")
    open(md, "w") do io
        println(io, "# BP schedule convergence report")
        println(io)
        println(io, "Generated by `benchmark/report_schedules.jl`; see that file's header for how")
        println(io, "to read the `iters` / `time_to_tol` pair.")
        println(io)
        println(io, "- host: `$(gethostname())`  ·  CPU: `$(Sys.CPU_NAME)`  ·  julia `$(VERSION)`")
        println(io, "- `Threads.nthreads() = $(Threads.nthreads())`, `BLAS.get_num_threads() = $(BLAS.get_num_threads())`")
        println(io, "- git sha: `$(_bench_gitsha())`")
        println(io, "- `SCHED_TOL = $(SCHED_TOL)`, `REPORT_MAXITER = $(REPORT_MAXITER)`" *
            (REPORT_MAXITER == SCHED_MAXITER ? " (same cap as `SUITE[\"schedule\"]`)" :
             " vs `SCHED_MAXITER = $(SCHED_MAXITER)` in `SUITE[\"schedule\"]`"))
        println(io, "  — `converged = NO` means \"not within $(REPORT_MAXITER) iterations\", not \"diverged\"")
        println(io, "- states are built at `Random.seed!(BENCH_SEED = $(BENCH_SEED))`, the same instance")
        println(io, "  `SUITE` measures, so `iters` here explains the `SUITE[\"schedule\"]` key of the same name")
        println(io, "- timings: minimum of $NREPS repetitions, single-threaded")
        println(io)
        println(io, "**Timings are machine-specific.** Only compare rows within one generated file,")
        println(io, "or two files generated back-to-back on the same host. The `iters` column is")
        println(io, "machine-independent and is the column later phases are gated on.")
        println(io)
        if isempty(failed)
            println(io, "All $(length(rows)) `(geom, sym, χ, schedule)` combinations converged.")
        else
            println(io, "## Did not converge within `REPORT_MAXITER`")
            println(io)
            for r in failed
                println(io, "- `$(r.geom)` / `$(r.sym)` / χ=$(r.chi) / `$(r.sched)`")
            end
            println(io)
            println(io, "For these rows `iters` and `time_to_tol` are cap artefacts and are shown")
            println(io, "as `—`; `time_per_sweep_s` is still a valid per-iteration cost.")
        end
        println(io)
        println(io, "## Results")
        println(io)
        print(io, markdown_table(rows))
    end

    print(markdown_table(rows))
    if !isempty(failed)
        println("\nNOT CONVERGED:")
        for r in failed
            println("  $(r.geom) / $(r.sym) / χ=$(r.chi) / $(r.sched)")
        end
    end
    println("\nwrote $csv\nwrote $md")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
