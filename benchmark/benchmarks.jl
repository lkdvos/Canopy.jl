# Belief-propagation benchmark suite for Canopy.
#
# Always run single-threaded, with BLAS pinned (`setup.jl` pins BLAS itself, but
# the env var also covers whatever runs before it is loaded):
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=benchmark -e '
#       include("benchmark/benchmarks.jl")
#       results = run(SUITE; verbose = true)
#       display(median(results))'
#
# `CANOPY_BENCH_FULL=1` enables the expensive tail of every χ grid.
#
# Record a baseline. `tune!` + a saved `params.json` matters: without
# `loadparams!`, `run` re-tunes `evals` per key on every invocation and injects
# variance between the two runs you are trying to compare.
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=benchmark -e '
#       include("benchmark/benchmarks.jl"); tune!(SUITE)
#       BenchmarkTools.save("benchmark/params.json", params(SUITE))
#       sha = readchomp(`git rev-parse --short=7 HEAD`)
#       BenchmarkTools.save("benchmark/baselines/main_$(sha)_t1.json", run(SUITE; verbose = true))'
#
# Compare against it — note `judge` needs the *same* `params.json` on both sides:
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=benchmark -e '
#       include("benchmark/benchmarks.jl")
#       loadparams!(SUITE, BenchmarkTools.load("benchmark/params.json")[1], :evals, :samples, :seconds)
#       base = BenchmarkTools.load("benchmark/baselines/main_<sha>_t1.json")[1]
#       new  = run(SUITE; verbose = true)
#       j = judge(median(new), median(base); time_tolerance = 0.05)
#       display(regressions(j)); display(improvements(j))'
#
# Save the full `results`, never `median(results)` — the medians can be taken
# later, the samples cannot be recovered. See `benchmark/baselines/README.md` for
# the (important) constraint on which pairs of runs may legitimately be compared.
#
# Suite groups:
#   SUITE["message"]     — `compute_message` cost, single-edge and vertex-batched
#   SUITE["sweep"]       — single `AI.step!` cost, per update schedule
#   SUITE["convergence"] — `belief_propagation` at fixed `maxiter`, per schedule
#   SUITE["schedule"]    — `belief_propagation` to fixed `tol`, per schedule
#   SUITE["allocator"]   — default vs Bumper allocator (time & allocations)
#
# Scalar (non-timing) reports live outside `SUITE`, since BenchmarkTools cannot
# record them. Run them explicitly:
#   benchmark/report_structure.jl  — the symmetry-block census
#   benchmark/report_schedules.jl  — (iters, converged, time-to-tol) per schedule
#   benchmark/profile_kernel.jl    — pprof profiles of the message kernel

using BenchmarkTools
using Canopy
import AlgorithmsInterface as AI

include("setup.jl")

# Provenance. Every saved result must be self-describing: a `judge` comparison is
# only meaningful between runs that agree on all of these.
function _bench_gitsha()
    try
        return readchomp(
            Cmd(`git rev-parse --short=7 HEAD`; dir = dirname(@__DIR__))
        )
    catch
        return "unknown"
    end
end

# 1/5/15-minute load average. This is a **shared workstation**: co-tenant jobs
# contend for memory bandwidth and last-level cache even when cores are free, and
# that shows up directly in the timings. Record it with every run — a `judge`
# comparison between two runs taken at very different loads is not trustworthy,
# and it is the first thing to check when a "regression" appears in a group that
# the change cannot have touched.
_bench_loadavg() = try
    # /proc/loadavg is "1min 5min 15min running/total lastpid"; keep the three averages.
    join(split(read("/proc/loadavg", String))[1:3], " ")
catch
    "unknown"
end

let
    println("""
    ── Canopy benchmark suite ───────────────────────────────────────────────
      git sha           : $(_bench_gitsha())
      julia             : $(VERSION)
      Threads.nthreads  : $(Threads.nthreads())
      BLAS.get_num_threads: $(BLAS.get_num_threads())
      BLAS config       : $(BLAS.get_config())
      Sys.CPU_NAME      : $(Sys.CPU_NAME)  ($(Sys.CPU_THREADS) logical CPUs)
      hostname          : $(gethostname())
      loadavg (1/5/15)  : $(_bench_loadavg())
      CANOPY_BENCH_FULL : $(BENCH_FULL)
    ─────────────────────────────────────────────────────────────────────────""")
end

const SUITE = BenchmarkGroup()
SUITE["message"] = BenchmarkGroup()
SUITE["sweep"] = BenchmarkGroup()
SUITE["convergence"] = BenchmarkGroup()
SUITE["schedule"] = BenchmarkGroup()
SUITE["allocator"] = BenchmarkGroup()

include("bench_message.jl")
include("bench_sweep.jl")
include("bench_convergence.jl")
include("bench_schedule.jl")
include("bench_allocator.jl")
