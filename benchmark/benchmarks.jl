# Belief-propagation benchmark suite for Canopy.
#
# Run the full suite:
#   julia --project=benchmark -e '
#       include("benchmark/benchmarks.jl");
#       results = run(SUITE; verbose = true);
#       display(median(results))'
#
# Save / load / compare a baseline:
#   BenchmarkTools.save("baseline.json", median(results))
#   baseline = BenchmarkTools.load("baseline.json")[1]
#   judge(median(run(SUITE)), baseline)
#
# Suite groups:
#   SUITE["message"]     — single-edge `compute_message` cost
#   SUITE["sweep"]       — single `AI.step!` cost, per update schedule
#   SUITE["convergence"] — `belief_propagation` at fixed `maxiter`, per schedule
#   SUITE["schedule"]    — `belief_propagation` to fixed `tol`, per schedule
#   SUITE["allocator"]   — default vs Bumper allocator (time & allocations)

using BenchmarkTools
using Canopy
import AlgorithmsInterface as AI

include("setup.jl")

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
