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
#   SUITE["sweep"]       — single `AI.step!` cost (one full BP iteration)
#   SUITE["convergence"] — `belief_propagation` at fixed `maxiter`

using BenchmarkTools
using Canopy
import AlgorithmsInterface as AI

include("setup.jl")

const SUITE = BenchmarkGroup()
SUITE["message"] = BenchmarkGroup()
SUITE["sweep"] = BenchmarkGroup()
SUITE["convergence"] = BenchmarkGroup()

include("bench_message.jl")
include("bench_sweep.jl")
include("bench_convergence.jl")
