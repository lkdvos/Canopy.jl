# Shared state-construction helpers for the BP benchmark suite.
#
# Assumes `Canopy`, `TensorKit`, `Graphs`, `Random`, and
# `AlgorithmsInterface as AI` are already brought into scope by the caller
# (see `benchmarks.jl`).

using Canopy: randn_state, BPMessages, belief_propagation
using Canopy: SynchronousSchedule, SpanningTreeSchedule, ResidualSchedule, GreedySampler
using TensorKit: ComplexSpace
using TensorKit: TO
using Graphs: cycle_graph, grid
using Random
import Bumper

Random.seed!(0)

# Allocators compared by the allocator benchmark group: plain heap allocation
# versus the production Bumper `ResizeBuffer`, which warms up to the peak
# intermediate size and reclaims temporaries across repeated contractions.
const BENCH_ALLOCATORS = (
    :default => TO.DefaultAllocator(),
    :bumper => Bumper.default_buffer(Bumper.ResizeBuffer),
)

ring_state(L::Int, Dmax::Int; T::Type = ComplexF64) =
    randn_state(T, cycle_graph(L), ComplexSpace(2), ComplexSpace(Dmax))

square_state(n::Int, m::Int, Dmax::Int; T::Type = ComplexF64) =
    randn_state(T, grid([n, m]), ComplexSpace(2), ComplexSpace(Dmax))

# Run a few BP iterations so kernel benchmarks measure cost on
# typical-shape messages rather than identity-initialised ones.
function warm_messages(state; maxiter::Int = 20)
    msgs = BPMessages(state)
    return belief_propagation(msgs, state; maxiter = maxiter, tol = 0)
end

# Build the `(problem, alg, bp_state)` triple needed to call `AI.step!`
# directly, avoiding `belief_propagation`'s solve scaffolding inside the
# timing loop. Stopping criterion is `StopAfterIteration(1)` so a single
# `step!` represents one full sweep. `schedule` selects the message-update
# order benchmarked.
function bp_kernel_setup(state; schedule = SynchronousSchedule(), allocator = Bumper.default_buffer(Bumper.ResizeBuffer))
    problem = Canopy.BPProblem(state)
    alg = Canopy.BeliefPropagation(AI.StopAfterIteration(1); schedule = schedule, allocator = allocator)
    bp_state = AI.initialize_state(problem, alg; messages = BPMessages(state))
    return problem, alg, bp_state
end

# The message-update schedules compared across the suite. Random schedules use a
# fixed-seed RNG so benchmark runs are reproducible. `ndirected(state)` is the
# total number of directed edges = a full-sweep batch size for the residual
# schedule.
ndirected(state) = length(BPMessages(state).messages)
const BENCH_SCHEDULES = (
    :sync => SynchronousSchedule(),
    :tree => SpanningTreeSchedule(; rng = MersenneTwister(0)),
    :residual => ResidualSchedule(GreedySampler(8)),
)
