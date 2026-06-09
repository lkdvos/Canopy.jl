# Shared state-construction helpers for the BP benchmark suite.
#
# Assumes `Canopy`, `TensorKit`, `Graphs`, `Dictionaries`, `Random`, and
# `AlgorithmsInterface as AI` are already brought into scope by the caller
# (see `benchmarks.jl`).

using Canopy: TensorNetworkState, BPMessages, belief_propagation, UndirectedEdge
using Canopy: SynchronousSchedule, SpanningTreeSchedule, ResidualSchedule, GreedySampler
using TensorKit: ComplexSpace
using Graphs: cycle_graph, grid, edges, src, dst
using Dictionaries: Dictionary
using Random

Random.seed!(0)

function ring_state(L::Int, Dmax::Int; T::Type = ComplexF64, S::Type = ComplexSpace)
    g = cycle_graph(L)
    pspaces = Dictionary{Int, S}(1:L, [ComplexSpace(2) for _ in 1:L])
    ekeys = [UndirectedEdge(src(e), dst(e)) for e in edges(g)]
    vspaces = Dictionary{UndirectedEdge{Int}, S}(ekeys, [ComplexSpace(Dmax) for _ in ekeys])
    st = TensorNetworkState{T}(undef, pspaces, vspaces)
    Random.randn!(st)
    return st
end

function square_state(n::Int, m::Int, Dmax::Int; T::Type = ComplexF64)
    g = grid([n, m])
    st = TensorNetworkState{T}(undef, g, ComplexSpace(2), ComplexSpace(Dmax))
    Random.randn!(st)
    return st
end

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
function bp_kernel_setup(state; schedule = SynchronousSchedule())
    problem = Canopy.BPProblem(state)
    alg = Canopy.BeliefPropagation(AI.StopAfterIteration(1); schedule = schedule)
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
