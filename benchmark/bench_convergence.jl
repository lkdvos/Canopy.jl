# BP convergence benchmarks.
#
# Times `belief_propagation` with fixed `maxiter` and `tol = 0`, so
# timing is purely `N × sweep_cost` and reflects neither
# iteration-count noise nor convergence-quality changes — those should be
# tracked separately. Each entry caps `samples`/`seconds` so a full SUITE
# run stays in O(minutes).

using Canopy: belief_propagation, BPMessages

SUITE["convergence"]["ring_L16_D4"] = @benchmarkable(
    belief_propagation(msgs, state; maxiter = 100, tol = 0),
    setup = (state = ring_state(16, 4); msgs = BPMessages(state)),
    evals = 1, samples = 10, seconds = 30,
)

SUITE["convergence"]["ring_L32_D8"] = @benchmarkable(
    belief_propagation(msgs, state; maxiter = 100, tol = 0),
    setup = (state = ring_state(32, 8); msgs = BPMessages(state)),
    evals = 1, samples = 10, seconds = 30,
)

SUITE["convergence"]["square_3x3_D4"] = @benchmarkable(
    belief_propagation(msgs, state; maxiter = 50, tol = 0),
    setup = (state = square_state(3, 3, 4); msgs = BPMessages(state)),
    evals = 1, samples = 10, seconds = 30,
)
