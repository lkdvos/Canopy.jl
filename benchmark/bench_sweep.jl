# Single-sweep BP benchmarks.
#
# Times one `AI.step!` — a full BP iteration over every directed edge —
# on a freshly-initialised state each sample. `evals=1` is required
# because `step!` mutates `bp_state`; multiple evaluations per sample
# would accumulate iterations and distort the timing.

SUITE["sweep"]["ring_L16_D8"] = @benchmarkable(
    AI.step!(problem, alg, bp_state),
    setup = ((problem, alg, bp_state) = bp_kernel_setup(ring_state(16, 8))),
    evals = 1,
)

SUITE["sweep"]["ring_L32_D8"] = @benchmarkable(
    AI.step!(problem, alg, bp_state),
    setup = ((problem, alg, bp_state) = bp_kernel_setup(ring_state(32, 8))),
    evals = 1,
)

SUITE["sweep"]["square_3x3_D4"] = @benchmarkable(
    AI.step!(problem, alg, bp_state),
    setup = ((problem, alg, bp_state) = bp_kernel_setup(square_state(3, 3, 4))),
    evals = 1,
)
