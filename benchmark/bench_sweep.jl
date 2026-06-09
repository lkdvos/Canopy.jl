# Single-sweep BP benchmarks.
#
# Times one `AI.step!` — one BP iteration under each message-update schedule —
# on a warmed state. `evals=1` is required because `step!` mutates `bp_state`;
# multiple evaluations per sample would accumulate iterations and distort the
# timing. The setup runs one `step!` first so the measured step reflects the
# steady-state per-iteration cost (in particular, the residual schedule seeds
# its residuals with a synchronous sweep on its very first step).

for (topo, statefn) in (
        ("ring_L16_D8", () -> ring_state(16, 8)),
        ("ring_L32_D8", () -> ring_state(32, 8)),
        ("square_3x3_D4", () -> square_state(3, 3, 4)),
    )
    for (tag, sched) in BENCH_SCHEDULES
        SUITE["sweep"][topo, tag] = @benchmarkable(
            AI.step!(problem, alg, bp_state),
            setup = (
                (problem, alg, bp_state) = bp_kernel_setup($statefn(); schedule = $sched);
                AI.step!(problem, alg, bp_state)
            ),
            evals = 1,
        )
    end
end
