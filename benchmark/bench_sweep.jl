# Single-sweep BP benchmarks.
#
# Times one `AI.step!` — one BP iteration under each message-update schedule —
# on a warmed state. `evals=1` is required because `step!` mutates `bp_state`;
# multiple evaluations per sample would accumulate iterations and distort the
# timing. The setup runs one `step!` first so the measured step reflects the
# steady-state per-iteration cost (in particular, the residual schedule seeds
# its residuals with a synchronous sweep on its very first step).
#
# Note this is *per-sweep* cost, not time-to-solution: a schedule doing more
# directed updates per sweep regresses here even when it converges in fewer
# sweeps. `SUITE["schedule"]` is the metric that captures the trade-off.

# χ = 32 on the honeycomb fixture is gated: four schedules × two symmetries is
# the most expensive block of this group.
const SWEEP_HEX_CHIS = BENCH_FULL ? (8, 32) : (8,)

_sweep_topologies() = (
    ("ring_L16_D8", () -> ring_state(16, 8)),
    ("ring_L32_D8", () -> ring_state(32, 8)),
    ("square_3x3_D4", () -> square_state(3, 3, 4)),
    (
        ("hex_$(sym)_chi$(χ)", () -> hex_state(sym, χ))
            for sym in (:trivial, :fz2_u1), χ in SWEEP_HEX_CHIS
    )...,
)

for (topo, statefn) in _sweep_topologies()
    for (tag, schedfn) in BENCH_SCHEDULES
        SUITE["sweep"][topo, tag] = @benchmarkable(
            AI.step!(problem, alg, bp_state),
            # The schedule is built fresh per sample (see `BENCH_SCHEDULES` in
            # `setup.jl`): a shared RNG-carrying instance made consecutive samples
            # of the same key take different spanning trees.
            setup = (
                Random.seed!($BENCH_SEED);
                (problem, alg, bp_state) = bp_kernel_setup($statefn(); schedule = $schedfn());
                AI.step!(problem, alg, bp_state)
            ),
            evals = 1,
        )
    end
end
