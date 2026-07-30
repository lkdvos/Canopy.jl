# BP convergence benchmarks.
#
# Times `belief_propagation` with fixed `maxiter` and `tol = 0`, so timing is
# purely `N × sweep_cost` for each message-update schedule and reflects neither
# iteration-count noise nor convergence-quality changes — those are tracked by
# the `schedule` group in `bench_schedule.jl`. Each entry caps `samples` /
# `seconds` so a full SUITE run stays in O(minutes).

using Canopy: belief_propagation, BPMessages

for (topo, statefn, maxiter) in (
        ("ring_L16_D4", () -> ring_state(16, 4), 100),
        ("ring_L32_D8", () -> ring_state(32, 8), 100),
        ("square_3x3_D4", () -> square_state(3, 3, 4), 50),
    )
    for (tag, schedfn) in BENCH_SCHEDULES
        SUITE["convergence"][topo, tag] = @benchmarkable(
            belief_propagation(msgs, state; maxiter = $maxiter, tol = 0, schedule = sched),
            # `BENCH_SCHEDULES` holds factories rather than instances; the schedule
            # is built per sample so an RNG-carrying one cannot drift between
            # samples. See the note in `setup.jl`.
            setup = (
                Random.seed!($BENCH_SEED);
                state = $statefn(); msgs = BPMessages(state); sched = $schedfn()
            ),
            evals = 1, samples = 10, seconds = 30,
        )
    end
end
