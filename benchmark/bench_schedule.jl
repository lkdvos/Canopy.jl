# BP schedule comparison benchmarks.
#
# Unlike the `convergence` group (fixed `maxiter`, `tol = 0`), this group times
# `belief_propagation` to a fixed *tolerance*: with `tol = SCHED_TOL` the solve
# stops as soon as `max(residuals) < tol`, so the wall-time captures the
# convergence advantage (or cost) of each message-update schedule on loopy
# networks, where the schedules actually differ. `maxiter` is a high cap.
#
# This group is the **decision metric** for the vertex-batched spanning-tree
# schedule: a schedule that performs more directed updates per sweep is allowed to
# regress `SUITE["sweep"]` as long as it wins here. Always read it next to
# `benchmark/report_schedules.jl`, which reports iteration counts alongside the
# times — time-to-tol improving while the iteration count rises sharply means the
# schedule got cheaper by getting weaker, not better.
#
# `SCHED_TOL`, `SCHED_MAXITER` and `iterations_to_tol` live in `setup.jl` so the
# reporter can use them without pulling in `SUITE`. For an ad-hoc iteration count:
#
#   include("benchmark/benchmarks.jl")
#   for (tag, schedfn) in BENCH_SCHEDULES      # factories, not instances — see setup.jl
#       r = iterations_to_tol(ring_state(32, 8), schedfn())
#       println(tag, " => ", r.iters, r.converged ? " iterations" : " (NOT converged)")
#   end

# The honeycomb entries are the primary fixture (coordination 3, matching
# production), via `hex_sched_state` — the *open* cell. The fully periodic cell
# does not converge to `SCHED_TOL` at all for `:fz2_u1`; see the long comment on
# `hex_sched_state` in `setup.jl`. Using it here would turn every graded entry in
# this group into a `maxiter × sweep` cap artefact, which is both uninformative
# and the most expensive thing the suite can do.
#
# χ = 32 is gated: with four schedules × two symmetries it is the single most
# expensive block of the suite, and at that χ only `:tree` and `:splash` converge
# even on the open cell (`benchmark/reports/schedules.md` records which).
const SCHED_HEX_CHIS = BENCH_FULL ? (8, 32) : (8,)

_sched_topologies() = (
    ("ring_L16_D4", () -> ring_state(16, 4)),
    ("ring_L32_D8", () -> ring_state(32, 8)),
    ("square_3x3_D4", () -> square_state(3, 3, 4)),
    (
        ("hex_$(sym)_chi$(χ)", () -> hex_sched_state(sym, χ))
            for sym in (:trivial, :fz2_u1), χ in SCHED_HEX_CHIS
    )...,
)

for (topo, statefn) in _sched_topologies()
    for (tag, schedfn) in BENCH_SCHEDULES
        SUITE["schedule"][topo, tag] = @benchmarkable(
            belief_propagation(
                msgs, state; maxiter = SCHED_MAXITER, tol = SCHED_TOL, schedule = sched,
            ),
            # `sched` is constructed per sample, not shared: see the note on
            # `BENCH_SCHEDULES` in `setup.jl`. Sharing one RNG-carrying instance
            # put a ±55% floor under this group.
            setup = (
                Random.seed!($BENCH_SEED);
                state = $statefn(); msgs = BPMessages(state); sched = $schedfn()
            ),
            # `seconds` is the *total* budget per entry, so it doubles as the cap a
            # non-converging (schedule, fixture) pair burns. Kept tight because the
            # group has `length(BENCH_SCHEDULES)` entries per topology.
            evals = 1, samples = 10, seconds = 12,
        )
    end
end
