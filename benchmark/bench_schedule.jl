# BP schedule comparison benchmarks.
#
# Unlike the `convergence` group (fixed `maxiter`, `tol = 0`), this group times
# `belief_propagation` to a fixed *tolerance*: with `tol = SCHED_TOL` the solve
# stops as soon as `max(residuals) < tol`, so the wall-time captures the
# convergence advantage (or cost) of each message-update schedule on loopy
# networks, where the schedules actually differ. `maxiter` is a high cap.
#
# To compare *iteration counts* to tolerance (the other half of the story),
# `iterations_to_tol` drives the schedule manually and returns the number of
# iterations needed. For an ad-hoc report:
#
#   include("benchmark/benchmarks.jl")
#   for (tag, sched) in BENCH_SCHEDULES
#       k = iterations_to_tol(ring_state(32, 8), sched; tol = SCHED_TOL)
#       println(tag, " => ", k, " iterations")
#   end

const SCHED_TOL = 1.0e-8
const SCHED_MAXITER = 5000

# Number of iterations a schedule needs to drive `max(residuals)` below `tol`,
# or `maxiter` if it does not converge in time. Mirrors the `AI.solve` loop but
# checks the residuals directly so it works for any schedule.
function iterations_to_tol(state, sched; tol = SCHED_TOL, maxiter = SCHED_MAXITER)
    problem = Canopy.BPProblem(state)
    alg = Canopy.BeliefPropagation(AI.StopAfterIteration(maxiter); schedule = sched)
    bp_state = AI.initialize_state(problem, alg; messages = BPMessages(state))
    for k in 1:maxiter
        AI.step!(problem, alg, bp_state)
        maximum(values(bp_state.residuals)) < tol && return k
    end
    return maxiter
end

for (topo, statefn) in (
        ("ring_L16_D4", () -> ring_state(16, 4)),
        ("ring_L32_D8", () -> ring_state(32, 8)),
        ("square_3x3_D4", () -> square_state(3, 3, 4)),
    )
    for (tag, sched) in BENCH_SCHEDULES
        SUITE["schedule"][topo, tag] = @benchmarkable(
            belief_propagation(
                msgs, state; maxiter = SCHED_MAXITER, tol = SCHED_TOL, schedule = $sched,
            ),
            setup = (state = $statefn(); msgs = BPMessages(state)),
            evals = 1, samples = 10, seconds = 30,
        )
    end
end
