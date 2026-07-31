"""
    SpanningTreeSchedule(; rng = Random.Xoshiro(0))

Default schedule. Each iteration draws a fresh random BFS vertex order
([`Canopy.random_bfs_order`](@ref)) and walks it twice — once inward
(reverse order) and once outward (order) — recomputing **all** of a vertex's
outgoing messages in one vertex-centric batch per visit. On a tree that is exact
in a single iteration; on a loopy network the loop-closing edges are covered
automatically, since the outward pass writes every directed edge.

`residuals[e]` is the change over the *whole* iteration (the same quantity
[`SynchronousSchedule`](@ref) records), measured against a start-of-iteration
snapshot rather than against the intermediate inward value, so `tol` means the
same thing under either schedule.

Messages are updated **in place**: the [`BPMessages`](@ref) returned by
[`belief_propagation`](@ref) aliases the one passed in.

# Performance trade-off

An iteration performs `3|E|` directed message updates against `2|E|` for a
per-edge schedule, but shares each vertex's absorptions across its whole outgoing
batch, so the cost *per update* falls roughly like `d` for coordination `d`. The
two effects cut in opposite directions and the crossover is around `d = 3`:

  * **high coordination — a large win.** On a graded honeycomb (`d = 3`,
    `fℤ₂ ⊠ U1Irrep`, χ = 32) this needs ~4× fewer iterations than
    [`SynchronousSchedule`](@ref) *and* a ~2.5× cheaper sweep, for a ~3× lower
    time-to-tolerance than the per-edge spanning-tree schedule it replaced.
  * **low coordination — a modest loss.** On degree-2 rings and chains a
    spanning-tree sweep already converged in a handful of iterations, so there is
    no iteration count left to trade away and the ~1.5× heavier sweep shows up
    directly: expect ~1.3× the time-to-tolerance. The absolute cost is small
    (a few iterations over small tensors), which is why this is still the default.

Pass `schedule = SynchronousSchedule()` if you are working exclusively with
low-coordination lattices and want the older cost profile back.

The default `rng` is seeded, so `belief_propagation` is reproducible and does not
perturb the global RNG stream. Pass `rng = Random.default_rng()` to opt into
globally-seeded randomization instead. Note a schedule object holds *mutable* RNG
state, so a single instance is neither thread-safe nor reusable across
measurements that need identical orders.
"""
struct SpanningTreeSchedule{RNG} <: BPSchedule
    rng::RNG
end
SpanningTreeSchedule(; rng = Random.Xoshiro(0)) = SpanningTreeSchedule(rng)

# A deep snapshot of the messages at the start of each iteration, used as the
# residual reference by the outward pass. The `copy` is required: the outward pass
# consumes `last_used[e]` destructively, which would otherwise destroy the
# caller's own tensors on the very first iteration.
init_last_used(::SpanningTreeSchedule, messages) = copy(messages)

# Two vertex-batched passes over a fresh random BFS order.
#
# Exactness: the batch computes every output of `v` from one snapshot of `v`'s
# incoming messages, and a message *out of* `v` is never an input *at* `v`, so a
# batch is identical to a per-edge loop over `v`'s outgoing edges in any order.
# Then, inward, when `v` is visited every `child → v` is already exact, so
# `v → parent(v)` is exact; outward, `parent → v` was written earlier in the same
# pass and every `child → v` still holds its inward value, so all of `v`'s
# outgoing messages are exact.
#
# WHY THE INWARD PASS IS PRUNED to `{v → n : pos[n] < pos[v]}` (on a tree: exactly
# `{v → parent(v)}`). The other inward writes are provably *dead*, so dropping
# them is observationally identical and takes the iteration from 4|E| to 3|E|
# directed updates:
#
#   * within the inward pass, `msgs[v → n]` is read only when `n` is visited
#     later in that pass, i.e. only when `pos[n] < pos[v]`;
#   * within the outward pass, `msgs[v → n]` is read when `n` is visited, at
#     `pos[n]`. If `pos[n] > pos[v]` the outward visit of `v` has already
#     overwritten it, so the inward value never reaches a reader;
#   * the outward pass writes every directed edge, so no inward value survives
#     to the end of the iteration either.
#
# Note the *converse* writes (`pos[n] < pos[v]`) are exactly the ones the outward
# pass consumes, which is why the predicate is `<` and not "parent only": on a
# loopy graph a vertex can have several already-visited neighbours.
function update_messages!(
        sched::SpanningTreeSchedule, problem::BPProblem, alg::BeliefPropagation, state::BPState
    )
    network = problem.network
    msgs = state.iterate
    order, pos = random_bfs_order(network, sched.rng)

    # One target buffer for the whole iteration: `outgoing_edges` is a lazy
    # generator while the batch kernel needs an `AbstractVector`.
    targets = DirectedEdge{keytype(network)}[]

    # Inward pass: leaves → roots, only the messages the outward pass will read.
    for v in Iterators.reverse(order)
        empty!(targets)
        pv = pos[v]
        for n in neighbors(network, v)
            pos[n] < pv && push!(targets, DirectedEdge(v, n))
        end
        update_messages_at!(
            msgs, network, targets, alg.backend, alg.allocator; timer = alg.timer,
        )
    end

    # Outward pass: roots → leaves, every outgoing message of every vertex. This
    # is each edge's single residual-recording write of the iteration, and it
    # happens after any inward write to the same edge, so `state.last_used` still
    # holds the start-of-iteration value when it is measured against.
    for v in order
        empty!(targets)
        for n in neighbors(network, v)
            push!(targets, DirectedEdge(v, n))
        end
        update_messages_at!(
            msgs, network, targets, alg.backend, alg.allocator;
            residuals = state.residuals, snapshot = state.last_used, timer = alg.timer,
        )
    end
    return state
end
