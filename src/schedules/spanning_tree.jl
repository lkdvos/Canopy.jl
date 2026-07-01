"""
    SpanningTreeSchedule(; rng = Random.default_rng())

Each iteration sample a random spanning tree of the network, solve it exactly
with a sequential leaves→root then root→leaves sweep (so updates see freshly
computed neighbors), and finally update the remaining loop-closing (cotree)
edges from the just-updated messages. The tree is resampled every iteration.
"""
struct SpanningTreeSchedule{RNG} <: BPSchedule
    rng::RNG
end
SpanningTreeSchedule(; rng = Random.default_rng()) = SpanningTreeSchedule(rng)

# Spanning-tree schedule: exact two-pass sweep along a fresh random spanning
# tree, then update the cotree edges from the just-updated messages.
function update_messages!(
        sched::SpanningTreeSchedule, problem::BPProblem, alg::BeliefPropagation, state::BPState
    )
    network = problem.network
    msgs = state.iterate
    order, parent, cotree = random_spanning_tree(network, sched.rng)
    # inward pass: leaves → root
    for v in Iterators.reverse(order)
        haskey(parent, v) || continue
        e = DirectedEdge(v, parent[v])
        state.residuals[e] = update_message!(msgs, network, e, alg.backend, alg.allocator; timer = alg.timer)
    end
    # outward pass: root → leaves
    for v in order
        haskey(parent, v) || continue
        e = DirectedEdge(parent[v], v)
        state.residuals[e] = update_message!(msgs, network, e, alg.backend, alg.allocator; timer = alg.timer)
    end
    # cotree (loop-closing) edges: both directions
    for ce in cotree
        e_fwd = DirectedEdge(ce)
        state.residuals[e_fwd] = update_message!(msgs, network, e_fwd, alg.backend, alg.allocator; timer = alg.timer)
        e_bwd = reverse(e_fwd)
        state.residuals[e_bwd] = update_message!(msgs, network, e_bwd, alg.backend, alg.allocator; timer = alg.timer)
    end
    return state
end
