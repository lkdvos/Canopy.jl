"""
    ResidualSplashSchedule(; height = 2)

Residual-scheduled Splash schedule. Each iteration performs one *splash* rooted
at the vertex with the largest inbound residual: a height-limited BFS tree is
built around the root, messages are recomputed inward (leaves → root) and then
outward (root → leaves, over the full graph neighborhood so loop-closing and
boundary messages are sent). Residuals are maintained incrementally against a
`last_used` snapshot, so no message is ever computed merely to measure a
residual. Exact on trees when `height` exceeds the graph diameter.
"""
struct ResidualSplashSchedule <: BPSchedule
    height::Int
end
ResidualSplashSchedule(; height::Int = 2) = ResidualSplashSchedule(height)

# A deep snapshot of the current messages: `residuals[e]` tracks the change on
# `e` since its consumer last used it.
init_last_used(::ResidualSplashSchedule, messages) = copy(messages)

# One splash per step, rooted at the maximum-inbound-residual vertex.
function update_messages!(
        sched::ResidualSplashSchedule, problem::BPProblem, alg::BeliefPropagation, state::BPState
    )
    network = problem.network
    root = select_root(network, state.residuals)
    splash!(state, network, alg, root, sched.height)
    return state
end

# O(V) scan: the vertex whose largest inbound residual is maximal.
function select_root(network, residuals)
    best = nothing
    best_r = -Inf
    for v in vertices(network)
        r = maximum(residuals[DirectedEdge(k, v)] for k in neighbors(network, v))
        if r > best_r
            best_r = r
            best = v
        end
    end
    return best
end

# One splash: inward toward the root, then outward over the full neighborhood.
function splash!(state, network, alg, root, height::Int)
    msgs = state.iterate
    order, parent = bfs_tree(network, root, height)

    # Inward: leaves → root, one message toward the parent each.
    for v in Iterators.reverse(order)
        v == root && continue
        e = DirectedEdge(v, parent[v])
        new_msg = recompute_message(msgs, network, e, alg.backend, alg.allocator; timer = alg.timer)
        msgs.messages[e] = new_msg
        state.residuals[e] = tr_distance(new_msg, state.last_used[e]; is_hermitian = true)
    end

    # Outward: root → leaves. Service each vertex (refresh its inbound snapshots),
    # then recompute all non-parent outbound messages together.
    for v in order
        for k in neighbors(network, v)
            e = DirectedEdge(k, v)
            state.last_used.messages[e] = copy(msgs[e])
            state.residuals[e] = 0.0
        end
        targets = [DirectedEdge(v, j) for j in neighbors(network, v) if j != get(parent, v, nothing)]
        new_msgs = recompute_messages(msgs, network, targets, alg.backend, alg.allocator; timer = alg.timer)
        for (e, new_msg) in zip(targets, new_msgs)
            msgs.messages[e] = new_msg
            state.residuals[e] = tr_distance(new_msg, state.last_used[e]; is_hermitian = true)
        end
    end
    return state
end
