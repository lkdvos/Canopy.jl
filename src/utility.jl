iterate_difference(previous_iterate, iterate) =
    iterate_difference!(previous_iterate, iterate)

"""
    @maybe_timeit timer name expr

Like `TimerOutputs.@timeit`, but a no-op when `timer === nothing` — lets
callers thread `timer = nothing` (the default) through `apply!` / `belief_propagation`
to disable timing entirely.
"""
macro maybe_timeit(timer, name, expr)
    return quote
        if $(esc(timer)) === nothing
            $(esc(expr))
        else
            @timeit $(esc(timer)) $(esc(name)) $(esc(expr))
        end
    end
end

# Graph helpers
# -------------

# Randomized BFS vertex order over the network's own adjacency, as
# `(order, pos)` with `pos[order[i]] == i`. Component roots are visited in a
# random order and each vertex's neighbours are shuffled, so a fresh order is
# drawn on every call.
#
# BFS is *restarted per connected component*, i.e. this spans a forest rather
# than a tree. [`SpanningTreeSchedule`](@ref) drives every vertex from `order`,
# so a single-rooted BFS would silently leave a disconnected network's other
# components untouched — their residuals would stay `Inf` and `StopWhenStable`
# would never fire.
#
# `pos` doubles as the visited marker, and the invariant callers rely on is that
# every vertex other than its component's root has at least one neighbour with a
# strictly smaller `pos` (its BFS parent).
function random_bfs_order(state::AbstractTensorNetwork, rng)
    V = keytype(state)
    roots = collect(vertices(state))
    Random.shuffle!(rng, roots)
    order = V[]
    sizehint!(order, length(roots))
    pos = Dictionary{V, Int}()
    head = 1
    for root in roots
        haskey(pos, root) && continue
        insert!(pos, root, length(order) + 1)
        push!(order, root)
        while head <= length(order)
            v = order[head]
            head += 1
            nbrs = collect(neighbors(state, v))
            Random.shuffle!(rng, nbrs)
            for n in nbrs
                haskey(pos, n) && continue
                insert!(pos, n, length(order) + 1)
                push!(order, n)
            end
        end
    end
    return order, pos
end

# Height-limited BFS from `root` (height counted in edges from root). Returns the
# BFS vertex order (root first) and the `parent` map (root absent). Modeled on
# the BFS loop in `random_bfs_order`, but deterministic, depth-bounded, and
# single-rooted (a splash is local by construction).
function bfs_tree(network::AbstractTensorNetwork, root, height::Int)
    V = keytype(network)
    parent = Dictionary{V, V}()
    depth = Dictionary{V, Int}()
    order = V[root]
    insert!(depth, root, 0)
    head = 1
    while head <= length(order)
        v = order[head]
        head += 1
        depth[v] >= height && continue
        for n in neighbors(network, v)
            haskey(depth, n) && continue
            insert!(depth, n, depth[v] + 1)
            insert!(parent, n, v)
            push!(order, n)
        end
    end
    return order, parent
end

# Propagate a change `δ` along edge `e = (s → r)` to the downstream edges that
# consume `msgs[e]` — the outgoing edges `r → t` with `t ≠ s`.
function propagate_change!(residuals, network, e::DirectedEdge, δ)
    s, r = first(e), last(e)
    for t in neighbors(network, r)
        t == s && continue
        residuals[DirectedEdge(r, t)] += δ
    end
    return residuals
end
