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

# Randomized BFS spanning tree over the network's own adjacency. Returns the BFS
# vertex order (root first), the `parent` map (root absent), and the cotree
# (loop-closing) undirected edges. Assumes a connected network.
function random_spanning_tree(state::AbstractTensorNetwork, rng)
    V = keytype(state)
    verts = collect(vertices(state))
    root = rand(rng, verts)
    parent = Dictionary{V, V}()
    visited = Set{V}((root,))
    order = V[root]
    tree_edges = Set{UndirectedEdge{V}}()
    head = 1
    while head <= length(order)
        v = order[head]
        head += 1
        nbrs = collect(neighbors(state, v))
        Random.shuffle!(rng, nbrs)
        for n in nbrs
            n in visited && continue
            push!(visited, n)
            insert!(parent, n, v)
            push!(tree_edges, UndirectedEdge(v, n))
            push!(order, n)
        end
    end
    cotree = [e for e in edges(state) if !(e in tree_edges)]
    return order, parent, cotree
end

# Height-limited BFS from `root` (height counted in edges from root). Returns the
# BFS vertex order (root first) and the `parent` map (root absent). Modeled on
# the BFS loop in `random_spanning_tree`, but deterministic and depth-bounded.
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
