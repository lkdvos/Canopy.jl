# Product-state construction
# ---------------------------
# Build a `TensorNetworkState` that is a product over its vertices. Each site
# state is specified as `sector => coefficients`: a definite charge `c_v` and a
# coefficient vector over that sector's degeneracy in the physical space. A
# nontrivial-charge site state cannot be written as a bare ket `P ← oneunit`
# (that map only reaches the trivial sector), so the bond (virtual) spaces must
# be *deduced* to carry the charges that make every on-site tensor consistent —
# the charge balance.
#
# With 1-dimensional bonds and an abelian symmetry, the per-edge charges solve a
# linear system over the symmetry group: at every vertex `v`,
#   c_v = ⊗_{neighbors w} σ(v, w) q_e,   σ(v, w) = identity if v < w else dual,
# where `q_e` is the charge stored on the canonical (smaller-vertex) side of the
# edge. This is solvable iff each connected component is charge-neutral
# (`⊗ c_v == one`); a spanning tree then fixes the tree-edge charges while loop
# edges stay trivial.

"""
    _deduce_bond_charges(adj, charges) -> Dictionary{UndirectedEdge{V}, I}

Deduce the per-edge charge (on the canonical, smaller-vertex side) of a 1-dim
product state from the per-vertex charges `charges::Dictionary{V, I}` and the
adjacency `adj`. Throws if any connected component is not charge-neutral.
"""
function _deduce_bond_charges(adj::Dictionary{V, Vector{V}}, charges::Dictionary{V, I}) where {V, I}
    qedge = Dictionary{UndirectedEdge{V}, I}()
    parent = Dictionary{V, V}()
    visited = Set{V}()

    for root in keys(adj)
        root in visited && continue
        # breadth-first spanning tree of this component
        push!(visited, root)
        order = [root]
        head = 1
        while head ≤ length(order)
            u = order[head]
            head += 1
            for w in adj[u]
                e = UndirectedEdge(u, w)
                haskey(qedge, e) || insert!(qedge, e, one(I))  # default: loop edges stay trivial
                if !(w in visited)
                    push!(visited, w)
                    insert!(parent, w, u)
                    push!(order, w)
                end
            end
        end

        # charge balance: a 1-dim product state exists iff the component is neutral
        total = foldl((c, v) -> only(c ⊗ charges[v]), order; init = one(I))
        total == one(I) ||
            throw(ArgumentError(lazy"product state is not charge-neutral on the component containing $root (total charge $total ≠ $(one(I))); only the trivial total charge is representable"))

        # solve tree edges leaves→root: each vertex fixes its parent edge from the
        # residual of its already-determined incident edges (children + loop edges)
        for idx in length(order):-1:1
            v = order[idx]
            haskey(parent, v) || continue  # root: nothing left to fix
            p = parent[v]
            ep = UndirectedEdge(v, p)
            res = charges[v]
            for w in adj[v]
                e = UndirectedEdge(v, w)
                e == ep && continue
                contrib = v < w ? qedge[e] : dual(qedge[e])
                res = only(res ⊗ dual(contrib))
            end
            qedge[ep] = v < p ? res : dual(res)
        end
    end
    return qedge
end

"""
    product_state([T], edges, pspaces, localstates) -> TensorNetworkState
    product_state([T], edges, P::IndexSpace, localstate::Pair) -> TensorNetworkState

Build a product [`TensorNetworkState`](@ref) on the graph spanned by `edges`
(e.g. from [`square_lattice`](@ref)).

Each vertex's local state is given as `sector => coefficients`: the symmetry
sector `c_v` (definite charge) and a coefficient vector over that sector's
degeneracy in the physical space. Pass per-vertex `pspaces::Dictionary{V, S}`
and `localstates::Dictionary{V, <:Pair}`, or uniform `P` and a single
`localstate` applied to every vertex.

The 1-dimensional bond (virtual) spaces are deduced so that every on-site tensor
is charge-consistent. Only abelian symmetries are supported, and each connected
component must be charge-neutral (`⊗ c_v == one`); both conditions throw a
descriptive error otherwise.

`T` defaults to `float` of the coefficient `eltype`.
"""
function product_state(
        ::Type{T}, edges::AbstractVector{<:UndirectedEdge{V}}, pspaces::Dictionary{V, S},
        localstates::Dictionary{V, <:Pair},
    ) where {T, V, S <: IndexSpace}
    I = sectortype(S)
    FusionStyle(I) isa UniqueFusion ||
        throw(ArgumentError(lazy"product_state only supports abelian symmetries; sectortype $I is non-abelian"))

    adj = adjacency(Indices(edges))
    issetequal(keys(pspaces), keys(adj)) ||
        throw(ArgumentError("vertices of `pspaces` do not match the vertices spanned by `edges`"))
    issetequal(keys(localstates), keys(adj)) ||
        throw(ArgumentError("vertices of `localstates` do not match the vertices spanned by `edges`"))

    charges = map(p -> convert(I, first(p)), localstates)
    for v in keys(adj)
        d = dim(pspaces[v], charges[v])
        ncoeff = length(last(localstates[v]))
        ncoeff == d ||
            throw(ArgumentError(lazy"vertex $v: sector $(charges[v]) has degeneracy $d in its physical space, but $ncoeff coefficients were given"))
    end

    qedge = _deduce_bond_charges(adj, charges)
    vspaces = map(q -> S(q => 1), qedge)
    state = TensorNetworkState{T}(undef, pspaces, vspaces)

    for v in keys(adj)
        t = state[v]
        zerovector!(t)
        copyto!(block(t, charges[v]), last(localstates[v]))
    end
    return state
end

function product_state(
        ::Type{T}, edges::AbstractVector{<:UndirectedEdge{V}}, P::S, localstate::Pair,
    ) where {T, V, S <: IndexSpace}
    verts = keys(adjacency(Indices(edges)))
    pspaces = Dictionary(verts, fill(P, length(verts)))
    localstates = Dictionary(verts, fill(localstate, length(verts)))
    return product_state(T, edges, pspaces, localstates)
end

function product_state(
        edges::AbstractVector{<:UndirectedEdge}, pspaces::Dictionary, localstates::Dictionary,
    )
    T = float(mapreduce(p -> eltype(last(p)), promote_type, localstates))
    return product_state(T, edges, pspaces, localstates)
end

function product_state(edges::AbstractVector{<:UndirectedEdge}, P::IndexSpace, localstate::Pair)
    return product_state(float(eltype(last(localstate))), edges, P, localstate)
end
