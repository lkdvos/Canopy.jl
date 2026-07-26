# Charge bookkeeping for state construction
# -----------------------------------------
# A site state is specified as `sector => coefficients`: a definite charge `c_v`
# and a coefficient vector over that sector's degeneracy in the physical space. A
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
#
# A component whose charges do not balance can still be realized by attaching a
# *charge bath*: one auxiliary vertex carrying the compensating charge on a
# 1-dimensional bond, which makes the augmented component neutral. This is opt-in
# through `total_charge`, since it adds a vertex to the state.

# --- local state specifications -----------------------------------------------
"""
    _localstate(P, spec) -> Pair{I, <:AbstractVector}

Canonicalize a user local-state specification against the physical space `P` into
`sector => coefficients`. Accepted forms:

- `sector => coefficients`, with one coefficient per degeneracy of `sector` in `P`,
- a bare coefficient vector, only if `sectortype(P) === Trivial`,
- anything convertible to a sector (e.g. `fℤ₂(1)`, or `1` for `U1Irrep`), if that
  sector is 1-dimensional in `P`.
"""
function _localstate(P::S, spec::Pair) where {S <: IndexSpace}
    c = _tosector(sectortype(S), first(spec))
    coeffs = last(spec)
    d = _sector_degeneracy(P, c)
    length(coeffs) == d ||
        throw(ArgumentError(lazy"sector $c has degeneracy $d in the physical space $P, but $(length(coeffs)) coefficients were given"))
    return c => coeffs
end

function _localstate(P::S, coeffs::AbstractVector) where {S <: IndexSpace}
    I = sectortype(S)
    I === Trivial ||
        throw(ArgumentError(lazy"a bare coefficient vector does not have a definite charge for sectortype $I; name the sector as `sector => coefficients`"))
    return _localstate(P, one(I) => coeffs)
end

function _localstate(P::S, charge) where {S <: IndexSpace}
    c = _tosector(sectortype(S), charge)
    d = _sector_degeneracy(P, c)
    isone(d) ||
        throw(ArgumentError(lazy"sector $c has degeneracy $d in the physical space $P; pass `$c => coefficients` to select a state within it"))
    return c => [true]
end

# `convert` would report a bare `MethodError` on the common mistake of naming a
# sector that belongs to a different symmetry than the physical space
function _tosector(::Type{I}, charge) where {I <: Sector}
    charge isa Sector && !(charge isa I) &&
        throw(ArgumentError(lazy"sector $charge has type $(typeof(charge)), but the physical space has sectortype $I"))
    return convert(I, charge)
end

function _sector_degeneracy(P::IndexSpace, c)
    d = dim(P, c)
    iszero(d) &&
        throw(ArgumentError(lazy"sector $c does not appear in the physical space $P"))
    return d
end

# scalartype of a spec, without needing the physical space (a bare charge means
# unit coefficients, hence `Bool`)
_speceltype(specs::Dictionary) = mapreduce(_speceltype, promote_type, specs)
_speceltype(spec::Pair) = eltype(last(spec))
_speceltype(spec::AbstractVector) = eltype(spec)
_speceltype(_) = Bool

# --- charge bath --------------------------------------------------------------
"""
    auxiliary_vertex(vertices) -> V
    auxiliary_vertex(::Type{V}, vertices) -> V

Derive a vertex token distinct from every vertex in `vertices`, to label the
charge-bath site of a state with nontrivial total charge (see
[`product_state`](@ref)). Derived labels sort before every existing vertex.

Methods exist for integer and `NTuple{N, Int}` vertex tokens; add one for a custom
vertex type, or pass `auxiliary` explicitly at the call site.
"""
auxiliary_vertex(verts) = auxiliary_vertex(eltype(verts), verts)
auxiliary_vertex(::Type{V}, verts) where {V <: Integer} = minimum(verts) - one(V)
function auxiliary_vertex(::Type{NTuple{N, Int}}, verts) where {N}
    return ntuple(i -> isone(i) ? minimum(first, verts) - 1 : 0, N)
end
function auxiliary_vertex(::Type{V}, verts) where {V}
    throw(ArgumentError(lazy"cannot derive an auxiliary vertex label for vertex type $V; pass `auxiliary` explicitly"))
end

"""
    _components(adj) -> Vector{Vector{V}}

Connected components of `adj`, each in breadth-first order from its first vertex.
"""
function _components(adj::Dictionary{V, Vector{V}}) where {V}
    comps = Vector{V}[]
    visited = Set{V}()
    for root in keys(adj)
        root in visited && continue
        push!(visited, root)
        order = [root]
        head = 1
        while head ≤ length(order)
            u = order[head]
            head += 1
            for w in adj[u]
                w in visited && continue
                push!(visited, w)
                push!(order, w)
            end
        end
        push!(comps, order)
    end
    return comps
end

_fuse_charges(::Type{I}, charges, verts) where {I} =
    foldl((c, v) -> only(c ⊗ charges[v]), verts; init = one(I))

"""
    _bath_component(adj, charges, total_charge) -> Union{Nothing, Vector{V}}

Determine whether a charge bath is needed to realize `total_charge`, returning the
connected component it must attach to, or `nothing` if the charges already balance.

`total_charge === nothing` demands neutrality of every component — the bath is
opt-in. Otherwise exactly one component may carry a nontrivial charge, and it must
equal `total_charge`.
"""
function _bath_component(
        adj::Dictionary{V, Vector{V}}, charges::Dictionary{V, I}, total_charge
    ) where {V, I}
    comps = _components(adj)
    totals = [_fuse_charges(I, charges, comp) for comp in comps]
    charged = findall(!=(one(I)), totals)
    q = isnothing(total_charge) ? nothing : _tosector(I, total_charge)

    if isempty(charged)
        isnothing(q) || q == one(I) ||
            throw(ArgumentError(lazy"the local charges fuse to $(one(I)), but `total_charge = $q` was requested; with 1-dimensional bonds the local charges fix the total"))
        return nothing
    end

    isnothing(total_charge) &&
        throw(ArgumentError(lazy"product state is not charge-neutral on the component containing $(first(comps[first(charged)])) (total charge $(totals[first(charged)]) ≠ $(one(I))); pass `total_charge` to attach a charge-bath site carrying the compensating charge"))

    isone(length(charged)) ||
        throw(ArgumentError(lazy"a single `total_charge` cannot describe $(length(charged)) charged components (containing $([first(comps[i]) for i in charged]), with charges $(totals[charged])); bonds never cross components, so each would need its own charge bath"))

    i = only(charged)
    totals[i] == q ||
        throw(ArgumentError(lazy"the local charges fuse to $(totals[i]) on the component containing $(first(comps[i])), but `total_charge = $q` was requested"))
    return comps[i]
end

"""
    _bath_neighbor(adj, verts) -> V

Vertex of `verts` with the fewest neighbors, tie-broken by vertex order.

The charge bath attaches here so that it is least likely to raise the state's
maximum coordination number `N`, which would give *every* on-site tensor an extra
`oneunit`-padded domain leg.
"""
function _bath_neighbor(adj::Dictionary{V, Vector{V}}, verts) where {V}
    return reduce(verts) do u, v
        return isless((length(adj[v]), v), (length(adj[u]), u)) ? v : u
    end
end

"""
    _augment(dict, key, value) -> Dictionary

A fresh dictionary with one extra entry. `insert!` is not an option: dictionaries
built on `keys(adj)` *share* that `Indices` object, so growing one in place silently
desynchronizes the others.
"""
_augment(dict::Dictionary, key, value) =
    Dictionary(vcat(collect(keys(dict)), [key]), vcat(collect(dict), [value]))

"""
    _bath_vertex(adj, verts, auxiliary) -> V

Label for a charge-bath vertex attached to `verts`, either `auxiliary` or derived
from the vertex type. Throws if it collides with an existing vertex.
"""
function _bath_vertex(adj::Dictionary{V, Vector{V}}, verts, auxiliary) where {V}
    aux = isnothing(auxiliary) ? auxiliary_vertex(V, verts) : convert(V, auxiliary)
    haskey(adj, aux) &&
        throw(ArgumentError(lazy"auxiliary vertex $aux is already a vertex of the state; pass a different `auxiliary`"))
    return aux
end

"""
    _deduce_bond_charges(adj, charges) -> Dictionary{UndirectedEdge{V}, I}

Deduce the per-edge charge (on the canonical, smaller-vertex side) of a 1-dim product state from the per-vertex charges `charges::Dictionary{V, I}` and the adjacency `adj`.
Throws if any connected component is not charge-neutral.
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
        total = _fuse_charges(I, charges, order)
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
