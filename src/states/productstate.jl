# Product-state construction
# --------------------------
# Build a `TensorNetworkState` that is a product over its vertices. The charge
# bookkeeping — local state specifications, bond-charge deduction and the charge
# bath — lives in `charges.jl`.

# expand a uniform specification into one entry per vertex, or validate a given one
function _pervertex(spec::Dictionary, verts, name)
    issetequal(keys(spec), verts) ||
        throw(ArgumentError(lazy"vertices of `$name` do not match the vertices spanned by the topology"))
    return spec
end
_pervertex(spec, verts, name) = Dictionary(verts, fill(spec, length(verts)))

"""
    product_state([T], topology, pspaces, localstates; total_charge = nothing, auxiliary = nothing)

Build a product [`TensorNetworkState`](@ref) on `topology`, which is either a
vector of [`UndirectedEdge`](@ref)s (e.g. from [`square_lattice`](@ref)) or a
`Graphs.AbstractGraph`.

`pspaces` gives the physical space of every vertex and `localstates` its local
state, each either as a `Dictionary` keyed by vertex or as a single value applied
to every vertex. A local state is one of

- `sector => coefficients`: a definite charge and a coefficient vector over that
  sector's degeneracy in the physical space,
- a bare coefficient vector, only if the physical space has `sectortype` `Trivial`,
- a bare sector (or anything convertible to one, e.g. `1` for `U1Irrep`), if that
  sector is 1-dimensional in the physical space.

Coefficients are used as given; the result is not normalized.

The 1-dimensional bond (virtual) spaces are deduced so that every on-site tensor is
charge-consistent. This requires each connected component to be charge-neutral
(`⊗ c_v == one`). To realize a nontrivial total charge instead, pass
`total_charge`: the local charges must then fuse to it, and a *charge-bath* vertex
carrying the compensating charge is attached to the lattice by a 1-dimensional
bond. It is an ordinary vertex of the resulting state — it will show up in
`vertices(state)` and in `TensorMap(state)`, and should be left out of gate lists
and observables. Its label comes from [`auxiliary_vertex`](@ref) unless `auxiliary`
is given, and it attaches to a vertex of minimal degree to avoid raising the
state's maximum coordination number.

Only abelian symmetries are supported; a non-abelian `sectortype` throws.

`T` defaults to `float` of the coefficient `eltype`.
"""
function product_state(
        ::Type{T}, topology, pspaces, localstates;
        total_charge = nothing, auxiliary = nothing,
    ) where {T}
    es = _edgelist(topology)
    adj = adjacency(Indices(es))
    ps = _pervertex(pspaces, keys(adj), "pspaces")
    ls = _pervertex(localstates, keys(adj), "localstates")

    S = valtype(ps)
    I = sectortype(S)
    FusionStyle(I) isa UniqueFusion ||
        throw(ArgumentError(lazy"product_state only supports abelian symmetries; sectortype $I is non-abelian"))

    specs = map(pairs(ls)) do (vertex, spec)
        return _localstate(ps[vertex], spec)
    end
    charges = Dictionary(keys(specs), I[first(specs[v]) for v in keys(specs)])

    component = _bath_component(adj, charges, total_charge)
    if !isnothing(component)
        aux = _bath_vertex(adj, keys(adj), auxiliary)
        neighbor = _bath_neighbor(adj, component)
        q = dual(_tosector(I, total_charge))

        adj = adjacency(Indices(vcat(es, [UndirectedEdge(aux, neighbor)])))
        ps = _augment(ps, aux, S(q => 1))
        charges = _augment(charges, aux, q)
        specs = _augment(specs, aux, q => [true])
    end

    qedge = _deduce_bond_charges(adj, charges)
    vspaces = map(q -> S(q => 1), qedge)
    state = TensorNetworkState{T}(undef, ps, vspaces)

    for (vertex, spec) in pairs(specs)
        t = state[vertex]
        zerovector!(t)
        copyto!(block(t, first(spec)), last(spec))
    end
    return state
end

function product_state(topology, pspaces, localstates; kwargs...)
    T = float(_speceltype(localstates))
    return product_state(T, topology, pspaces, localstates; kwargs...)
end
