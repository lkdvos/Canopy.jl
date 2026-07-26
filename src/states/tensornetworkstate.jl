const StateTensor{T <: Number, S <: IndexSpace, N, A <: DenseVector{T}} =
    TensorMap{T, S, 1, N, A}

"""
    TensorNetworkState{T, S, N, A, V}

Tensor network state on an undirected graph. Each vertex holds a `TensorMap`
with one physical leg in the codomain and up to `N` virtual legs in the
domain; unused virtual legs are padded with `oneunit(S)`.

Type parameters:
- `T <: Number`: scalar type of the tensor entries.
- `S <: IndexSpace`: index space type (e.g. `ComplexSpace`, `Z2Space`).
- `N`: maximum coordination number (fixed size of every tensor's domain).
- `A <: DenseVector{T}`: dense storage type for tensor blocks.
- `V`: vertex token type, the key type of `adjacency` and `vertices`.
"""
struct TensorNetworkState{T <: Number, S <: IndexSpace, N, A <: DenseVector{T}, V}
    # TODO: small vector?
    adjacency::Dictionary{V, Vector{V}}
    vertices::Dictionary{V, StateTensor{T, S, N, A}}

    function TensorNetworkState{T, S, N, A, V}(
            adjacency::Dictionary{V, Vector{V}}, vertices::Dictionary{V, StateTensor{T, S, N, A}}
        ) where {T, S, A, N, V}
        @assert keys(adjacency) === keys(vertices) "adjacency and vertices should share keys"
        maximum(length, adjacency) <= N || throw(DimensionMismatch("non-matching coordination number"))
        return new{T, S, N, A, V}(adjacency, vertices)
    end
end

function TensorNetworkState(
        adjacency::Dictionary{V, Vector{V}}, vertices::Dictionary{V, StateTensor{T, S, N, A}}
    ) where {T, S, N, A, V}
    return TensorNetworkState{T, S, N, A, V}(adjacency, vertices)
end

# Constructors
# ------------
"""
    TensorNetworkState{T, S, N, A, V}(undef, pspaces, vspaces)
    TensorNetworkState{T, S, N, A}(undef, pspaces, vspaces)
    TensorNetworkState{T, S, N}(undef, pspaces, vspaces)
    TensorNetworkState{T, S}(undef, pspaces, vspaces)
    TensorNetworkState{T}(undef, pspaces, vspaces)
    TensorNetworkState(undef, pspaces, vspaces)

Allocate an uninitialized [`TensorNetworkState`](@ref) from per-vertex physical spaces `pspaces::Dictionary{V, S}` and per-edge virtual spaces `vspaces::Dictionary{UndirectedEdge{V}, S}`.
The adjacency is inferred from the edge keys, and `keys(pspaces)` must match the resulting vertex set.

`vspaces` must contain non-dual spaces; for each edge `(u, v)` with `u < v` the virtual space is used as-is on the `u` side and dualised on the `v` side.
Each tensor's domain is padded with `oneunit(S)` up to `N` legs.

Omitted type parameters take defaults:
- `T = Float64` is the scalartype of the state.
- `S` is the spacetype of the state, inferred from the input spaces.
- `N` is the maximum coordination of the inferred adjacency.
- `A = Vector{T}` is the storagetype for the tensor entries.
- `V` is the key type for labeling the vertices, inferred from the input dictionaries.
"""
function TensorNetworkState{T, S, N, A, V}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, N, A, V}
    adj = adjacency(keys(vspaces))
    issetequal(keys(pspaces), keys(adj)) || throw(ArgumentError("Incompatible vertices and edges"))

    @assert !any(isdual, vspaces) "TensorNetworkState expects non-dual virtual spaces"

    vertices = map(pairs(adj)) do (vertex, neighbors)
        P = pspaces[vertex]
        Vs = ntuple(N) do i
            return if i <= length(neighbors)
                Vspace = vspaces[UndirectedEdge(vertex, neighbors[i])]
                vertex < neighbors[i] ? Vspace : dual(Vspace)
            else
                oneunit(S)
            end
        end
        return StateTensor{T, S, N, A}(undef, P ← ⊗(Vs...))
    end

    return TensorNetworkState{T, S, N, A, V}(adj, vertices)
end

function TensorNetworkState{T, S, N, A}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, N, A, V}
    return TensorNetworkState{T, S, N, A, V}(undef, pspaces, vspaces)
end
function TensorNetworkState{T, S, N}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, N, V}
    return TensorNetworkState{T, S, N, Vector{T}}(undef, pspaces, vspaces)
end
function TensorNetworkState{T, S}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, V}
    adj = adjacency(keys(vspaces))
    N = maximum(length, adj)
    return TensorNetworkState{T, S, N}(undef, pspaces, vspaces)
end
function TensorNetworkState{T}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, V}
    return TensorNetworkState{T, S}(undef, pspaces, vspaces)
end
function TensorNetworkState(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {V, S <: IndexSpace}
    return TensorNetworkState{Float64}(undef, pspaces, vspaces)
end

# --- topology and uniform spaces ----------------------------------------------
# A topology is either a vector of `UndirectedEdge`s or a `Graphs.AbstractGraph`.
_edgelist(edges::AbstractVector{<:UndirectedEdge}) = edges
_edgelist(g::Graphs.AbstractGraph) = [UndirectedEdge(src(e), dst(e)) for e in edges(g)]

# expand uniform physical and virtual spaces over a topology
function _spaces(g::Graphs.AbstractGraph, P::S, V::S) where {S <: IndexSpace}
    pspaces = Dictionary(vertices(g), fill(P, nv(g)))
    vspaces = Dictionary(_edgelist(g), fill(V, ne(g)))
    return pspaces, vspaces
end
function _spaces(es::AbstractVector{<:UndirectedEdge}, P::S, V::S) where {S <: IndexSpace}
    verts = vertices(es)
    return Dictionary(verts, fill(P, length(verts))), Dictionary(es, fill(V, length(es)))
end

"""
    TensorNetworkState{T}(undef, g::Graphs.AbstractGraph, P::S, V::S)
    TensorNetworkState(undef, g::Graphs.AbstractGraph, P::S, V::S)

Convenience constructor for a [`TensorNetworkState`](@ref) on a
`Graphs.jl` graph `g`, with uniform per-vertex physical space `P` and
uniform per-edge virtual space `V`.

Vertex tokens are taken as `vertices(g)` (an `Int` range for the standard
`Graphs.SimpleGraph` family). Edges are converted to canonical
[`UndirectedEdge`](@ref)s. The on-site tensors are left uninitialized;
call `Random.randn!`/`Random.rand!` on the result to fill them.
"""
function TensorNetworkState{T}(
        ::UndefInitializer, g::Graphs.AbstractGraph, P::S, V::S,
    ) where {T, S <: IndexSpace}
    return TensorNetworkState{T}(undef, _spaces(g, P, V)...)
end

function TensorNetworkState(
        ::UndefInitializer, g::Graphs.AbstractGraph, P::S, V::S,
    ) where {S <: IndexSpace}
    return TensorNetworkState{Float64}(undef, g, P, V)
end

"""
    TensorNetworkState{T}(undef, edges::AbstractVector{<:UndirectedEdge}, P::S, V::S)
    TensorNetworkState(undef, edges::AbstractVector{<:UndirectedEdge}, P::S, V::S)

Convenience constructor for a [`TensorNetworkState`](@ref) on the graph spanned
by `edges`, with uniform per-vertex physical space `P` and uniform per-edge
virtual space `V`. Vertices are the endpoints appearing in `edges` (any isolated
vertices are not represented). Useful together with the lattice constructors
([`square_lattice`](@ref), [`triangular_lattice`](@ref), [`hexagonal_lattice`](@ref)).
"""
function TensorNetworkState{T}(
        ::UndefInitializer, edges::AbstractVector{<:UndirectedEdge}, P::S, V::S,
    ) where {T, S <: IndexSpace}
    return TensorNetworkState{T}(undef, _spaces(edges, P, V)...)
end

function TensorNetworkState(
        ::UndefInitializer, edges::AbstractVector{<:UndirectedEdge}, P::S, V::S,
    ) where {S <: IndexSpace}
    return TensorNetworkState{Float64}(undef, edges, P, V)
end

for f! in (:rand!, :randn!)
    @eval begin
        Random.$f!(state::TensorNetworkState) =
            Random.$f!(Random.default_rng(), state)
        Random.$f!(rng::Random.AbstractRNG, state::TensorNetworkState) =
            (foreach(Base.Fix1(Random.$f!, rng), values(state.vertices)); state)
    end
end

# --- random states ------------------------------------------------------------
# attach a charge-bath site carrying `dual(total_charge)` through a 1-dimensional
# bond, so that the physical legs of the remaining vertices fuse to `total_charge`
function _attach_bath(
        pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S},
        total_charge, auxiliary,
    ) where {V, S <: IndexSpace}
    I = sectortype(S)
    q = dual(_tosector(I, total_charge))
    q == one(I) && return pspaces, vspaces

    adj = adjacency(keys(vspaces))
    aux = _bath_vertex(adj, keys(adj), auxiliary)
    neighbor = _bath_neighbor(adj, keys(adj))
    Q = S(q => 1)

    return _augment(pspaces, aux, Q), _augment(vspaces, UndirectedEdge(aux, neighbor), Q)
end

for (f, f!) in ((:rand_state, :rand!), (:randn_state, :randn!))
    @eval begin
        function $f(
                rng::Random.AbstractRNG, ::Type{T}, pspaces::Dictionary{V, S},
                vspaces::Dictionary{UndirectedEdge{V}, S};
                total_charge = nothing, auxiliary = nothing,
            ) where {T, V, S <: IndexSpace}
            if !isnothing(total_charge)
                pspaces, vspaces = _attach_bath(pspaces, vspaces, total_charge, auxiliary)
            end
            return Random.$f!(rng, TensorNetworkState{T}(undef, pspaces, vspaces))
        end
        function $f(
                rng::Random.AbstractRNG, ::Type{T}, topology, P::S, V::S; kwargs...
            ) where {T, S <: IndexSpace}
            return $f(rng, T, _spaces(topology, P, V)...; kwargs...)
        end

        $f(::Type{T}, pspaces::Dictionary, vspaces::Dictionary; kwargs...) where {T} =
            $f(Random.default_rng(), T, pspaces, vspaces; kwargs...)
        $f(::Type{T}, topology, P::IndexSpace, V::IndexSpace; kwargs...) where {T} =
            $f(Random.default_rng(), T, topology, P, V; kwargs...)

        $f(rng::Random.AbstractRNG, pspaces::Dictionary, vspaces::Dictionary; kwargs...) =
            $f(rng, Float64, pspaces, vspaces; kwargs...)
        $f(rng::Random.AbstractRNG, topology, P::IndexSpace, V::IndexSpace; kwargs...) =
            $f(rng, Float64, topology, P, V; kwargs...)

        $f(pspaces::Dictionary, vspaces::Dictionary; kwargs...) =
            $f(Random.default_rng(), Float64, pspaces, vspaces; kwargs...)
        $f(topology, P::IndexSpace, V::IndexSpace; kwargs...) =
            $f(Random.default_rng(), Float64, topology, P, V; kwargs...)
    end
end

@doc """
    randn_state([rng], [T], pspaces, vspaces; total_charge = nothing, auxiliary = nothing)
    randn_state([rng], [T], topology, P::IndexSpace, V::IndexSpace; kwargs...)

Allocate a [`TensorNetworkState`](@ref) and fill it with normally distributed
entries. Spaces are given either per vertex and per edge (`pspaces`, `vspaces`, as
for the `undef` constructor) or uniformly as `P` and `V` over a `topology`, which is
a vector of [`UndirectedEdge`](@ref)s or a `Graphs.AbstractGraph`.

Filling graded tensors at random already yields a state of *trivial* total charge,
since every on-site tensor conserves charge. To land in a different global sector,
pass `total_charge`: a charge-bath vertex carrying the compensating charge is
attached by a 1-dimensional bond, exactly as in [`product_state`](@ref) — it is an
ordinary vertex of the result and should be left out of gate lists and observables.
Note that the whole charge enters through that single 1-dimensional bond, so the
result is generic within the target sector only up to that bottleneck.

See also [`rand_state`](@ref), which draws uniformly instead.
""" randn_state

@doc """
    rand_state([rng], [T], pspaces, vspaces; total_charge = nothing, auxiliary = nothing)
    rand_state([rng], [T], topology, P::IndexSpace, V::IndexSpace; kwargs...)

As [`randn_state`](@ref), but with uniformly distributed entries.
""" rand_state


# Properties
# ----------
Base.eltype(::Type{TensorNetworkState{T, S, N, A, V}}) where {T, S, N, A, V} = TensorMap{T, S, 1, N, A}

Base.keytype(state::TensorNetworkState) = keytype(typeof(state))
Base.keytype(::Type{TensorNetworkState{T, S, N, A, V}}) where {T, S, N, A, V} = V

VectorInterface.scalartype(::Type{T}) where {T <: TensorNetworkState} = scalartype(eltype(T))
TensorKit.storagetype(::Type{T}) where {T <: TensorNetworkState} = storagetype(eltype(T))
TensorKit.spacetype(::Type{T}) where {T <: TensorNetworkState} = spacetype(eltype(T))

Adapt.adapt_structure(to, state::TensorNetworkState) =
    TensorNetworkState(state.adjacency, map(adapt(to), state.vertices))

"""
    physicalspace(state, vertex) -> S

Return the physical (codomain) space of the on-site tensor at `vertex`.
"""
physicalspace(state::TensorNetworkState, vertex) = space(state[vertex], 1)

"""
    virtualspace(state, edge) -> S

Return the virtual space of `edge` as seen from `first(edge)`.

By convention `virtualspace(state, edge) == dual(virtualspace(state, reverse(edge)))`.
The *stored* space of an edge is non-dual on its smaller-vertex side, but `space` dualizes
domain legs, so the space returned here is non-dual when `first(edge) > last(edge)`.
Throws `ArgumentError` if `edge` is not incident on `first(edge)`.
"""
function virtualspace(state::TensorNetworkState, edge)
    tensor = state[first(edge)]
    leg_id = leg_index(state, edge)
    isnothing(leg_id) && throw(KeyError(edge))
    return space(tensor, leg_id + 1)
end

Base.length(state::TensorNetworkState) = length(state.vertices)

"""
    state[v] -> TensorMap

Return the on-site tensor at vertex `v`.
"""
@inline Base.getindex(state::TensorNetworkState, v) = state.vertices[v]


# --- consistency check --------------------------------------------------------
"""
    check_consistency(state) -> Bool

Return `true` if `state` is internally consistent:

- every vertex has at most as many neighbors as its tensor has domain legs, and the domain legs beyond its neighbors are unit spaces
- every edge's virtual space is the dual of its reverse, and is stored non-dual on the canonical (`first < last`) side.
"""
function check_consistency(state::TensorNetworkState)
    for v in vertices(state)
        t = state[v]
        adj = neighbors(state, v)
        length(adj) <= numin(t) || return false

        for i in (length(adj) + 1):numin(t)
            isunitspace(domain(t)[i]) || return false
        end
    end
    for edge in edges(state)
        # `reverse` is the identity on an `UndirectedEdge`, so orient it to compare the two
        # sides; `virtualspace` dualizes, hence the non-dual side is the *larger* vertex
        e = DirectedEdge(edge)
        isdual(virtualspace(state, e)) || return false
        virtualspace(state, e) == dual(virtualspace(state, reverse(e))) || return false
    end
    return true
end

# --- iteration accessors ------------------------------------------------------
"""
    vertices(state)

Return the collection of vertices of `state`.
"""
vertices(state::TensorNetworkState) = keys(state.vertices)

"""
    edges(state)

Return the set of undirected edges of `state` in canonical orientation
(`first(e) < last(e)`).
"""
function edges(state::TensorNetworkState)
    es = Indices{UndirectedEdge{keytype(state)}}()
    for v₁ in vertices(state), v₂ in neighbors(state, v₁)
        v₁ < v₂ && insert!(es, UndirectedEdge(v₁, v₂))
    end
    return es
end

"""
    neighbors(state, v)

Return the list of neighbors of vertex `v`, in the order matching the virtual legs of `state[v]`.
"""
neighbors(state::TensorNetworkState, v) = state.adjacency[v]

"""
    has_vertex(state, v) -> Bool

Return `true` if `v` is a vertex of `state`.
"""
has_vertex(state::TensorNetworkState, v) = haskey(state.adjacency, v)

"""
    has_edge(state, u, v) -> Bool
    has_edge(state, e::UndirectedEdge) -> Bool

Return `true` if `state` has an edge between `u` and `v` (equivalently, if
`v` is a neighbor of `u`).
"""
function has_edge(state::TensorNetworkState, u, v)
    return haskey(state.adjacency, u) && v in state.adjacency[u]
end
has_edge(state::TensorNetworkState, e::UndirectedEdge) =
    has_edge(state, first(e), last(e))

"""
    degree(state, v) -> Int

Return the number of neighbors of vertex `v` in `state`, i.e. its local
coordination number. This equals `length(neighbors(state, v))`.
"""
degree(state::TensorNetworkState, v) = length(neighbors(state, v))

"""
    incoming_edges(state, site; exclude=()) -> generator

Iterator over the directed edges `DirectedEdge(n, site)` for every neighbor
`n` of `site` in `state`, optionally skipping any neighbor present in
`exclude` (which must be an iterable container of vertex tokens — pass
`(other_vertex,)` to skip a single neighbor).

The result is a lazy generator suitable for `attach_messages`, `map`, and
`for` loops. Order matches `neighbors(state, site)`.
"""
function incoming_edges(state::TensorNetworkState, site; exclude = ())
    return (DirectedEdge(n, site) for n in neighbors(state, site) if !(n in exclude))
end

"""
    outgoing_edges(state, site; exclude=()) -> generator

Iterator over the directed edges `DirectedEdge(site, n)` for every neighbor
`n` of `site` in `state`, optionally skipping any neighbor present in
`exclude`. The reverse of [`incoming_edges`](@ref); order matches
`neighbors(state, site)`. `collect` it to pass to the vector form of
[`compute_message`](@ref).
"""
function outgoing_edges(state::TensorNetworkState, site; exclude = ())
    return (DirectedEdge(site, n) for n in neighbors(state, site) if !(n in exclude))
end

"""
    leg_index(state, edge) -> Int

Return the 1-based position of `edge` within `neighbors(state, first(edge))`, i.e. the domain-leg index occupied by `edge` in `state[first(edge)]`.
Throws `ArgumentError` if `edge` is not incident on `first(edge)`.
"""
function leg_index(state::TensorNetworkState, edge::AbstractEdge)
    idx = findfirst(==(last(edge)), neighbors(state, first(edge)))
    isnothing(idx) && throw(ArgumentError(lazy"edge $edge does not exist"))
    return idx
end

# --- dense materialization ----------------------------------------------------
"""
    TensorMap(state::TensorNetworkState) -> TensorMap

Materialize `state` as a dense many-body wavefunction by contracting every
virtual bond and dropping the `oneunit`-padded domain legs of each on-site
tensor.

The returned tensor has `length(state)` codomain legs and no domain legs:
codomain leg `i` is the physical leg of the `i`-th vertex in the iteration
order of `vertices(state)`.

This is exponentially expensive in the number of vertices and is intended
for small systems (tests, debugging, comparison against exact diagonalization).
"""
function TensorKit.TensorMap(state::TensorNetworkState)
    V = keytype(typeof(state))
    tensors = AbstractTensorMap[]
    indices = Vector{Int}[]
    edge_label = Dict{UndirectedEdge{V}, Int}()
    next_label = Ref(0)

    for (i, v) in enumerate(vertices(state))
        T = state[v]
        adj = neighbors(state, v)
        d = length(adj)

        Tstripped = T
        for k in numin(T):-1:(d + 1)
            Tstripped = removeunit(Tstripped, 1 + k)
        end

        labels = Vector{Int}(undef, 1 + d)
        labels[1] = -i
        for k in 1:d
            e = UndirectedEdge(v, adj[k])
            labels[k + 1] = get!(edge_label, e) do
                next_label[] += 1
                return next_label[]
            end
        end
        push!(tensors, Tstripped)
        push!(indices, labels)
    end

    return ncon(tensors, indices)
end

# --- pretty printing ---------------------------------------------------------
function Base.show(io::IO, ::MIME"text/plain", state::TensorNetworkState)
    summary(io, state)
    print(io, " with ", length(state), " vertices:")
    indent = '\t'
    inner = IOContext(io, :typeinfo => eltype(state))
    for v in vertices(state)
        println(io)
        println(io)
        print(io, " vertex ", v, ":")
        T = state[v]
        d = length(neighbors(state, v))
        for k in numin(T):-1:(d + 1)
            T = removeunit(T, 1 + k)
        end
        buf = IOBuffer()
        show(IOContext(buf, inner), MIME"text/plain"(), T)
        for line in eachline(IOBuffer(take!(buf)))
            println(io)
            print(io, indent, line)
        end
    end
    return nothing
end
