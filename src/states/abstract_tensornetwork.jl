# --- shared tensor-network layer ----------------------------------------------
#
# `TensorNetworkState` and `TensorNetworkOperator` differ only in how many *physical*
# legs sit in the codomain of each on-site tensor: one for a state, two for an
# operator. Everything else — the adjacency, the bond-duality convention, the
# `oneunit`-padded domain, the graph interface — is shared, and lives here.
#
# ## Leg layout
#
# An on-site tensor is a `TensorMap{T, S, NP, N, A}`:
#
#     slots 1 … NP        physical legs (codomain)
#     slots NP+1 … NP+N   virtual legs (domain), one per neighbor, padded with oneunit(S)
#
# so virtual leg `k` (as returned by `leg_index`) sits at tensor slot `k + NP`. Code
# reads `NP` off the tensor with `numout`, never by hard-coding it; `num_physical`
# exists for the places that only have the network, not a tensor.
#
# ## Scope
#
# This supertype is deliberately *closed*: it exists to share method bodies and to
# derive the `T, S, N, A, V` traits from one set of parameters, over the two
# representations this package owns. It is not the data-agnostic graph layer that
# `docs/src/design.md` argues against — it makes no promise to outside subtypes, and
# every method here reaches directly into the `adjacency` / `vertices` fields.

"""
    AbstractTensorNetwork{T, S, N, A, V}

Supertype of [`TensorNetworkState`](@ref) and [`TensorNetworkOperator`](@ref): a
tensor network on an undirected graph, holding one `TensorMap` per vertex with a
fixed number of physical legs in the codomain and up to `N` virtual legs in the
domain.

Type parameters:
- `T <: Number`: scalar type of the tensor entries.
- `S <: IndexSpace`: index space type (e.g. `ComplexSpace`, `Z2Space`).
- `N`: maximum coordination number (fixed size of every tensor's domain).
- `A <: DenseVector{T}`: dense storage type for tensor blocks.
- `V`: vertex token type, the key type of `adjacency` and `vertices`.

Subtypes are expected to hold `adjacency::Dictionary{V, Vector{V}}` and
`vertices::Dictionary{V, <:AbstractTensorMap}`, and to define
[`num_physical`](@ref), `eltype` and `Adapt.adapt_structure`.
"""
abstract type AbstractTensorNetwork{T <: Number, S <: IndexSpace, N, A <: DenseVector{T}, V} end

"""
    num_physical(network) -> Int
    num_physical(::Type{<:AbstractTensorNetwork}) -> Int

Number of physical legs in the codomain of each on-site tensor: `1` for a
[`TensorNetworkState`](@ref), `2` for a [`TensorNetworkOperator`](@ref).

This is also the offset between a virtual leg index and its tensor slot: virtual leg
`k` of `network[v]` sits at slot `k + num_physical(network)`. Code holding the tensor
itself should prefer `numout(network[v])`, which is the same number.
"""
num_physical(network::AbstractTensorNetwork) = num_physical(typeof(network))

# --- shared constructor helper ------------------------------------------------
# The `N` domain spaces of the tensor at `vertex`: the per-edge space for each of its
# `nbrs` (used as-is on the smaller-vertex side, dualised on the larger), then `oneunit(S)`
# padding up to `N`. Shared by the `TensorNetworkState` and `TensorNetworkOperator`
# `undef` constructors, which differ only in their codomain.
function _padded_virtualspaces(
        ::Type{S}, ::Val{N}, vertex, nbrs, vspaces
    ) where {S <: IndexSpace, N}
    return ntuple(Val(N)) do i
        return if i <= length(nbrs)
            Vspace = vspaces[UndirectedEdge(vertex, nbrs[i])]
            vertex < nbrs[i] ? Vspace : dual(Vspace)
        else
            oneunit(S)
        end
    end
end

# Properties
# ----------
Base.keytype(network::AbstractTensorNetwork) = keytype(typeof(network))
Base.keytype(::Type{<:AbstractTensorNetwork{T, S, N, A, V}}) where {T, S, N, A, V} = V

VectorInterface.scalartype(::Type{<:AbstractTensorNetwork{T}}) where {T} = T
TensorKit.storagetype(::Type{<:AbstractTensorNetwork{T, S, N, A}}) where {T, S, N, A} = A
TensorKit.spacetype(::Type{<:AbstractTensorNetwork{T, S}}) where {T, S} = S

Base.length(network::AbstractTensorNetwork) = length(network.vertices)

"""
    network[v] -> TensorMap

Return the on-site tensor at vertex `v`.
"""
@inline Base.getindex(network::AbstractTensorNetwork, v) = network.vertices[v]

for f! in (:rand!, :randn!)
    @eval begin
        Random.$f!(network::AbstractTensorNetwork) =
            Random.$f!(Random.default_rng(), network)
        Random.$f!(rng::Random.AbstractRNG, network::AbstractTensorNetwork) =
            (foreach(Base.Fix1(Random.$f!, rng), values(network.vertices)); network)
    end
end

# Spaces
# ------
"""
    physicalspace(network, vertex) -> S
    physicalspace(network, vertex, i::Int) -> S

Return the `i`-th physical (codomain) space of the on-site tensor at `vertex`,
defaulting to `i = 1`.

For a [`TensorNetworkState`](@ref) there is only one, so the index is redundant. For a
[`TensorNetworkOperator`](@ref), slot `1` is the ket (row) space and slot `2` the bra
(column) space, which by the vectorization convention is `dual` of the row space — see
[`isvectorized`](@ref Canopy.isvectorized).
"""
physicalspace(network::AbstractTensorNetwork, vertex, i::Int = 1) = space(network[vertex], i)

"""
    physicalspaces(network, vertex) -> NTuple{num_physical(network), S}

Return all physical (codomain) spaces of the on-site tensor at `vertex`, in slot order.
"""
physicalspaces(network::AbstractTensorNetwork, vertex) =
    ntuple(i -> space(network[vertex], i), num_physical(network))

"""
    virtualspace(network, edge) -> S

Return the virtual space of `edge` as seen from `first(edge)`.

By convention `virtualspace(net, edge) == dual(virtualspace(net, reverse(edge)))`.
The *stored* space of an edge is non-dual on its smaller-vertex side, but `space` dualizes
domain legs, so the space returned here is non-dual when `first(edge) > last(edge)`.
Throws `ArgumentError` if `edge` is not incident on `first(edge)`.
"""
function virtualspace(network::AbstractTensorNetwork, edge)
    tensor = network[first(edge)]
    return space(tensor, leg_index(network, edge) + numout(tensor))
end

# --- consistency check --------------------------------------------------------
"""
    check_consistency(network) -> Bool

Return `true` if `network` is internally consistent:

- every vertex has at most as many neighbors as its tensor has domain legs, and the domain legs beyond its neighbors are unit spaces
- every edge's virtual space is the dual of its reverse, and is stored non-dual on the canonical (`first < last`) side.
"""
function check_consistency(network::AbstractTensorNetwork)
    for v in vertices(network)
        t = network[v]
        adj = neighbors(network, v)
        length(adj) <= numin(t) || return false

        for i in (length(adj) + 1):numin(t)
            isunitspace(domain(t)[i]) || return false
        end
    end
    for edge in edges(network)
        # `reverse` is the identity on an `UndirectedEdge`, so orient it to compare the two
        # sides; `virtualspace` dualizes, hence the non-dual side is the *larger* vertex
        e = DirectedEdge(edge)
        isdual(virtualspace(network, e)) || return false
        virtualspace(network, e) == dual(virtualspace(network, reverse(e))) || return false
    end
    return true
end

# --- iteration accessors ------------------------------------------------------
"""
    vertices(network)

Return the collection of vertices of `network`.
"""
vertices(network::AbstractTensorNetwork) = keys(network.vertices)

"""
    edges(network)

Return the set of undirected edges of `network` in canonical orientation
(`first(e) < last(e)`).
"""
function edges(network::AbstractTensorNetwork)
    es = Indices{UndirectedEdge{keytype(network)}}()
    for v₁ in vertices(network), v₂ in neighbors(network, v₁)
        v₁ < v₂ && insert!(es, UndirectedEdge(v₁, v₂))
    end
    return es
end

"""
    neighbors(network, v)

Return the list of neighbors of vertex `v`, in the order matching the virtual legs of `network[v]`.
"""
neighbors(network::AbstractTensorNetwork, v) = network.adjacency[v]

"""
    has_vertex(network, v) -> Bool

Return `true` if `v` is a vertex of `network`.
"""
has_vertex(network::AbstractTensorNetwork, v) = haskey(network.adjacency, v)

"""
    has_edge(network, u, v) -> Bool
    has_edge(network, e::UndirectedEdge) -> Bool

Return `true` if `network` has an edge between `u` and `v` (equivalently, if
`v` is a neighbor of `u`).
"""
function has_edge(network::AbstractTensorNetwork, u, v)
    return haskey(network.adjacency, u) && v in network.adjacency[u]
end
has_edge(network::AbstractTensorNetwork, e::UndirectedEdge) =
    has_edge(network, first(e), last(e))

"""
    degree(network, v) -> Int

Return the number of neighbors of vertex `v` in `network`, i.e. its local
coordination number. This equals `length(neighbors(network, v))`.
"""
degree(network::AbstractTensorNetwork, v) = length(neighbors(network, v))

"""
    incoming_edges(network, site; exclude=()) -> generator

Iterator over the directed edges `DirectedEdge(n, site)` for every neighbor
`n` of `site` in `network`, optionally skipping any neighbor present in
`exclude` (which must be an iterable container of vertex tokens — pass
`(other_vertex,)` to skip a single neighbor).

The result is a lazy generator suitable for `attach_messages`, `map`, and
`for` loops. Order matches `neighbors(network, site)`.
"""
function incoming_edges(network::AbstractTensorNetwork, site; exclude = ())
    return (DirectedEdge(n, site) for n in neighbors(network, site) if !(n in exclude))
end

"""
    outgoing_edges(network, site; exclude=()) -> generator

Iterator over the directed edges `DirectedEdge(site, n)` for every neighbor
`n` of `site` in `network`, optionally skipping any neighbor present in
`exclude`. The reverse of [`incoming_edges`](@ref); order matches
`neighbors(network, site)`. `collect` it to pass to the vector form of
[`compute_message`](@ref).
"""
function outgoing_edges(network::AbstractTensorNetwork, site; exclude = ())
    return (DirectedEdge(site, n) for n in neighbors(network, site) if !(n in exclude))
end

"""
    leg_index(network, edge) -> Int

Return the 1-based position of `edge` within `neighbors(network, first(edge))`, i.e. the
domain-leg index occupied by `edge` in `network[first(edge)]`. The corresponding *tensor
slot* is `leg_index(network, edge) + num_physical(network)`.
Throws `ArgumentError` if `edge` is not incident on `first(edge)`.
"""
function leg_index(network::AbstractTensorNetwork, edge::AbstractEdge)
    idx = findfirst(==(last(edge)), neighbors(network, first(edge)))
    isnothing(idx) && throw(ArgumentError(lazy"edge $edge does not exist"))
    return idx
end

# --- pretty printing ---------------------------------------------------------
function Base.show(io::IO, ::MIME"text/plain", network::AbstractTensorNetwork)
    summary(io, network)
    print(io, " with ", length(network), " vertices:")
    indent = '\t'
    inner = IOContext(io, :typeinfo => eltype(network))
    for v in vertices(network)
        println(io)
        println(io)
        print(io, " vertex ", v, ":")
        T = network[v]
        np = numout(T)
        d = length(neighbors(network, v))
        for k in numin(T):-1:(d + 1)
            T = removeunit(T, np + k)
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
