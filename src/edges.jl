abstract type AbstractEdge{V} end

"""
    UndirectedEdge(u, v)

An undirected edge between vertex tokens `u` and `v`, stored in canonical orientation: the
constructor swaps its arguments so that `first(e) < last(e)` always holds, making
`UndirectedEdge(u, v) == UndirectedEdge(v, u)` and giving each edge a single identity as a
dictionary key.

That ordering also carries the conventions: virtual spaces are stored non-dual on the
smaller-vertex side, and the vertex order is the fermion order (see
[Fermionic correctness](@ref)). `reverse` is therefore the identity on an `UndirectedEdge`
— convert to a [`DirectedEdge`](@ref) to talk about one side of a bond.
"""
struct UndirectedEdge{V} <: AbstractEdge{V}
    src::V
    dst::V
    UndirectedEdge{V}(u::V, v::V) where {V} = ifelse(isless(v, u), new{V}(v, u), new{V}(u, v))
end
UndirectedEdge(u::V, v::V) where {V} = UndirectedEdge{V}(u, v)
UndirectedEdge(u, v) = UndirectedEdge(promote(u, v)...)

"""
    DirectedEdge(u, v)

An oriented edge from `u` to `v`, preserving its argument order. Used wherever a side of a
bond matters: BP messages are keyed by `DirectedEdge`, read as *sender → receiver*, and
`leg_index`/`virtualspace` locate a leg within `first(e)`'s tensor.

Converts to and from [`UndirectedEdge`](@ref).
"""
struct DirectedEdge{V} <: AbstractEdge{V}
    src::V
    dst::V
end
DirectedEdge(u, v) = DirectedEdge(promote(u, v)...)

Base.first(e::AbstractEdge) = e.src
Base.last(e::AbstractEdge) = e.dst

Base.Tuple(e::AbstractEdge) = (e.src, e.dst)

Base.convert(::Type{UndirectedEdge{V}}, x::Tuple{V, V}) where {V} = UndirectedEdge(x...)
Base.convert(::Type{DirectedEdge{V}}, x::Pair{V, V}) where {V} = DirectedEdge(x...)

DirectedEdge(e::UndirectedEdge) = DirectedEdge(first(e), last(e))
UndirectedEdge(e::DirectedEdge) = UndirectedEdge(first(e), last(e))

Base.convert(::Type{DirectedEdge{V}}, e::UndirectedEdge{V}) where {V} = DirectedEdge(e)
Base.convert(::Type{UndirectedEdge{V}}, e::DirectedEdge{V}) where {V} = UndirectedEdge(e)

Base.reverse(e::DirectedEdge) = DirectedEdge(e.dst, e.src)
Base.reverse(e::UndirectedEdge) = e

Base.:(==)(e1::E, e2::E) where {E <: AbstractEdge} = Tuple(e1) == Tuple(e2)

"""
    vertices(edges::AbstractVector{<:UndirectedEdge})

Return the vertex tokens spanned by `edges`, in first-encountered order — the same
order in which a state built on `edges` iterates its vertices. Useful to key the
dictionaries expected by the [`TensorNetworkState`](@ref) constructors, e.g.
`Dictionary(vertices(es), ...)`.
"""
vertices(edges::AbstractVector{<:UndirectedEdge}) = collect(keys(adjacency(Indices(edges))))

function adjacency(edges::Indices{UndirectedEdge{V}}) where {V}
    adj = Dictionary{V, Vector{V}}()
    for edge in edges
        vs = get!(() -> V[], adj, first(edge))
        push!(vs, last(edge))
        ws = get!(() -> V[], adj, last(edge))
        push!(ws, first(edge))
    end
    return adj
end
