abstract type AbstractEdge{V} end

struct UndirectedEdge{V} <: AbstractEdge{V}
    src::V
    dst::V
    UndirectedEdge{V}(u::V, v::V) where {V} = ifelse(isless(v, u), new{V}(v, u), new{V}(u, v))
end
UndirectedEdge(u::V, v::V) where {V} = UndirectedEdge{V}(u, v)
UndirectedEdge(u, v) = UndirectedEdge(promote(u, v)...)

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
