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

# Properties
# ----------
Base.eltype(::Type{TensorNetworkState{T, S, N, A, V}}) where {T, S, N, A, V} = TensorMap{T, S, 1, N, A}

Base.keytype(state::TensorNetworkState) = keytype(typeof(state))
Base.keytype(::Type{TensorNetworkState{T, S, N, A, V}}) where {T, S, N, A, V} = V

VectorInterface.scalartype(::Type{T}) where {T <: TensorNetworkState} = scalartype(eltype(T))
TensorKit.storagetype(::Type{T}) where {T <: TensorNetworkState} = storagetype(eltype(T))
TensorKit.spacetype(::Type{T}) where {T <: TensorNetworkState} = spacetype(eltype(T))

"""
    physicalspace(state, vertex) -> S

Return the physical (codomain) space of the on-site tensor at `vertex`.
"""
physicalspace(state::TensorNetworkState, vertex) = space(state[vertex], 1)

"""
    virtualspace(state, edge) -> S

Return the virtual space of `edge` as seen from `first(edge)`.

By convention `virtualspace(state, edge) == dual(virtualspace(state, reverse(edge)))`:
the non-dual side is the one with the smaller vertex.
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

- every tensor has no more virtual legs occupied than its vertex has neighbors, and any extra domain legs are unit spaces
- every edge's virtual space is non-dual on the canonical (`first < last`) side and the dual of its reverse.
"""
function check_consistency(state::TensorNetworkState)
    for v in vertices(state)
        t = state[v]
        adj = neighbors(state, v)
        numin(t) <= length(adj) || return false

        for i in (length(adj) + 1):numin(t)
            @assert isunitspace(domain(t)[i])
        end

    end
    for edge in edges(state)
        !isdual(virtualspace(state, edge)) || return false
        virtualspace(state, edge) == dual(virtualspace(state, reverse(edge))) || return false
    end
    return true
end

# --- iteration accessors ------------------------------------------------------
"""
    vertices(state)

Return the collection of vertices of `state`.
"""
Graphs.vertices(state::TensorNetworkState) = keys(state.vertices)

"""
    edges(state)

Return the set of undirected edges of `state` in canonical orientation
(`first(e) < last(e)`).
"""
function Graphs.edges(state::TensorNetworkState)
    es = Indices{UndirectedEdge{keytype(state)}}()
    for v₁ in Graphs.vertices(state), v₂ in neighbors(state, v₁)
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
    leg_index(state, edge) -> Int

Return the 1-based position of `edge` within `neighbors(state, first(edge))`, i.e. the domain-leg index occupied by `edge` in `state[first(edge)]`.
Throws `ArgumentError` if `edge` is not incident on `first(edge)`.
"""
function leg_index(state::TensorNetworkState, edge::DirectedEdge)
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

    for (i, v) in enumerate(Graphs.vertices(state))
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
