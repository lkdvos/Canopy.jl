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

See [`AbstractTensorNetwork`](@ref) for the shared interface, and
[`TensorNetworkOperator`](@ref) for the two-physical-leg counterpart.
"""
struct TensorNetworkState{T <: Number, S <: IndexSpace, N, A <: DenseVector{T}, V} <:
    AbstractTensorNetwork{T, S, N, A, V}
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
        Vs = _padded_virtualspaces(S, Val(N), vertex, neighbors, vspaces)
        return StateTensor{T, S, N, A}(undef, pspaces[vertex] ← ⊗(Vs...))
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
num_physical(::Type{<:TensorNetworkState}) = 1

Base.eltype(::Type{<:TensorNetworkState{T, S, N, A}}) where {T, S, N, A} = StateTensor{T, S, N, A}

Adapt.adapt_structure(to, state::TensorNetworkState) =
    TensorNetworkState(state.adjacency, map(adapt(to), state.vertices))

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
