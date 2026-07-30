const OperatorTensor{T <: Number, S <: IndexSpace, N, A <: DenseVector{T}} =
    TensorMap{T, S, 2, N, A}

"""
    TensorNetworkOperator{T, S, N, A, V}

Tensor network operator on an undirected graph — a global operator or density matrix.
Each vertex holds a `TensorMap` with **two** physical legs in the codomain and up to `N`
virtual legs in the domain; unused virtual legs are padded with `oneunit(S)`.

The two physical legs are the *vectorization* of a linear map: slot 1 is the ket (row)
space `P` and slot 2 the bra (column) space `dual(P)`, so that

    op[v] :: P_v ⊗ dual(P_v) ← V_1 ⊗ … ⊗ V_N

Keeping both legs in the codomain is what makes an operator reusable by belief
propagation without moving data — see `TensorNetworkState(op)`.

Type parameters are as for [`TensorNetworkState`](@ref); see
[`AbstractTensorNetwork`](@ref) for the shared interface.

A tensor whose slot 2 is *not* `dual` of slot 1 is still a legal value of this type — it
is a purification (ancilla picture) rather than a vectorized operator. Two-sided gate
application requires the vectorized form; see [`isvectorized`](@ref Canopy.isvectorized).
"""
struct TensorNetworkOperator{T <: Number, S <: IndexSpace, N, A <: DenseVector{T}, V} <:
    AbstractTensorNetwork{T, S, N, A, V}
    adjacency::Dictionary{V, Vector{V}}
    vertices::Dictionary{V, OperatorTensor{T, S, N, A}}

    function TensorNetworkOperator{T, S, N, A, V}(
            adjacency::Dictionary{V, Vector{V}}, vertices::Dictionary{V, OperatorTensor{T, S, N, A}}
        ) where {T, S, A, N, V}
        @assert keys(adjacency) === keys(vertices) "adjacency and vertices should share keys"
        maximum(length, adjacency) <= N || throw(DimensionMismatch("non-matching coordination number"))
        return new{T, S, N, A, V}(adjacency, vertices)
    end
end

function TensorNetworkOperator(
        adjacency::Dictionary{V, Vector{V}}, vertices::Dictionary{V, OperatorTensor{T, S, N, A}}
    ) where {T, S, N, A, V}
    return TensorNetworkOperator{T, S, N, A, V}(adjacency, vertices)
end

# Constructors
# ------------
"""
    TensorNetworkOperator{T, S, N, A, V}(undef, pspaces, vspaces)
    TensorNetworkOperator{T, S, N, A}(undef, pspaces, vspaces)
    TensorNetworkOperator{T, S, N}(undef, pspaces, vspaces)
    TensorNetworkOperator{T, S}(undef, pspaces, vspaces)
    TensorNetworkOperator{T}(undef, pspaces, vspaces)
    TensorNetworkOperator(undef, pspaces, vspaces)

Allocate an uninitialized [`TensorNetworkOperator`](@ref) from per-vertex physical spaces
`pspaces::Dictionary{V, S}` and per-edge virtual spaces
`vspaces::Dictionary{UndirectedEdge{V}, S}`.

`pspaces[v]` gives the *ket* space `P_v`; the codomain of the on-site tensor is
`P_v ⊗ dual(P_v)`, so operators built this way are square. The type itself allows
`P_out ≠ P_in`, which a one-site [`LeftGate`](@ref) can produce.

Everything else matches the [`TensorNetworkState`](@ref) constructor: the adjacency is
inferred from the edge keys, `keys(pspaces)` must match the resulting vertex set,
`vspaces` must be non-dual and is dualised on the larger-vertex side of each edge, and
each tensor's domain is padded with `oneunit(S)` up to `N` legs.

Omitted type parameters take the same defaults as for [`TensorNetworkState`](@ref).
"""
function TensorNetworkOperator{T, S, N, A, V}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, N, A, V}
    adj = adjacency(keys(vspaces))
    issetequal(keys(pspaces), keys(adj)) || throw(ArgumentError("Incompatible vertices and edges"))

    @assert !any(isdual, vspaces) "TensorNetworkOperator expects non-dual virtual spaces"

    vertices = map(pairs(adj)) do (vertex, neighbors)
        P = pspaces[vertex]
        Vs = _padded_virtualspaces(S, Val(N), vertex, neighbors, vspaces)
        return OperatorTensor{T, S, N, A}(undef, (P ⊗ dual(P)) ← ⊗(Vs...))
    end

    return TensorNetworkOperator{T, S, N, A, V}(adj, vertices)
end

function TensorNetworkOperator{T, S, N, A}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, N, A, V}
    return TensorNetworkOperator{T, S, N, A, V}(undef, pspaces, vspaces)
end
function TensorNetworkOperator{T, S, N}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, N, V}
    return TensorNetworkOperator{T, S, N, Vector{T}}(undef, pspaces, vspaces)
end
function TensorNetworkOperator{T, S}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, V}
    N = maximum(length, adjacency(keys(vspaces)))
    return TensorNetworkOperator{T, S, N}(undef, pspaces, vspaces)
end
function TensorNetworkOperator{T}(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {T, S, V}
    return TensorNetworkOperator{T, S}(undef, pspaces, vspaces)
end
function TensorNetworkOperator(
        ::UndefInitializer, pspaces::Dictionary{V, S}, vspaces::Dictionary{UndirectedEdge{V}, S}
    ) where {V, S <: IndexSpace}
    return TensorNetworkOperator{Float64}(undef, pspaces, vspaces)
end

"""
    TensorNetworkOperator{T}(undef, topology, P::S)
    TensorNetworkOperator(undef, topology, P::S)

Convenience constructor with a uniform physical space `P` and uniform bond space `V` over a
`topology` — a vector of [`UndirectedEdge`](@ref)s or a `Graphs.AbstractGraph`. The on-site
codomain is `P ⊗ dual(P)`.
"""
function TensorNetworkOperator{T}(
        ::UndefInitializer, topology, P::S, V::S,
    ) where {T, S <: IndexSpace}
    return TensorNetworkOperator{T}(undef, _spaces(topology, P, V)...)
end
function TensorNetworkOperator(::UndefInitializer, topology, P::S, V::S) where {S <: IndexSpace}
    return TensorNetworkOperator{Float64}(undef, topology, P, V)
end

# --- random operators ---------------------------------------------------------
for (f, f!) in ((:rand_operator, :rand!), (:randn_operator, :randn!))
    @eval begin
        function $f(
                rng::Random.AbstractRNG, ::Type{T}, pspaces::Dictionary{V, S},
                vspaces::Dictionary{UndirectedEdge{V}, S},
            ) where {T, V, S <: IndexSpace}
            return Random.$f!(rng, TensorNetworkOperator{T}(undef, pspaces, vspaces))
        end
        function $f(
                rng::Random.AbstractRNG, ::Type{T}, topology, P::S, V::S
            ) where {T, S <: IndexSpace}
            return $f(rng, T, _spaces(topology, P, V)...)
        end

        $f(::Type{T}, pspaces::Dictionary, vspaces::Dictionary) where {T} =
            $f(Random.default_rng(), T, pspaces, vspaces)
        $f(::Type{T}, topology, P::IndexSpace, V::IndexSpace) where {T} =
            $f(Random.default_rng(), T, topology, P, V)

        $f(rng::Random.AbstractRNG, pspaces::Dictionary, vspaces::Dictionary) =
            $f(rng, Float64, pspaces, vspaces)
        $f(rng::Random.AbstractRNG, topology, P::IndexSpace, V::IndexSpace) =
            $f(rng, Float64, topology, P, V)

        $f(pspaces::Dictionary, vspaces::Dictionary) =
            $f(Random.default_rng(), Float64, pspaces, vspaces)
        $f(topology, P::IndexSpace, V::IndexSpace) =
            $f(Random.default_rng(), Float64, topology, P, V)
    end
end

@doc """
    randn_operator([rng], [T], pspaces, vspaces)
    randn_operator([rng], [T], topology, P::IndexSpace, V::IndexSpace)

Allocate a [`TensorNetworkOperator`](@ref) and fill it with normally distributed entries.
Spaces are given either per vertex and per edge (`pspaces`, `vspaces`, as for the `undef`
constructor) or uniformly as `P` and `V` over a `topology`.

The result is a generic two-physical-leg network: it is neither Hermitian nor positive,
and its trace is not normalized. For a physically meaningful density matrix start from
[`identity_operator`](@ref) and evolve it with [`SandwichGate`](@ref).

Unlike [`randn_state`](@ref) there is no `total_charge` keyword — the operator analogue of
a charge bath is a charge-*shifting* operator, which needs its own semantics.

See also [`rand_operator`](@ref), which draws uniformly instead.
""" randn_operator

@doc """
    rand_operator([rng], [T], pspaces, vspaces)
    rand_operator([rng], [T], topology, P::IndexSpace, V::IndexSpace)

As [`randn_operator`](@ref), but with uniformly distributed entries.
""" rand_operator

# --- identity operator --------------------------------------------------------
"""
    identity_operator([T], topology, pspaces)

The identity operator `𝟙` on `topology`, i.e. the β = 0 (infinite-temperature) density
matrix up to normalization. `topology` is a vector of [`UndirectedEdge`](@ref)s or a
`Graphs.AbstractGraph`; `pspaces` is the physical space of every vertex, either as a
`Dictionary` keyed by vertex or as a single space applied to all of them — the same
convention as [`product_state`](@ref). `T` defaults to `Float64`.

Every bond is `oneunit(S)`, so this is a product operator and needs no bond-charge solve:
the identity carries trivial total charge, unlike the nontrivially-charged local states
[`product_state`](@ref) has to accommodate. The 1-dimensional bonds also make belief
propagation *exact* here, since every message is rank 1.

The result is **not** trace-normalized: `tr(𝟙) = ∏_v dim(P_v)`.

```julia
ρ = identity_operator(ComplexF64, square_lattice(3, 3), ComplexSpace(2))
```
"""
function identity_operator(::Type{T}, topology, pspaces) where {T}
    es = _edgelist(topology)
    ps = _pervertex(pspaces, keys(adjacency(Indices(es))), "pspaces")
    S = valtype(ps)
    vspaces = Dictionary(es, fill(oneunit(S), length(es)))
    return _fill_identity!(TensorNetworkOperator{T}(undef, ps, vspaces), ps)
end
identity_operator(topology, pspaces) = identity_operator(Float64, topology, pspaces)

# Build `id(P)` in the vectorized layout `P ⊗ dual(P) ← oneunit ⊗ … ⊗ oneunit`, matching the
# duality of each already-allocated domain leg. `flip` on a unit leg is the 1×1 identity
# isomorphism, so it changes the space without touching the content.
function _fill_identity!(op::TensorNetworkOperator{T}, pspaces) where {T}
    for v in vertices(op)
        target = op[v]
        t = repartition(id(T, pspaces[v]), 2, 0)
        for _ in 1:numin(target)
            t = insertleftunit(t, numind(t) + 1)
        end
        for k in 1:numin(target)
            isdual(domain(target)[k]) && (t = flip(t, 2 + k))
        end
        space(t) == space(target) ||
            throw(SpaceMismatch(lazy"identity_operator built $(space(t)), expected $(space(target))"))
        op.vertices[v] = t
    end
    return op
end

# --- lift a state -------------------------------------------------------------
"""
    TensorNetworkOperator(state::TensorNetworkState)

View `state` as a [`TensorNetworkOperator`](@ref) by inserting a trivial (`oneunit`) bra
leg on every site, i.e. `|ψ⟩` regarded as a map `P ← oneunit`.

This is *not* the density matrix `|ψ⟩⟨ψ|`, which would need bond dimension `χ²`. It is
useful as a purification with a one-dimensional ancilla, and as a cross-check: because a
trivial second leg fuses to nothing, this operator's BP messages equal `state`'s own.
"""
function TensorNetworkOperator(state::TensorNetworkState)
    return TensorNetworkOperator(
        state.adjacency, map(t -> insertrightunit(t, 1), state.vertices)
    )
end

# Properties
# ----------
num_physical(::Type{<:TensorNetworkOperator}) = 2

Base.eltype(::Type{<:TensorNetworkOperator{T, S, N, A}}) where {T, S, N, A} = OperatorTensor{T, S, N, A}

Adapt.adapt_structure(to, op::TensorNetworkOperator) =
    TensorNetworkOperator(op.adjacency, map(adapt(to), op.vertices))

"""
    isvectorized(op::TensorNetworkOperator) -> Bool

Return `true` if every on-site tensor has `physicalspace(op, v, 2) == dual(physicalspace(op, v, 1))`,
i.e. if `op` is the vectorization of a *square* linear map.

This is the precondition for two-sided gate application ([`SandwichGate`](@ref)): only
then do the left action on slot 1 and the right action on slot 2 both typecheck. A network
that fails this is a perfectly good purification — one physical leg and one ancilla — and
supports [`LeftGate`](@ref) freely; it just has no `ρ ↦ GρG†`.
"""
isvectorized(op::TensorNetworkOperator) =
    all(v -> physicalspace(op, v, 2) == dual(physicalspace(op, v, 1)), vertices(op))

# --- the fused state view -----------------------------------------------------
# Fusing the two physical legs of a `TensorMap{T,S,2,N}` into one is a pure reindexing of
# the *same* storage: the fusion trees of `P₁ ⊗ P₂ → c` enumerate exactly the degeneracy
# basis of `fuse(P₁, P₂)` in sector `c`, in the same (sorted-`blocksectors`) order, and
# `unitary(fuse(P₁⊗P₂) ← P₁⊗P₂)` is block-wise the identity. So the fused tensor shares
# `data` with the original and copies nothing.
#
# This is a deliberate dependence on TensorKit's internal block layout — it is not part of
# TensorKit's documented contract. The payoff is that belief propagation needs no operator
# awareness whatsoever. `test/test_operators.jl` guards the assumption; if TensorKit ever
# changes its layout, that test fails loudly rather than BP silently going wrong.
_fuse_physical(t::OperatorTensor{T}) where {T} =
    TensorMap{T}(t.data, fuse(codomain(t)) ← domain(t))

"""
    TensorNetworkState(op::TensorNetworkOperator)

View `op` as a [`TensorNetworkState`](@ref) whose physical space is the *fusion* of `op`'s
two physical legs. Copies nothing — the returned state shares its tensor storage with
`op`.

This is how an operator reaches belief propagation: because the fusion map is unitary and
BP closes every physical leg between the ket and the bra at the same vertex, the messages
of the fused state are *identical* to the ones the operator itself defines. Every BP
entry point (`BPMessages`, `belief_propagation`) accepts a
[`TensorNetworkOperator`](@ref) and forwards through this view.

!!! warning
    The view aliases `op`'s storage, so mutating one mutates the other, and it goes
    **stale** as soon as `apply!` rebinds a vertex tensor. Construct it where it is used
    rather than caching it.

Note the physical space of the view is `fuse(P ⊗ dual(P))`, not `P`, so a
`reduced_density_matrix` taken on it is expressed in fused indices.
"""
TensorNetworkState(op::TensorNetworkOperator) =
    TensorNetworkState(op.adjacency, map(_fuse_physical, op.vertices))

# The BP entry points that forward through this view live with the functions they extend:
# `BPMessages(op)` and `check_consistency(op, msgs)` in `messages.jl`, and
# `belief_propagation(msgs, op)` in `beliefpropagation.jl`.

# --- dense materialization ----------------------------------------------------
"""
    TensorMap(op::TensorNetworkOperator) -> TensorMap

Materialize `op` as a dense many-body operator by contracting every virtual bond and
dropping the `oneunit`-padded domain legs of each on-site tensor.

The returned tensor has `2 * length(op)` codomain legs and no domain legs, ordered so that
`repartition(TensorMap(op), n, n)` with `n = length(op)` is the operator
`⊗_v P_v ← ⊗_v P_v`, with vertex `i` at slot `i` on both sides (`i` indexing
`vertices(op)`). Concretely: ket legs run forwards (`-1, …, -n`) and bra legs backwards
(`-2n, …, -(n+1)`), matching `repartition`'s cyclic leg order.

This is exponentially expensive in the number of vertices and is intended for small
systems (tests, debugging, comparison against exact diagonalization).
"""
function TensorKit.TensorMap(op::TensorNetworkOperator)
    V = keytype(typeof(op))
    n = length(op)
    tensors = AbstractTensorMap[]
    indices = Vector{Int}[]
    edge_label = Dict{UndirectedEdge{V}, Int}()
    next_label = Ref(0)

    for (i, v) in enumerate(vertices(op))
        T = op[v]
        adj = neighbors(op, v)
        d = length(adj)

        Tstripped = T
        for k in numin(T):-1:(d + 1)
            Tstripped = removeunit(Tstripped, 2 + k)
        end

        labels = Vector{Int}(undef, 2 + d)
        labels[1] = -i                  # ket  leg of vertex i
        labels[2] = -(2n + 1 - i)       # bra  leg of vertex i, counted from the far end
        for k in 1:d
            e = UndirectedEdge(v, adj[k])
            labels[k + 2] = get!(edge_label, e) do
                next_label[] += 1
                return next_label[]
            end
        end
        push!(tensors, Tstripped)
        push!(indices, labels)
    end

    return ncon(tensors, indices)
end
