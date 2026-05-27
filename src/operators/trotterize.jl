"""
    TrotterScheme

Supertype for product-formula splittings used by [`trotterize`](@ref).
"""
abstract type TrotterScheme end

"""
    Strang(coloring=nothing)

Symmetric second-order (Strang) splitting. `coloring` is an iterable of independent edge classes (no two edges in a class sharing a vertex).
When `nothing`, a greedy first-fit coloring is computed from the input edges.
"""
struct Strang{C} <: TrotterScheme
    coloring::C
end
Strang() = Strang(nothing)

"""
    edge_coloring(edges) -> Vector{Vector{eltype(edges)}}

Greedy first-fit edge coloring of an iterable of [`UndirectedEdge`](@ref)s.
Returns a vector of colour classes, each an independent set of edges.
"""
function edge_coloring(edges)
    edges_list = collect(edges)
    nE = length(edges_list)
    colors = fill(0, nE)
    for i in 1:nE
        e = edges_list[i]
        used = Set{Int}()
        for j in 1:nE
            (j == i || colors[j] == 0) && continue
            e2 = edges_list[j]
            if e.src == e2.src || e.src == e2.dst ||
                    e.dst == e2.src || e.dst == e2.dst
                push!(used, colors[j])
            end
        end
        c = 1
        while c in used
            c += 1
        end
        colors[i] = c
    end
    K = isempty(colors) ? 0 : maximum(colors)
    classes = [eltype(edges_list)[] for _ in 1:K]
    for i in 1:nE
        push!(classes[colors[i]], edges_list[i])
    end
    return classes
end

"""
    trotterize(bond_hams, dτ, alg::Strang) -> Circuit

Build a [`Circuit`](@ref) implementing one Strang step of size `dτ` for the
sum of two-site Hamiltonians `bond_hams::AbstractDict{<:UndirectedEdge, <:AbstractTensorMap}`.

For `K = length(coloring)` colour classes the returned circuit holds the symmetric sequence

    [CG(½, class₁), …, CG(½, class_{K-1}), CG(1, class_K),
     CG(½, class_{K-1}), …, CG(½, class₁)]

where `CG(τ, class)` is the [`CompositeGate`](@ref) of `exp(-τ·dτ·h_e)` over the edges `e` in `class`.
Half-step layers are reused in reverse.
"""
function trotterize(
        bond_hams::AbstractDict{<:UndirectedEdge, <:AbstractTensorMap},
        dτ::Real, alg::Strang,
    )
    coloring = isnothing(alg.coloring) ? edge_coloring(keys(bond_hams)) : alg.coloring
    build_layer(class, dτ_eff) = CompositeGate(
        [LocalGate((e.src, e.dst), exp(-dτ_eff * bond_hams[e])) for e in class]
    )
    K = length(coloring)
    K == 1 && return Circuit([build_layer(coloring[1], dτ)])
    half_layers = [build_layer(coloring[k], dτ / 2) for k in 1:(K - 1)]
    full_layer = build_layer(coloring[K], dτ)
    return Circuit([half_layers; full_layer; reverse(half_layers)])
end
