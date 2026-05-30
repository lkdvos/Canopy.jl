using Canopy
using Canopy: UndirectedEdge, edge_coloring, trotterize, Strang, Circuit
using TensorKit
using Graphs: path_graph, cycle_graph, edges, src, dst
using Test


@testset "edge_coloring class counts" begin
    # A path needs 2 colours (alternating edges share a vertex).
    classes = edge_coloring([UndirectedEdge(src(e), dst(e)) for e in edges(path_graph(6))])
    @test length(classes) == 2

    # An even cycle is 2-edge-colourable.
    classes = edge_coloring([UndirectedEdge(src(e), dst(e)) for e in edges(cycle_graph(6))])
    @test length(classes) == 2

    # An odd cycle requires 3 colours.
    classes = edge_coloring([UndirectedEdge(src(e), dst(e)) for e in edges(cycle_graph(5))])
    @test length(classes) == 3

    # Each class is an independent set of edges (no vertex repeated within a class).
    for c in classes
        verts = Int[]
        for e in c
            append!(verts, (e.src, e.dst))
        end
        @test allunique(verts)
    end
end


@testset "trotterize Strang produces symmetric 2K-1 layer structure" begin
    P = ComplexSpace(2)
    h = let m = randn(ComplexF64, P ⊗ P, P ⊗ P); (m + m') / 2 end

    # `Dict` iteration order is hash-seed-dependent, which feeds into the
    # greedy `edge_coloring` and changes the resulting K. Assert the
    # structural relation `length(gatelist) == 2K - 1` against the actual K.
    for g in (path_graph(6), cycle_graph(5))
        es = [UndirectedEdge(src(e), dst(e)) for e in edges(g)]
        bond_hams = Dict(e => h for e in es)
        K = length(edge_coloring(keys(bond_hams)))
        circ = trotterize(bond_hams, 0.05, Strang())
        @test circ isa Circuit
        @test length(circ.gatelist) == 2K - 1
    end
end
