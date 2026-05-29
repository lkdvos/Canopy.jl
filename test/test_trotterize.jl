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


@testset "trotterize Strang produces symmetric K-1, K, K-1 layer structure" begin
    P = ComplexSpace(2)
    h = let m = randn(ComplexF64, P ⊗ P, P ⊗ P); (m + m') / 2 end

    # Path graph → 2 classes → expected layer count 1 + 1 + 1 = 3.
    es_path = [UndirectedEdge(src(e), dst(e)) for e in edges(path_graph(6))]
    bond_hams = Dict(e => h for e in es_path)
    circ = trotterize(bond_hams, 0.05, Strang())
    @test circ isa Circuit
    @test length(circ.gatelist) == 3

    # Odd cycle → 3 classes → expected layer count 2 + 1 + 2 = 5.
    es_odd = [UndirectedEdge(src(e), dst(e)) for e in edges(cycle_graph(5))]
    bond_hams = Dict(e => h for e in es_odd)
    circ = trotterize(bond_hams, 0.05, Strang())
    @test length(circ.gatelist) == 5
end
