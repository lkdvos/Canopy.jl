# Lattice edge constructors
# -------------------------
# Hand-rolled builders for the common 2D lattices, modeled on the networkx
# lattice-graph family. Each returns a `Vector{UndirectedEdge{V}}` with
# coordinate-tuple vertex labels `V`; the vertex set is recovered downstream via
# `adjacency`. Boundary conditions are selected per axis through
# `periodic::NTuple{2,Bool}` (open / cylinder / torus).


function _check_periodic(periodic::NTuple{2, Bool}, sizes::NTuple{2, Int}, mins::NTuple{2, Int}, name)
    for d in 1:2
        if periodic[d] && sizes[d] < mins[d]
            throw(ArgumentError(lazy"$name: periodic dimension $d requires size ≥ $(mins[d]), got $(sizes[d])"))
        end
    end
    return nothing
end

"""
    square_lattice(m, n; periodic=(false, false)) -> Vector{UndirectedEdge{NTuple{2,Int}}}

Edges of an `m × n` square lattice. Vertices are labeled `(i, j)` with
`i ∈ 1:m`, `j ∈ 1:n`; every interior vertex has coordination 4 (nearest
neighbors in both axes).

`periodic[1]` / `periodic[2]` wrap the first / second axis (open boundaries by
default), giving a plane, cylinder, or torus. A periodic axis must have size
`≥ 3` to avoid duplicated bonds.
"""
function square_lattice(m::Int, n::Int; periodic::NTuple{2, Bool} = (false, false))
    (m ≥ 1 && n ≥ 1) || throw(ArgumentError("square_lattice: dimensions must be ≥ 1"))
    _check_periodic(periodic, (m, n), (3, 3), "square_lattice")
    es = UndirectedEdge{NTuple{2, Int}}[]
    for i in 1:m, j in 1:n
        if j < n
            push!(es, UndirectedEdge((i, j), (i, j + 1)))
        elseif periodic[2]
            push!(es, UndirectedEdge((i, n), (i, 1)))
        end
        if i < m
            push!(es, UndirectedEdge((i, j), (i + 1, j)))
        elseif periodic[1]
            push!(es, UndirectedEdge((m, j), (1, j)))
        end
    end
    return es
end

"""
    triangular_lattice(m, n; periodic=(false, false)) -> Vector{UndirectedEdge{NTuple{2,Int}}}

Edges of an `m × n` triangular lattice. Vertices are labeled `(i, j)` with
`i ∈ 1:m`, `j ∈ 1:n`: a square grid augmented with one diagonal bond per cell
(`(i, j)`–`(i+1, j+1)`), so every interior vertex has coordination 6.

`periodic[1]` / `periodic[2]` wrap the first / second axis (plane / cylinder / torus).
A periodic axis must have size `≥ 3` to avoid duplicated bonds.
"""
function triangular_lattice(m::Int, n::Int; periodic::NTuple{2, Bool} = (false, false))
    (m ≥ 1 && n ≥ 1) || throw(ArgumentError("triangular_lattice: dimensions must be ≥ 1"))
    _check_periodic(periodic, (m, n), (3, 3), "triangular_lattice")
    es = UndirectedEdge{NTuple{2, Int}}[]
    for i in 1:m, j in 1:n
        ir = i < m ? i + 1 : (periodic[1] ? 1 : nothing)
        jr = j < n ? j + 1 : (periodic[2] ? 1 : nothing)
        isnothing(jr) || push!(es, UndirectedEdge((i, j), (i, jr)))       # horizontal
        isnothing(ir) || push!(es, UndirectedEdge((i, j), (ir, j)))       # vertical
        (isnothing(ir) || isnothing(jr)) || push!(es, UndirectedEdge((i, j), (ir, jr)))  # diagonal
    end
    return es
end

"""
    hexagonal_lattice(m, n; periodic=(false, false)) -> Vector{UndirectedEdge{NTuple{3,Int}}}

Edges of a honeycomb (hexagonal) lattice of `m × n` unit cells, each carrying two sublattice sites.
Vertices are labeled `(i, j, s)` with `i ∈ 1:m`, `j ∈ 1:n`, sublattice `s ∈ {1, 2}`.
Each `A = (i, j, 1)` bonds to `B`-sites `(i, j, 2)`, `(i+1, j, 2)`, `(i, j+1, 2)`, giving coordination 3 in the bulk.

`periodic[1]` / `periodic[2]` wrap the first / second axis (plane / cylinder / torus).
A periodic axis must have size `≥ 2` to avoid duplicated bonds.
"""
function hexagonal_lattice(m::Int, n::Int; periodic::NTuple{2, Bool} = (false, false))
    (m ≥ 1 && n ≥ 1) || throw(ArgumentError("hexagonal_lattice: dimensions must be ≥ 1"))
    _check_periodic(periodic, (m, n), (2, 2), "hexagonal_lattice")
    es = UndirectedEdge{NTuple{3, Int}}[]
    for i in 1:m, j in 1:n
        a = (i, j, 1)
        push!(es, UndirectedEdge(a, (i, j, 2)))
        if i < m
            push!(es, UndirectedEdge(a, (i + 1, j, 2)))
        elseif periodic[1]
            push!(es, UndirectedEdge(a, (1, j, 2)))
        end
        if j < n
            push!(es, UndirectedEdge(a, (i, j + 1, 2)))
        elseif periodic[2]
            push!(es, UndirectedEdge(a, (i, 1, 2)))
        end
    end
    return es
end
