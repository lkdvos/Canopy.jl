# Constructing states

There are three ways to get a [`TensorNetworkState`](@ref): allocate it uninitialized and
fill the tensors yourself, draw its entries at random, or build a product state from
per-vertex local states. All of them share the same two ingredients — a *topology* and a
choice of *spaces* — so this page introduces those first.

## Topologies

A topology is either a vector of `UndirectedEdge`s or a `Graphs.AbstractGraph`. The built-in
lattice constructors return edge vectors:

```julia
using Canopy

square_lattice(3, 4)                        # open boundaries
square_lattice(3, 4; periodic = (false, true))
triangular_lattice(3, 3)
hexagonal_lattice(2, 2)                     # (i, j, s) vertex tokens, s the sublattice
```

Vertex tokens are whatever the topology uses: `Int` for a `Graphs.jl` graph, `NTuple{2,Int}`
for the square and triangular lattices, `NTuple{3,Int}` for the hexagonal one. Any type works
as long as it supports `isless` and `promote` — the ordering is what fixes the bond-duality
and fermion-ordering conventions (see [Fermionic correctness](@ref)).

To recover the vertices an edge vector spans, use `vertices`:

```julia
es = square_lattice(2, 3)
verts = vertices(es)                        # Vector, in first-encountered order
```

That order is the one a state built on `es` iterates its vertices in, which makes it safe to
key dictionaries with:

```julia
using Dictionaries
pspaces = Dictionary(verts, [something(v) for v in verts])
```

Only vertices touched by an edge are represented, so an edge vector cannot express an
isolated vertex; a `Graphs.AbstractGraph` can.

## Spaces

Every vertex carries a physical space, and every edge a virtual space. Both are given either
per vertex/edge as dictionaries, or uniformly as a single space each:

```julia
using TensorKit

P = ComplexSpace(2)                         # physical
V = ComplexSpace(4)                         # virtual
```

Virtual spaces are stored once per undirected edge and must be passed **non-dual**. For a
canonical edge `(u, v)` with `u < v`, the `u`-side carries `V_e` and the `v`-side
`dual(V_e)`, which is what makes the two sides of every bond a matched contraction pair.

Each on-site tensor has one physical leg in the codomain and exactly `N` virtual legs in the
domain, where `N` is the *maximum* coordination number over the whole graph; a vertex with
fewer neighbors has its trailing legs padded with `oneunit(S)`. `check_consistency(state)`
verifies both invariants.

## Uninitialized and random states

```julia
state = TensorNetworkState{ComplexF64}(undef, es, P, V)
Random.randn!(state)
```

`randn_state` and `rand_state` do this in one call, and additionally accept an `rng` and the
`(pspaces, vspaces)` dictionary pair. Both `rng` and the scalar type are optional:

```julia
randn_state(es, P, V)                              # Float64
randn_state(ComplexF64, es, P, V)
randn_state(MersenneTwister(1234), ComplexF64, es, P, V)
randn_state(ComplexF64, cycle_graph(8), P, V)      # graph topology
randn_state(ComplexF64, pspaces, vspaces)          # per-vertex/per-edge spaces
```

The remaining type parameters of `TensorNetworkState` are inferred or defaulted: the space
type `S` from the input spaces, `N` from the maximum coordination, the storage type
`A = Vector{T}`, and the vertex token type `V` from the keys.

For a symmetric state the *content* of the virtual space — how the bond dimension is split
across sectors — is yours to choose:

```julia
V = Vect[fℤ₂](0 => cld(Dmax, 2), 1 => fld(Dmax, 2))
```

## Product states

A product state is specified by its topology, its physical spaces, and one local state per
vertex. `pspaces` and `localstates` are each independently either a `Dictionary` keyed by
vertex or a single value applied to every vertex:

```julia
product_state(es, ComplexSpace(2), [0.3, 0.4])              # uniform everywhere
product_state(es, P, Dictionary(verts, mystates))           # uniform space, per-vertex states
product_state(ComplexF64, es, pspaces, localstates)         # explicit scalar type
```

A local state is one of three forms:

```julia
Trivial() => [0.3, 0.4]     # a sector and a coefficient per degeneracy of that sector in P
[0.3, 0.4]                  # bare vector — only when sectortype(P) === Trivial
fℤ₂(1)                      # bare sector — only when it is 1-dimensional in P
1                           # anything convertible to a sector, e.g. for U1Irrep
```

Coefficients are used exactly as given: the result is **not** normalized.

The bare-vector form is restricted to `Trivial` on purpose. Under a symmetry, a vector
spanning several sectors has no definite charge, and there would be no correct bond charge to
assign to it — so the sector must be named explicitly.

### Why the bond charges are deduced

Bonds of a product state are 1-dimensional, and their charges are *solved for* rather than
supplied. The reason is that a nontrivial-charge site state cannot be written as a bare ket
`P ← oneunit`: that map only reaches the trivial sector. The bonds therefore have to carry
whatever charge makes each on-site tensor consistent. For an abelian symmetry the constraint
at every vertex `v` is

```
c_v = ⊗_{neighbors w} σ(v, w) q_e,     σ(v, w) = identity if v < w else dual
```

with `q_e` stored on the canonical (smaller-vertex) side of each edge. Canopy solves this by
taking a breadth-first spanning tree of each connected component and fixing the tree-edge
charges leaves→root, leaving loop edges trivial. The system is solvable exactly when each
component is charge-neutral, `⊗ c_v == one`, which is why an unbalanced pattern throws unless
you opt in to `total_charge` below.

Only abelian symmetries are supported; a non-abelian `sectortype` throws, since the solve
relies on `⊗` having a unique outcome.

## Nontrivial total charge

A closed network of charge-conserving tensors has trivial total charge — contracting every
bond leaves a vector in `⊗_v P_v` whose charge is the unit. So a state at, say, fixed nonzero
particle number is not representable as-is, whether it is a product state or a random one.

Passing `total_charge` makes it representable by attaching a *charge bath*: one auxiliary
vertex whose 1-dimensional physical space carries `dual(total_charge)`, joined to the lattice
by a 1-dimensional bond. The real vertices then fuse to `total_charge`.

```julia
Q = sum(occupation(v) for v in verts)
state = product_state(
    ComplexF64, es, fermion_space(U1Irrep),
    Dictionary(verts, [fℤ₂(occupation(v)) ⊠ U1Irrep(occupation(v)) for v in verts]);
    total_charge = fℤ₂(mod(Q, 2)) ⊠ U1Irrep(Q),
)
```

For a product state the local charges *determine* the total, so `total_charge` is checked
against them and must agree. For `randn_state`/`rand_state` there is nothing to check —
filling graded tensors at random already yields a trivial total charge, and the bath is
simply what selects a different global sector:

```julia
randn_state(ComplexF64, es, fermion_space(U1Irrep), V; total_charge = fℤ₂(1) ⊠ U1Irrep(3))
```

Three things to know about the bath:

- **It is an ordinary vertex.** It appears in `vertices(state)`, in `length(state)` and in
  `TensorMap(state)`, and it is *your* responsibility to leave it out of gate lists,
  Trotter schemes and observables. Building those over your own vertex list rather than over
  `vertices(state)` does this automatically.
- **It attaches to a minimum-degree vertex.** Attaching to an interior site of a
  coordination-3 lattice would raise the state's `N` from 3 to 4 and so give *every* tensor
  in the network an extra padded leg. Boundary sites usually have slack; on a fully periodic
  lattice all degrees are equal and `N` grows regardless.
- **Its label is derived from the vertex type** by `Canopy.auxiliary_vertex`, which has
  methods for integer and `NTuple{N, Int}` tokens and produces a label sorting before every
  existing vertex (`0`, `(0, 0)`, `(0, 0, 0)` for 1-based lattices). For any other vertex
  type, or to choose the label yourself, pass `auxiliary`.

The charge enters through a single 1-dimensional bond, so a random state built this way is
generic within the target sector only up to that bottleneck. Note also that the target charge
has to be able to *flow* through the virtual spaces you chose: if `V` carries no sectors that
can route it to the rest of the lattice, the result is a state whose tensors are all zero.
Giving `V` sectors of both signs avoids this.

## A worked symmetric example

Spinless fermions at fixed particle number on an open honeycomb lattice, in a checkerboard
occupation pattern:

```julia
using Canopy, TensorKit, Dictionaries
using TensorKitTensors.FermionOperators: fermion_space, f_num

es = hexagonal_lattice(4, 6)
verts = sort(vertices(es))
occ(v) = (sum(v) % 4 == 0) ? 0 : 1

P = fermion_space(U1Irrep)                  # sectors fℤ₂(n) ⊠ U1Irrep(n)
localstates = Dictionary(verts, [fℤ₂(occ(v)) ⊠ U1Irrep(occ(v)) for v in verts])
Q = sum(occ, verts)

state = product_state(
    ComplexF64, es, P, localstates; total_charge = fℤ₂(mod(Q, 2)) ⊠ U1Irrep(Q)
)

msgs = belief_propagation(BPMessages(state), state; maxiter = 50)
n = sum(expect(state, msgs, f_num(ComplexF64, U1Irrep), (v,)) for v in verts)   # == Q
```

Belief propagation is exact on a product state regardless of loops, since 1-dimensional bonds
make every message rank-1 — so the occupations come out exactly right here. Note the `(v,)`:
with tuple-labeled vertices, `expect(state, msgs, op, (i, j, s))` would be read as a
*multi-site* request, so a single vertex must be wrapped.

## API reference

```@docs
Canopy.UndirectedEdge
Canopy.DirectedEdge
vertices
TensorNetworkState
product_state
randn_state
rand_state
square_lattice
triangular_lattice
hexagonal_lattice
Canopy.auxiliary_vertex
physicalspace
virtualspace
Canopy.check_consistency
```
