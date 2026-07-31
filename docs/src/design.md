# Tensor Network Library — Data Structure Design

A TensorKit.jl-based library for tensor network states on arbitrary graphs, targeting belief-propagation and simple-update algorithms.

## Overview

The library is built in two pieces:

1. **Tensor network state** — a struct that directly holds a per-vertex adjacency list (in canonical leg order) and a per-vertex `TensorMap` (one physical leg + a padded fixed number of virtual legs). Undirected edges and bond spaces are derived on demand.
2. **BP messages** — a flat dictionary keyed by directed edge, holding a 1↔1 message tensor on the receiver's side of each bond. Kept separate from the state because messages are algorithm state, not physical state.

There is no separate generic graph type. The state's per-vertex adjacency list lives directly on `TensorNetworkState` — that is what gives O(degree) adjacency — and a data-agnostic graph layer cannot expose this without leaking concrete details. We avoid maintaining a parallel adjacency structure by inlining the dictionaries directly on each concrete type.

## Conventions

- **Vertex keys** are arbitrary user-chosen types `V` (typically `Int`); they are used directly as keys into `Dictionaries.jl` containers.
- **Containers are static.** No structural mutation of the vertex/adjacency dictionaries after construction; tensor values can be updated freely.
- **Edge orientation in the state** is fixed by vertex ordering: for undirected edge `(u, v)` we store the key with `u ≤ v`. The smaller-keyed endpoint is the source.
- **Codomain/domain split.** Every vertex tensor has its physical legs in the codomain and a fixed number `N` of virtual legs in the domain, where `N` is the maximum coordination of the graph. A [`TensorNetworkState`](@ref) has **one** physical leg, a [`TensorNetworkOperator`](@ref) has **two**; virtual leg `k` therefore sits at tensor slot `k + numout`. Virtual legs beyond the vertex's actual degree are padded with `oneunit(S)` and contribute trivially to contractions.
- **Duality.** Virtual spaces are stored once per undirected edge as non-dual `S` values. For an edge `e = (u, v)` (canonical, `u < v`) with stored space `V_e`, the `u`-side carries `V_e` and the `v`-side carries `dual(V_e)`. The constructor enforces non-dual input; `check_consistency` verifies the duality invariant across the whole network. This duality convention is also what makes the fermionic signs consistent — see [Fermionic correctness](@ref).
- **No multigraphs, no self-loops.**

## Graph primitives

`TensorNetworkState` and `BPMessages` share the same edge-key shapes and adjacency conventions, captured here once.

### Edge types

```julia
abstract type AbstractEdge{V} end

struct UndirectedEdge{V} <: AbstractEdge{V}
    src::V    # by convention, src ≤ dst
    dst::V
end

struct DirectedEdge{V} <: AbstractEdge{V}
    src::V
    dst::V
end

```

A canonical-form constructor for `UndirectedEdge` enforces `src ≤ dst`.
`DirectedEdge` preserves order as given.
Both types support `first(e)` and `last(e)` (the source and destination), `Tuple(e)`, and `reverse(e)` — for `UndirectedEdge`, `reverse` is the identity. `==` compares the underlying tuple, so directed and undirected edges with the same endpoints are still distinct types.

`adjacency(edges::Indices{UndirectedEdge{V}})` builds the per-vertex neighbor lists in one pass; this is how the state constructor turns an edge dictionary into a `Dictionary{V, Vector{V}}` adjacency map.

### Sparse adjacency via `Dictionaries.jl`

There are no integer tokens in the public surface — vertex keys `V` and edge keys (`UndirectedEdge{V}` or `DirectedEdge{V}`) are used directly. `Dictionaries.jl` gives O(1) `getindex` and `haskey`, so the dictionary keying *is* the sparse adjacency / incidence structure; no separate adjacency or incidence matrix is stored.

## Tensor network state

### Per-vertex data

`TensorNetworkState` stores two parallel dictionaries keyed by vertex `V`:

- `adjacency[v]` — the ordered neighbor list of `v`. Its position is canonical: domain leg `k` of `vertices[v]` connects to `adjacency[v][k]`. So `adjacency` simultaneously serves as the neighbor list and the leg-index map.
- `vertices[v]` — the on-site `TensorMap` with shape `P_v ← V_1 ⊗ ... ⊗ V_N`.

There is no separate `edges` field. The undirected edge set is derived on demand from `adjacency` via `Graphs.edges(state)`.

### State type

```julia
const StateTensor{T <: Number, S <: IndexSpace, N, A <: DenseVector{T}} =
    TensorMap{T, S, 1, N, A}

struct TensorNetworkState{T <: Number, S <: IndexSpace, N, A <: DenseVector{T}, V}
    adjacency::Dictionary{V, Vector{V}}
    vertices::Dictionary{V, StateTensor{T, S, N, A}}
end
```

`N` is the *maximum* coordination of the graph and is fixed for the whole network — every tensor has exactly `N` domain legs, with the trailing `N - length(adjacency[v])` legs padded with `oneunit(S)`. This keeps tensor types uniform across vertices (and therefore inference-friendly) even on graphs with non-uniform degrees. Bond spaces may differ from edge to edge, allowing inhomogeneous bond dimensions.

### Graphs.jl interop

`TensorNetworkState` does not subtype `Graphs.AbstractGraph`, but it implements the relevant interface methods directly: `Graphs.vertices(state)` returns the vertex keys; `Graphs.edges(state)` returns the canonical (`first < last`) undirected edge set as an `Indices{UndirectedEdge{V}}`.

### Helpers

```julia
state[v]                   # on-site TensorMap at vertex v
length(state)              # number of vertices
physicalspace(state, v)    # codomain (physical) space of state[v]
virtualspace(state, e)     # virtual space of `e` as seen from `first(e)`; works for
                           # UndirectedEdge or DirectedEdge
neighbors(state, v)        # ordered neighbor list, matching the domain legs of state[v]
leg_index(state, e)        # 1-based domain-leg position of DirectedEdge `e` within
                           # neighbors(state, first(e))
check_consistency(state)   # verify duality invariant and unit padding of trailing legs
TensorMap(state)           # dense materialization (exponentially expensive)
```

## BP messages

A flat dictionary keyed by `DirectedEdge{V}`, with two entries per undirected state edge:

```julia
const MessageTensor{T <: Number, S <: IndexSpace, A <: DenseVector{T}} =
    TensorMap{T, S, 1, 1, A}

struct BPMessages{T <: Number, S <: IndexSpace, A <: DenseVector{T}, V}
    messages::Dictionary{DirectedEdge{V}, MessageTensor{T, S, A}}
end
```

There is no explicit vertex field — the vertex set is induced by the directed-edge keys, which in turn come from the state. The full geometric convention is documented at the top of `src/messages.jl`; the structural facts are repeated here.

### Message types and spaces

A directed edge `e = (s, r)` is read as *sender → receiver*. The message `msgs[e]` lives on the receiver's side of the underlying bond and has space `V_r ← V_r`, where `V_r = virtualspace(state, DirectedEdge(r, s))`. Concretely, for canonical undirected edge `(u, v)` with stored space `V_e` (so `u < v` and `dual(V_e)` sits on the `v`-side):

- the `u → v` message has space `dual(V_e) ← dual(V_e)`;
- the `v → u` message has space `V_e ← V_e`.

The codomain (bra layer) and domain (ket layer) carry the same vector space; `attach_messages` contracts the domain against `state[r]`'s ket virtual leg and exposes the codomain in its place.

### Identity initialization

`BPMessages(state)` populates every directed edge with the identity on the receiver's side. This is the standard BP initialization and is *exact* on trees — a single sweep along the tree reaches the fixed point.

### Compatibility with a state

Messages reference state vertex keys and edge endpoints, not state tensors. This means messages compose freely with tensor updates (tensors change, messages stay valid as long as bond spaces don't). `check_consistency(state, msgs)` verifies that every undirected state edge has both directed messages present and that each message space matches the corresponding receiver-side bond.

## Construction ergonomics

The primary constructor takes per-vertex physical spaces and per-edge virtual spaces as dictionaries and allocates uninitialized tensors:

```julia
TensorNetworkState{T, S, N, A, V}(undef,
                                  pspaces::Dictionary{V, S},
                                  vspaces::Dictionary{UndirectedEdge{V}, S})
```

Most type parameters can be omitted and are inferred or defaulted: `T = Float64`, `S` from the input spaces, `N` from the maximum coordination of the inferred adjacency, `A = Vector{T}`, and `V` from the dictionary key types. The shortest form is `TensorNetworkState(undef, pspaces, vspaces)`.

The constructor:
1. Builds the adjacency `Dictionary{V, Vector{V}}` from `keys(vspaces)`, choosing each vertex's neighbor ordering as the edge keys are encountered.
2. Checks that `keys(pspaces) == keys(adjacency)` and that all `vspaces` values are non-dual.
3. For each vertex `v`, allocates a `StateTensor` with codomain `pspaces[v]` and domain `V_1 ⊗ ... ⊗ V_N`, where the first `length(adjacency[v])` factors are the (possibly dualized) `vspaces` entries and the rest are `oneunit(S)`.

Tensors are then filled by the user, typically with `Random.randn!(state)` or `Random.rand!(state)`, both of which delegate to the underlying TensorKit tensors.

The constructor stress-tests the data structure: anything awkward to build here points at a design problem.

### Entry points

Everything else funnels into that constructor. `randn_state` and `rand_state` allocate and fill in one call; `product_state` builds a state that is a product over its vertices. Each accepts a *topology* — a vector of `UndirectedEdge`s, as produced by `square_lattice`, `triangular_lattice` and `hexagonal_lattice`, or a `Graphs.AbstractGraph` — in place of the dictionary pair. See [Constructing states](@ref) for the full interface.

Two design decisions are worth recording here.

**Product-state bond charges are deduced, not supplied.** A nontrivial-charge site state cannot be written as a bare ket `P ← oneunit`, since that map only reaches the trivial sector; the 1-dimensional bonds must therefore carry whatever charge makes each on-site tensor consistent. For an abelian symmetry this is a linear system over the symmetry group, solved on a breadth-first spanning tree per connected component, and solvable exactly when each component is charge-neutral. This is what confines `product_state` to abelian sectortypes: the solve relies on `⊗` having a unique outcome.

**The charge bath is an ordinary vertex, not a tracked one.** A closed network of charge-conserving tensors has trivial total charge, so a state in a nontrivial global sector needs one auxiliary vertex carrying the compensating charge on a 1-dimensional bond. That vertex is deliberately *not* recorded in the struct: it appears in `vertices(state)` and `TensorMap(state)` like any other, and callers exclude it from gate lists and observables themselves. Tracking it would add a field and touch every place that iterates vertices, for a construction-time concern — see the open questions below. It attaches to a minimum-degree vertex so that it does not raise `N` and thereby give every tensor in the network an extra `oneunit`-padded leg.

## Tensor network operator

`TensorNetworkOperator` holds **two** physical legs per site, both in the codomain, as the
vectorization of a linear map: slot 1 is the ket (row) index carrying `P_v` and slot 2 the bra
(column) index carrying `dual(P_v)`. Everything else — adjacency, bond duality, `oneunit`
padding — is shared with `TensorNetworkState` via the `AbstractTensorNetwork` supertype, and
[`Canopy.num_physical`](@ref) supplies the one number that differs. See
[Operators and density matrices](@ref) for the full interface.

`AbstractTensorNetwork` is deliberately a *closed* internal supertype over the two
representations this package owns. It is not the data-agnostic graph layer rejected above: it
makes no promise to outside subtypes, and every method on it reaches directly into the
`adjacency`/`vertices` fields.

### Belief propagation reuses the state path verbatim

Keeping both physical legs in the codomain means fusing them is a pure reinterpretation of the
same storage — the fusion trees of `P₁ ⊗ P₂ → c` enumerate exactly the degeneracy basis of
`fuse(P₁, P₂)` in sector `c`, in the same order, and `unitary(fuse ← P₁⊗P₂)` is block-wise the
identity. So `TensorNetworkState(op)` shares `data` with `op`, and because BP closes every
physical leg between ket and bra at the same vertex, the fused state's messages are *identical*
to the operator's. The message kernels, schedules and `belief_propagation` are untouched.

**This is a deliberate dependence on TensorKit's internal block layout**, not on its documented
contract. It is load-bearing, and the price is paid explicitly: `test/test_operators.jl` asserts
`_fuse_physical(t) ≈ unitary(fuse ← codomain(t)) * t` and `.data === t.data` across `Trivial`,
`U₁`, `SU₂` and `fℤ₂`, so a layout change in TensorKit fails there loudly rather than corrupting
BP silently. The alternative — threading a physical-leg offset through every message kernel —
was rejected as the worse trade.

Note what BP on a two-leg network computes: the environment of `tr(ρ†ρ)`, not `tr(ρ)`. A true
trace is a *single-layer* contraction needing vector-valued messages, i.e. a different
algorithm.

### Gate application

The only genuinely new machinery. The `GateAction` enum — `LeftAction`, `RightAction`,
`SandwichAction`, passed as the `action` keyword of `apply!` and defaulting to the two-sided one
— records which physical slots a gate acts on; those slots are the single parameter the two-site
kernel needs. The action is union-split by hand into three calls, each passing a literal slot
tuple and its own `θ` contraction as a `do` block, so the kernel's leg bookkeeping — all of it
derived from `length(acted)` — infers as if the action were static.
Legs the gate does not touch stay in the QR environment, so a one-sided operator gate costs
exactly what the same gate costs on a state — only `SandwichAction` pays `d²`.

The duality convention removes all transposes: a dual codomain leg contracts directly against a
non-dual gate leg, so right-multiplication just consumes the gate's *codomain* instead of its
domain.

## Open questions / future work

- **Fermionic operator networks.** `apply!` on a `TensorNetworkOperator` throws for fermionic
  sectortypes. The operator path adds braid sites the state analysis does not cover — the extra
  codomain leg crossing the QR/SVD partition, and the bra-side gate contraction — and the open
  question is whether right-multiplication on a dual codomain leg needs a compensating `twist!`.
  The type, its constructors and the fused view are already fermion-safe.
- **Single-layer BP** for a true `trace(ρ)` and for operator expectation values
  `tr(ρO)/tr(ρ)`: vector-valued messages and a separate fixed-point iteration.
- **`product_operator`** with per-site local operators, and `|ψ⟩⟨ψ|` from a state (needs `A ⊗ Ā`
  with fused bonds, i.e. `χ²`).
- **Message bond dimensions** independent of state bond dimensions (for loop-corrected BP, boundary-MPS environments). The current shape supports this — `BPMessages` is parameterized independently — but the constructor and `check_consistency` check will need refinement.
- **Auxiliary-vertex tracking** so that observables, gate lists and Trotter schemes can skip charge-bath sites automatically instead of by caller convention. Would add a field to the struct and touch every place that iterates vertices.
- **Non-abelian product states**, which need degeneracy and multiplicity indices to be distinguished in the local-state specification, and a fusion-tree solve rather than the current unique-fusion one.
- **Mutable graphs** for coarse-graining / RG workflows. Out of scope for v1; would force `adjacency` and `vertices` to support structural updates.
- **Adjacency matrix cache** as a derived field for spectral graph operations. Add only if profiling justifies.
- **More than two outer legs per site**, with symbolic naming. Two are supported (see [Tensor network operator](@ref)); a third would want the leg count as a type parameter rather than a third struct.
