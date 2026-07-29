# Real-time Hubbard quench on Canopy.jl — physics layer
# ======================================================
#
# Everything model-specific for a spinful Hubbard quench evolved by belief propagation +
# bond truncation. The CLI driver (`run.jl`) owns all I/O; this module owns the physics.
#
#     H = t Σ_<ij>,σ (c†_iσ c_jσ + h.c.)  +  U Σ_i n_i↑ n_i↓
#
# The central design choice is that a **quench pattern is symmetry-independent
# combinatorics**: a `Dict{V,Symbol}` over `:emp / :up / :dn / :updn`. The mapping onto
# graded-space sectors is a separate layer ([`localstate`](@ref)), so the tensor-network
# state, the startup self-check and the free-fermion reference all read the *same* pattern.
# That is what makes a symmetry sweep trustworthy: only one small table changes.

module HubbardQuench

using Canopy
using Canopy: belief_propagation, UndirectedEdge, DirectedEdge, tr_distance,
    recompute_message, DefaultBackend, default_allocator
using TensorKit
using TensorKitTensors.HubbardOperators: hubbard_space, u_num, d_num, e_num, ud_num,
    half_ud_num, e_hopping
using MatrixAlgebraKit: truncrank, trunctol
using Dictionaries: Dictionary
using LinearAlgebra: Diagonal, diag, transpose

export Lattice, lattice, quench_pattern, graph_center, bulk_region
export sectortypes, operators, localstate, occupation, build_state, check_initial_state
export build_layers, truncation, default_cutoff, bp_schedule, bp_residual, maxvirtualdim,
    bond_sector_structure
export measure, SiteObservables, free_fermion_reference

# ---------------------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------------------

const SECTOR_NAMES = ("trivial", "u1")

"""
    sectortypes(particle::AbstractString, spin::AbstractString) -> (P, S)

Map CLI symmetry names onto TensorKit sector types. Only abelian symmetries are usable:
`product_state` throws for non-abelian sectortypes, and several Hubbard operators are not
implemented under SU(2) particle symmetry. That is no physical loss here — an AFM/CDW
product state breaks spin-rotation symmetry, so it cannot live in an SU(2) sector anyway.
"""
function sectortypes(particle::AbstractString, spin::AbstractString)
    lookup(name, which) =
        name == "trivial" ? Trivial :
        name == "u1" ? U1Irrep :
        name in ("su2", "su2irrep") ? throw(
            ArgumentError(
                "SU(2) $which symmetry is not supported: `product_state` only handles abelian \
                 sectors, and an AFM/CDW product state breaks spin rotation regardless. \
                 Use one of $(SECTOR_NAMES)."
            )
        ) :
        throw(ArgumentError("unknown $which symmetry `$name`; expected one of $(SECTOR_NAMES)"))
    return lookup(particle, "particle"), lookup(spin, "spin")
end

"""
    operators([T = ComplexF64], P, S; interaction = :ud) -> NamedTuple

The operator set needed for the gates, the observables and the self-check. `interaction`
selects `U n↑n↓` (`:ud`) or the particle-hole-symmetric `U (n↑-½)(n↓-½)` (`:half_ud`);
at fixed particle number the two differ only by a constant and a chemical-potential term,
i.e. by a global phase, but the choice changes what `E_int` means.
"""
function operators(::Type{T}, P::Type{<:Sector}, S::Type{<:Sector}; interaction::Symbol = :ud) where {T}
    interaction in (:ud, :half_ud) ||
        throw(ArgumentError("unknown interaction `$interaction`; expected :ud or :half_ud"))
    docc = ud_num(T, P, S)
    # Share the tensor when the interaction *is* `ud_num`, so `measure` can skip the second
    # expectation value via an `===` identity check.
    int = interaction === :ud ? docc : half_ud_num(T, P, S)
    return (;
        nup = u_num(T, P, S),
        ndn = d_num(T, P, S),
        ntot = e_num(T, P, S),
        docc, int,
        hop = e_hopping(T, P, S),
        pspace = hubbard_space(P, S),
    )
end
operators(P::Type{<:Sector}, S::Type{<:Sector}; kwargs...) =
    operators(ComplexF64, P, S; kwargs...)

# ---------------------------------------------------------------------------------------
# Local states
# ---------------------------------------------------------------------------------------
#
# `hubbard_space` orders its four basis states |0>, |↑>, |↓>, |↑↓>, but *which* (sector,
# degeneracy index) each one lands on depends on the symmetry. This table is not documented
# upstream; it was derived from — and is re-verified by `test/runtests.jl` against — the
# diagonal blocks of `u_num` / `d_num` / `ud_num`, which are exactly the occupations of each
# basis state. A wrong degeneracy index silently yields a *different physical state*, so
# `check_initial_state` re-asserts it at every run rather than trusting this table.
#
# Specifications are always returned in the fully explicit `sector => coefficients` form
# accepted by `product_state` (`Canopy/src/states/charges.jl`), including for 1-dimensional
# sectors where a bare sector would also do — uniformity keeps the table readable.

"""
    occupation(which::Symbol) -> (n↑, n↓)

Spin occupations of a local basis state, independent of symmetry. The single source of
truth shared by the self-check and the free-fermion reference.
"""
function occupation(which::Symbol)
    which === :emp && return (0, 0)
    which === :up && return (1, 0)
    which === :dn && return (0, 1)
    which === :updn && return (1, 1)
    throw(ArgumentError("unknown local state `$which`; expected :emp, :up, :dn or :updn"))
end

const LOCAL_STATES = Symbol[:emp, :up, :dn, :updn]

"""
    localstate(P, S, which::Symbol) -> Pair{<:Sector, Vector{Float64}}

The `sector => coefficients` specification of local basis state `which` in
`hubbard_space(P, S)`, ready to hand to `product_state`.
"""
function localstate(::Type{Trivial}, ::Type{Trivial}, which::Symbol)
    which === :emp && return fℤ₂(0) => [1.0, 0.0]
    which === :updn && return fℤ₂(0) => [0.0, 1.0]
    which === :up && return fℤ₂(1) => [1.0, 0.0]
    which === :dn && return fℤ₂(1) => [0.0, 1.0]
    return _badlocalstate(which)
end

function localstate(::Type{Trivial}, ::Type{U1Irrep}, which::Symbol)
    which === :emp && return (fℤ₂(0) ⊠ U1Irrep(0)) => [1.0, 0.0]
    which === :updn && return (fℤ₂(0) ⊠ U1Irrep(0)) => [0.0, 1.0]
    which === :up && return (fℤ₂(1) ⊠ U1Irrep(1 // 2)) => [1.0]
    which === :dn && return (fℤ₂(1) ⊠ U1Irrep(-1 // 2)) => [1.0]
    return _badlocalstate(which)
end

function localstate(::Type{U1Irrep}, ::Type{Trivial}, which::Symbol)
    which === :emp && return (fℤ₂(0) ⊠ U1Irrep(0)) => [1.0]
    which === :up && return (fℤ₂(1) ⊠ U1Irrep(1)) => [1.0, 0.0]
    which === :dn && return (fℤ₂(1) ⊠ U1Irrep(1)) => [0.0, 1.0]
    which === :updn && return (fℤ₂(0) ⊠ U1Irrep(2)) => [1.0]
    return _badlocalstate(which)
end

function localstate(::Type{U1Irrep}, ::Type{U1Irrep}, which::Symbol)
    which === :emp && return (fℤ₂(0) ⊠ U1Irrep(0) ⊠ U1Irrep(0)) => [1.0]
    which === :up && return (fℤ₂(1) ⊠ U1Irrep(1) ⊠ U1Irrep(1 // 2)) => [1.0]
    which === :dn && return (fℤ₂(1) ⊠ U1Irrep(1) ⊠ U1Irrep(-1 // 2)) => [1.0]
    which === :updn && return (fℤ₂(0) ⊠ U1Irrep(2) ⊠ U1Irrep(0)) => [1.0]
    return _badlocalstate(which)
end

localstate(P::Type{<:Sector}, S::Type{<:Sector}, ::Symbol) =
    throw(ArgumentError("no local-state table for symmetry `($P, $S)`"))

_badlocalstate(which) =
    throw(ArgumentError("unknown local state `$which`; expected one of $(LOCAL_STATES)"))

# ---------------------------------------------------------------------------------------
# Lattice, sublattice sign, measurement regions
# ---------------------------------------------------------------------------------------

const LATTICE_NAMES = ("hex", "square")

"""
    Lattice

An edge list plus the derived bookkeeping the driver needs: sorted vertices, a plain
adjacency map, and the bipartite sublattice sign `ε_v ∈ {+1,-1}` that defines the staggered
order parameter.
"""
struct Lattice{V}
    kind::Symbol
    m::Int
    n::Int
    periodic::NTuple{2, Bool}
    edges::Vector{UndirectedEdge{V}}
    verts::Vector{V}
    adj::Dict{V, Vector{V}}
    sublattice::Dict{V, Int}
end

Base.length(lat::Lattice) = length(lat.verts)

"""
    lattice(kind, m, n; periodic = (false, false)) -> Lattice

Build a lattice from Canopy's generators. `kind` is `"hex"` (honeycomb, `2mn` sites, vertex
labels `(i,j,s)` with the sublattice in `s`) or `"square"` (`mn` sites, labels `(i,j)`).

`triangular_lattice` is deliberately not offered: it is not bipartite, so the staggered
order parameter this driver measures would be meaningless on it.

Note that `hexagonal_lattice(m, n)` has `2mn` sites, whereas the TNQS reference script's
`named_hexagonal_lattice_graph(5, 5)` has ~70. Choose `m, n` by target *site count* rather
than by copying `(5, 5)` — e.g. `(6, 6)` gives 72 sites.
"""
function lattice(kind::AbstractString, m::Int, n::Int; periodic::NTuple{2, Bool} = (false, false))
    if kind == "hex"
        es = hexagonal_lattice(m, n; periodic)
        sign_of = v -> isone(v[3]) ? 1 : -1
    elseif kind == "square"
        es = square_lattice(m, n; periodic)
        sign_of = v -> iseven(v[1] + v[2]) ? 1 : -1
    elseif kind in ("triangular", "tri")
        throw(
            ArgumentError(
                "the triangular lattice is not bipartite, so the staggered order parameter \
                 measured here is not defined on it; use one of $(LATTICE_NAMES)"
            )
        )
    else
        throw(ArgumentError("unknown lattice `$kind`; expected one of $(LATTICE_NAMES)"))
    end

    verts = sort(Canopy.vertices(es))
    V = eltype(verts)
    adj = Dict{V, Vector{V}}(v => V[] for v in verts)
    for e in es
        push!(adj[e.src], e.dst)
        push!(adj[e.dst], e.src)
    end
    for v in verts
        sort!(adj[v])           # deterministic neighbour choice for the doublon quench
    end
    sublattice = Dict{V, Int}(v => sign_of(v) for v in verts)

    # Periodic wrapping can close odd cycles and destroy bipartiteness, which would silently
    # corrupt the order parameter. Check the assignment directly against every edge.
    for e in es
        sublattice[e.src] == -sublattice[e.dst] || throw(
            ArgumentError(
                "lattice $kind($m,$n) with periodic=$periodic is not bipartite (edge \
                 $(e.src)–$(e.dst) joins one sublattice to itself); the staggered order \
                 parameter is undefined. Use open boundaries or an even extent."
            )
        )
    end

    return Lattice(Symbol(kind), m, n, periodic, es, verts, adj, sublattice)
end

# Multi-source BFS; returns distances (typemax(Int) for unreachable).
function _bfs_distances(lat::Lattice{V}, sources) where {V}
    dist = Dict{V, Int}(v => typemax(Int) for v in lat.verts)
    frontier = V[]
    for s in sources
        dist[s] = 0
        push!(frontier, s)
    end
    while !isempty(frontier)
        next = V[]
        for v in frontier, w in lat.adj[v]
            if dist[w] == typemax(Int)
                dist[w] = dist[v] + 1
                push!(next, w)
            end
        end
        frontier = next
    end
    return dist
end

"""
    graph_center(lat) -> Vector{V}

Vertices of minimum eccentricity, sorted. Canopy has no equivalent of NamedGraphs' `center`,
which the TNQS reference script uses both to pick the doublon site and to select a
boundary-free measurement region. O(N²) — negligible at the sizes reachable here.
"""
function graph_center(lat::Lattice)
    ecc = Dict(v => maximum(values(_bfs_distances(lat, (v,)))) for v in lat.verts)
    best = minimum(values(ecc))
    return sort([v for v in lat.verts if ecc[v] == best])
end

"""
    bulk_region(lat, depth) -> Vector{V}

Vertices at graph distance `≥ depth` from the open boundary, where the boundary is the set
of under-coordinated vertices. `depth = 0` returns every vertex; a fully periodic lattice
has no boundary and so is entirely bulk.

More tunable than [`graph_center`](@ref), and it degrades gracefully on cylinders and tori.
"""
function bulk_region(lat::Lattice, depth::Int)
    depth >= 0 || throw(ArgumentError("bulk depth must be ≥ 0, got $depth"))
    depth == 0 && return copy(lat.verts)
    Δ = maximum(length(lat.adj[v]) for v in lat.verts)
    boundary = [v for v in lat.verts if length(lat.adj[v]) < Δ]
    isempty(boundary) && return copy(lat.verts)
    dist = _bfs_distances(lat, boundary)
    region = sort([v for v in lat.verts if dist[v] >= depth])
    isempty(region) && throw(
        ArgumentError(
            "bulk depth $depth leaves no sites on $(lat.kind)($(lat.m),$(lat.n)); \
             lower --bulk-depth or enlarge the lattice"
        )
    )
    return region
end

# ---------------------------------------------------------------------------------------
# Quench patterns
# ---------------------------------------------------------------------------------------

const QUENCH_NAMES = ("cdw", "doublon")

"""
    quench_pattern(kind, lat) -> Dict{V,Symbol}

The initial product state as symmetry-independent local-state labels.

- `"cdw"`: the global AFM / spin-density wave, `:up` on one sublattice and `:dn` on the
  other. (The TNQS script calls this a CDW; the staggered quantity that is actually
  measured is `⟨n↑ − n↓⟩`.)
- `"doublon"`: the same background with a doublon `:updn` on a central site and a hole
  `:emp` on one of its neighbours. Both sites are chosen deterministically (graph centre,
  then smallest-labelled neighbour) so runs are reproducible.
"""
function quench_pattern(kind::AbstractString, lat::Lattice{V}) where {V}
    pat = Dict{V, Symbol}(v => (lat.sublattice[v] > 0 ? :up : :dn) for v in lat.verts)
    kind == "cdw" && return pat
    if kind == "doublon"
        vc = first(graph_center(lat))
        vn = first(lat.adj[vc])
        pat[vc] = :updn
        pat[vn] = :emp
        return pat
    end
    throw(ArgumentError("unknown quench `$kind`; expected one of $(QUENCH_NAMES)"))
end

# ---------------------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------------------

"""
    build_state([T = ComplexF64], lat, pat, P, S) -> (state, total_charge, bath_attached)

Build the initial product state. The per-site charges are fused to get the total charge and
`total_charge` is passed to `product_state` **only when it is non-trivial** — Canopy throws
both when a non-neutral state omits it and when a neutral state supplies it.

When it is non-trivial (particle-U(1) at half filling has `Q = N ≠ 0`, whereas a balanced
AFM has `Sz = 0` and needs nothing) Canopy attaches a *charge-bath vertex*. That is an
ordinary extra vertex: it shows up in `vertices(state)`, `length(state)` and
`edges(state)`. Everything downstream must therefore iterate `lat.verts` / `lat.edges`,
never the state's own vertex or edge sets. `bath_attached` is returned so the driver can
record it.
"""
function build_state(
        ::Type{T}, lat::Lattice, pat::Dict, P::Type{<:Sector}, S::Type{<:Sector}
    ) where {T}
    pspace = hubbard_space(P, S)
    I = sectortype(pspace)
    specs = [localstate(P, S, pat[v]) for v in lat.verts]
    ls = Dictionary(lat.verts, specs)

    q = foldl((a, b) -> only(a ⊗ b), (first(s) for s in specs); init = one(I))
    total_charge = q == one(I) ? nothing : q

    state = product_state(T, lat.edges, pspace, ls; total_charge)
    bath_attached = length(Canopy.vertices(state)) > length(lat.verts)
    return state, q, bath_attached
end
build_state(lat::Lattice, pat::Dict, P::Type{<:Sector}, S::Type{<:Sector}) =
    build_state(ComplexF64, lat, pat, P, S)

"""
    check_initial_state(state, msgs, lat, pat, ops; atol = 1e-8) -> NamedTuple

Assert that the constructed state really is the intended product state: per site,
`⟨n↑⟩`, `⟨n↓⟩` and `⟨n↑n↓⟩` must match [`occupation`](@ref) of `pat[v]`.

This is the guard on the [`localstate`](@ref) table. A wrong degeneracy index produces a
perfectly valid state of the *wrong* physical configuration, which no amount of downstream
inspection would reveal, so this runs on every invocation rather than only in the tests.
"""
function check_initial_state(state, msgs, lat::Lattice, pat::Dict, ops; atol::Real = 1.0e-8)
    bad = String[]
    nup_tot = 0.0
    ndn_tot = 0.0
    for v in lat.verts
        ρ = reduced_density_matrix((v,), state, msgs)
        got = (real(tr(ops.nup * ρ)), real(tr(ops.ndn * ρ)), real(tr(ops.docc * ρ)))
        nu, nd = occupation(pat[v])
        want = (float(nu), float(nd), float(nu * nd))
        if any(abs.(got .- want) .> atol)
            length(bad) < 5 && push!(
                bad,
                "  $v ($(pat[v])): (n↑,n↓,n↑n↓) = $(round.(got; digits = 8)), expected $want",
            )
        end
        nup_tot += got[1]
        ndn_tot += got[2]
    end
    isempty(bad) || error(
        "initial state does not match the requested quench pattern — the `localstate` \
         table is wrong for this symmetry:\n" * join(bad, "\n")
    )
    return (; nup_total = nup_tot, ndn_total = ndn_tot)
end

# ---------------------------------------------------------------------------------------
# Trotter layers (second order)
# ---------------------------------------------------------------------------------------

"""
    build_layers(lat, ops, U, t, dt) -> (single, hoplayers, ncolors)

Second-order symmetric Strang splitting of `H = H_U + Σ_c H_c` over the `K` edge-colour
classes:

    single(dt/2)
    hop_1(dt/2) … hop_{K-1}(dt/2)  hop_K(dt)  hop_{K-1}(dt/2) … hop_1(dt/2)
    single(dt/2)

mirroring the `2K-1` layer structure of Canopy's `Strang`, which cannot be reused directly
because `trotterize` only builds imaginary-time gates `exp(-dτ h)`.

`hoplayers` is returned as a plain `Vector{CompositeGate}` rather than a `Circuit` on
purpose: `apply!(…, ::Circuit)` reports only the *maximum* truncation error over its layers,
while applying each class separately gives both the max and the sum per step.

`single` is `nothing` at `U = 0`, where the interaction layer is the identity.
"""
function build_layers(lat::Lattice, ops, U::Real, t::Real, dt::Real)
    classes = edge_coloring(lat.edges)
    K = length(classes)

    single = if iszero(U)
        nothing
    else
        g = exp(-im * (U * dt / 2) * ops.int)
        CompositeGate([LocalGate((v,), g) for v in lat.verts])
    end

    g_half = exp(-im * (t * dt / 2) * ops.hop)
    g_full = exp(-im * (t * dt) * ops.hop)
    classgate(class, g) = CompositeGate([LocalGate((e.src, e.dst), g) for e in class])

    hoplayers = CompositeGate[]
    for k in 1:(K - 1)
        push!(hoplayers, classgate(classes[k], g_half))
    end
    push!(hoplayers, classgate(classes[K], g_full))
    for k in (K - 1):-1:1
        push!(hoplayers, classgate(classes[k], g_half))
    end

    return single, hoplayers, K
end

"""
    default_cutoff([T = ComplexF64]) -> Float64

Safe default discard threshold: two orders of magnitude above Canopy's gauge tolerance.

**This is not a cosmetic default.** `apply!` gauges the bond in and out through the
pseudo-inverse of the BP message spectrum, flooring eigenvalues at
`default_gauge_tol(T) = eps(real(T))^(3/4) ≈ 1.8e-12`. A rank-only truncation (`cutoff = 0`)
keeps numerically-null Schmidt directions once `χ` exceeds the rank the state actually needs,
and the gauge inverse then amplifies noise in those directions catastrophically.

The failure mode is nasty because it is silent and *anti-convergent*: the reported truncation
error `info.ϵ` keeps shrinking (→ 1e-10) while the observables go wrong, and the damage grows
with `χ`. Measured on a 4-site ring at `T = 0.3`, error in the staggered order parameter
against the exact free-fermion result:

    cutoff    χ=8       χ=16      χ=32      χ=64
    0         1.0e-02   1.1e-02   6.6e-03   3.5e-01   ← blows up
    1e-12     1.0e-02   1.1e-02   1.0e-02   3.6e-01   ← still blows up (≈ gauge_tol)
    1e-10     1.0e-02   1.1e-02   1.0e-02   1.0e-02   ← stable, χ-independent

The residual 1.0e-2 is the genuine BP loop error on that ring, not truncation.

Note that the TNQS reference script's `cutoff = 1e-14` is *below* this floor and so removes
nothing — it must not be carried over to Canopy unchanged.
"""
default_cutoff(::Type{T} = ComplexF64) where {T <: Number} =
    100 * Canopy.default_gauge_tol(T[])

"""
    truncation(χ, cutoff; elt = ComplexF64)

The MatrixAlgebraKit truncation strategy for `apply!`. Canopy has no `maxdim`/`cutoff`
keywords of its own; `trunc` is forwarded straight to `svd_trunc!`.

Warns when `cutoff` sits at or below the gauge tolerance — see [`default_cutoff`](@ref) for
why that silently corrupts large-`χ` runs.
"""
function truncation(χ::Int, cutoff::Real; elt::Type{<:Number} = ComplexF64)
    χ > 0 || throw(ArgumentError("χ must be positive, got $χ"))
    floor_ = Canopy.default_gauge_tol(elt[])
    if cutoff < 10 * floor_
        @warn """
        Truncation cutoff is at or below Canopy's gauge tolerance; large-χ runs can go \
        silently wrong while the reported truncation error keeps shrinking.
        """ cutoff gauge_tol = floor_ recommended = default_cutoff(elt)
    end
    return iszero(cutoff) ? truncrank(χ) : truncrank(χ) & trunctol(; atol = cutoff)
end

# ---------------------------------------------------------------------------------------
# Belief propagation helpers
# ---------------------------------------------------------------------------------------

const SCHEDULE_NAMES = ("sync", "spanningtree", "residual", "splash")

"""
    bp_schedule(name; batchsize = 8, height = 2) -> BPSchedule

Note that `ResidualSchedule`'s residual is an input-change surrogate, so its `tol` lives on
a different scale from the other schedules' — do not compare convergence thresholds across
schedules.
"""
function bp_schedule(name::AbstractString; batchsize::Int = 8, height::Int = 2)
    name == "sync" && return SynchronousSchedule()
    name == "spanningtree" && return SpanningTreeSchedule()
    name == "residual" && return ResidualSchedule(GreedySampler(batchsize))
    name == "splash" && return ResidualSplashSchedule(; height)
    throw(ArgumentError("unknown BP schedule `$name`; expected one of $(SCHEDULE_NAMES)"))
end

"""
    bp_residual(msgs, state, lat) -> Float64

Maximum single-sweep change in any message, i.e. how converged BP actually is.

`belief_propagation` returns only the new messages — no iteration count and no residual — so
silently saturating `maxiter` every step is invisible without this. It uses
`recompute_message` (not the raw `compute_message`) and `is_hermitian = true`, matching what
`update_message!` does internally, so the number lands on exactly the same scale as the
`tol` handed to `belief_propagation`. Costs one extra sweep (~3% at `maxiter = 30`).

Ranges over `lat.edges`, so a charge-bath bond is excluded; its message is 1-dimensional and
static, hence trivially converged.
"""
function bp_residual(msgs, state, lat::Lattice)
    backend = DefaultBackend()
    allocator = default_allocator(state)
    res = 0.0
    for e in lat.edges, de in (DirectedEdge(e.src, e.dst), DirectedEdge(e.dst, e.src))
        fresh = recompute_message(msgs, state, de, backend, allocator)
        res = max(res, tr_distance(msgs[de], fresh; is_hermitian = true))
    end
    return res
end

"""
    maxvirtualdim(state, lat) -> Int

Largest total bond dimension over the *physical* edges. Canopy has no helper for this, and
iterating the state's own edges would include a charge-bath bond.
"""
maxvirtualdim(state, lat::Lattice) =
    maximum(dim(virtualspace(state, e)) for e in lat.edges)

"""
    bond_sector_structure(state, lat) -> (nsectors, maxsecdim)

Sector count and largest per-sector dimension on the widest bond. At equal total `χ`,
symmetry only pays off once the per-sector blocks are large enough that block `gemm`
dominates the per-block bookkeeping, so these two numbers are what explain the walltime
comparison between symmetries.
"""
function bond_sector_structure(state, lat::Lattice)
    best = argmax(e -> dim(virtualspace(state, e)), lat.edges)
    V = virtualspace(state, best)
    secs = collect(sectors(V))
    return length(secs), maximum(c -> dim(V, c), secs; init = 0)
end

# ---------------------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------------------

"""
    SiteObservables

Per-site `⟨n↑⟩`, `⟨n↓⟩`, `⟨n↑n↓⟩`, in `lat.verts` order.
"""
struct SiteObservables{V}
    verts::Vector{V}
    nup::Vector{Float64}
    ndn::Vector{Float64}
    docc::Vector{Float64}
end

"""
    measure(state, msgs, lat, ops, region; t, U, energy = true) -> (scalars, site)

All observables for one measurement point.

The staggered order parameter over a region `R` is

    m_s(R) = (1/|R|) Σ_{v ∈ R} ε_v ⟨n_v↑ − n_v↓⟩

which is **exactly 1.0** for the perfect AFM product state by construction, whatever the
sublattice balance of `R`. (The TNQS reference script seeds its trajectory with `1.0` while
the `Sz`-based quantity it computes actually starts at `0.5`; defining the normalization
into the observable avoids inheriting that mismatch.)

Energies use `expect` on nearest-neighbour bonds. At `U ≠ 0` there is no exact reference,
so drift in `E_tot` is the only available signal that `χ` or BP has failed — it is not
monotone, because BP expectation values are not variational, but a growing drift is
diagnostic.

One 1-site reduced density matrix is built per site and reused across the operators rather
than calling `expect` once per (site, operator).
"""
function measure(
        state, msgs, lat::Lattice, ops, region::AbstractVector;
        t::Real, U::Real, energy::Bool = true
    )
    nsites = length(lat.verts)
    nup = Vector{Float64}(undef, nsites)
    ndn = Vector{Float64}(undef, nsites)
    docc = Vector{Float64}(undef, nsites)
    int_sum = 0.0
    same_int = ops.int === ops.docc

    for (k, v) in enumerate(lat.verts)
        ρ = reduced_density_matrix((v,), state, msgs)
        nup[k] = real(tr(ops.nup * ρ))
        ndn[k] = real(tr(ops.ndn * ρ))
        docc[k] = real(tr(ops.docc * ρ))
        energy && (int_sum += same_int ? docc[k] : real(tr(ops.int * ρ)))
    end

    index = Dict(v => k for (k, v) in enumerate(lat.verts))
    staggered(R) = sum(lat.sublattice[v] * (nup[index[v]] - ndn[index[v]]) for v in R) / length(R)

    E_kin = if energy
        t * sum(real(expect(state, msgs, ops.hop, e)) for e in lat.edges)
    else
        NaN
    end
    E_int = energy ? U * int_sum : NaN

    scalars = (;
        m_s_all = staggered(lat.verts),
        m_s_bulk = staggered(region),
        n_mean = sum(nup .+ ndn) / nsites,
        docc_mean = sum(docc) / nsites,
        E_kin, E_int, E_tot = E_kin + E_int,
    )
    return scalars, SiteObservables(lat.verts, nup, ndn, docc)
end

# ---------------------------------------------------------------------------------------
# Free-fermion reference (U = 0 only)
# ---------------------------------------------------------------------------------------
#
# At U = 0 the model is quadratic, so the exact trajectory follows from the single-particle
# correlation matrices C_σ[i,j] = <c†_iσ c_jσ>. Both spin sectors are tracked explicitly
# rather than assuming they are mirror images: that only holds for the CDW quench (by
# bipartite symmetry) and fails for the doublon quench.

apply_one_mode!(C, a, ph) = (@views(C[a, :] .*= conj(ph)); @views(C[:, a] .*= ph); C)

function apply_two_mode!(C, a, b, w)
    blk = [a, b]
    @views C[blk, :] .= conj(w) * C[blk, :]
    @views C[:, blk] .= C[:, blk] * transpose(w)
    return C
end

"""
    free_fermion_reference(lat, pat, t, dt, nsteps) -> (circuit, cont)

Two truncation-free reference trajectories of `m_s_all`, each of length `nsteps + 1`:

- `circuit` applies the **identical second-order Trotter circuit** used by the tensor-network
  run, so its residual against the simulation is *pure bond-truncation error*;
- `cont` applies `exp(-i h T)` in one shot (what the TNQS script does), so
  `circuit − cont` isolates the residual Trotter error at this `dt`.

Valid only at `U = 0`; the caller is responsible for that.
"""
function free_fermion_reference(lat::Lattice, pat::Dict, t::Real, dt::Real, nsteps::Int)
    verts = lat.verts
    N = length(verts)
    idx = Dict(v => k for (k, v) in enumerate(verts))
    ε = Float64[lat.sublattice[v] for v in verts]

    occ_up = Float64[occupation(pat[v])[1] for v in verts]
    occ_dn = Float64[occupation(pat[v])[2] for v in verts]
    order(Cu, Cd) = sum(ε .* (real.(diag(Cu)) .- real.(diag(Cd)))) / N

    # --- circuit-matched: same 2K-1 half/full-step colour-class sandwich as `build_layers`
    classes = edge_coloring(lat.edges)
    K = length(classes)
    w_half = exp(-im * (t * dt / 2) * ComplexF64[0 1; 1 0])
    w_full = exp(-im * (t * dt) * ComplexF64[0 1; 1 0])
    schedule = Tuple{Int, Matrix{ComplexF64}}[]
    for k in 1:(K - 1)
        push!(schedule, (k, w_half))
    end
    push!(schedule, (K, w_full))
    for k in (K - 1):-1:1
        push!(schedule, (k, w_half))
    end

    Cu = complex(Matrix(Diagonal(occ_up)))
    Cd = complex(Matrix(Diagonal(occ_dn)))
    circuit = zeros(Float64, nsteps + 1)
    circuit[1] = order(Cu, Cd)
    for step in 1:nsteps
        for (k, w) in schedule, e in classes[k]
            a, b = idx[e.src], idx[e.dst]
            apply_two_mode!(Cu, a, b, w)
            apply_two_mode!(Cd, a, b, w)
        end
        circuit[step + 1] = order(Cu, Cd)
    end

    # --- continuous: C(T) = conj(V) C₀ transpose(V) with V = exp(-i T h)
    h = zeros(ComplexF64, N, N)
    for e in lat.edges
        a, b = idx[e.src], idx[e.dst]
        h[a, b] = t
        h[b, a] = t
    end
    Cu0 = complex(Matrix(Diagonal(occ_up)))
    Cd0 = complex(Matrix(Diagonal(occ_dn)))
    cont = zeros(Float64, nsteps + 1)
    for step in 0:nsteps
        Vt = exp(-im * (step * dt) * h)
        cont[step + 1] = order(conj(Vt) * Cu0 * transpose(Vt), conj(Vt) * Cd0 * transpose(Vt))
    end

    return circuit, cont
end

end # module
