# Shared state-construction helpers for the BP benchmark suite.
#
# Assumes `Canopy`, `TensorKit`, `Graphs`, `Random`, and
# `AlgorithmsInterface as AI` are already brought into scope by the caller
# (see `benchmarks.jl`).
#
# BLAS is pinned to one thread here, matching
# `benchmark/realtime_timing/run_timings.jl:32` and `examples/make.jl:16`. The
# symmetric fixtures below contract many *small* blocks, so with multithreaded
# BLAS every timing depends on OpenBLAS's per-call threading heuristics and
# nothing is comparable across χ or across symmetries. Export
# `OPENBLAS_NUM_THREADS=1` in the run environment as well — the pin below only
# takes effect once this file is loaded.

using Canopy: randn_state, BPMessages, belief_propagation, hexagonal_lattice
using Canopy: SynchronousSchedule, SpanningTreeSchedule, ResidualSchedule,
    ResidualSplashSchedule, GreedySampler
using TensorKit: ComplexSpace, Vect, Z2Irrep, U1Irrep, SU2Irrep, fℤ₂, ⊠
using TensorKit: TO
import TensorKit
using Graphs: cycle_graph, grid
using LinearAlgebra: BLAS
using Random
import Bumper

BLAS.set_num_threads(1)

Random.seed!(0)

# Every `@benchmarkable` `setup` block below starts with `Random.seed!(BENCH_SEED)`.
#
# BenchmarkTools re-runs `setup` once per *sample*, and `randn_state` draws from the
# global RNG, so without an explicit seed each sample builds a **different random
# instance**. For the fixed-work groups that merely adds variance; for
# `SUITE["schedule"]`, whose measured quantity is time to a tolerance, different
# instances need different numbers of iterations, and the group's reported spread
# is then genuine instance-to-instance variance rather than measurement noise.
#
# MEASURED: without seeding, the noise floor — two runs of identical code judged
# against each other — was ±59% on `SUITE["schedule"]` and ±22% on
# `SUITE["allocator"]` (the `apply_gate` keys, whose SVD work depends on the
# instance), and `judge` flagged even the fully deterministic `:sync` keys. With
# seeding, every sample of a key does the same work and the floor drops to
# machine noise.
const BENCH_SEED = 0

# `CANOPY_BENCH_FULL=1` enables the expensive tail of every χ grid (χ = 64 in
# the message group, χ = 32 on the honeycomb schedule / sweep fixtures). The
# default suite targets ≤15 min single-threaded, the full suite ≤45 min.
const BENCH_FULL = get(ENV, "CANOPY_BENCH_FULL", "0") == "1"

# Allocators compared by the allocator benchmark group: plain heap allocation
# versus the production Bumper `ResizeBuffer`, which warms up to the peak
# intermediate size and reclaims temporaries across repeated contractions.
const BENCH_ALLOCATORS = (
    :default => TO.DefaultAllocator(),
    :bumper => Bumper.default_buffer(Bumper.ResizeBuffer),
)

# Symmetry sectors
# ----------------
# `BENCH_SPACES` mirrors `_MSG_SPACES` (`test/test_messages.jl:23-27`) but is
# χ-parametrised, so entries are comparable at equal *total* bond dimension:
# every `V(χ)` below satisfies `dim(V(χ)) == χ` exactly (asserted at load time).
# Physical spaces are built from TensorKit alone — `TensorKitTensors` is
# deliberately *not* a benchmark dependency, since the physical space content is
# irrelevant to `randn_state` cost.
#
# The entries separate the two axes along which symmetric blocks get harder:
#
#   * blocks *growing*     — `:z2` / `:fz2` keep 2 sectors at every χ, so the
#     per-sector degeneracy grows linearly and BLAS eventually dominates.
#   * blocks *multiplying* — `:fz2_u1` grows its sector count like `log2(χ)`, so
#     blocks stay small and per-block bookkeeping (`twist!`, subblock lookups,
#     LRU hits) dominates. This is the regime the project targets.
#
# `:fz2_u1` alone cannot separate them: from χ = 8 to 64 it moves sector count
# (7→13) and mean subblock size (1.6→234.7) *together*, so a threshold fitted
# against it is fitted against a mixed axis. `:fz2_u1_flat` is the control — the
# same symmetry group and the same total χ with the sector count pinned at 4, so
# on the honeycomb vertex `ntrees` stays 16 over the whole χ grid while `meanblk`
# grows 7.5→3840. It is in `BENCH_SPACES`, not census-only, precisely so that the
# separation exists in the *timing* data and not only in the census.

"""
    _peaked_dims(χ, nsec) -> Vector{Int}

Split a total dimension `χ` over `nsec` charge sectors with binomial
(≈Gaussian) weights. Exact — `sum(_peaked_dims(χ, nsec)) == χ` — and every entry
is `≥ 1`, so at small χ the split degrades to "one state per sector" rather than
silently dropping sectors, which would change the number of blocks being
measured. Largest-remainder apportionment on top of a floor of 1.
"""
function _peaked_dims(χ::Int, nsec::Int)
    nsec ≤ χ || throw(ArgumentError("cannot split χ = $χ over $nsec sectors"))
    w = [binomial(nsec - 1, i - 1) for i in 1:nsec]
    tot = sum(w)
    raw = [χ * wi / tot for wi in w]
    d = [max(1, floor(Int, r)) for r in raw]
    while sum(d) > χ                       # the floor of 1 can overshoot at small χ
        i = argmax(d)
        d[i] > 1 || break
        d[i] -= 1
    end
    ord = sort(collect(eachindex(d)); by = i -> -(raw[i] - floor(raw[i])))
    k = 0
    while sum(d) < χ
        d[ord[mod1(k += 1, nsec)]] += 1
    end
    return d
end

_trivial_virtual(χ::Int) = ComplexSpace(χ)
_z2_virtual(χ::Int) = Vect[Z2Irrep](0 => cld(χ, 2), 1 => fld(χ, 2))
_fz2_virtual(χ::Int) = Vect[fℤ₂](0 => cld(χ, 2), 1 => fld(χ, 2))

# U(1) charges `n ∈ -nmax:nmax` with fermion parity tied to `mod(n, 2)`. `nmax`
# grows like `log2(χ)`, so the *sector count* is itself a function of χ (7/9/11/13
# at χ = 8/16/32/64) — that is the axis this fixture exists to sweep. Capped at
# `(χ - 1) ÷ 2` so the floor-of-1 in `_peaked_dims` always fits inside χ.
_fz2u1_nmax(χ::Int) = max(1, min(floor(Int, log2(χ)), (χ - 1) ÷ 2))
_fz2u1_space(ns, d) =
    Vect[fℤ₂ ⊠ U1Irrep](fℤ₂(mod(n, 2)) ⊠ U1Irrep(n) => di for (n, di) in zip(ns, d))

function _fz2u1_virtual(χ::Int)
    ns = (-_fz2u1_nmax(χ)):_fz2u1_nmax(χ)
    return _fz2u1_space(ns, _peaked_dims(χ, length(ns)))
end

# Companion to `:fz2_u1`: the *same* symmetry group with the sector count pinned
# at 4, so growing χ grows the degeneracies instead of multiplying the sectors.
# This is the degeneracy-heavy regime the kernel profile in the project plan was
# taken on. `:z2` / `:fz2` also hold their block count fixed, but in a different
# symmetry group; this one does it *inside* `fℤ₂ ⊠ U1Irrep`, so it is the only
# fixture directly comparable to `:fz2_u1` — see the axis note above
# `BENCH_SPACES`.
const _FZ2U1_FLAT_NS = -2:1
_fz2u1_flat_virtual(χ::Int) =
    _fz2u1_space(_FZ2U1_FLAT_NS, _peaked_dims(χ, length(_FZ2U1_FLAT_NS)))

# Two-state physical space: one even/charge-0 and one odd/charge-1 state, i.e. a
# single spinless fermion mode. Shape-equivalent to
# `TensorKitTensors.FermionOperators.fermion_space(U1Irrep)` without the dep.
const _P_FZ2U1 = _fz2u1_space((0, 1), (1, 1))

# Non-abelian fixtures
# --------------------
# `uses_blocked_kernel` does not require abelian fusion (it never did need to —
# see `src/backends.jl`), so the A/B has to cover `UniqueFusion() === false` too.
# These are the *only* fixtures where a relayout pays TensorKit's
# `GenericTreeTransformer` instead of the cached `AbelianTreeTransformer`, which is
# the one place the blocked kernel could plausibly lose ground it holds elsewhere —
# though the pairwise arm pays the same tax, so a tie is the expected outcome.
#
# They sweep the *same* "blocks multiplying" axis as `:fz2_u1`: the spin cutoff
# grows with χ, so the sector count grows while the degeneracies stay small.
#
# Half-integer spins are carried as `twoj = 2j ∈ 0:tjmax`, so the sector's quantum
# dimension is `twoj + 1` and its fermion parity is `mod(twoj, 2)` — a half-integer
# total spin is an odd number of electrons, which is exactly the physical
# spin/parity tie in `TensorKitTensors.HubbardOperators.hubbard_space(_, SU2Irrep)`.

# Largest `twoj` that fits: every sector needs degeneracy ≥ 1, so the spins
# `0 … tjmax/2` already cost `Σ (twoj + 1) = (tjmax+1)(tjmax+2)/2` dimensions. Grow
# like `log2(χ)` as `:fz2_u1` does, but never past what χ can hold — at χ = 8 the
# triangular bound binds (3 sectors, not 4) and above it `log2` does.
function _su2_tjmax(χ::Int)
    fits(tj) = (tj + 1) * (tj + 2) ÷ 2 ≤ χ
    fits(1) || throw(ArgumentError("χ = $χ is too small for an SU(2) fixture"))
    tj = min(floor(Int, log2(χ)), 1)
    while fits(tj + 1) && tj + 1 ≤ floor(Int, log2(χ))
        tj += 1
    end
    return tj
end

"""
    _su2_degeneracies(χ, tjmax) -> Vector{Int}

Degeneracies for spins `twoj/2`, `twoj ∈ 0:tjmax`, whose total *quantum* dimension
is exactly `χ`. Every entry is `≥ 1` (as in [`_peaked_dims`](@ref), so no sector is
silently dropped and the block count stays on the axis being swept), the surplus is
apportioned by binomial weight in whole multiples of each sector's quantum
dimension, and the `twoj = 0` sector — quantum dimension 1, so it can absorb any
remainder — takes what is left over.
"""
function _su2_degeneracies(χ::Int, tjmax::Int)
    qd = collect(1:(tjmax + 1))                     # quantum dim of spin twoj/2
    n = length(qd)
    sum(qd) ≤ χ || throw(ArgumentError("cannot fit spins 0:$tjmax//2 in χ = $χ"))
    d = ones(Int, n)
    surplus = χ - sum(qd)
    w = [binomial(n - 1, i - 1) for i in 1:n]
    tot = sum(w)
    for i in sortperm(w; rev = true)
        k = min(surplus, floor(Int, χ * w[i] / tot)) ÷ qd[i]
        d[i] += k
        surplus -= k * qd[i]
    end
    d[1] += surplus
    return d
end

function _su2_virtual(χ::Int)
    tj = _su2_tjmax(χ)
    d = _su2_degeneracies(χ, tj)
    return Vect[SU2Irrep](SU2Irrep(i // 2) => d[i + 1] for i in 0:tj)
end

function _fz2su2_virtual(χ::Int)
    tj = _su2_tjmax(χ)
    d = _su2_degeneracies(χ, tj)
    return Vect[fℤ₂ ⊠ SU2Irrep](
        fℤ₂(mod(i, 2)) ⊠ SU2Irrep(i // 2) => d[i + 1] for i in 0:tj
    )
end

# Four-state physical space: the spin-SU(2) Hubbard site `|0⟩, |↑↓⟩` (even, spin 0)
# and `|↑⟩, |↓⟩` (odd, spin ½). Shape-equivalent to
# `TensorKitTensors.HubbardOperators.hubbard_space(Trivial, SU2Irrep)` without the
# dep, as `_P_FZ2U1` is to `fermion_space(U1Irrep)`.
const _P_FZ2SU2 = Vect[fℤ₂ ⊠ SU2Irrep](
    fℤ₂(0) ⊠ SU2Irrep(0) => 2, fℤ₂(1) ⊠ SU2Irrep(1 // 2) => 1
)

# `(tag, physical_space, χ -> virtual_space)`.
const BENCH_SPACES = (
    (:trivial, ComplexSpace(2), _trivial_virtual),
    (:z2, Vect[Z2Irrep](0 => 1, 1 => 1), _z2_virtual),
    (:fz2, Vect[fℤ₂](0 => 1, 1 => 1), _fz2_virtual),
    (:fz2_u1, _P_FZ2U1, _fz2u1_virtual),
    (:fz2_u1_flat, _P_FZ2U1, _fz2u1_flat_virtual),
)

# The spaces `report_structure.jl` censuses, and the ones `bench_space` resolves
# against. A superset of `BENCH_SPACES`: the census is free (pure structure, no
# timings), so it can cover spaces the timed `SUITE` cannot afford, and anything
# here that is *not* in `BENCH_SPACES` gets no `SUITE` timings.
#
# The two non-abelian entries are deliberately census-only. They exist for
# `bench_backend_ab.jl` — the *ratio* harness, which is the only place a
# kernel-selection claim may be measured (`SUITE` cannot: see that file's header) —
# and putting them in `BENCH_SPACES` would lengthen every `SUITE` group with fixtures
# no `SUITE` decision reads.
const CENSUS_SPACES = (
    BENCH_SPACES...,
    (:su2, Vect[SU2Irrep](SU2Irrep(0) => 1, SU2Irrep(1 // 2) => 1), _su2_virtual),
    (:fz2_su2, _P_FZ2SU2, _fz2su2_virtual),
)

# The full χ grid every splitter is validated against.
const BENCH_CHIS = (8, 16, 32, 64)

"""
    bench_space(sym) -> (P, V)

The `(physical_space, χ -> virtual_space)` pair tagged `sym` in `CENSUS_SPACES`.
"""
function bench_space(sym::Symbol)
    i = findfirst(t -> first(t) === sym, CENSUS_SPACES)
    isnothing(i) && throw(ArgumentError("unknown benchmark space $sym"))
    return CENSUS_SPACES[i][2], CENSUS_SPACES[i][3]
end

# Splitter self-check. A splitter that misses the requested total measures at a
# different χ than it claims, and a splitter whose *sector count* drifts off the
# axis it is supposed to hold measures the wrong axis entirely; assert both here
# rather than discovering it in a report.
#
# The two graded-U(1) entries are asserted in *opposite* directions, and that is
# the point of having both: `:fz2_u1` must multiply its sectors as χ grows, while
# `:fz2_u1_flat` must keep them fixed so χ only grows the degeneracies. Applying
# the growth check to `:fz2_u1_flat` would contradict its entire purpose. The two
# `SU(2)` entries are on the multiplying axis, so they are asserted with `:fz2_u1`.
#
# `dim(V(χ)) == χ` is the check that matters most for the non-abelian entries: there
# a sector contributes `degeneracy × (2j + 1)`, so an off-by-one in
# `_su2_degeneracies` would measure at a χ it does not claim rather than fail.
let
    for (sym, _, V) in CENSUS_SPACES
        for χ in BENCH_CHIS
            d = TensorKit.dim(V(χ))
            d == χ || error("CENSUS_SPACES[$sym]: dim(V($χ)) = $d ≠ $χ")
        end
        nsec = [length(collect(TensorKit.sectors(V(χ)))) for χ in BENCH_CHIS]
        if sym === :fz2_u1 || sym === :su2 || sym === :fz2_su2
            (issorted(nsec) && last(nsec) > first(nsec)) ||
                error("CENSUS_SPACES[:$sym]: sector count does not grow with χ ($nsec)")
        elseif sym === :fz2_u1_flat
            allequal(nsec) ||
                error("BENCH_SPACES[:fz2_u1_flat]: sector count is not fixed in χ ($nsec)")
        end
    end
end

# Fixtures
# --------
# `hex_state` is the *primary* fixture: coordination 3, matching production
# (`scripts/hubbard_quench`, `examples/realtime/main.jl`,
# `benchmark/realtime_timing/run_timings.jl`), whereas every other fixture here
# is degree 2 or 4. `hexagonal_lattice(2, 2; periodic = (true, true))` gives 8
# vertices, 12 edges, every vertex degree 3, and is loopy, so schedules differ.
# Degree 3 is also what makes χ = 64 affordable: the on-site tensor holds `2χ³`
# entries (8 MB at χ = 64) against `2χ⁴` (512 MB) for a degree-4 interior vertex.

ring_state(L::Int, Dmax::Int; T::Type = ComplexF64) =
    randn_state(T, cycle_graph(L), ComplexSpace(2), ComplexSpace(Dmax))

square_state(n::Int, m::Int, Dmax::Int; T::Type = ComplexF64) =
    randn_state(T, grid([n, m]), ComplexSpace(2), ComplexSpace(Dmax))

hex_state(
    m::Int, n::Int, χ::Int; P, V, T::Type = ComplexF64,
    periodic::NTuple{2, Bool} = (true, true),
) = randn_state(T, hexagonal_lattice(m, n; periodic = periodic), P, V(χ))

hex_state(
    sym::Symbol, χ::Int; m::Int = 2, n::Int = 2, T::Type = ComplexF64,
    periodic::NTuple{2, Bool} = (true, true),
) = ((P, V) = bench_space(sym); hex_state(m, n, χ; P = P, V = V, T = T, periodic = periodic))

# Honeycomb fixture for the groups that run BP *to a tolerance*.
#
# MEASURED, NOT ASSUMED: on the fully periodic 2×2 cell, a `fℤ₂ ⊠ U1Irrep` random
# state does not converge to `SCHED_TOL = 1e-8` under *any* schedule. The quantity
# below is the *true fixed-point* residual — recompute every message from the
# current set and compare (`tr_distance(…; is_hermitian = true)`), not the per-step
# change `bp_state.residuals` reports — and it decays algebraically, not
# geometrically:
#
#        k =        10       50       100      300      1000
#   χ=8,  :sync   9.9e-2   1.2e-2   3.0e-3   3.5e-4   3.1e-5
#   χ=32, :sync   1.1e-1   1.3e-2   3.6e-3   4.5e-4   4.5e-5
#
# `:tree` behaves the same (1.3e-5 at χ=8, 3.2e-5 at χ=32, both at k=1000). So
# after 1000 iterations it is still >3 orders of magnitude above 1e-8, at both χ
# and under both schedules; the fitted decay is ≈ k^-1.75, which puts 1e-8 near
# 1e5 iterations. Every `SUITE["schedule"]` entry on that fixture would therefore
# measure `maxiter × sweep` — duplicating the `convergence` group — at the maximum
# possible cost.
#
# MEASURED, and narrower than it looks: this is a property of *this charge
# distribution*, not of gradedness, not of the sector count, and not of the torus.
# `:fz2_u1_flat` (same symmetry group, sector count pinned at 4) also fails on the
# periodic cell, so it is not the growing sector count. But the same geometry, same
# symmetry group and same total χ = 8 with charges all *non-negative* —
# `(0,0)=>1, (1,1)=>2, (0,2)=>2, (1,3)=>2, (0,4)=>1` — reaches 2.4e-16 within 10
# iterations (`iterations_to_tol` 9 `:sync` / 6 `:tree`), and `:trivial` on the same
# cell needs 18. The discriminant across those three graded distributions is whether
# the charges straddle zero — `_fz2u1_virtual` spans `-nmax:nmax` and
# `_fz2u1_flat_virtual` spans `-2:1`, and both fail — but that is three data points,
# so treat it as the working hypothesis rather than a theorem. What is *not* in
# doubt is the negative result: do **not** read this as "graded + periodic is
# unusable" when choosing future fixtures. That is false, and acting on it would
# cost real coverage.
#
# Dropping the periodicity fixes it. On the *open* 2×2 cell **all four** schedules
# converge at **both** χ (source: `benchmark/reports/schedules.csv`, `hex_open` /
# `fz2_u1` — read it from there rather than restating it, so it cannot drift again):
#
#   χ = 8    sync 436   tree 139   residual 454   splash 135
#   χ = 32   sync 429   tree 126   residual 431   splash 138
#
# It stays loopy (8 vertices, 8 edges → one independent cycle), and its degree
# sequence is `[1, 1, 2, 2, 2, 2, 3, 3]` against `N = 3`, so it is the only
# honeycomb fixture that exercises `oneunit`-padded domain legs at all — the
# periodic cell is degree 3 everywhere, i.e. `d == N` at every vertex and never a
# padded leg. Extra coverage, not less.
#
# The `message` and `sweep` groups keep the fully periodic cell: neither runs to a
# tolerance, and there every vertex has degree exactly 3.
hex_sched_state(sym::Symbol, χ::Int; T::Type = ComplexF64) =
    hex_state(sym, χ; T = T, periodic = (false, false))

# A representative degree-3 vertex of `hex_state`. Fixed rather than "first" so
# reports and profiles are reproducible; on the periodic 2×2 cell every vertex
# has degree 3, but the A/B sublattices differ in which virtual legs are dual.
const HEX_VERTEX = (1, 1, 1)

# The geometries censused by `report_structure.jl` / `report_schedules.jl`,
# spanning coordination 2, 3 and 4. The third field is the representative site of
# maximal degree.
const BENCH_GEOMETRIES = (
    (:ring, () -> cycle_graph(8), 1),                                             # degree 2
    (:hex, () -> hexagonal_lattice(2, 2; periodic = (true, true)), HEX_VERTEX),    # degree 3 throughout
    (:hex_open, () -> hexagonal_lattice(2, 2), HEX_VERTEX),                       # degree 3 at HEX_VERTEX, 1-2 on the boundary
    (:square, () -> grid([3, 3]), 5),                                             # degree 4 centre
)

"""
    bench_state(geom::Symbol, sym::Symbol, χ) -> (state, vertex)

The `(geom, sym, χ)` fixture and its representative maximal-degree vertex.
"""
function bench_state(geom::Symbol, sym::Symbol, χ::Int; T::Type = ComplexF64)
    i = findfirst(t -> first(t) === geom, BENCH_GEOMETRIES)
    isnothing(i) && throw(ArgumentError("unknown benchmark geometry $geom"))
    _, topofn, v = BENCH_GEOMETRIES[i]
    P, V = bench_space(sym)
    return randn_state(T, topofn(), P, V(χ)), v
end

# Messages
# --------
# Run a few BP iterations so kernel benchmarks measure cost on typical-shape
# messages rather than identity-initialised ones. `schedule` is pinned
# explicitly: the `belief_propagation` default is about to change, and an
# implicit default here would hide that behaviour change in an unrelated fixture.
function warm_messages(state; maxiter::Int = 20)
    msgs = BPMessages(state)
    return belief_propagation(
        msgs, state; maxiter = maxiter, tol = 0, schedule = SynchronousSchedule()
    )
end

# Identity-initialised messages, for the `message` group. Neither
# `compute_message!` path branches on tensor *values* — only on spaces — so
# kernel walltime is a function of the fixture's spaces alone, and warming is
# pure setup cost that dominates at χ ≥ 32. The schedule / convergence /
# allocator groups keep `warm_messages`: their residual-driven paths and gauge
# factorizations genuinely do depend on values.
cold_messages(state) = BPMessages(state)

# Build the `(problem, alg, bp_state)` triple needed to call `AI.step!`
# directly, avoiding `belief_propagation`'s solve scaffolding inside the
# timing loop. Stopping criterion is `StopAfterIteration(1)` so a single
# `step!` represents one full sweep. `schedule` selects the message-update
# order benchmarked.
function bp_kernel_setup(state; schedule = SynchronousSchedule(), allocator = Bumper.default_buffer(Bumper.ResizeBuffer))
    problem = Canopy.BPProblem(state)
    alg = Canopy.BeliefPropagation(AI.StopAfterIteration(1); schedule = schedule, allocator = allocator)
    bp_state = AI.initialize_state(problem, alg; messages = BPMessages(state))
    return problem, alg, bp_state
end

# The message-update schedules compared across the suite, as `tag => factory`.
#
# **Factories, not instances, and this matters.** `SpanningTreeSchedule` holds a
# mutable RNG, so a single shared instance advances its state from one
# BenchmarkTools sample to the next: consecutive samples of the *same* key take
# different spanning trees and therefore need different numbers of iterations to
# reach `tol`. That is genuine algorithmic variance being reported as measurement
# noise, and it is not even reproducible across processes, because anything that
# draws first (notably `tune!`) shifts the whole sequence.
#
# MEASURED: with one shared instance, the noise floor — two runs of identical
# code, judged against each other — was ±55% on `SUITE["schedule"]` and ±18% on
# `SUITE["sweep"]`, and *every* worst-case key was `:tree` / `:residual` /
# `:splash` while `:sync` never appeared. Constructing the schedule inside each
# benchmark's `setup` (excluded from the timing) makes every sample start from the
# same RNG state and brings both groups' floors down to the single digits.
#
# `:splash` uses `height = 4`, not the default 2: on `cycle_graph(32)` a height-2
# splash refreshes only a handful of messages per iteration, so it runs into
# `SCHED_MAXITER` and burns the full per-entry time cap with no information.
ndirected(state) = length(BPMessages(state).messages)
const BENCH_SCHEDULES = (
    :sync => () -> SynchronousSchedule(),
    :tree => () -> SpanningTreeSchedule(; rng = MersenneTwister(0)),
    :residual => () -> ResidualSchedule(GreedySampler(8)),
    :splash => () -> ResidualSplashSchedule(; height = 4),
)

# Convergence target shared by `bench_schedule.jl` and `report_schedules.jl`.
#
# `SCHED_MAXITER` is 500, not the 5000 it was before the symmetric fixtures
# existed. `maxiter` is the cap a *non-converging* (fixture, schedule) pair burns,
# and a `belief_propagation` call that hits the cap measures `maxiter × sweep` —
# exactly what the `convergence` group already measures — so a high cap buys no
# information and costs the most. Measured: at 5000 the graded honeycomb entries
# alone took ~145 s per sample and the whole suite ran well over its 15-minute
# budget. 1000 leaves headroom above the slowest combination that *does* converge
# on these fixtures (`hex_open` / `fz2_u1` / `:sync` needs 429-436 iterations, see
# `benchmark/reports/schedules.md`), so a schedule getting somewhat slower to
# converge shows up as a longer time rather than silently tipping over the cap.
const SCHED_TOL = 1.0e-8
const SCHED_MAXITER = 1000

"""
    iterations_to_tol(state, sched; tol, maxiter) -> (; iters, converged)

Number of iterations `sched` needs to drive `max(residuals)` below `tol`. Mirrors
the `AI.solve` loop but checks the residuals directly, so it works for any
schedule.

`converged` is `false` when `maxiter` was reached without meeting `tol`; then
`iters == maxiter` and the value says nothing about the convergence rate. Callers
must branch on `converged` rather than comparing `iters` to `maxiter`, since a
schedule may legitimately converge on exactly the last allowed iteration.
"""
function iterations_to_tol(state, sched; tol = SCHED_TOL, maxiter = SCHED_MAXITER)
    problem = Canopy.BPProblem(state)
    alg = Canopy.BeliefPropagation(AI.StopAfterIteration(maxiter); schedule = sched)
    bp_state = AI.initialize_state(problem, alg; messages = BPMessages(state))
    for k in 1:maxiter
        AI.step!(problem, alg, bp_state)
        maximum(values(bp_state.residuals)) < tol && return (; iters = k, converged = true)
    end
    return (; iters = maxiter, converged = false)
end
