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
using TensorKit: ComplexSpace, Vect, Z2Irrep, U1Irrep, fℤ₂, ⊠
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
# The four entries separate the two axes along which symmetric blocks get harder:
#
#   * blocks *growing*     — `:z2` / `:fz2` keep 2 sectors at every χ, so the
#     per-sector degeneracy grows linearly and BLAS eventually dominates.
#   * blocks *multiplying* — `:fz2_u1` grows its sector count like `log2(χ)`, so
#     blocks stay small and per-block bookkeeping (`twist!`, subblock lookups,
#     LRU hits) dominates. This is the regime the project targets.
#
# `report_structure.jl` additionally censuses `:fz2_u1_flat` (below), which holds
# the sector count *fixed* at 4, so the two axes can be compared within one
# symmetry group rather than only across symmetry groups.

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

# Census-only companion to `:fz2_u1`: the *same* symmetry group with the sector
# count pinned at 4, so growing χ grows the degeneracies instead of multiplying
# the sectors. This is the degeneracy-heavy regime the kernel profile in the
# project plan was taken on. Keeping it out of `BENCH_SPACES` keeps `SUITE`
# inside its time budget.
const _FZ2U1_FLAT_NS = -2:1
_fz2u1_flat_virtual(χ::Int) =
    _fz2u1_space(_FZ2U1_FLAT_NS, _peaked_dims(χ, length(_FZ2U1_FLAT_NS)))

# Two-state physical space: one even/charge-0 and one odd/charge-1 state, i.e. a
# single spinless fermion mode. Shape-equivalent to
# `TensorKitTensors.FermionOperators.fermion_space(U1Irrep)` without the dep.
const _P_FZ2U1 = _fz2u1_space((0, 1), (1, 1))

# `(tag, physical_space, χ -> virtual_space)`.
const BENCH_SPACES = (
    (:trivial, ComplexSpace(2), _trivial_virtual),
    (:z2, Vect[Z2Irrep](0 => 1, 1 => 1), _z2_virtual),
    (:fz2, Vect[fℤ₂](0 => 1, 1 => 1), _fz2_virtual),
    (:fz2_u1, _P_FZ2U1, _fz2u1_virtual),
)

# `BENCH_SPACES` plus the fixed-sector-count `:fz2_u1` variant; see above.
const CENSUS_SPACES = (BENCH_SPACES..., (:fz2_u1_flat, _P_FZ2U1, _fz2u1_flat_virtual))

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

# Splitter self-check. A splitter that misses the requested total, or that
# silently yields the same number of sectors at every χ, measures the wrong axis;
# assert both here rather than discovering it in a report.
let
    for (sym, _, V) in CENSUS_SPACES
        for χ in BENCH_CHIS
            d = TensorKit.dim(V(χ))
            d == χ || error("BENCH_SPACES[$sym]: dim(V($χ)) = $d ≠ $χ")
        end
        nsec = [length(collect(TensorKit.sectors(V(χ)))) for χ in BENCH_CHIS]
        if sym === :fz2_u1
            (issorted(nsec) && last(nsec) > first(nsec)) ||
                error("BENCH_SPACES[:fz2_u1]: sector count does not grow with χ ($nsec)")
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
# state does not converge to `SCHED_TOL = 1e-8` under *any* schedule. Its residual
# decays algebraically, roughly like `1/k` — 1.2e-1 at k=10, 9.7e-3 at k=50,
# 3.2e-3 at k=100, 4.4e-4 at k=300 — so 1e-8 is thousands of iterations away, and
# every `SUITE["schedule"]` entry on that fixture would measure `maxiter × sweep`
# (i.e. duplicate the `convergence` group) at the maximum possible cost. The same
# geometry with trivial symmetry reaches 4e-16 in ~18 iterations, so this is a
# property of the graded state on a small torus, not of the geometry alone.
#
# Dropping the periodicity fixes it: on the *open* 2×2 cell all four schedules
# converge at χ = 8 (17–133 iterations) and `:tree` / `:splash` still converge at
# χ = 32 (242 / 84). It stays loopy (8 vertices, 8 edges → one independent cycle)
# and it adds `oneunit`-padded legs at the boundary (degrees 1, 2, 3 with `N = 3`),
# which is extra coverage rather than less.
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
