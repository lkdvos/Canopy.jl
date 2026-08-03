# The symmetry-block census.
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=benchmark benchmark/report_structure.jl
#
# BenchmarkTools records timings, not scalars, so this is a separate script. It
# never runs as part of `run(SUITE)`. Output goes to
# `benchmark/reports/structure.{csv,md}`, both committed.
#
# WHY THIS EXISTS
# ---------------
# "The message contractions get smaller and more numerous as symmetry is added"
# is the premise of the whole performance project, and it is two independent
# claims. The census separates them:
#
#   * blocks *growing*      — sector count fixed, per-sector degeneracy ∝ χ.
#     `:z2` / `:fz2` (2 sectors at every χ) and `:fz2_u1_flat` (4 sectors at
#     every χ, same symmetry group as `:fz2_u1`) live here. BLAS eventually
#     dominates; overheads that scale with the *number* of blocks fade out.
#   * blocks *multiplying*  — sector count grows with χ, degeneracies stay small.
#     `:fz2_u1` lives here. Per-block bookkeeping (`twist!`, per-subblock
#     `Dictionary` lookups, LRU hits) dominates and stays dominant.
#
# A threshold fitted on one axis does not transfer to the other, which is the
# reason both are in the table. Compare `:fz2_u1` against `:fz2_u1_flat` at equal
# χ: same symmetry group, same total bond dimension, opposite regime.
#
# COLUMN MEANINGS
# ---------------
# All per-vertex quantities are for the representative maximal-degree vertex of
# the geometry (`BENCH_GEOMETRIES`), whose on-site tensor has space
# `P ← V₁ ⊗ … ⊗ V_N` with `M = N + 1` indices.
#
#   d, M         site degree; number of tensor indices (`M = N + 1`, `N` = the
#                lattice's *maximum* coordination, so `d < N` means padded legs)
#   nsectors     `length(sectors(V(χ)))` — sectors on one virtual leg
#   maxsecdim    `maximum(dim(V, c))`    — largest per-sector degeneracy
#                (this pair is the `bond_sector_structure` vocabulary of
#                `scripts/hubbard_quench/HubbardQuench.jl:566-578` and
#                `benchmark/realtime_timing/compare_symmetry.jl:105-109`)
#   ntrees       `length(fusiontrees(state[v]))` — number of stored subblocks
#   nblocksec    `length(blocksectors(space(state[v])))` — coupled sectors of the
#                on-site tensor. For a `1 ← N` tensor this is bounded by the
#                *physical* space's sector count, so it is NOT the number of
#                gemms the kernel performs; see `close_*` below.
#   dim          `dim(space(state[v]))` — total stored scalars
#   meanblk      `dim / ntrees` — mean subblock size, the quantity a blocked
#                kernel's overhead ratio is a function of
#   close_nblk   coupled sectors of the *closing* layout
#                `space(T, k+1) ← ⊗(space(T, j) for j ≠ k+1)`, i.e. the number of
#                gemms one closing performs (one per coupled sector). `k` is the
#                middle target leg.
#   close_min/med/max
#                element counts (`nrow × ncol`) of those closing blocks. This is
#                the actual gemm size distribution, min/median/max over sectors.

using Canopy
import AlgorithmsInterface as AI
using TensorKit
using Statistics: median
using Printf

include("setup.jl")

using Canopy: outgoing_edges, leg_index
using Graphs: neighbors

const REPORT_DIR = joinpath(@__DIR__, "reports")
mkpath(REPORT_DIR)

# Blocks of the closing contraction, as element counts. `compute_message!` closes
# `prefix[k] : V_k ← (P, others)` against `suffix[k]`, one gemm per coupled sector
# of that layout, so the block shapes of `prefix[k]` are the gemm shapes. Built
# by actually permuting the tensor rather than reconstructing the `HomSpace` by
# hand, so the numbers cannot drift from what the kernel does.
function closing_blocks(t, k::Int)
    M = TensorKit.numind(t)
    others = Tuple(i for i in 1:M if i != k + 1)
    p = permute(t, ((k + 1,), others))
    return [length(b) for (_, b) in blocks(p)]
end

struct CensusRow
    geom::Symbol
    sym::Symbol
    chi::Int
    d::Int
    M::Int
    nsectors::Int
    maxsecdim::Int
    ntrees::Int
    nblocksec::Int
    dim::Int
    meanblk::Float64
    close_nblk::Int
    close_min::Int
    close_med::Float64
    close_max::Int
end

function census_row(geom::Symbol, sym::Symbol, χ::Int)
    state, v = bench_state(geom, sym, χ)
    _, V = bench_space(sym)
    Vχ = V(χ)
    t = state[v]
    d = length(neighbors(state, v))
    M = TensorKit.numind(t)
    ntrees = length(collect(fusiontrees(t)))
    nbs = length(collect(blocksectors(space(t))))
    dm = dim(space(t))
    secs = collect(sectors(Vχ))
    # Middle target leg: representative of the closings the kernel performs.
    cb = closing_blocks(t, cld(d, 2))
    return CensusRow(
        geom, sym, χ, d, M,
        length(secs), maximum(c -> dim(Vχ, c), secs; init = 0),
        ntrees, nbs, dm, dm / ntrees,
        length(cb),
        isempty(cb) ? 0 : minimum(cb), isempty(cb) ? 0.0 : median(cb),
        isempty(cb) ? 0 : maximum(cb),
    )
end

const HEADER = (
    "geom", "sym", "chi", "d", "M", "nsectors", "maxsecdim", "ntrees",
    "nblocksec", "dim", "meanblk", "close_nblk", "close_min", "close_med", "close_max",
)

_cells(r::CensusRow) = (
    string(r.geom), string(r.sym), string(r.chi), string(r.d), string(r.M),
    string(r.nsectors), string(r.maxsecdim), string(r.ntrees), string(r.nblocksec),
    string(r.dim), @sprintf("%.1f", r.meanblk), string(r.close_nblk),
    string(r.close_min), @sprintf("%.1f", r.close_med), string(r.close_max),
)

function markdown_table(rows)
    cells = map(_cells, rows)
    w = [maximum(length, (HEADER[i], (c[i] for c in cells)...)) for i in eachindex(HEADER)]
    pad(s, i) = rpad(s, w[i])
    io = IOBuffer()
    println(io, "| ", join((pad(HEADER[i], i) for i in eachindex(HEADER)), " | "), " |")
    println(io, "|", join(("-"^(w[i] + 2) for i in eachindex(HEADER)), "|"), "|")
    for c in cells
        println(io, "| ", join((pad(c[i], i) for i in eachindex(c)), " | "), " |")
    end
    return String(take!(io))
end

# The axis-separation summary: for each (geom, sym), how the census scales from
# the smallest to the largest χ censused.
#
# The discriminating statistic is the growth of `nsectors` (and, downstream of it,
# `ntrees` — the *number* of stored subblocks). It is NOT the growth of `meanblk`:
# mean block size grows in every regime, because total `dim` grows like χ^d
# whatever the symmetry. What distinguishes the regimes is whether the block
# *count* comes along:
#
#   nsectors ≈ const → "growing only": degeneracies scale with χ, `ntrees` is
#                      constant, and per-block overhead amortises away as χ rises
#   nsectors ↑       → "multiplying":  the block count scales with χ too, so
#                      overheads proportional to it never amortise
#
# A blocked kernel's overhead ratio is `c / (meanblk · m̄)`, so read `meanblk` in
# the full census below for the absolute threshold and read this table for which
# direction each fixture moves.
function axis_table(rows)
    io = IOBuffer()
    println(io, "| geom | sym | χ range | nsectors | maxsecdim | ntrees | meanblk | close_nblk | close_med | regime |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for geom in (g[1] for g in BENCH_GEOMETRIES), sym in (s[1] for s in CENSUS_SPACES)
        rs = filter(r -> r.geom === geom && r.sym === sym, rows)
        length(rs) < 2 && continue
        lo, hi = first(rs), last(rs)
        treesx = hi.ntrees / lo.ntrees
        secx = hi.nsectors / lo.nsectors
        regime = if sym === :trivial
            "n/a (unsymmetric)"
        elseif secx ≥ 1.5
            "**multiplying** (and growing)"
        else
            "growing only"
        end
        @printf(
            io, "| %s | %s | %d→%d | %d→%d (%.1f×) | %d→%d (%.1f×) | %d→%d (**%.1f×**) | %.1f→%.1f (%.0f×) | %d→%d | %.0f→%.0f | %s |\n",
            geom, sym, lo.chi, hi.chi,
            lo.nsectors, hi.nsectors, hi.nsectors / lo.nsectors,
            lo.maxsecdim, hi.maxsecdim, hi.maxsecdim / lo.maxsecdim,
            lo.ntrees, hi.ntrees, treesx,
            lo.meanblk, hi.meanblk, hi.meanblk / lo.meanblk,
            lo.close_nblk, hi.close_nblk,
            lo.close_med, hi.close_med,
            regime,
        )
    end
    return String(take!(io))
end

function main()
    rows = CensusRow[]
    for (geom, _, _) in BENCH_GEOMETRIES, (sym, _, _) in CENSUS_SPACES, χ in BENCH_CHIS
        # A degree-4 vertex at χ = 64 with trivial symmetry is 512 MB of on-site
        # tensor; skip the combinations that only measure the allocator.
        geom === :square && χ > 32 && continue
        push!(rows, census_row(geom, sym, χ))
    end

    csv = joinpath(REPORT_DIR, "structure.csv")
    open(csv, "w") do io
        println(io, join(HEADER, ","))
        for r in rows
            println(io, join(_cells(r), ","))
        end
    end

    md = joinpath(REPORT_DIR, "structure.md")
    open(md, "w") do io
        println(io, "# Symmetry-block census")
        println(io)
        println(io, "Generated by `benchmark/report_structure.jl`. Column meanings and the")
        println(io, "growing-vs-multiplying distinction are documented in that file's header.")
        println(io)
        println(io, "- host: `$(gethostname())`  ·  CPU: `$(Sys.CPU_NAME)`  ·  julia `$(VERSION)`")
        println(io, "- git sha: `$(_bench_gitsha())`")
        println(io)
        println(io, "Timings are **not** in this table; it is pure structure and is therefore")
        println(io, "machine-independent. `:fz2_u1_flat` is `fℤ₂ ⊠ U1Irrep` with the sector count")
        println(io, "pinned at 4, so it isolates the growing axis inside the same symmetry group as")
        println(io, "`:fz2_u1`. It is in `BENCH_SPACES` too, so its `SUITE[\"message\"][\"hex_vertex\",")
        println(io, ":fz2_u1_flat, χ]` timings can be read directly against these rows.")
        println(io)
        println(io, "Two observations that hold across every row and matter for the blocked kernel:")
        println(io)
        println(io, "1. `close_nblk == nsectors` exactly — the closing performs one gemm per")
        println(io, "   *coupled* sector of `V_k`, so 7-13 gemms per closing on the `:fz2_u1`")
        println(io, "   fixtures, not one per uncoupled tuple (`ntrees` is 73-253 there).")
        println(io, "2. `nblocksec` is 2 for every graded fixture and 1 for `:trivial`, because the")
        println(io, "   on-site tensor is `P ← V₁⊗…⊗V_N` and its coupled sector is fixed by the")
        println(io, "   *physical* leg. It is therefore **not** a count of the kernel's gemms;")
        println(io, "   `close_nblk` is.")
        println(io)
        println(io, "## Axis separation")
        println(io)
        println(io, "Scaling from the smallest to the largest χ censused, per (geometry, symmetry).")
        println(io)
        print(io, axis_table(rows))
        println(io)
        println(io, "## Full census")
        println(io)
        print(io, markdown_table(rows))
    end

    print(markdown_table(rows))
    println()
    print(axis_table(rows))
    println("\nwrote $csv\nwrote $md")
    return rows
end

# `_bench_gitsha` lives in `benchmarks.jl`, which this script does not include.
function _bench_gitsha()
    try
        return readchomp(Cmd(`git rev-parse --short=7 HEAD`; dir = dirname(@__DIR__)))
    catch
        return "unknown"
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
