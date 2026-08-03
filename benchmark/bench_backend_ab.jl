# Interleaved blocked-vs-pairwise A/B on the vertex-batched message kernel.
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#       julia --project=benchmark benchmark/bench_backend_ab.jl [χ...]
#
# WHY THIS IS NOT A `SUITE` GROUP
# -------------------------------
# `SUITE["message"]["hex_vertex", sym, χ, backend]` has the backend axis inside
# one `run(SUITE)`, which was *supposed* to make the blocked/pairwise ratio
# immune to machine drift. It does not: `BenchmarkGroup` is a `Dict`, so `run`
# does **not** execute keys in insertion order, and the two arms of a pair end up
# far apart in time. At χ = 64 one key takes ~10 s, so on a shared box the drift
# between the two arms swamps a 20% effect. Measured consequence: the committed
# `SUITE` run reported `fz2_u1_flat` at χ = 64 as **0.970** (a regression) where
# the interleaved measurement below gives **1.199**.
#
# So: anything that gates on a *ratio* gets measured here, with the two arms
# adjacent in time, on the same fixture, the same `out` buffers and the same
# bump allocator.
#
# METHOD
# ------
#   * one fixture per (sym, χ): the degree-3 honeycomb vertex `HEX_VERTEX` of the
#     periodic 2×2 cell — production coordination, and the same fixture the
#     `SUITE["message"]` decision metric uses;
#   * both arms are compiled once before timing starts;
#   * per repetition: `min` of `AB_INNER` back-to-back calls of arm A, then the
#     same for arm B, so a scheduler hiccup has to land inside one 5-call burst
#     to bias the pair;
#   * reported statistic: `min` over `AB_REPS` repetitions of those per-rep
#     minima — minimum is the right estimator for "how fast is this code", the
#     upper tail is contention;
#   * `:trivial` is the **control**. `uses_blocked_kernel` excludes `Trivial`, so
#     both arms run byte-identical code there and its ratio is this harness's
#     measurement floor. Quote it with every table.
#
# Correctness and allocator hygiene are checked per fixture too (the arms must
# agree to `rtol`, the bump buffer must come back empty with no overflow), so a
# ratio can never be reported for a fixture whose two arms disagree.

using Canopy
import AlgorithmsInterface as AI
using TensorKit
using LinearAlgebra: norm
using Printf

include("setup.jl")

using Canopy: compute_message, compute_message!, outgoing_edges,
    buffer_stats, buffer_isempty

# χ grid. Positional args override; the default pair is the one the
# auto-selection decision needs — χ = 64 is the largest point phase 2 measured
# (where `z2` had fallen to 1.028) and χ = 128 is the extrapolation check.
const AB_CHIS = isempty(ARGS) ? (64, 128) : Tuple(parse(Int, a) for a in ARGS)

# `:su2` / `:fz2_su2` are the non-abelian rows. `uses_blocked_kernel` does not
# require `UniqueFusion` (see `src/backends.jl`), so they take the blocked path and
# belong in this A/B; they are also the only rows where a relayout goes through
# TensorKit's `GenericTreeTransformer` rather than the cached
# `AbelianTreeTransformer`. Both arms pay that, so the expectation is a tie or a
# win, not a loss — which is exactly the claim this table has to settle.
const AB_SYMS = (:trivial, :z2, :fz2, :fz2_u1, :fz2_u1_flat, :su2, :fz2_su2)

const AB_REPS = parse(Int, get(ENV, "CANOPY_AB_REPS", "9"))
const AB_INNER = parse(Int, get(ENV, "CANOPY_AB_INNER", "5"))

# Skip guard. A degree-3 on-site tensor holds `dim(P) · χ³` scalars, so `:trivial`
# at χ = 128 is 67 MB per site and 537 MB for the 8-site cell, before the kernel's
# ~2d chain links of the same size. The graded fixtures are far smaller
# (`fz2_u1` at χ = 64 is 59 369 entries against `:trivial`'s 524 288). Rather than
# hard-coding which combinations are affordable, estimate and compare.
const AB_MAXGB = parse(Float64, get(ENV, "CANOPY_AB_MAXGB", "24"))

const REPORT_DIR = joinpath(@__DIR__, "reports")
const REPORT_FILE = joinpath(REPORT_DIR, "backend_ab.md")

_loadavg() = try
    join(split(read("/proc/loadavg", String))[1:3], " ")
catch
    "unknown"
end

_gitsha() = try
    readchomp(Cmd(`git rev-parse --short=7 HEAD`; dir = dirname(@__DIR__)))
catch
    "unknown"
end

"""
    best(f, n) -> Float64

Minimum wall time, in microseconds, of `n` back-to-back calls of `f`.
"""
best(f, n) = minimum(begin
        t = time_ns()
        f()
        (time_ns() - t) / 1.0e3
    end for _ in 1:n)

struct ABRow
    sym::Symbol
    chi::Int
    ntrees::Int
    dim::Int
    meanblk::Float64
    pairwise::Float64      # µs
    blocked::Float64       # µs
    ratio::Float64         # pairwise / blocked, >1 means blocked is faster
    spread::Float64        # (max-min)/min over reps, of the *ratio*'s arms
    peak_p::Int            # bump-buffer high-water mark, bytes
    peak_b::Int
    maxdiff::Float64       # relative disagreement between the two arms
end

# Estimated live bytes of the fixture: the 8-site cell plus a generous allowance
# for the kernel's chain links (up to `2d` links of `dim(space(T))` each) and the
# `d` output messages.
function _estimate_bytes(sym::Symbol, χ::Int)
    P, V = bench_space(sym)
    Vχ = V(χ)
    site = dim(P) * dim(Vχ)^3                      # degree-3 honeycomb vertex
    return 16 * (8 * site + 8 * site + 3 * dim(Vχ)^2)
end

function ab_row(sym::Symbol, χ::Int; reps = AB_REPS, inner = AB_INNER)
    Random.seed!(BENCH_SEED)
    P, V = bench_space(sym)
    dim(V(χ)) == χ || error("bench_space($sym): dim(V($χ)) ≠ $χ")
    state = hex_state(2, 2, χ; P = P, V = V)
    T = state[HEX_VERTEX]
    msgs = cold_messages(state)
    edges = collect(outgoing_edges(state, HEX_VERTEX))

    # Correctness first: a ratio between two arms that disagree is meaningless.
    ref_p = compute_message(msgs, state, edges, PairwiseBackend(), TO.DefaultAllocator())
    ref_b = compute_message(msgs, state, edges, BlockedBackend(), TO.DefaultAllocator())
    maxdiff = maximum(
        norm(ref_b[i] - ref_p[i]) / max(norm(ref_p[i]), eps()) for i in eachindex(ref_p)
    )
    maxdiff < 1.0e-12 || error("$sym χ=$χ: arms disagree by $maxdiff")

    out = compute_message(msgs, state, edges)
    buf = Bumper.default_buffer(Bumper.ResizeBuffer)

    # Compile both arms, then take each arm's peak from a *reset* buffer —
    # `max_offset` is a process-lifetime high-water mark.
    Bumper.reset_buffer!(buf)
    compute_message!(out, msgs, state, edges, PairwiseBackend(), buf)
    st_p = buffer_stats(buf)
    buffer_isempty(buf) || error("$sym χ=$χ: pairwise left the buffer non-empty")
    st_p.noverflow == 0 || error("$sym χ=$χ: pairwise overflowed the buffer")

    Bumper.reset_buffer!(buf)
    compute_message!(out, msgs, state, edges, BlockedBackend(), buf)
    st_b = buffer_stats(buf)
    buffer_isempty(buf) || error("$sym χ=$χ: blocked left the buffer non-empty")
    st_b.noverflow == 0 || error("$sym χ=$χ: blocked overflowed the buffer")

    tp = Float64[]
    tb = Float64[]
    for _ in 1:reps
        push!(tp, best(() -> compute_message!(out, msgs, state, edges, PairwiseBackend(), buf), inner))
        push!(tb, best(() -> compute_message!(out, msgs, state, edges, BlockedBackend(), buf), inner))
    end

    mp, mb = minimum(tp), minimum(tb)
    spread = max((maximum(tp) - mp) / mp, (maximum(tb) - mb) / mb)
    ntrees = length(collect(fusiontrees(T)))
    dm = dim(space(T))
    return ABRow(
        sym, χ, ntrees, dm, dm / ntrees, mp, mb, mp / mb, spread,
        st_p.peak, st_b.peak, maxdiff,
    )
end

const HEADER = (
    "sym", "chi", "ntrees", "dim", "meanblk", "pairwise_us", "blocked_us",
    "ratio", "spread", "peak_pairwise", "peak_blocked", "maxdiff",
)

_cells(r::ABRow) = (
    string(r.sym), string(r.chi), string(r.ntrees), string(r.dim),
    @sprintf("%.1f", r.meanblk), @sprintf("%.1f", r.pairwise),
    @sprintf("%.1f", r.blocked), @sprintf("%.3f", r.ratio),
    @sprintf("%.1f%%", 100 * r.spread), string(r.peak_p), string(r.peak_b),
    @sprintf("%.1e", r.maxdiff),
)

function markdown_table(rows)
    cells = map(_cells, rows)
    w = [maximum(length, (HEADER[i], (c[i] for c in cells)...)) for i in eachindex(HEADER)]
    io = IOBuffer()
    println(io, "| ", join((rpad(HEADER[i], w[i]) for i in eachindex(HEADER)), " | "), " |")
    println(io, "|", join(("-"^(w[i] + 2) for i in eachindex(HEADER)), "|"), "|")
    for c in cells
        println(io, "| ", join((rpad(c[i], w[i]) for i in eachindex(c)), " | "), " |")
    end
    return String(take!(io))
end

function main()
    load0 = _loadavg()
    println("""
    ── interleaved blocked/pairwise A/B ─────────────────────────────────────
      git sha           : $(_gitsha())
      julia             : $(VERSION)
      Threads.nthreads  : $(Threads.nthreads())
      BLAS threads      : $(BLAS.get_num_threads())
      Sys.CPU_NAME      : $(Sys.CPU_NAME)
      hostname          : $(gethostname())
      loadavg at start  : $load0
      χ grid            : $(AB_CHIS)
      reps × inner      : $(AB_REPS) × $(AB_INNER)
    ─────────────────────────────────────────────────────────────────────────""")

    rows = ABRow[]
    skipped = Tuple{Symbol, Int, Float64}[]
    for χ in AB_CHIS, sym in AB_SYMS
        gb = _estimate_bytes(sym, χ) / 2^30
        if gb > AB_MAXGB
            @printf("skip  %-13s χ=%-4d  (≈%.1f GB > CANOPY_AB_MAXGB=%.1f)\n", sym, χ, gb, AB_MAXGB)
            push!(skipped, (sym, χ, gb))
            continue
        end
        r = ab_row(sym, χ)
        push!(rows, r)
        @printf(
            "%-13s χ=%-4d  pairwise %9.1f µs   blocked %9.1f µs   ratio %6.3f   (meanblk %.1f, ≈%.2f GB)\n",
            r.sym, r.chi, r.pairwise, r.blocked, r.ratio, r.meanblk, gb
        )
        GC.gc()
    end
    load1 = _loadavg()

    println()
    print(markdown_table(rows))
    println()
    println("loadavg at start: $load0")
    println("loadavg at end  : $load1")
    isempty(skipped) || println("skipped (memory): $skipped")

    mkpath(REPORT_DIR)
    open(REPORT_FILE, "w") do io
        println(io, "# Blocked vs pairwise message kernel, interleaved A/B")
        println(io)
        println(io, "Generated by `benchmark/bench_backend_ab.jl`. Method, and why this is not a")
        println(io, "`SUITE` group, are documented in that file's header.")
        println(io)
        println(io, "- host: `$(gethostname())`  ·  CPU: `$(Sys.CPU_NAME)`  ·  julia `$(VERSION)`")
        println(io, "- git sha: `$(_gitsha())`  ·  BLAS threads: 1  ·  `Threads.nthreads()` = $(Threads.nthreads())")
        println(io, "- loadavg (1/5/15): `$load0` at start, `$load1` at end")
        println(io, "- $(AB_REPS) reps × $(AB_INNER) inner calls; reported time is min-over-reps of min-over-inner")
        println(io)
        println(io, "`ratio = pairwise / blocked`, so **> 1 means the blocked kernel is faster**.")
        println(io, "`:trivial` is the control: `uses_blocked_kernel` excludes `Trivial`, so both arms")
        println(io, "run identical code and its ratio is this harness's measurement floor.")
        println(io, "`spread` is the worst arm's `(max - min) / min` over reps; `maxdiff` is the")
        println(io, "relative disagreement between the two arms' outputs.")
        println(io)
        print(io, markdown_table(rows))
        if !isempty(skipped)
            println(io)
            println(io, "Skipped as too large (`CANOPY_AB_MAXGB` = $(AB_MAXGB)):")
            for (sym, χ, gb) in skipped
                @printf(io, "- `%s` at χ = %d (≈%.1f GB)\n", sym, χ, gb)
            end
        end
    end
    println("wrote $REPORT_FILE")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
