# Kernel-only profiler for the vertex-batched BP message kernel.
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=benchmark benchmark/profile_kernel.jl
#   # optional: restrict the sweep
#   ... benchmark/profile_kernel.jl --sym fz2_u1 --chi 8 32
#
# Loops `compute_message!` on the same `BENCH_SPACES × χ × hex` fixtures the
# `SUITE["message"]["hex_vertex", …]` benchmarks use, under `Profile.@profile`,
# and writes one pprof profile per fixture to
# `benchmark/profiles/kernel_<sym>_chi<χ>.pb.gz`. View with:
#
#   julia -e 'using PProf; PProf.refresh(file="benchmark/profiles/kernel_fz2_u1_chi32.pb.gz")'
#   # or: go tool pprof -http=: benchmark/profiles/kernel_fz2_u1_chi32.pb.gz
#
# The `.pb.gz` files are **not committed** (`benchmark/.gitignore`) — the full sweep
# is ~12 MB of regenerable binary. What is committed is the numbers extracted from
# them, in `benchmark/reports/kernel_profile.md`.
#
# The one number to record in a PR is the **gemm fraction of BP self-time**:
# the share of samples inside `mul!` / `gemm_wrapper!` / `gemm!`. The whole point
# of the reformulation work is that this fraction is low (single-digit percent at
# small χ), so moving it up is the phase result statement. Each run also prints an
# inclusive-share breakdown by function-name match over the same categories the
# project plan tabulates; the shares overlap by construction (`tensoradd!` sits
# under `add_transform_kernel!`), so do not sum them. The whole table is written to
# `benchmark/reports/kernel_profile.md`, which is committed so phases are diffable.
#
# The end-to-end counterpart is `benchmark/realtime_timing/profile_runs.jl`, which
# warms to the bond plateau on a production honeycomb model. This script exists
# because that environment lacks `Graphs` and `BenchmarkTools` and so cannot
# include `benchmark/setup.jl`.
#
# ALLOCATION CAVEAT — read before making any allocation claim
# -----------------------------------------------------------
# `@allocated`, `@time`'s byte count, BenchmarkTools' `memory`/`allocs` columns
# and Julia's allocation profiler **cannot see Bumper's off-heap temporaries**
# (`benchmark/realtime_timing/README.md:121-128`). Canopy bump-allocates every
# contraction intermediate into a `Bumper.ResizeBuffer`, and those bytes never
# reach the GC. All of the above therefore measure **GC pressure only** — the
# persistent heap objects that escape — not the working set.
#
# Every allocation claim must be paired with the buffer's own accounting
# (`src/bumper.jl:43-51`):
#
#   buffer_stats(buf).peak       high-water bump offset, i.e. the real temporary
#                                working-set size in bytes
#   buffer_stats(buf).noverflow  number of live heap overflow blocks; > 0 means
#                                the buffer was too small and fell back to the heap
#   buffer_isempty(buf)          all temporaries freed (offset back to 0, no
#                                overflow blocks live) — the hygiene assertion
#
# `report_buffers` below prints these next to the heap `@allocated` for each
# fixture, so the two are never quoted apart.

using Canopy
import AlgorithmsInterface as AI
using TensorKit
using Printf
using Profile
using PProf
import Bumper

include("setup.jl")

using Canopy: compute_message, compute_message!, outgoing_edges, buffer_stats, buffer_isempty
using TensorKit.TO: DefaultBackend

const PROFILE_DIR = joinpath(@__DIR__, "profiles")
const REPORT_DIR = joinpath(@__DIR__, "reports")

# Inclusive sample shares by function-name match, mirroring the categories in the
# project plan's baseline table. Overlapping by construction — do not sum.
const GEMM_CATEGORY = "gemm (mul!/gemm_wrapper!/gemm!)"
const CATEGORIES = (
    GEMM_CATEGORY => ("mul!", "gemm_wrapper!", "gemm!"),
    "blas_contract!" => ("blas_contract!",),
    "add_transform_kernel! / tensoradd!" => ("add_transform_kernel!", "tensoradd!"),
    "permute!" => ("permute!",),
    "_mapreduce_block! (Strided)" => ("_mapreduce_block!",),
    "twist!" => ("twist!",),
    "subblock" => ("subblock",),
    "LRU get!" => ("get!",),
)

function fixture(sym::Symbol, χ::Int)
    P, V = bench_space(sym)
    state = hex_state(2, 2, χ; P = P, V = V)
    msgs = cold_messages(state)
    edges = collect(outgoing_edges(state, HEX_VERTEX))
    out = compute_message(msgs, state, edges)
    return state, msgs, edges, out
end

# Inclusive share of samples whose backtrace contains a frame matching any of a
# category's patterns.
#
# `data` must come from `Profile.fetch(; include_meta = false)`, i.e. a flat
# instruction-pointer stream delimited by 0. Do **not** pass `Profile.retrieve()`'s
# buffer here: since Julia 1.8 that interleaves per-sample metadata (task id,
# thread id, cpu clock, sleep state) between the backtraces, and a naive walk would
# count those blocks as extra samples and undercount every share.
function inclusive_shares(data, lidict)
    total = 0
    hits = Dict(k => 0 for (k, _) in CATEGORIES)
    frames = String[]
    for ip in data
        if ip == 0
            total += 1
            for (k, pats) in CATEGORIES
                any(f -> any(p -> occursin(p, f), pats), frames) && (hits[k] += 1)
            end
            empty!(frames)
        else
            for sf in get(lidict, ip, ())
                push!(frames, string(sf.func))
            end
        end
    end
    return total, hits
end

# Heap allocation *and* bump-buffer accounting, side by side. See the caveat above:
# neither number alone is a memory claim.
#
# `buffer_stats(buf).peak` is `buf.max_offset`, a **high-water mark over the
# buffer's whole lifetime** — nothing lowers it on its own. The buffer here is the
# task-local default, shared by every fixture in this process, so without an
# explicit reset a small fixture would report the largest previous fixture's peak.
# Reset first, then run once to establish the mark. (`reset_buffer!` also discards
# warmed-up *capacity*, which is fine here — `max_offset` tracks the bump offset,
# not the allocation, so the figure stays accurate — but it is the reason a pooled
# design in a threaded phase should reset `offset` only.)
function _reset_peak!(buf)
    try
        Bumper.reset_buffer!(buf)
        return true
    catch
        return false
    end
end

function report_buffers(io, sym, χ)
    state, msgs, edges, out = fixture(sym, χ)
    buf = Bumper.default_buffer(Bumper.ResizeBuffer)
    reset_ok = _reset_peak!(buf)
    compute_message!(out, msgs, state, edges, DefaultBackend(), buf)      # establish the mark
    heap = @allocated compute_message!(out, msgs, state, edges, DefaultBackend(), buf)
    st = buffer_stats(buf)
    line = @sprintf(
        "heap @allocated %d B (GC pressure only) · bump peak %d B%s · noverflow %d · empty %s",
        heap, st.peak, reset_ok ? "" : " (NOT reset)", st.noverflow, buffer_isempty(buf),
    )
    println(io, "  ", line)
    return line
end

function profile_one(sym::Symbol, χ::Int; seconds::Float64 = 3.0)
    mkpath(PROFILE_DIR)
    state, msgs, edges, out = fixture(sym, χ)

    println("── $sym, χ = $χ ", "─"^40)
    println("  vertex $(HEX_VERTEX): d = $(length(edges)), dim(space) = $(dim(space(state[HEX_VERTEX]))), ",
        "ntrees = $(length(collect(fusiontrees(state[HEX_VERTEX]))))")

    compute_message!(out, msgs, state, edges)         # compile
    t = @elapsed compute_message!(out, msgs, state, edges)
    ncalls = max(10, ceil(Int, seconds / max(t, 1.0e-6)))
    @printf("  %.1f µs/call → %d calls\n", 1.0e6 * t, ncalls)

    # 10 kHz for `seconds` of wall time is O(10⁴) samples — plenty for percentage
    # shares — and stays well inside the buffer. Sampling faster (the 100 kHz this
    # script originally used) overflows the buffer mid-loop, which both truncates
    # the measurement and leaves `Profile.retrieve()` in a state PProf rejects.
    Profile.clear()
    Profile.init(; n = 10_000_000, delay = 1.0e-4)
    Profile.@profile for _ in 1:ncalls
        compute_message!(out, msgs, state, edges)
    end

    flat = Profile.fetch(; include_meta = false)
    lidict = Profile.getdict(flat)
    total, hits = inclusive_shares(flat, lidict)
    @printf("  samples: %d\n", total)
    gemm = 0.0
    if total > 0
        for (k, _) in CATEGORIES
            share = 100 * hits[k] / total
            k == GEMM_CATEGORY && (gemm = share)
            @printf("    %-38s %5.1f%%\n", k, share)
        end
        @printf("  ==> GEMM FRACTION OF BP SELF-TIME: %.1f%%  (the number to record in a PR)\n", gemm)
    end

    # Hand PProf the same metadata-free stream. `Profile.retrieve()`'s buffer
    # interleaves per-sample metadata that PProf reports as "Unexpected 0 in data".
    path = joinpath(PROFILE_DIR, "kernel_$(sym)_chi$(χ).pb.gz")
    pprof(flat, lidict; out = path, web = false)
    println("  wrote $path")

    bufline = report_buffers(stdout, sym, χ)
    return (;
        sym, chi = χ, d = length(edges),
        dim = dim(space(state[HEX_VERTEX])),
        ntrees = length(collect(fusiontrees(state[HEX_VERTEX]))),
        us_per_call = 1.0e6 * t, samples = total,
        shares = Dict(k => (total > 0 ? 100 * hits[k] / total : NaN) for (k, _) in CATEGORIES),
        bufline,
    )
end

function parse_args(args)
    syms = Symbol[]
    chis = Int[]
    mode = :none
    for a in args
        if a == "--sym"
            mode = :sym
        elseif a == "--chi"
            mode = :chi
        elseif mode === :sym
            push!(syms, Symbol(a))
        elseif mode === :chi
            push!(chis, parse(Int, a))
        else
            error("unrecognised argument $a (expected --sym <tag>... --chi <int>...)")
        end
    end
    isempty(syms) && (syms = [s for (s, _, _) in BENCH_SPACES])
    isempty(chis) && (chis = collect(BENCH_CHIS))
    return syms, chis
end

# Commit the summary alongside the `.pb.gz` files: the flame graphs are the detail,
# but the gemm fraction and the overhead shares are the numbers a PR quotes, and
# they need to be diffable across phases.
function write_report(rows)
    mkpath(REPORT_DIR)
    path = joinpath(REPORT_DIR, "kernel_profile.md")
    open(path, "w") do io
        println(io, "# BP message-kernel profile (honeycomb, degree 3)")
        println(io)
        println(io, "Generated by `benchmark/profile_kernel.jl`. Flame graphs for each row are in")
        println(io, "`benchmark/profiles/kernel_<sym>_chi<χ>.pb.gz`.")
        println(io)
        println(io, "- host: `$(gethostname())`  ·  CPU: `$(Sys.CPU_NAME)`  ·  julia `$(VERSION)`")
        println(io, "- `Threads.nthreads() = $(Threads.nthreads())`, `BLAS.get_num_threads() = $(BLAS.get_num_threads())`")
        println(io, "- vertex `$(HEX_VERTEX)` of `hexagonal_lattice(2, 2; periodic = (true, true))`")
        println(io)
        println(io, "Shares are **inclusive** sample fractions by function-name match, so they overlap")
        println(io, "(`tensoradd!` sits under `add_transform_kernel!`, everything sits under")
        println(io, "`blas_contract!`). **Do not sum them.** The single number to quote per phase is")
        println(io, "the gemm fraction.")
        println(io)
        println(io, "| sym | χ | ntrees | dim | µs/call | **gemm %** | blas_contract! | add_transform/tensoradd | permute! | Strided | twist! | subblock | LRU get! |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows
            sh = r.shares
            @printf(
                io, "| %s | %d | %d | %d | %.0f | **%.1f** | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |\n",
                r.sym, r.chi, r.ntrees, r.dim, r.us_per_call,
                sh[GEMM_CATEGORY], sh["blas_contract!"],
                sh["add_transform_kernel! / tensoradd!"], sh["permute!"],
                sh["_mapreduce_block! (Strided)"], sh["twist!"], sh["subblock"], sh["LRU get!"],
            )
        end
        println(io)
        println(io, "## Allocator hygiene")
        println(io)
        println(io, "`@allocated` and Julia's allocation profiler **cannot see Bumper's off-heap")
        println(io, "temporaries**, so the heap column is GC pressure only; the bump figures are the")
        println(io, "real temporary working set. See this script's header.")
        println(io)
        for r in rows
            println(io, "- `$(r.sym)` χ=$(r.chi): $(r.bufline)")
        end
    end
    println("\nwrote $path")
    return path
end

function main(args = ARGS)
    syms, chis = parse_args(args)
    println("BLAS threads = $(BLAS.get_num_threads()), Threads.nthreads() = $(Threads.nthreads())")
    rows = [profile_one(sym, χ) for sym in syms for χ in chis]
    write_report(rows)
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
