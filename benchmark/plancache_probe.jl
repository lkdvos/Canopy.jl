# Phase 4a — plan-cache feasibility for the blocked message kernel.
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#       julia --project=benchmark benchmark/plancache_probe.jl [--count] [--profile] [--cheat]
#
# THE QUESTION
# ------------
# Every `mul!` and every `permute!` in `_blocked_message!` reaches TensorKit's
# space-structure layer, which is a `GlobalLRUCache` behind a `SpinLock` taken on
# **every lookup including hits** (`LRUCache/src/LRUCache.jl:19,215`):
#
#   * `blocks(t)`   → `blockstructure(space(t))`
#                   → `blocksectors` (`sectorstructure`, LRU) + `degeneracystructure` (LRU)
#     …and `mul!(C, A, B)` calls `blocks` **three times**
#     (`TensorKit/src/tensors/linalg.jl:336-338`).
#   * `permute!`    → `braid!` → `treebraider(Vdst, Vsrc, p, levels)` (LRU, keyed
#     on *two* `HomSpace`s) → `add_transform!` → `spacecheck_transform` (which
#     rebuilds `permute(Vsrc, p)` and compares).
#   * `tensoralloc_add` → `dim(::HomSpace)` → `degeneracystructure` (LRU).
#
# A `MessagePlan` would hoist all of that above the call. This script measures
# how much is actually there.
#
# THREE MEASUREMENTS
# ------------------
# `--count`   exact lookups per vertex call, from `LRUCache.cache_info` deltas on
#             `TensorKit.GLOBAL_CACHES` (hits + misses is a lookup counter), plus
#             the standalone cost of one lookup of each kind on the same fixture.
# `--profile` inclusive sample shares for each structure-layer function, from a
#             `Profile.@profile` loop over the *current default* kernel.
# `--cheat`   an **upper bound**: a hand-rolled kernel that precomputes every
#             transformer and every block range once, outside the timing loop, and
#             then does nothing but `TO.tensoradd!` on `StridedView`s and `mul!`
#             on `reshape(view(data, rng))`. It performs the same subblock copies
#             and the same gemms as the real kernel — it just never asks a
#             `HomSpace` anything. A real plan cache pays lookup, validation and
#             generality on top, so `t_real / t_cheat` is a **ceiling**, not a
#             forecast. Correctness is asserted against the real kernel first.
#
# Measurement follows `benchmark/bench_backend_ab.jl`: both arms in one process on
# one fixture, alternating within each repetition, min-of-inner then min-over-reps,
# with a `control` arm that runs the cheat kernel against itself so its deviation
# from 1.0 is the noise floor.

using Canopy
import AlgorithmsInterface as AI
using TensorKit
using LinearAlgebra
using Printf
using Profile
import Bumper

include("setup.jl")

using Canopy: compute_message, compute_message!, outgoing_edges, leg_index,
    neighbors, DirectedEdge
using TensorKit: TupleTools, TO, StridedView
const VZero = TensorKit.Zero

const REPORT_DIR = joinpath(@__DIR__, "reports")

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

best(f, n) = minimum(begin
        t = time_ns()
        f()
        (time_ns() - t) / 1.0e3
    end for _ in 1:n)

function fixture(sym::Symbol, χ::Int)
    Random.seed!(BENCH_SEED)
    P, V = bench_space(sym)
    state = hex_state(2, 2, χ; P = P, V = V)
    msgs = cold_messages(state)
    edges = collect(outgoing_edges(state, HEX_VERTEX))
    out = compute_message(msgs, state, edges)
    return state, msgs, edges, out
end

const PROBE_SYMS = (:fz2, :fz2_u1, :fz2_u1_flat)
const PROBE_CHIS = (8, 32, 64, 128)

# ---------------------------------------------------------------------------
# 1. Lookup counting
# ---------------------------------------------------------------------------
# `LRUCache` is not a direct benchmark dependency; reach it through TensorKit,
# which brings the module into scope with `using LRUCache`.
const cache_info = TensorKit.LRUCache.cache_info

_cachecounts() = Dict(
    name => (i = cache_info(cache); i.hits + i.misses)
        for (name, cache) in TensorKit.GLOBAL_CACHES
)

function count_lookups(sym::Symbol, χ::Int; ncalls::Int = 20)
    state, msgs, edges, out = fixture(sym, χ)
    buf = Bumper.default_buffer(Bumper.ResizeBuffer)
    compute_message!(out, msgs, state, edges, BlockedBackend(), buf)   # warm every cache
    before = _cachecounts()
    for _ in 1:ncalls
        compute_message!(out, msgs, state, edges, BlockedBackend(), buf)
    end
    after = _cachecounts()
    return Dict(k => (after[k] - before[k]) / ncalls for k in keys(before))
end

# Standalone cost of one *hit* of each cached lookup, on the same fixture: this is
# what a plan cache would save per lookup.
function lookup_costs(sym::Symbol, χ::Int; n::Int = 2000)
    state, _, _, _ = fixture(sym, χ)
    T = state[HEX_VERTEX]
    W = space(T)
    M = numind(T)
    allinds = ntuple(identity, M)
    p1 = (TupleTools.deleteat(allinds, 2), (2,))
    W1 = permute(W, p1)
    levels = ntuple(identity, M)
    lv = (TupleTools.getindices(levels, (1,)), TupleTools.getindices(levels, ntuple(i -> i + 1, M - 1)))
    # warm
    TensorKit.sectorstructure(W1); TensorKit.degeneracystructure(W1)
    TensorKit.blockstructure(W1); TensorKit.subblockstructure(W1)
    TensorKit.treebraider(W1, W, p1, lv)
    return (
        sectorstructure = best(() -> TensorKit.sectorstructure(W1), n),
        degeneracystructure = best(() -> TensorKit.degeneracystructure(W1), n),
        blocksectors = best(() -> TensorKit.blocksectors(W1), n),
        blockstructure = best(() -> TensorKit.blockstructure(W1), n),
        subblockstructure = best(() -> TensorKit.subblockstructure(W1), n),
        treebraider = best(() -> TensorKit.treebraider(W1, W, p1, lv), n),
        hash_homspace = best(() -> hash(W1), n),
        permute_homspace = best(() -> permute(W, p1), n),
        blocks = best(() -> TensorKit.blocks(T), n),
    )
end

# ---------------------------------------------------------------------------
# 2. Profile shares
# ---------------------------------------------------------------------------
const CATEGORIES = (
    "mul!/gemm" => ("mul!", "gemm_wrapper!", "gemm!"),
    "tensoradd!/add_transform" => ("tensoradd!", "add_transform_kernel!", "add_transform!"),
    "_mapreduce_block! (Strided)" => ("_mapreduce_block!",),
    "sectorstructure" => ("sectorstructure",),
    "degeneracystructure" => ("degeneracystructure",),
    "blocksectors" => ("blocksectors",),
    "blockstructure" => ("blockstructure",),
    "subblockstructure" => ("subblockstructure",),
    "treebraider/treetransformer" => ("treebraider", "treetransformer", "TreeTransformer"),
    "hash" => ("hash",),
    "LRU get!" => ("get!",),
    "spacecheck/permute(HomSpace)" => ("spacecheck_transform", "_permute", "compose"),
    "tensoralloc" => ("tensoralloc",),
)

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

function profile_one(sym::Symbol, χ::Int; seconds::Float64 = 3.0)
    state, msgs, edges, out = fixture(sym, χ)
    buf = Bumper.default_buffer(Bumper.ResizeBuffer)
    call() = compute_message!(out, msgs, state, edges, BlockedBackend(), buf)
    call()
    t = @elapsed call()
    ncalls = max(10, ceil(Int, seconds / max(t, 1.0e-6)))
    Profile.clear()
    Profile.init(; n = 10_000_000, delay = 1.0e-4)
    Profile.@profile for _ in 1:ncalls
        call()
    end
    flat = Profile.fetch(; include_meta = false)
    lidict = Profile.getdict(flat)
    total, hits = inclusive_shares(flat, lidict)
    shares = Dict(k => (total > 0 ? 100 * hits[k] / total : NaN) for (k, _) in CATEGORIES)
    return (; sym, chi = χ, us_per_call = 1.0e6 * t, samples = total, shares)
end

# ---------------------------------------------------------------------------
# 3. The "cheating" kernel
# ---------------------------------------------------------------------------
#
# Every transformer and every block range is resolved once, at plan time. The
# call itself touches only `Vector{T}`s, integer ranges and isbits plan entries —
# no `HomSpace`, no `TensorMap`, no cache.

# One resolved copy schedule is a `(transformer_data, permutation)` pair, where
# `transformer_data` is the `(coeff, dst_struct, src_struct)` vector TensorKit
# itself would have looked up. Every field is concretely typed (parameters
# inferred at construction) so the timed loop dispatches nothing dynamically.
struct CheatPlan{TRM, TRL, TRS}
    d::Int
    legmin::Int
    legmax::Int
    dimT::Int
    dimmsg::Vector{Int}
    dimout::Vector{Int}                          # per *target*, not per leg
    target_legs::Vector{Int}
    msg_tr::Vector{TRM}                          # per leg, twist folded in
    enter_ket::TRL
    enter_bra::TRL
    up::Vector{TRS}                              # k -> Layout(k) → Layout(k+1)
    down::Vector{TRS}                            # k -> Layout(k+1) → Layout(k)
    absorb_ket::Vector{Vector{NTuple{4, Int}}}   # (off, offmsg, m, n) per sector
    absorb_bra::Vector{Vector{NTuple{4, Int}}}
    absorb_zero::Vector{Vector{UnitRange{Int}}}  # layout sectors the message lacks
    close::Vector{Vector{NTuple{5, Int}}}        # (offO, offBra, offKet, d1, d2)
    close_zero::Vector{Vector{UnitRange{Int}}}
end

# `treebraider` with exactly the arguments `permute!`/`TO.tensoradd!` would use.
function _braider(Wdst, Wsrc, p)
    M = length(codomain(Wsrc)) + length(domain(Wsrc))
    levels = ntuple(identity, M)
    N₁ = length(codomain(Wsrc))
    lv = (
        TupleTools.getindices(levels, ntuple(identity, N₁)),
        TupleTools.getindices(levels, ntuple(i -> N₁ + i, M - N₁)),
    )
    return TensorKit.treebraider(Wdst, Wsrc, p, lv)
end

# `(coeff, dst_struct, src_struct)` triples, optionally rescaled per *source*
# subblock by `θ(f_src)`. The transformer is built in `pairs(subblockstructure)`
# order, so index `i` of `.data` is the `i`-th fusion tree pair of `Wsrc`
# (`TensorKit/src/tensors/treetransformers.jl:23-30`).
function _transport(Wdst, Wsrc, p, θ = nothing)
    tr = _braider(Wdst, Wsrc, p)
    data = copy(tr.data)
    if θ !== nothing
        trees = collect(keys(TensorKit.subblockstructure(Wsrc)))
        for i in eachindex(data)
            c, sd, ss = data[i]
            data[i] = (c * θ(trees[i]), sd, ss)
        end
    end
    return (data, p)
end

function transport!(tr, dst::Vector{T}, src::Vector{T}, backend, allocator) where {T}
    data, p = tr
    @inbounds for i in eachindex(data)
        coeff, sd, ss = data[i]
        TO.tensoradd!(
            StridedView(dst, sd...), StridedView(src, ss...), p, false,
            coeff, VZero(), backend, allocator
        )
    end
    return nothing
end

function absorb!(sect::Vector{NTuple{4, Int}}, zeros_, dst::Vector{T}, src::Vector{T}, msg::Vector{T}, adj::Bool) where {T}
    # `mul!` zero-fills a destination sector missing from either operand via
    # `rmul!(C, false)`; reproduce that or the composition inherits garbage. On
    # every fixture measured here these lists are empty (`nblocks == nsectors`),
    # so this costs nothing — it is here so the cheat cannot be accidentally
    # right only on the fixtures that happen to have no gaps.
    @inbounds for r in zeros_
        fill!(view(dst, r), zero(T))
    end
    @inbounds for i in eachindex(sect)
        off, offm, m, n = sect[i]
        C = reshape(view(dst, (off + 1):(off + m * n)), m, n)
        A = reshape(view(src, (off + 1):(off + m * n)), m, n)
        B = reshape(view(msg, (offm + 1):(offm + n * n)), n, n)
        adj ? mul!(C, A, adjoint(B)) : mul!(C, A, B)
    end
    return nothing
end

function close!(sect::Vector{NTuple{5, Int}}, zeros_, o::Vector{T}, bra::Vector{T}, ket::Vector{T}) where {T}
    @inbounds for r in zeros_
        fill!(view(o, r), zero(T))
    end
    @inbounds for i in eachindex(sect)
        offO, offB, offK, d₁, d₂ = sect[i]
        C = reshape(view(o, (offO + 1):(offO + d₂ * d₂)), d₂, d₂)
        A = reshape(view(bra, (offB + 1):(offB + d₁ * d₂)), d₁, d₂)
        B = reshape(view(ket, (offK + 1):(offK + d₁ * d₂)), d₁, d₂)
        mul!(C, adjoint(A), B)
    end
    return nothing
end

function build_plan(state, msgs, edges)
    v = first(first(edges))
    T = state[v]
    Tn = scalartype(T)
    W = space(T)
    M = numind(T)
    nbrs = neighbors(state, v)
    d = length(nbrs)
    target_legs = map(e -> leg_index(state, e), edges)
    legmin, legmax = extrema(target_legs)

    allinds = ntuple(identity, M)
    layout(k) = (TupleTools.deleteat(allinds, k + 1), (k + 1,))
    pswap(j) = (ntuple(i -> ifelse(i == j, M, i), M - 1), (j,))
    pmsg = ((2,), (1,))
    dual_phys = isdual(space(T, 1))

    incoming = map(n -> msgs[DirectedEdge(n, v)], nbrs)
    Wmsg = map(space, incoming)
    Wmt = map(Wm -> permute(Wm, pmsg), Wmsg)
    WL = [permute(W, layout(k)) for k in 1:d]
    Wout = [space(msgs[DirectedEdge(v, nbrs[k])]) for k in 1:d]

    # transposed messages, with the per-coupled-sector twist folded into the copy
    msg_tr = [_transport(Wmt[j], Wmsg[j], pmsg, f -> twist(f[1].coupled)) for j in 1:d]

    enter_ket = _transport(WL[1], W, layout(1), dual_phys ? (f -> twist(f[1].uncoupled[1])) : nothing)
    enter_bra = _transport(WL[d], W, layout(d))
    # unused chain slots get an empty schedule of the same type rather than
    # `nothing`, so `up`/`down` stay concretely typed and the timed loop does not
    # dispatch dynamically per step.
    _blank(tr) = (empty(tr[1]), tr[2])
    up1 = _transport(WL[min(2, d)], WL[1], pswap(2))
    up = [k <= legmax - 1 ? _transport(WL[k + 1], WL[k], pswap(k + 1)) : _blank(up1) for k in 1:d]
    down = [(legmin <= k <= d - 1) ? _transport(WL[k], WL[k + 1], pswap(k + 1)) : _blank(up1) for k in 1:d]

    bsL = [TensorKit.blockstructure(WL[k]) for k in 1:d]
    bsM = [TensorKit.blockstructure(Wmt[k]) for k in 1:d]
    bsO = [TensorKit.blockstructure(Wout[k]) for k in 1:d]

    absorb(k, j) = NTuple{4, Int}[
        (first(rA) - 1, first(bsM[j][c][2]) - 1, dd[1], dd[2])
            for (c, (dd, rA)) in pairs(bsL[k]) if haskey(bsM[j], c)
    ]
    absorb_ket = [k <= legmax - 1 ? absorb(k, k) : NTuple{4, Int}[] for k in 1:d]
    absorb_bra = [(legmin <= k <= d - 1) ? absorb(k + 1, k + 1) : NTuple{4, Int}[] for k in 1:d]
    # `absorb_zero[k]` is indexed by the *layout* whose sectors are written, i.e.
    # `k` for a ket step and `k+1` for a bra step; both are covered by keying on
    # the layout index directly.
    absorb_zero = [
        [r for (c, ((_, _), r)) in pairs(bsL[k]) if !haskey(bsM[k], c)] for k in 1:d
    ]

    close = Vector{NTuple{5, Int}}[]
    close_zero = Vector{UnitRange{Int}}[]
    for k in target_legs
        sect = NTuple{5, Int}[]
        present = Set()
        for (c, (dd, r)) in pairs(bsL[k])
            haskey(bsO[k], c) || continue
            (odd, orng) = bsO[k][c]
            odd == (dd[2], dd[2]) ||
                error("closing block mismatch at leg $k: out $(odd) vs layout $(dd)")
            push!(present, c)
            push!(sect, (first(orng) - 1, first(r) - 1, first(r) - 1, dd[1], dd[2]))
        end
        push!(close, sect)
        push!(close_zero, [r for (c, ((_, _), r)) in pairs(bsO[k]) if !(c in present)])
    end

    return CheatPlan(
        d, legmin, legmax, dim(W), [dim(Wm) for Wm in Wmsg],
        [dim(Wout[k]) for k in target_legs],
        collect(target_legs), msg_tr, enter_ket, enter_bra, up, down,
        absorb_ket, absorb_bra, absorb_zero, close, close_zero,
    )
end

struct CheatBuffers{T}
    ket::Vector{Vector{T}}
    bra::Vector{Vector{T}}
    tmp::Vector{T}
    mt::Vector{Vector{T}}
    out::Vector{Vector{T}}
end

function CheatBuffers(::Type{T}, pl::CheatPlan) where {T}
    return CheatBuffers{T}(
        [Vector{T}(undef, pl.dimT) for _ in 1:pl.d],
        [Vector{T}(undef, pl.dimT) for _ in 1:pl.d],
        Vector{T}(undef, pl.dimT),
        [Vector{T}(undef, n) for n in pl.dimmsg],
        [Vector{T}(undef, n) for n in pl.dimout],
    )
end

function cheat_call!(pl::CheatPlan, b::CheatBuffers, Tdata, msgdata, backend, allocator)
    d, legmin, legmax = pl.d, pl.legmin, pl.legmax
    for j in 1:d
        transport!(pl.msg_tr[j], b.mt[j], msgdata[j], backend, allocator)
    end
    transport!(pl.enter_ket, b.ket[1], Tdata, backend, allocator)
    for k in 1:(legmax - 1)
        absorb!(pl.absorb_ket[k], pl.absorb_zero[k], b.tmp, b.ket[k], b.mt[k], false)
        transport!(pl.up[k], b.ket[k + 1], b.tmp, backend, allocator)
    end
    transport!(pl.enter_bra, b.bra[d], Tdata, backend, allocator)
    for k in (d - 1):-1:legmin
        absorb!(pl.absorb_bra[k], pl.absorb_zero[k + 1], b.tmp, b.bra[k + 1], b.mt[k + 1], true)
        transport!(pl.down[k], b.bra[k], b.tmp, backend, allocator)
    end
    for i in eachindex(pl.target_legs)
        k = pl.target_legs[i]
        close!(pl.close[i], pl.close_zero[i], b.out[i], b.bra[k], b.ket[k])
    end
    return b.out
end

# ---------------------------------------------------------------------------
struct CheatRow
    sym::Symbol
    chi::Int
    dim::Int
    ntrees::Int
    t_real::Float64
    t_cheat::Float64
    t_control::Float64
    ratio::Float64        # real / cheat, > 1 means the cheat is faster
    control_dev::Float64  # |control/cheat - 1|
    maxdiff::Float64
end

const AB_REPS = parse(Int, get(ENV, "CANOPY_PROBE_REPS", "9"))
const AB_INNER = parse(Int, get(ENV, "CANOPY_PROBE_INNER", "5"))

function cheat_row(sym::Symbol, χ::Int; reps = AB_REPS, inner = AB_INNER)
    state, msgs, edges, out = fixture(sym, χ)
    v = HEX_VERTEX
    T = state[v]
    nbrs = neighbors(state, v)
    backend = TO.DefaultBackend()
    allocator = TO.DefaultAllocator()

    pl = build_plan(state, msgs, edges)
    b = CheatBuffers(scalartype(T), pl)
    msgdata = [msgs[DirectedEdge(n, v)].data for n in nbrs]

    ref = compute_message(msgs, state, edges, BlockedBackend(), allocator)
    cheat_call!(pl, b, T.data, msgdata, backend, allocator)
    maxdiff = maximum(
        norm(b.out[i] - ref[i].data) / max(norm(ref[i].data), eps())
            for i in eachindex(ref)
    )
    maxdiff < 1.0e-12 || error("$sym χ=$χ: cheat kernel disagrees by $maxdiff")

    buf = Bumper.default_buffer(Bumper.ResizeBuffer)
    real_call() = compute_message!(out, msgs, state, edges, BlockedBackend(), buf)
    cheat() = cheat_call!(pl, b, T.data, msgdata, backend, allocator)
    real_call(); cheat()

    tr = Float64[]; tc = Float64[]; tk = Float64[]
    for _ in 1:reps
        push!(tr, best(real_call, inner))
        push!(tc, best(cheat, inner))
        push!(tk, best(cheat, inner))
    end
    mr, mc, mk = minimum(tr), minimum(tc), minimum(tk)
    return CheatRow(
        sym, χ, dim(space(T)), length(collect(fusiontrees(T))),
        mr, mc, mk, mr / mc, abs(mk / mc - 1), maxdiff,
    )
end

function cheat_table(io, rows)
    println(io, "| sym | χ | dim | ntrees | real µs | cheat µs | control µs | **real/cheat** | control dev | maxdiff |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for r in rows
        @printf(
            io, "| %s | %d | %d | %d | %.1f | %.1f | %.1f | **%.3f** | %.1f%% | %.1e |\n",
            r.sym, r.chi, r.dim, r.ntrees, r.t_real, r.t_cheat, r.t_control,
            r.ratio, 100 * r.control_dev, r.maxdiff,
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
function main(args = ARGS)
    want(x) = isempty(args) || (x in args)
    load0 = _loadavg()
    println("""
    ── plan-cache probe ─────────────────────────────────────────────────────
      git sha           : $(_gitsha())
      julia             : $(VERSION)
      Threads.nthreads  : $(Threads.nthreads())
      BLAS threads      : $(BLAS.get_num_threads())
      Sys.CPU_NAME      : $(Sys.CPU_NAME)
      hostname          : $(gethostname())
      loadavg at start  : $load0
    ─────────────────────────────────────────────────────────────────────────""")

    counts = Any[]
    if want("--count")
        for χ in PROBE_CHIS, sym in PROBE_SYMS
            c = count_lookups(sym, χ)
            lc = lookup_costs(sym, χ)
            push!(counts, (; sym, chi = χ, counts = c, costs = lc))
            println("$sym χ=$χ  lookups/call: ", sort(collect(c); by = first))
            println("            hit cost µs: ", lc)
            GC.gc()
        end
        println()
    end

    profs = Any[]
    if want("--profile")
        for χ in PROBE_CHIS, sym in PROBE_SYMS
            p = profile_one(sym, χ)
            push!(profs, p)
            @printf("%-13s χ=%-4d %8.1f µs/call, %d samples\n", sym, χ, p.us_per_call, p.samples)
            for (k, _) in CATEGORIES
                @printf("    %-32s %5.1f%%\n", k, p.shares[k])
            end
            GC.gc()
        end
        println()
    end

    cheats = CheatRow[]
    if want("--cheat")
        for χ in PROBE_CHIS, sym in PROBE_SYMS
            r = cheat_row(sym, χ)
            push!(cheats, r)
            @printf(
                "%-13s χ=%-4d  real %9.1f µs   cheat %9.1f µs   ratio %6.3f  (control dev %.1f%%)\n",
                r.sym, r.chi, r.t_real, r.t_cheat, r.ratio, 100 * r.control_dev
            )
            GC.gc()
        end
        println()
        cheat_table(stdout, cheats)
        println()
    end

    load1 = _loadavg()
    println("loadavg at start: $load0")
    println("loadavg at end  : $load1")

    mkpath(REPORT_DIR)
    path = joinpath(REPORT_DIR, "plancache_probe.md")
    open(path, "w") do io
        println(io, "# Plan-cache feasibility probe")
        println(io)
        println(io, "Generated by `benchmark/plancache_probe.jl`; method in that file's header.")
        println(io)
        println(io, "- host: `$(gethostname())`  ·  CPU: `$(Sys.CPU_NAME)`  ·  julia `$(VERSION)`")
        println(io, "- git sha: `$(_gitsha())`  ·  BLAS threads: $(BLAS.get_num_threads())  ·  `Threads.nthreads()` = $(Threads.nthreads())")
        println(io, "- loadavg (1/5/15): `$load0` at start, `$load1` at end")
        if !isempty(counts)
            println(io, "\n## Cached lookups per vertex call\n")
            keysall = sort(collect(keys(counts[1].counts)); by = string)
            println(io, "| sym | χ | ", join(string.(keysall), " | "), " | total |")
            println(io, "|---|---|", join(("---" for _ in keysall), "|"), "|---|")
            for c in counts
                tot = sum(c.counts[k] for k in keysall)
                @printf(io, "| %s | %d | ", c.sym, c.chi)
                for k in keysall
                    @printf(io, "%.1f | ", c.counts[k])
                end
                @printf(io, "**%.1f** |\n", tot)
            end
            println(io, "\n### Cost of a single cache *hit*, same fixture (µs)\n")
            ks = collect(keys(counts[1].costs))
            println(io, "| sym | χ | ", join(string.(ks), " | "), " |")
            println(io, "|---|---|", join(("---" for _ in ks), "|"), "|")
            for c in counts
                @printf(io, "| %s | %d | ", c.sym, c.chi)
                println(io, join((@sprintf("%.3f", getfield(c.costs, k)) for k in ks), " | "), " |")
            end
        end
        if !isempty(profs)
            println(io, "\n## Profile shares (inclusive, overlapping — do not sum)\n")
            cats = [k for (k, _) in CATEGORIES]
            println(io, "| sym | χ | µs/call | ", join(cats, " | "), " |")
            println(io, "|---|---|---|", join(("---" for _ in cats), "|"), "|")
            for p in profs
                @printf(io, "| %s | %d | %.0f | ", p.sym, p.chi, p.us_per_call)
                println(io, join((@sprintf("%.1f", p.shares[k]) for k in cats), " | "), " |")
            end
        end
        if !isempty(cheats)
            println(io, "\n## Cheating-prototype upper bound\n")
            println(io, "`real/cheat > 1` is the factor a *perfect* plan cache could buy — the cheat")
            println(io, "kernel does the same subblock copies and the same gemms but resolves every")
            println(io, "transformer and every block range at plan time. It is an **upper bound**.")
            println(io, "`control` is a second cheat arm; its deviation from the first is the floor.\n")
            cheat_table(io, cheats)
        end
    end
    println("wrote $path")
    return (; counts, profs, cheats)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
