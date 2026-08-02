# Phase 4a — threading feasibility probe for the blocked message kernel.
#
#   JULIA_NUM_THREADS=n OPENBLAS_NUM_THREADS=1 \
#       julia --project=benchmark --threads=n benchmark/blockloop_probe.jl [--census|--probe|--upstream]
#
# WHAT "THE BLOCK LOOP" IS
# ------------------------
# The blocked kernel (`src/messages.jl:641`, `_blocked_message!`) reduces a
# degree-`d` vertex update to
#
#   * `d`     transposed messages   (`tensoradd!`, χ² each)
#   * 2 entry braids + `2d-2` chain re-permutes (`tensoradd!`, `dim(space(T))` each)
#   * `2d-2`  absorptions  `mul!(tmp, link, mt)`
#   * `d`     closings     `mul!(out[i], adjoint(bra[k]), ket[k])`
#
# Every `mul!` here is `LinearAlgebra.mul!(::AbstractTensorMap, ...)`
# (`TensorKit/src/tensors/linalg.jl:330`), a **serial merge-join over coupled
# sectors** carrying the comment `# TODO: consider spawning threads for different
# blocks`. That per-coupled-sector loop is the candidate for phase 4b, and it is
# what this script censuses and times.
#
# The complementary loop — `tforeach(transformer.data)` inside
# `add_transform_kernel!` (`TensorKit/src/tensors/indexmanipulations.jl:659`) —
# is already threadable **upstream** via `TensorKit.set_num_transformer_threads`,
# so `--upstream` measures that instead of reimplementing it.
#
# METHOD
# ------
# `--census` is pure structure: it rebuilds the `Layout(k)` `HomSpace`s from the
# fixture's space signature and reads `blockstructure`, so it costs nothing and
# covers χ = 128 at degree 4 where the tensors would be 8 GB.
#
# `--probe` builds the real chain links with Canopy's own internals
# (`_transposed_messages`, `_blocked_step`), flattens *every* `mul!` of one
# vertex call into a list of raw `(C, A, B)` block triples, and then times that
# one list under six arms interleaved within each repetition, min-of-inner then
# min-over-reps, exactly as `benchmark/bench_backend_ab.jl` does. `control` is
# byte-identical code to `serial`, so its deviation is the noise floor.
#
# Two families of threaded arm, and the distinction is the whole point:
#
#   * **flat** — one parallel region over every block of the call. Not
#     achievable (the chain is sequential), reported as an optimistic bound.
#   * **grouped** — one parallel region per `mul!`, `nblk/mul!` blocks wide.
#     This is what phase 4b could actually build.

using Canopy
import AlgorithmsInterface as AI
using TensorKit
using LinearAlgebra
using Printf
using Statistics: median
import Bumper

include("setup.jl")

using Canopy: compute_message, compute_message!, outgoing_edges, leg_index,
    neighbors, DirectedEdge
using Canopy: _transposed_messages, _blocked_step
using TensorKit: TupleTools, TO
const OMT = TensorKit.OhMyThreads

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

# ---------------------------------------------------------------------------
# Census — pure structure, no tensor allocation
# ---------------------------------------------------------------------------
#
# `randn_state` dualises the virtual leg on the larger-labelled endpoint
# (`src/states/tensornetworkstate.jl:71-79`) and pads to `N = max coordination`
# with `oneunit(S)`. Rather than reimplement that, build the fixture once at the
# cheapest χ, read off `(d, N, isdual pattern)` at the representative vertex, and
# re-materialise the `HomSpace` at any χ from that signature.

struct SiteSig
    P::Any
    duals::Vector{Bool}    # per *virtual* leg, length d
    N::Int                 # padded domain arity
    d::Int
end

function site_signature(geom::Symbol, sym::Symbol)
    state, v = bench_state(geom, sym, first(BENCH_CHIS))
    W = space(state[v])
    N = length(domain(W))
    d = length(neighbors(state, v))
    duals = [isdual(domain(W)[i]) for i in 1:d]
    return SiteSig(codomain(W)[1], duals, N, d)
end

function site_space(sig::SiteSig, V, χ::Int)
    Vχ = V(χ)
    S = typeof(Vχ)
    Vs = ntuple(sig.N) do i
        return i <= sig.d ? (sig.duals[i] ? dual(Vχ) : Vχ) : oneunit(S)
    end
    return sig.P ← ⊗(Vs...)
end

# The `Layout(k)` family and the per-leg message space, from the site `HomSpace`.
_layouts(W) = (M = length(codomain(W)) + length(domain(W));
    allinds = ntuple(identity, M);
    k -> permute(W, (TupleTools.deleteat(allinds, k + 1), (k + 1,))))

# The *composable* form of incoming message `j`, i.e. what the chain actually
# multiplies: `mt = permute(msg, ((2,),(1,)))` lands on `domain(W)[j] ←
# domain(W)[j]`, so its coupled sectors are `sectors(domain(W)[j])`.
#
# Getting this duality wrong is not cosmetic: with `V' ← V'` instead, the
# intersection silently loses every sector whose conjugate is absent from `V`,
# which for `:fz2_u1_flat` (charges `-2:1`, deliberately *not* symmetric about 0)
# undercounts the block loop by one sector out of four. Cross-checked against the
# real `blocks(...)` triple count in `--probe`.
_msgspace(W, j) = (Vj = domain(W)[j]; Vj ← Vj)

"""
    gemm_shapes(W, d, target_legs) -> Vector{(kind, leg, Vector{(m,n,k)})}

Every `mul!` one `_blocked_message!` call performs, as the list of per-coupled-
sector gemm shapes it iterates. `kind ∈ (:absorb_ket, :absorb_bra, :close)`.
"""
function gemm_shapes(W, d::Int, target_legs)
    legmin, legmax = extrema(target_legs)
    L = _layouts(W)
    bs(Wk) = TensorKit.blockstructure(Wk)
    out = Tuple{Symbol, Int, Vector{NTuple{3, Int}}}[]

    # absorptions: `mul!(tmp, link::Layout(k), mt_j)` with `mt_j` block `(dc, dc)`
    # and `link` block `(d₁, dc)`; the iterated sectors are those present in both.
    function absorb(kind, k, j)
        bl = bs(L(k))
        bm = bs(_msgspace(W, j))
        shapes = NTuple{3, Int}[]
        for (c, ((d₁, d₂), _)) in pairs(bl)
            haskey(bm, c) || continue
            push!(shapes, (d₁, d₂, d₂))
        end
        return push!(out, (kind, j, shapes))
    end
    for k in 1:(legmax - 1)
        absorb(:absorb_ket, k, k)
    end
    for k in (d - 1):-1:legmin
        absorb(:absorb_bra, k + 1, k + 1)
    end

    # closings: `mul!(out[i], adjoint(bra[k])::(d₂,d₁), ket[k]::(d₁,d₂))`
    for k in target_legs
        bl = bs(L(k))
        bo = bs(_msgspace(W, k))
        shapes = NTuple{3, Int}[]
        for (c, ((d₁, d₂), _)) in pairs(bl)
            haskey(bo, c) || continue
            push!(shapes, (d₂, d₂, d₁))
        end
        push!(out, (:close, k, shapes))
    end
    return out
end

struct CensusRow
    geom::Symbol
    sym::Symbol
    chi::Int
    d::Int
    nsectors::Int          # sectors of V(χ)
    ntrees::Int
    dim::Int
    nmul::Int              # `mul!` calls per vertex-batch call
    nblk_min::Int
    nblk_med::Float64
    nblk_max::Int
    mnk_min::NTuple{3, Int}
    mnk_med::NTuple{3, Int}
    mnk_max::NTuple{3, Int}
    flops_min::Int
    flops_med::Int
    flops_max::Int
    flops_tot::Int
end

function census_row(geom::Symbol, sym::Symbol, χ::Int, sig::SiteSig)
    P, V = bench_space(sym)
    W = site_space(sig, V, χ)
    targets = collect(1:sig.d)
    gs = gemm_shapes(W, sig.d, targets)
    nblk = [length(s) for (_, _, s) in gs]
    allshapes = reduce(vcat, (s for (_, _, s) in gs))
    flops = [2 * m * n * k for (m, n, k) in allshapes]
    ord = sortperm(flops)
    med(v) = isempty(v) ? 0 : v[ord[cld(length(ord), 2)]]
    return CensusRow(
        geom, sym, χ, sig.d,
        length(collect(TensorKit.sectors(V(χ)))),
        length(TensorKit.fusiontrees(W)), dim(W),
        length(gs), minimum(nblk), median(nblk), maximum(nblk),
        allshapes[ord[1]], allshapes[ord[cld(length(ord), 2)]], allshapes[ord[end]],
        minimum(flops), med(flops), maximum(flops), sum(flops),
    )
end

const CENSUS_SYMS = (:fz2, :fz2_u1, :fz2_u1_flat)
const CENSUS_CHIS = (8, 32, 64, 128)
const CENSUS_GEOMS = (:hex, :square)

function run_census()
    rows = CensusRow[]
    for geom in CENSUS_GEOMS, sym in CENSUS_SYMS
        sig = site_signature(geom, sym)
        for χ in CENSUS_CHIS
            push!(rows, census_row(geom, sym, χ, sig))
        end
    end
    return rows
end

function census_table(io, rows)
    println(io, "| geom | sym | χ | d | nsectors | ntrees | dim | n_mul! | nblocks min/med/max | (m,n,k) min | (m,n,k) med | (m,n,k) max | 2mnk min/med/max | Σ2mnk |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows
        @printf(
            io, "| %s | %s | %d | %d | %d | %d | %d | %d | %d/%.0f/%d | %s | %s | %s | %d/%d/%d | %.3g |\n",
            r.geom, r.sym, r.chi, r.d, r.nsectors, r.ntrees, r.dim, r.nmul,
            r.nblk_min, r.nblk_med, r.nblk_max,
            string(r.mnk_min), string(r.mnk_med), string(r.mnk_max),
            r.flops_min, r.flops_med, r.flops_max, float(r.flops_tot),
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Chain reconstruction — the real links, kept alive on the heap
# ---------------------------------------------------------------------------
#
# Mirrors `_blocked_message!` but with `TO.DefaultAllocator()` so every
# intermediate survives the call and its blocks can be handed to the probe.
function chain_links(msgs, state, edges)
    backend = TO.DefaultBackend()
    allocator = TO.DefaultAllocator()
    v = first(first(edges))
    T = state[v]
    M = numind(T)
    nbrs = neighbors(state, v)
    d = length(nbrs)
    target_legs = map(e -> leg_index(state, e), edges)
    legmin, legmax = extrema(target_legs)

    allinds = ntuple(identity, M)
    layout(k) = (TupleTools.deleteat(allinds, k + 1), (k + 1,))
    pid = (ntuple(identity, M - 1), (M,))
    pswap(j) = (ntuple(i -> ifelse(i == j, M, i), M - 1), (j,))
    pmsg = ((2,), (1,))

    incoming = map(n -> msgs[DirectedEdge(n, v)], nbrs)
    msgt = _transposed_messages(incoming, legmin, legmax, pmsg, backend, allocator)

    ket1 = permute(T, layout(1))
    isdual(space(T, 1)) && twist!(ket1, (1,))
    ket = Vector{typeof(ket1)}(undef, d)
    ket[1] = ket1
    for k in 1:(legmax - 1)
        ket[k + 1] = _blocked_step(ket[k], msgt[k], pid, pswap(k + 1), backend, allocator)
    end

    brad = permute(T, layout(d))
    bra = Vector{typeof(brad)}(undef, d)
    bra[d] = brad
    for k in (d - 1):-1:legmin
        bra[k] = _blocked_step(bra[k + 1], adjoint(msgt[k + 1]), pid, pswap(k + 1), backend, allocator)
    end

    return (; T, d, target_legs, legmin, legmax, msgt, ket, bra, pid, pswap, layout)
end

# Every `mul!` of one vertex call, flattened to raw per-sector block triples.
# `blocks(...)` is called here, in setup, so the timed loop touches no space at
# all — it multiplies `StridedView`s over already-allocated flat data, which is
# the best case any phase-4b implementation could hope to hand to a task.
#
# Three groups, because the three call sites differ in *which operand carries an
# adjoint* and therefore in the concrete `StridedView` type: keeping them in one
# `Vector{Any}` would add a dynamic dispatch per block, which at these block
# sizes is a measurable fraction of the block itself.
struct JobSet{J1, J2, J3}
    ket::Vector{J1}       # `mul!(tmp, ket[k],   msgt[k])`
    bra::Vector{J2}       # `mul!(tmp, bra[k+1], adjoint(msgt[k+1]))`
    close::Vector{J3}     # `mul!(out[i], adjoint(bra[k]), ket[k])`
    shapes::Vector{NTuple{3, Int}}
    bounds::Vector{UnitRange{Int}}   # one range per `mul!`, in global 1:n indexing
    keep::Vector{Any}     # destinations, kept alive
end

Base.length(js::JobSet) = length(js.ket) + length(js.bra) + length(js.close)

@inline function runjob(js::JobSet, i::Int)
    n1 = length(js.ket)
    n2 = n1 + length(js.bra)
    if i <= n1
        @inbounds j = js.ket[i]
        mul!(j[1], j[2], j[3])
    elseif i <= n2
        @inbounds j = js.bra[i - n1]
        mul!(j[1], j[2], j[3])
    else
        @inbounds j = js.close[i - n2]
        mul!(j[1], j[2], j[3])
    end
    return nothing
end

function _triples(C, A, B, shapes)
    bA = Dict(TensorKit.blocks(A))
    bB = Dict(TensorKit.blocks(B))
    tr = map(collect(TensorKit.blocks(C))) do (c, Cb)
        return (haskey(bA, c) && haskey(bB, c)) ? (Cb, bA[c], bB[c]) : nothing
    end
    keep = [t for t in tr if t !== nothing]
    for (Cb, Ab, _) in keep
        push!(shapes, (size(Cb, 1), size(Cb, 2), size(Ab, 2)))
    end
    return identity.(keep)
end

function block_jobs(cl, out)
    d, legmin, legmax = cl.d, cl.legmin, cl.legmax
    shapes = NTuple{3, Int}[]
    keep = Any[]
    counts = Int[]

    ketg = [
        begin
            tmp = similar(cl.ket[k])
            push!(keep, tmp)
            _triples(tmp, cl.ket[k], cl.msgt[k], shapes)
        end for k in 1:(legmax - 1)
    ]
    brag = [
        begin
            tmp = similar(cl.bra[k + 1])
            push!(keep, tmp)
            _triples(tmp, cl.bra[k + 1], adjoint(cl.msgt[k + 1]), shapes)
        end for k in (d - 1):-1:legmin
    ]
    closeg = [
        _triples(out[i], adjoint(cl.bra[cl.target_legs[i]]), cl.ket[cl.target_legs[i]], shapes)
            for i in eachindex(cl.target_legs)
    ]
    append!(counts, length.(ketg)); append!(counts, length.(brag)); append!(counts, length.(closeg))
    off = 0
    bounds = UnitRange{Int}[]
    for c in counts
        push!(bounds, (off + 1):(off + c))
        off += c
    end
    return JobSet(
        identity.(reduce(vcat, ketg)), identity.(reduce(vcat, brag)),
        identity.(reduce(vcat, closeg)), shapes, bounds, keep,
    )
end

# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------
function loop_serial(js)
    for i in 1:length(js)
        runjob(js, i)
    end
    return nothing
end

function loop_spawn(js, nchunks::Int)
    n = length(js)
    nc = min(nchunks, n)
    tasks = Vector{Task}(undef, nc)
    for c in 1:nc
        lo = div((c - 1) * n, nc) + 1
        hi = div(c * n, nc)
        tasks[c] = Threads.@spawn begin
            for i in lo:hi
                runjob(js, i)
            end
        end
    end
    foreach(wait, tasks)
    return nothing
end

function loop_tforeach(js, ntasks::Int)
    OMT.tforeach(1:length(js); ntasks = ntasks) do i
        runjob(js, i)
        return nothing
    end
    return nothing
end

# The arms above flatten *all* `mul!`s of the call into one parallel region.
# That is not achievable: `ket[k+1]` is computed from `ket[k]`, so the chain is
# strictly sequential and only the blocks *within one* `mul!` are independent
# (the `d` closings are mutually independent, but they are the last step). The
# two arms below are what phase 4b could actually build — one parallel region per
# `mul!` — and the gap between them and the flattened arms is the price of the
# dependency structure.
function loop_spawn_grouped(js, nchunks::Int)
    for rng in js.bounds
        n = length(rng)
        n == 0 && continue
        nc = min(nchunks, n)
        tasks = Vector{Task}(undef, nc)
        for c in 1:nc
            lo = first(rng) + div((c - 1) * n, nc)
            hi = first(rng) + div(c * n, nc) - 1
            tasks[c] = Threads.@spawn begin
                for i in lo:hi
                    runjob(js, i)
                end
            end
        end
        foreach(wait, tasks)
    end
    return nothing
end

function loop_tforeach_grouped(js, ntasks::Int)
    for rng in js.bounds
        isempty(rng) && continue
        OMT.tforeach(rng; ntasks = min(ntasks, length(rng))) do i
            runjob(js, i)
            return nothing
        end
    end
    return nothing
end

# Bare `@spawn`-and-wait round trip, `n` tasks, for calibration.
function spawn_roundtrip(n::Int)
    tasks = Vector{Task}(undef, n)
    for c in 1:n
        tasks[c] = Threads.@spawn nothing
    end
    foreach(wait, tasks)
    return nothing
end

# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
struct ProbeRow
    sym::Symbol
    chi::Int
    nthreads::Int
    nblocks::Int           # summed over *all* `mul!`s of the call
    nsec::Int              # blocks inside ONE `mul!` — the loop 4b would thread
    nclose::Int            # blocks in the `d` closings, the largest independent group
    t_kernel::Float64      # µs, full `compute_message!`
    t_serial::Float64      # µs, serial block loop
    t_serial2::Float64     # µs, control (identical code)
    t_spawn::Float64       # µs, `Threads.@spawn` per chunk, nchunks = nthreads, ALL blocks
    t_tf_nt::Float64       # µs, `tforeach`, ntasks = nthreads, ALL blocks
    t_spawn_g::Float64     # µs, `Threads.@spawn`, one region per `mul!` (achievable)
    t_tf_g::Float64        # µs, `tforeach`, one region per `mul!` (achievable)
    t_rt1::Float64         # µs, one-task spawn round trip
    t_rtn::Float64         # µs, nthreads-task spawn round trip
    mean_block::Float64    # µs, serial time / nblocks
    med_block::Float64     # µs, timed alone: the median-flops block
    amdahl::Float64        # t_serial / t_kernel
end

const PROBE_REPS = parse(Int, get(ENV, "CANOPY_PROBE_REPS", "9"))
const PROBE_INNER = parse(Int, get(ENV, "CANOPY_PROBE_INNER", "5"))

function probe_row(sym::Symbol, χ::Int; reps = PROBE_REPS, inner = PROBE_INNER)
    Random.seed!(BENCH_SEED)
    P, V = bench_space(sym)
    state = hex_state(2, 2, χ; P = P, V = V)
    msgs = cold_messages(state)
    edges = collect(outgoing_edges(state, HEX_VERTEX))
    out = compute_message(msgs, state, edges)
    buf = Bumper.default_buffer(Bumper.ResizeBuffer)

    cl = chain_links(msgs, state, edges)
    js = block_jobs(cl, out)
    nb = length(js)
    nt = Threads.nthreads()

    # Cross-check the *structural* census against what the real `blocks(...)`
    # merge-join actually yields: same number of gemms, same multiset of
    # `(m, n, k)`. If these disagree the census table is wrong, not the timings.
    gs = gemm_shapes(space(state[HEX_VERTEX]), cl.d, cl.target_legs)
    predicted = sort(reduce(vcat, (sh for (_, _, sh) in gs)))
    observed = sort(js.shapes)
    predicted == observed || error(
        "$sym χ=$χ: census predicts $(length(predicted)) gemms, kernel does $(length(observed))"
    )

    # compile every arm
    compute_message!(out, msgs, state, edges, BlockedBackend(), buf)
    loop_serial(js); loop_spawn(js, nt); loop_tforeach(js, nt)
    loop_spawn_grouped(js, nt); loop_tforeach_grouped(js, nt)
    spawn_roundtrip(1); spawn_roundtrip(nt)

    tk = Float64[]; ts = Float64[]; ts2 = Float64[]; tsp = Float64[]
    tfn = Float64[]; tspg = Float64[]; tfg = Float64[]; r1 = Float64[]; rn = Float64[]
    for _ in 1:reps
        push!(tk, best(() -> compute_message!(out, msgs, state, edges, BlockedBackend(), buf), inner))
        push!(ts, best(() -> loop_serial(js), inner))
        push!(tsp, best(() -> loop_spawn(js, nt), inner))
        push!(tfn, best(() -> loop_tforeach(js, nt), inner))
        push!(tspg, best(() -> loop_spawn_grouped(js, nt), inner))
        push!(tfg, best(() -> loop_tforeach_grouped(js, nt), inner))
        push!(ts2, best(() -> loop_serial(js), inner))
        push!(r1, best(() -> spawn_roundtrip(1), inner))
        push!(rn, best(() -> spawn_roundtrip(nt), inner))
    end
    m(v) = minimum(v)

    # The genuine median per-block time: time the median-flops gemm on its own.
    fl = [2 * m_ * n_ * k_ for (m_, n_, k_) in js.shapes]
    imed = sortperm(fl)[cld(length(fl), 2)]
    t1 = best(() -> runjob(js, imed), 3)
    ninner = clamp(round(Int, 2000 / max(t1, 1.0e-3)), 3, 200)
    tmed = minimum(best(() -> runjob(js, imed), ninner) for _ in 1:reps)

    nsec = maximum(length(sh) for (_, _, sh) in gs)
    nclose = sum(length(sh) for (kind, _, sh) in gs if kind === :close)
    return ProbeRow(
        sym, χ, nt, nb, nsec, nclose, m(tk), m(ts), m(ts2), m(tsp), m(tfn),
        m(tspg), m(tfg), m(r1), m(rn), m(ts) / nb, tmed, m(ts) / m(tk),
    )
end

const PROBE_SYMS = (:fz2_u1, :fz2_u1_flat, :fz2)
const PROBE_CHIS = (
    isempty(filter(a -> startswith(a, "--chi="), ARGS)) ? (32, 64, 128) :
        Tuple(parse(Int, s) for s in split(split(only(filter(a -> startswith(a, "--chi="), ARGS)), "=")[2], ","))
)

function probe_table(io, rows)
    println(io, "| sym | χ | nthr | nblk tot | **nblk/mul!** | nblk close | kernel µs | serial µs | control µs | flat @spawn | flat tforeach | **grouped @spawn** | **grouped tforeach** | rt(1) µs | rt(nt) µs | mean/blk µs | **med blk µs** | **t_loop/t_kernel** |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows
        @printf(
            io, "| %s | %d | %d | %d | **%d** | %d | %.1f | %.2f | %.2f | %.2f | %.2f | **%.2f** | **%.2f** | %.2f | %.2f | %.3f | **%.3f** | **%.3f** |\n",
            r.sym, r.chi, r.nthreads, r.nblocks, r.nsec, r.nclose,
            r.t_kernel, r.t_serial, r.t_serial2,
            r.t_spawn, r.t_tf_nt, r.t_spawn_g, r.t_tf_g, r.t_rt1, r.t_rtn,
            r.mean_block, r.med_block, r.amdahl,
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Upstream knobs — zero Canopy code
# ---------------------------------------------------------------------------
#
# `TensorKit.TRANSFORMER_THREADS` (`TensorKit.jl:226`) defaults to 1 and gates
# `use_threaded_transform`, which *additionally* requires
# `length(t.data) > Strided.MINTHREADLENGTH` (= 32768), so it is inert on small
# fixtures no matter what the knob says. `blockscheduler` /
# `with_blockscheduler` (`tensors/backends.jl:12-33`) exists but
# `default_blockscheduler` has **no callers** in 0.17.1 — verified by grep — so
# it can only be a no-op. Both facts are asserted at run time below.
struct UpstreamRow
    sym::Symbol
    chi::Int
    dim::Int
    threaded_transform_eligible::Bool
    tt::Vector{Pair{Int, Float64}}   # TRANSFORMER_THREADS => µs
    t_blocksched::Float64            # µs under with_blockscheduler
end

function upstream_row(sym::Symbol, χ::Int; reps = PROBE_REPS, inner = PROBE_INNER, ns = (1, 2, 4, 8))
    Random.seed!(BENCH_SEED)
    P, V = bench_space(sym)
    state = hex_state(2, 2, χ; P = P, V = V)
    msgs = cold_messages(state)
    edges = collect(outgoing_edges(state, HEX_VERTEX))
    out = compute_message(msgs, state, edges)
    buf = Bumper.default_buffer(Bumper.ResizeBuffer)
    T = state[HEX_VERTEX]
    eligible = length(T.data) > 32768

    call() = compute_message!(out, msgs, state, edges, BlockedBackend(), buf)
    call()
    acc = Dict(n => Float64[] for n in ns)
    tbs = Float64[]
    for _ in 1:reps
        for n in ns
            TensorKit.set_num_transformer_threads(min(n, Threads.nthreads()))
            push!(acc[n], best(call, inner))
        end
        TensorKit.set_num_transformer_threads(1)
        push!(tbs, best(() -> TensorKit.with_blockscheduler(call, OMT.DynamicScheduler()), inner))
    end
    TensorKit.set_num_transformer_threads(1)
    return UpstreamRow(
        sym, χ, dim(space(T)), eligible,
        [n => minimum(acc[n]) for n in ns], minimum(tbs),
    )
end

function upstream_table(io, rows)
    ns = isempty(rows) ? Int[] : first.(rows[1].tt)
    println(io, "| sym | χ | dim(T) | data > MINTHREADLENGTH | ", join(("TT=$n µs" for n in ns), " | "), " | with_blockscheduler µs |")
    println(io, "|---|---|---|---|", join(("---" for _ in ns), "|"), "|---|")
    for r in rows
        @printf(io, "| %s | %d | %d | %s | ", r.sym, r.chi, r.dim, r.threaded_transform_eligible)
        for (_, t) in r.tt
            @printf(io, "%.1f | ", t)
        end
        @printf(io, "%.1f |\n", r.t_blocksched)
    end
    return nothing
end

# ---------------------------------------------------------------------------
function main(args = ARGS)
    want(x) = isempty(filter(a -> !startswith(a, "--chi="), args)) || (x in args)
    load0 = _loadavg()
    println("""
    ── block-loop threading probe ───────────────────────────────────────────
      git sha           : $(_gitsha())
      julia             : $(VERSION)
      Threads.nthreads  : $(Threads.nthreads())
      BLAS threads      : $(BLAS.get_num_threads())
      TRANSFORMER_THREADS: $(TensorKit.get_num_transformer_threads())
      Sys.CPU_NAME      : $(Sys.CPU_NAME)
      hostname          : $(gethostname())
      loadavg at start  : $load0
    ─────────────────────────────────────────────────────────────────────────""")

    census = want("--census") ? run_census() : CensusRow[]
    isempty(census) || (census_table(stdout, census); println())

    probe = ProbeRow[]
    if want("--probe")
        for χ in PROBE_CHIS, sym in PROBE_SYMS
            r = probe_row(sym, χ)
            push!(probe, r)
            @printf(
                "%-13s χ=%-4d nthr=%d nblk=%3d(%d/mul)  kernel %9.1f  serial %8.2f  flat %8.2f  grouped %8.2f  amdahl %.3f\n",
                r.sym, r.chi, r.nthreads, r.nblocks, r.nsec, r.t_kernel, r.t_serial,
                min(r.t_spawn, r.t_tf_nt), min(r.t_spawn_g, r.t_tf_g), r.amdahl
            )
            GC.gc()
        end
        println()
        probe_table(stdout, probe)
        println()
    end

    upstream = UpstreamRow[]
    if want("--upstream")
        for χ in PROBE_CHIS, sym in PROBE_SYMS
            push!(upstream, upstream_row(sym, χ))
            GC.gc()
        end
        upstream_table(stdout, upstream)
        println()
    end

    load1 = _loadavg()
    println("loadavg at start: $load0")
    println("loadavg at end  : $load1")

    tag = "t$(Threads.nthreads())"
    mkpath(REPORT_DIR)
    path = joinpath(REPORT_DIR, "blockloop_probe_$tag.md")
    open(path, "w") do io
        println(io, "# Block-loop threading probe (`Threads.nthreads() = $(Threads.nthreads())`)")
        println(io)
        println(io, "Generated by `benchmark/blockloop_probe.jl`; method in that file's header.")
        println(io)
        println(io, "- host: `$(gethostname())`  ·  CPU: `$(Sys.CPU_NAME)`  ·  julia `$(VERSION)`")
        println(io, "- git sha: `$(_gitsha())`  ·  BLAS threads: $(BLAS.get_num_threads())")
        println(io, "- loadavg (1/5/15): `$load0` at start, `$load1` at end")
        println(io, "- $(PROBE_REPS) reps × $(PROBE_INNER) inner calls; min-over-reps of min-over-inner")
        if !isempty(census)
            println(io, "\n## Block census\n")
            census_table(io, census)
        end
        if !isempty(probe)
            println(io, "\n## Threading probe\n")
            println(io, "`control` runs byte-identical code to `serial`; `|control/serial - 1|` is the noise floor.\n")
            probe_table(io, probe)
        end
        if !isempty(upstream)
            println(io, "\n## Upstream knobs\n")
            upstream_table(io, upstream)
        end
    end
    println("wrote $path")
    return (; census, probe, upstream)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
