# §2 — subblock-outer-loop threading feasibility for the blocked message kernel.
#
#   JULIA_NUM_THREADS=N OPENBLAS_NUM_THREADS=1 \
#       julia --project=benchmark --threads=N benchmark/subblock_probe.jl \
#           [--census] [--serial] [--threaded] [--blas]
#
# THE DESIGN UNDER TEST
# ---------------------
# `benchmark/reports/blocked_kernel_followups.md` §2:
#
#   > Use the current kernel as the *inner* part of a multithreaded **outer** loop
#   > that unpacks each symmetry subblock and hands it to the same kernel. Accept a
#   > **per-thread set of output messages**, summed at the end.
#
# This is *not* what `phase4a.md` measured and rejected (that was the coupled-sector
# loop inside one `mul!`); see the scope warning at the top of that report.
#
# WHAT AN INDEPENDENT UNIT IS, AND WHY
# ------------------------------------
# Label every subblock by the map `leg id -> sector on that leg`. Two facts make
# that label the right unit:
#
#   * **It is invariant along both chains.** Messages are sector-diagonal, so
#     absorbing message `j` cannot change `σ_j`; and a relayout is a permutation,
#     which reorders legs and recouples the *internal* tree structure but never
#     changes which sector sits on which external leg. (True for a general fusion
#     style, not just abelian: F-moves and braids both preserve the external
#     labels. For non-abelian a unit holds the several trees sharing one label, and
#     a relayout mixes *within* the unit.)
#   * **It is what the closing sums over.** `out[k]_c = Σ_{f₁} bra[k][f₁,c]† ket[k][f₁,c]`
#     — independent over `f₁`, i.e. over units, and *additive* into `out`. That is
#     precisely why per-thread outputs plus a reduction is the answer and
#     partitioning the output is not.
#
# So the unit count is the **fusion-tree** count (73/121/181/253/337 for `fz2_u1` at
# χ = 8…128) rather than the 7-15 coupled sectors phase 4a was limited by, and the
# `ket[k] → ket[k+1]` dependency becomes sequencing *inside* a thread.
#
# THE TAX THIS PROBE EXISTS TO MEASURE
# ------------------------------------
# `mul!(::AbstractTensorMap, …)` issues **one gemm per coupled sector**, on the
# block that *merges* every fusion tree sharing that sector. Splitting the outer
# loop by unit splits those merged gemms along their `m` axis: at the production
# point `n_mul! × nsectors = 7 × 15 = 105` gemm calls per vertex call become
# `7 × 337 = 2359`, with the same total flops. The production gemms are already
# extremely skinny — census median `(m, n, k) = (2035, 3, 3)` at χ = 128, i.e.
# ≈0.13 flop/byte and squarely memory-bound — so this is a real serial cost that
# has to be paid *before* any threading gain, and §2 does not list it.
#
# `--serial` measures exactly that, and it is the gate: if the decomposition is
# already `x`× slower serially, threading must beat `x` before it breaks even.
#
# ARMS
# ----
# `--census`   unit count, per-unit work (copy elements and gemm flops), the
#              resulting load imbalance, and the gemm-call multiplication above.
#              Pure structure, so it is machine-independent.
# `--serial`   three arms on one fixture, interleaved: the real kernel, the
#              per-*sector* cheat kernel of `plancache_probe.jl` (bitwise identical
#              to the real one, and the honest baseline for a hand-rolled loop), and
#              the per-*unit* cheat kernel built here. Ratio = the decomposition tax.
# `--threaded` the per-unit arm at `Threads.nthreads()`, with per-thread output
#              buffers reduced in fixed order, against the *same call* at
#              `nthreads = 1`. Units are assigned longest-processing-time-first, from
#              the plan-time work estimate: contiguous chunking caps at 2.7× on 8
#              threads because heavy units cluster in enumeration order, and a
#              *dynamic* scheduler would balance as well but would break the bitwise
#              reproducibility §2 asks for (a unit landing on a different thread lands
#              in a different partial sum). Bitwise equality is asserted, not `≈`.
# `--blas`     `benchmark/reports/blocked_kernel_followups.md` §4.1: every
#              measurement in this project pinned `BLAS.set_num_threads(1)`, and the
#              block loop is 57-74 % of the kernel at χ = 128, so letting BLAS use
#              cores is untested free parallelism. It competes with the §2 work for
#              the same cores, which is why it is checked here rather than later.
#
# SCOPE — READ THIS BEFORE QUOTING ANY NUMBER FROM HERE
# -----------------------------------------------------
# **Abelian fixtures only.** The per-sector cheat kernel this probe extends assumes
# one destination subblock per source subblock (`transport!` writes with `β = 0`),
# which holds for `UniqueFusion` and not otherwise. `uses_blocked_kernel` *does*
# accept non-abelian fusion (see `src/backends.jl`), so the non-abelian path is
# genuinely uncovered here; it is measured against the pairwise kernel in
# `benchmark/reports/backend_ab.md` instead. A non-abelian unit is well defined (see
# above) but its relayout mixes trees inside the unit, which this plan cannot
# express.
#
# Measurement discipline follows `benchmark/bench_backend_ab.jl` §5: arms
# interleaved inside each repetition, min-of-inner then min-over-reps, one process
# one fixture, a control arm whose two sides run byte-identical code, and
# `/proc/loadavg` recorded at both ends.

using Canopy
import AlgorithmsInterface as AI
using TensorKit
using LinearAlgebra
using Printf
using Statistics: median
import Bumper

# Brings in `setup.jl`, and the per-sector cheat plan this probe extends:
# `CheatPlan`, `build_plan`, `transport!`, `absorb!`, `close!`, `_transport`,
# `fixture`, `best`. Its own `main()` does not run (its `PROGRAM_FILE` guard).
include("plancache_probe.jl")

const PROBE_DIR = joinpath(@__DIR__, "reports")

# Same fixtures phase 4a timed, so the two reports are directly comparable.
const SUB_SYMS = (:fz2, :fz2_u1, :fz2_u1_flat)
const SUB_CHIS = (32, 64, 128)

const SUB_REPS = parse(Int, get(ENV, "CANOPY_SUB_REPS", "9"))
const SUB_INNER = parse(Int, get(ENV, "CANOPY_SUB_INNER", "5"))

# ---------------------------------------------------------------------------
# Unit labelling
# ---------------------------------------------------------------------------
#
# `label[legid] = sector on leg legid`, canonicalised so it is the *same* tuple
# whichever layout the subblock is read in.
#
# The canonicalisation is the fiddly part and getting it wrong would silently merge
# units. A fusion tree records, for a **codomain** axis, the sector of that space,
# and for a **domain** axis the sector of its dual. Legs move between codomain and
# domain as the layout changes, so the raw `uncoupled` entry flips to its conjugate
# under `Layout(k) → Layout(k±1)` for the two legs that swapped. Conjugating the
# domain entries puts every axis into the one convention `space(W, a)` uses, which
# `permute` preserves. Asserted against `space(·, a)` in `_check_labels`.
function _leg_labels(Wl, legids::NTuple{M, Int}, ncod::Int) where {M}
    sbs = TensorKit.subblockstructure(Wl)
    inv = ntuple(i -> findfirst(==(i), legids)::Int, M)      # leg id -> axis
    labels = map(collect(keys(sbs))) do (g₁, g₂)
        unc = (g₁.uncoupled..., g₂.uncoupled...)
        return ntuple(i -> (inv[i] ≤ ncod ? unc[inv[i]] : dual(unc[inv[i]])), M)
    end
    return labels
end

# The label has to mean the same thing in every layout or two distinct units could
# share one and the partition would be unsound — so check it against something
# *independent* of how it was built: `space(Wl, a)`, the one view of a leg's space
# that `permute` leaves alone.
#
# Two assertions, both at plan time and both free:
#
#   1. axis `a` of the layout carries the same space as leg `legids[a]` of the
#      reference `Wref` — i.e. the layout really is a permutation of the legs we
#      think it is;
#   2. every label entry is a sector that space actually contains, which is what
#      pins down the conjugation convention (dropping the `dual` on the domain
#      entries fails this immediately on any fixture with a dual leg).
# `Wl[a]` is the `HomSpace` accessor `space(::AbstractTensorMap, a)` is built on: the
# space of axis `a` in the all-in-the-codomain convention, i.e. already dualised for
# a domain axis. That is the convention `_leg_labels` canonicalises to.
function _check_labels(Wl, Wref, legids::NTuple{M, Int}, labels) where {M}
    for a in 1:M
        Wl[a] == Wref[legids[a]] ||
            error("axis $a of $Wl is not leg $(legids[a]) of the reference space")
    end
    for lab in labels, a in 1:M
        s = lab[legids[a]]
        TensorKit.hassector(Wl[a], s) ||
            error("label sector $s is absent from axis $a ($(Wl[a])) — wrong convention")
    end
    return nothing
end

# Row range of a subblock inside its coupled-sector block.
#
# TensorKit lays a block out column-major with rows ordered by codomain tree and
# columns by domain tree, and every `Layout(k)` has exactly one domain leg, hence
# one domain tree — so a subblock is a *contiguous band of rows* spanning all
# columns, and `mul!` on `view(block, rows, :)` reaches BLAS with `lda = m` and
# copies nothing. `offset - blockoffset` is that band's start; asserted against the
# strided spec by the caller.
function _rowband(sub, blockoffset::Int, ncod::Int)
    sz, _, off = sub
    nrows = prod(ntuple(i -> sz[i], ncod))
    start = off - blockoffset
    return (start + 1):(start + nrows)
end

# ---------------------------------------------------------------------------
# The per-unit plan
# ---------------------------------------------------------------------------
#
# Every field is indexed by unit first, so one unit's whole chain is a walk down
# structs-of-vectors with no `HomSpace` in sight — the same discipline as
# `CheatPlan`, which is what makes the serial A/B against it a fair fight rather
# than a comparison of bookkeeping styles.

# One transport, partitioned by unit: `data[u]` is the `(coeff, dst, src)` triples
# whose *source* subblock belongs to unit `u`.
#
# `p` is a type parameter, not a plain `::Tuple` field: the timed loop passes it
# straight to `TO.tensoradd!`, so an abstract field type would make the unit arm pay
# a dynamic dispatch per subblock that the per-sector arm it is being compared
# against does not — i.e. it would manufacture the very tax this probe measures.
struct UnitTransport{TR, P}
    data::Vector{Vector{TR}}
    p::P
end

# One absorption, per unit: gemm `C[rows, :] = A[rows, :] * B` inside coupled
# sector `σ_k` of the layout, `B` the (transposed) message block.
struct UnitAbsorb
    rows::Vector{UnitRange{Int}}   # band inside the block
    blockoff::Vector{Int}          # block offset in the layout buffer
    m::Vector{Int}                 # block row count (the gemm's lda)
    n::Vector{Int}                 # block column count = message block size
    msgoff::Vector{Int}
end

# One closing, per unit: `out_c += bra[rows, :]' * ket[rows, :]`, accumulating.
struct UnitClose
    rows::Vector{UnitRange{Int}}
    blockoff::Vector{Int}
    m::Vector{Int}
    n::Vector{Int}
    outoff::Vector{Int}
end

struct UnitPlan{TRM, TE, TS}
    nunits::Int
    d::Int
    legmin::Int
    legmax::Int
    dimT::Int
    dimmsg::Vector{Int}
    dimout::Vector{Int}
    target_legs::Vector{Int}
    msg_tr::Vector{TRM}                 # per leg — NOT per unit, see below
    enter_ket::TE
    enter_bra::TE
    up::Vector{TS}
    down::Vector{TS}
    absorb_ket::Vector{UnitAbsorb}      # per chain step k
    absorb_bra::Vector{UnitAbsorb}
    close::Vector{UnitClose}            # per target
    # census
    work_copy::Vector{Int}              # elements copied, per unit
    work_flop::Vector{Int}              # 2mnk summed over the unit's gemms
    ngemm_unit::Int
    ngemm_sector::Int
end

# The `d` message transposes are *not* per-unit and must stay outside the parallel
# region: a message's subblock is labelled by one leg's sector alone, so it is
# shared by every unit carrying that sector. They are χ²-sized and there are `d` of
# them, against `≈2d` chain links of `dim(P)·χ^d`, so leaving them serial costs
# nothing measurable — and threading them would need a barrier before the chains.

function _unit_transport(Wdst, Wsrc, p, labels_src, ui, θ = nothing)
    data, _ = _transport(Wdst, Wsrc, p, θ)
    TR = eltype(data)
    parts = [TR[] for _ in 1:length(ui.units)]
    for i in eachindex(data)
        push!(parts[ui[labels_src[i]]], data[i])
    end
    return UnitTransport{TR, typeof(p)}(parts, p)
end

# Elements this transport writes on behalf of each unit — the copy half of the
# census. Counted from the destination subblock *sizes* the transport actually
# carries, so an unused chain slot (an empty schedule) contributes nothing and the
# figure tracks the work a call really does.
function _add_copy_work!(work::Vector{Int}, tr::UnitTransport)
    for u in eachindex(tr.data)
        for (_, sd, _) in tr.data[u]
            work[u] += prod(sd[1])
        end
    end
    return work
end

# Label -> unit index, plus the label list in a fixed order. The order is the
# enumeration order of `W`'s own subblocks, so it is deterministic and the
# reduction below can be done in a fixed order without sorting anything.
struct UnitIndex{L}
    units::Vector{L}
    index::Dict{L, Int}
end
UnitIndex(labels::Vector{L}) where {L} = UnitIndex{L}(labels, Dict(l => i for (i, l) in enumerate(labels)))
Base.getindex(ui::UnitIndex{L}, l::L) where {L} = ui.index[l]

function build_unit_plan(state, msgs, edges)
    v = first(first(edges))
    T = state[v]
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

    # leg ids in axis order, per space
    legids_W = allinds                                   # `W` is already `Layout`-0-ish: (phys, ℓ₁ … ℓ_N)
    legids_L(k) = (TupleTools.deleteat(allinds, k + 1)..., k + 1)

    # units are enumerated in `W`'s subblock order
    labels_W = _leg_labels(W, legids_W, 1)
    ui = UnitIndex(labels_W)
    nunits = length(ui.units)
    length(ui.index) == nunits || error("W's subblock labels are not distinct")
    _check_labels(W, W, legids_W, labels_W)

    incoming = map(n -> msgs[DirectedEdge(n, v)], nbrs)
    Wmsg = map(space, incoming)
    Wmt = map(Wm -> permute(Wm, pmsg), Wmsg)
    WL = [permute(W, layout(k)) for k in 1:d]
    Wout = [space(msgs[DirectedEdge(v, nbrs[k])]) for k in 1:d]

    # This is the load-bearing assertion of the whole probe: every layout must carry
    # exactly the same *set* of unit labels as `W`, one subblock each. If it did not,
    # the partition would not be closed under a chain step and units would not be
    # independent. `length == nunits` together with `Set(...) == Set(...)` is the
    # bijection — one alone would not be (abelian fusion is what makes it hold; a
    # non-abelian space would give several subblocks per label, which is why this
    # probe is abelian-only and why that shows up as a *failure* here rather than as
    # silently wrong numbers).
    labels_L = [_leg_labels(WL[k], legids_L(k), M - 1) for k in 1:d]
    for k in 1:d
        _check_labels(WL[k], W, legids_L(k), labels_L[k])
        length(labels_L[k]) == nunits ||
            error("layout $k has $(length(labels_L[k])) subblocks, W has $nunits (non-abelian?)")
        Set(labels_L[k]) == Set(ui.units) || error("layout $k does not preserve the unit labels")
    end

    msg_tr = [_transport(Wmt[j], Wmsg[j], pmsg, f -> twist(f[1].coupled)) for j in 1:d]

    enter_ket = _unit_transport(
        WL[1], W, layout(1), labels_W, ui,
        dual_phys ? (f -> twist(f[1].uncoupled[1])) : nothing,
    )
    enter_bra = _unit_transport(WL[d], W, layout(d), labels_W, ui)

    _blank(t::UnitTransport{TR, P}) where {TR, P} =
        UnitTransport{TR, P}([TR[] for _ in 1:nunits], t.p)
    up1 = _unit_transport(WL[min(2, d)], WL[1], pswap(2), labels_L[1], ui)
    up = [
        k ≤ legmax - 1 ?
            _unit_transport(WL[k + 1], WL[k], pswap(k + 1), labels_L[k], ui) :
            _blank(up1) for k in 1:d
    ]
    down = [
        (legmin ≤ k ≤ d - 1) ?
            _unit_transport(WL[k], WL[k + 1], pswap(k + 1), labels_L[k + 1], ui) :
            _blank(up1) for k in 1:d
    ]

    bsL = [TensorKit.blockstructure(WL[k]) for k in 1:d]
    sbL = [TensorKit.subblockstructure(WL[k]) for k in 1:d]
    bsM = [TensorKit.blockstructure(Wmt[k]) for k in 1:d]
    bsO = [TensorKit.blockstructure(Wout[k]) for k in 1:d]

    # Census, per unit. `work_copy` is over the transports a call actually issues —
    # `up`/`down` carry empty schedules for the chain slots a partial target set does
    # not reach, so those contribute nothing.
    work_copy = zeros(Int, nunits)
    work_flop = zeros(Int, nunits)
    _add_copy_work!(work_copy, enter_ket)
    _add_copy_work!(work_copy, enter_bra)
    for k in 1:d
        _add_copy_work!(work_copy, up[k])
        _add_copy_work!(work_copy, down[k])
    end

    # Row band of every subblock of layout `k`, indexed by unit. The coupled sector
    # of `Layout(k)` is the sector on leg `k`, so the unit label fixes which block a
    # unit lives in and each unit contributes exactly one band.
    function bands(k)
        rows = Vector{UnitRange{Int}}(undef, nunits)
        blockoff = zeros(Int, nunits)
        ms = zeros(Int, nunits)
        ns = zeros(Int, nunits)
        secs = Vector{Any}(undef, nunits)
        for (i, (fp, sub)) in enumerate(pairs(sbL[k]))
            u = ui[labels_L[k][i]]
            c = fp[1].coupled
            (dd, r) = bsL[k][c]
            bo = first(r) - 1
            band = _rowband(sub, bo, M - 1)
            # A subblock must be exactly `view(block, band, :)`, or the gemms below
            # are wrong. Two conditions: it spans all columns with the block's own
            # column stride (skipped when there is only one column, where the stride
            # is unconstrained), and it fits inside the block.
            (sub[1][M] == 1 || sub[2][M] == dd[1]) ||
                error("subblock column stride $(sub[2][M]) ≠ block rows $(dd[1]) in layout $k")
            sub[1][M] == dd[2] ||
                error("subblock spans $(sub[1][M]) of $(dd[2]) columns in layout $k")
            last(band) ≤ dd[1] || error("band $band overruns $(dd[1]) rows in layout $k")
            rows[u] = band
            blockoff[u] = bo
            ms[u] = dd[1]
            ns[u] = dd[2]
            secs[u] = c
        end
        return rows, blockoff, ms, ns, secs
    end

    banded = [bands(k) for k in 1:d]

    function absorb(k, j)
        rows, blockoff, ms, ns, secs = banded[k]
        msgoff = zeros(Int, nunits)
        for u in 1:nunits
            c = secs[u]
            haskey(bsM[j], c) ||
                error("message $j has no block for sector $c — the cheat's zero-fill path")
            msgoff[u] = first(bsM[j][c][2]) - 1
            work_flop[u] += 2 * length(rows[u]) * ns[u] * ns[u]
        end
        return UnitAbsorb(rows, blockoff, ms, ns, msgoff)
    end

    absorb_ket = [
        k ≤ legmax - 1 ? absorb(k, k) :
            UnitAbsorb(fill(1:0, nunits), zeros(Int, nunits), zeros(Int, nunits), zeros(Int, nunits), zeros(Int, nunits))
            for k in 1:d
    ]
    absorb_bra = [
        (legmin ≤ k ≤ d - 1) ? absorb(k + 1, k + 1) :
            UnitAbsorb(fill(1:0, nunits), zeros(Int, nunits), zeros(Int, nunits), zeros(Int, nunits), zeros(Int, nunits))
            for k in 1:d
    ]

    close = UnitClose[]
    for k in target_legs
        rows, blockoff, ms, ns, secs = banded[k]
        outoff = zeros(Int, nunits)
        for u in 1:nunits
            c = secs[u]
            haskey(bsO[k], c) || error("output at leg $k has no block for sector $c")
            (odd, orng) = bsO[k][c]
            odd == (ns[u], ns[u]) ||
                error("closing block mismatch at leg $k: out $odd vs layout column count $(ns[u])")
            outoff[u] = first(orng) - 1
            work_flop[u] += 2 * length(rows[u]) * ns[u] * ns[u]
        end
        push!(close, UnitClose(rows, blockoff, ms, ns, outoff))
    end

    ngemm_unit = nunits * (count(k -> k ≤ legmax - 1, 1:d) + count(k -> legmin ≤ k ≤ d - 1, 1:d) + length(target_legs))
    ngemm_sector = sum(length(bsL[k]) for k in 1:d if k ≤ legmax - 1; init = 0) +
        sum(length(bsL[k + 1]) for k in 1:d if legmin ≤ k ≤ d - 1; init = 0) +
        sum(length(bsO[k]) for k in target_legs; init = 0)

    return UnitPlan(
        nunits, d, legmin, legmax, dim(W), [dim(Wm) for Wm in Wmsg],
        [dim(Wout[k]) for k in target_legs], collect(target_legs),
        msg_tr, enter_ket, enter_bra, up, down, absorb_ket, absorb_bra, close,
        work_copy, work_flop, ngemm_unit, ngemm_sector,
    )
end

# ---------------------------------------------------------------------------
# The per-unit call
# ---------------------------------------------------------------------------

# Buffers a *thread* owns for the duration of a call. `ket`/`bra` are full-size
# layout buffers shared by every thread — each unit writes only its own disjoint
# bands, so no locking is needed — while `tmp` and `out` must be per-thread: `tmp`
# is reused across chain steps (two units at different `k` would race), and `out`
# is where the reduction happens.
struct UnitBuffers{T}
    ket::Vector{Vector{T}}
    bra::Vector{Vector{T}}
    tmp::Vector{Vector{T}}          # per thread
    mt::Vector{Vector{T}}
    out::Vector{Vector{Vector{T}}}  # [thread][target]
end

function UnitBuffers(::Type{T}, pl::UnitPlan, nthr::Int) where {T}
    return UnitBuffers{T}(
        [Vector{T}(undef, pl.dimT) for _ in 1:pl.d],
        [Vector{T}(undef, pl.dimT) for _ in 1:pl.d],
        [Vector{T}(undef, pl.dimT) for _ in 1:nthr],
        [Vector{T}(undef, n) for n in pl.dimmsg],
        [[Vector{T}(undef, n) for n in pl.dimout] for _ in 1:nthr],
    )
end

function unit_transport!(tr::UnitTransport, u::Int, dst::Vector{T}, src::Vector{T}, backend, allocator) where {T}
    data = tr.data[u]
    @inbounds for i in eachindex(data)
        coeff, sd, ss = data[i]
        TO.tensoradd!(
            StridedView(dst, sd...), StridedView(src, ss...), tr.p, false,
            coeff, VZero(), backend, allocator,
        )
    end
    return nothing
end

# `view(reshape(view(data, blk), m, n), rows, :)` is a `StridedMatrix` with
# `lda = m`, so `mul!` reaches `gemm!` directly and BLAS copies nothing — this is
# the whole reason a *band* is the right granularity rather than a repacked buffer.
@inline function _band(data::Vector{T}, off::Int, m::Int, n::Int, rows::UnitRange{Int}) where {T}
    blk = reshape(view(data, (off + 1):(off + m * n)), m, n)
    return view(blk, rows, :)
end

function unit_absorb!(a::UnitAbsorb, u::Int, dst::Vector{T}, src::Vector{T}, msg::Vector{T}, adj::Bool) where {T}
    @inbounds begin
        rows = a.rows[u]
        isempty(rows) && return nothing
        off, m, n = a.blockoff[u], a.m[u], a.n[u]
        C = _band(dst, off, m, n, rows)
        A = _band(src, off, m, n, rows)
        B = reshape(view(msg, (a.msgoff[u] + 1):(a.msgoff[u] + n * n)), n, n)
        adj ? mul!(C, A, adjoint(B)) : mul!(C, A, B)
    end
    return nothing
end

function unit_close!(c::UnitClose, u::Int, o::Vector{T}, bra::Vector{T}, ket::Vector{T}) where {T}
    @inbounds begin
        rows = c.rows[u]
        off, m, n = c.blockoff[u], c.m[u], c.n[u]
        A = _band(bra, off, m, n, rows)
        B = _band(ket, off, m, n, rows)
        C = reshape(view(o, (c.outoff[u] + 1):(c.outoff[u] + n * n)), n, n)
        mul!(C, adjoint(A), B, one(T), one(T))     # accumulate: many units per block
    end
    return nothing
end

# One unit's complete, independent chain: entry, ket chain up, bra chain down, and
# the closings accumulated into *this thread's* outputs.
function unit_chain!(pl::UnitPlan, b::UnitBuffers, thr::Int, u::Int, Tdata, backend, allocator)
    tmp = b.tmp[thr]
    unit_transport!(pl.enter_ket, u, b.ket[1], Tdata, backend, allocator)
    for k in 1:(pl.legmax - 1)
        unit_absorb!(pl.absorb_ket[k], u, tmp, b.ket[k], b.mt[k], false)
        unit_transport!(pl.up[k], u, b.ket[k + 1], tmp, backend, allocator)
    end
    unit_transport!(pl.enter_bra, u, b.bra[pl.d], Tdata, backend, allocator)
    for k in (pl.d - 1):-1:pl.legmin
        unit_absorb!(pl.absorb_bra[k], u, tmp, b.bra[k + 1], b.mt[k + 1], true)
        unit_transport!(pl.down[k], u, b.bra[k], tmp, backend, allocator)
    end
    for i in eachindex(pl.target_legs)
        unit_close!(pl.close[i], u, b.out[thr][i], b.bra[pl.target_legs[i]], b.ket[pl.target_legs[i]])
    end
    return nothing
end

# Sum the per-thread outputs into thread 1's, in **fixed** thread order so the
# result is bitwise reproducible: identical to the serial arm at `nthr == 1` and
# identical between two runs at the same `nthr`. Floating-point addition is not
# associative, so an unordered reduction would make the kernel non-deterministic.
function unit_reduce!(b::UnitBuffers, nthr::Int)
    for t in 2:nthr
        for i in eachindex(b.out[1])
            axpy!(true, b.out[t][i], b.out[1][i])
        end
    end
    return b.out[1]
end

function unit_call_serial!(pl::UnitPlan, b::UnitBuffers, Tdata, msgdata, backend, allocator)
    for j in 1:pl.d
        transport!(pl.msg_tr[j], b.mt[j], msgdata[j], backend, allocator)
    end
    for o in b.out[1]
        fill!(o, zero(eltype(o)))
    end
    for u in 1:pl.nunits
        unit_chain!(pl, b, 1, u, Tdata, backend, allocator)
    end
    return b.out[1]
end

# Deterministic LPT assignment over units, precomputed once. Contiguous chunking was
# tried first and the census bound says why it is wrong: heavy units *cluster* in
# enumeration order (neighbouring units share coupled-sector blocks), so contiguous
# chunks cap at 2.7× on 8 threads against a bound of 8.0×. Measuring that would have
# been measuring a strawman decomposition, which is the mistake `phase4a.md` made at
# a coarser granularity.
#
# `chunks` is a pure function of the plan, so the unit -> thread map — and hence the
# per-thread partial sums and the fixed-order reduction — are reproducible.
unit_chunks(pl::UnitPlan, nthr::Int) = _lpt_chunks(pl.work_copy .+ pl.work_flop, nthr)

function unit_call_threaded!(
        pl::UnitPlan, b::UnitBuffers, Tdata, msgdata, backend, allocator,
        nthr::Int, chunks::Vector{Vector{Int}},
    )
    for j in 1:pl.d
        transport!(pl.msg_tr[j], b.mt[j], msgdata[j], backend, allocator)
    end
    for t in 1:nthr, o in b.out[t]
        fill!(o, zero(eltype(o)))
    end
    tasks = map(1:nthr) do t
        Threads.@spawn begin
            for u in chunks[t]
                unit_chain!(pl, b, t, u, Tdata, backend, allocator)
            end
        end
    end
    foreach(wait, tasks)
    return unit_reduce!(b, nthr)
end

# Dynamic variant: `nthr` long-lived tasks pulling unit indices off one atomic
# counter. Self-balancing with no work model at all — which is the point. The
# deterministic LPT assignment above needs a plan-time estimate of each unit's cost,
# and that estimate weights copy elements against gemm flops 1:1, a proxy nobody has
# calibrated. If this arm matches LPT, the whole work model can be deleted.
#
# Not bitwise reproducible run to run, deliberately: which unit lands in which
# per-thread partial sum now depends on scheduling. (Fixed-order reduction is kept, so
# only the *partitioning* varies.)
#
# The risk this arm is measured against is **stragglers**, not overhead: pulling in
# index order gives the heaviest unit no priority, and at χ = 128 the heaviest
# `fz2_u1` unit is 16.6× the mean against a 42× per-thread share, so grabbing it last
# could add ~40 % to one thread's span. LPT sorts descending precisely to prevent that.
# One task per worker rather than one per unit, so the atomic is touched `nunits`
# times and `@spawn` only `nthr` times.
function unit_call_dynamic!(
        pl::UnitPlan, b::UnitBuffers, Tdata, msgdata, backend, allocator, nthr::Int,
    )
    for j in 1:pl.d
        transport!(pl.msg_tr[j], b.mt[j], msgdata[j], backend, allocator)
    end
    for t in 1:nthr, o in b.out[t]
        fill!(o, zero(eltype(o)))
    end
    next = Threads.Atomic{Int}(1)
    nunits = pl.nunits
    tasks = map(1:nthr) do t
        Threads.@spawn begin
            while true
                u = Threads.atomic_add!(next, 1)
                u > nunits && break
                unit_chain!(pl, b, t, u, Tdata, backend, allocator)
            end
        end
    end
    foreach(wait, tasks)
    return unit_reduce!(b, nthr)
end

# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

const SCHED_THREADS = (2, 4, 8, 16)

struct CensusRow
    sym::Symbol
    chi::Int
    d::Int
    nunits::Int
    dim::Int
    ngemm_sector::Int
    ngemm_unit::Int
    copy_min::Int
    copy_med::Float64
    copy_max::Int
    flop_min::Int
    flop_med::Float64
    flop_max::Int
    imbalance::Float64          # max unit / mean unit
    dyn_bound::Vector{Float64}  # per SCHED_THREADS: best speedup any schedule can reach
    stat_bound::Vector{Float64} # …contiguous chunking
    lpt_bound::Vector{Float64}  # …deterministic longest-processing-time-first
end

# Longest-processing-time-first assignment of units to `t` threads: sort by work
# descending, hand each unit to the currently-least-loaded thread.
#
# **Why not work stealing.** §2's hazard table asks for a fixed reduction order and
# bitwise reproducibility — `nthreads == 1` equal to serial, and two runs at the same
# `nthreads` equal to each other. Those two requirements rule out a dynamic
# scheduler: with per-thread output buffers, a unit that lands on a different thread
# between runs lands in a different partial sum, and floating-point addition is not
# associative, so the totals differ in the last bits. A *deterministic* balanced
# assignment gets the load balance without giving that up, which is why the work
# estimate is computed at plan time rather than discovered at run time.
#
# Ties broken by unit index (`sortperm` is stable) so the assignment is a pure
# function of the plan and nothing else.
function _lpt_chunks(w::Vector{Int}, t::Int)
    chunks = [Int[] for _ in 1:t]
    load = zeros(Int, t)
    for u in sortperm(w; rev = true)
        i = argmin(load)
        push!(chunks[i], u)
        load[i] += w[u]
    end
    return chunks
end

# Three scheduling bounds from the per-unit work vector. The gaps between them are
# the argument for how much scheduling machinery to build.
#
# `dyn` is the classic bound for independent tasks on `t` workers: no schedule can
# finish before `max(longest task, total/t)`. `stat` is *contiguous* chunking, the
# obvious first implementation. `lpt` is the deterministic balanced assignment above.
# All three are evaluated on the real work vector and the real assignment, so they
# are not models.
function _sched_bounds(w::Vector{Int}, nthreads)
    total = sum(w)
    n = length(w)
    dyn = Float64[]
    stat = Float64[]
    lpt = Float64[]
    for t in nthreads
        push!(dyn, total / max(maximum(w), total / t))
        contig = [(1 + ((i - 1) * n) ÷ t):((i * n) ÷ t) for i in 1:t]
        push!(stat, total / maximum(sum(@view w[c]) for c in contig))
        push!(lpt, total / maximum(sum(@view w[c]) for c in _lpt_chunks(w, t)))
    end
    return dyn, stat, lpt
end

function census_row(sym::Symbol, χ::Int)
    state, msgs, edges, _ = fixture(sym, χ)
    pl = build_unit_plan(state, msgs, edges)
    # Copy and flop counts are not in the same unit, and the kernel is a mix of the
    # two — at production χ roughly 40 % transports and 58 % gemms by direct timing
    # (`phase4a.md`). Weighting them 1:1 is a *proxy*, so read the bounds below as
    # "is the imbalance anywhere near binding" and not as a forecast.
    w = pl.work_copy .+ pl.work_flop
    dyn, stat, lpt = _sched_bounds(w, SCHED_THREADS)
    return CensusRow(
        sym, χ, pl.d, pl.nunits, pl.dimT, pl.ngemm_sector, pl.ngemm_unit,
        minimum(pl.work_copy), median(pl.work_copy), maximum(pl.work_copy),
        minimum(pl.work_flop), median(pl.work_flop), maximum(pl.work_flop),
        maximum(w) / (sum(w) / length(w)), dyn, stat, lpt,
    )
end

struct SerialRow
    sym::Symbol
    chi::Int
    nunits::Int
    t_real::Float64
    t_sector::Float64
    t_unit::Float64
    t_control::Float64
    tax::Float64            # t_unit / t_sector, > 1 means the decomposition costs
    vs_real::Float64        # t_unit / t_real
    control_dev::Float64
    maxdiff_sector::Float64
    maxdiff_unit::Float64
end

function _arms(sym::Symbol, χ::Int)
    state, msgs, edges, out = fixture(sym, χ)
    v = HEX_VERTEX
    T = state[v]
    nbrs = neighbors(state, v)
    backend = TO.DefaultBackend()
    allocator = TO.DefaultAllocator()

    plS = build_plan(state, msgs, edges)
    bS = CheatBuffers(scalartype(T), plS)
    plU = build_unit_plan(state, msgs, edges)
    msgdata = [msgs[DirectedEdge(n, v)].data for n in nbrs]

    ref = compute_message(msgs, state, edges, BlockedBackend(), allocator)
    cheat_call!(plS, bS, T.data, msgdata, backend, allocator)
    md_s = maximum(
        norm(bS.out[i] - ref[i].data) / max(norm(ref[i].data), eps()) for i in eachindex(ref)
    )
    return (; state, msgs, edges, out, T, plS, bS, plU, msgdata, backend, allocator, ref, md_s)
end

function serial_row(sym::Symbol, χ::Int; reps = SUB_REPS, inner = SUB_INNER)
    a = _arms(sym, χ)
    a.md_s < 1.0e-12 || error("$sym χ=$χ: per-sector cheat disagrees by $(a.md_s)")

    bU = UnitBuffers(scalartype(a.T), a.plU, 1)
    o = unit_call_serial!(a.plU, bU, a.T.data, a.msgdata, a.backend, a.allocator)
    md_u = maximum(
        norm(o[i] - a.ref[i].data) / max(norm(a.ref[i].data), eps()) for i in eachindex(a.ref)
    )
    md_u < 1.0e-12 || error("$sym χ=$χ: per-unit kernel disagrees by $md_u")

    buf = Bumper.default_buffer(Bumper.ResizeBuffer)
    real_call() = compute_message!(a.out, a.msgs, a.state, a.edges, BlockedBackend(), buf)
    sector() = cheat_call!(a.plS, a.bS, a.T.data, a.msgdata, a.backend, a.allocator)
    unit() = unit_call_serial!(a.plU, bU, a.T.data, a.msgdata, a.backend, a.allocator)
    real_call(); sector(); unit()

    tr = Float64[]; ts = Float64[]; tu = Float64[]; tc = Float64[]
    for _ in 1:reps
        push!(tr, best(real_call, inner))
        push!(ts, best(sector, inner))
        push!(tu, best(unit, inner))
        push!(tc, best(unit, inner))          # control: the unit arm against itself
    end
    mr, ms, mu, mc = minimum(tr), minimum(ts), minimum(tu), minimum(tc)
    return SerialRow(
        sym, χ, a.plU.nunits, mr, ms, mu, mc, mu / ms, mu / mr,
        abs(mc / mu - 1), a.md_s, md_u,
    )
end

struct ThreadRow
    sym::Symbol
    chi::Int
    nunits::Int
    nthreads::Int
    t_serial::Float64
    t_threaded::Float64
    t_control::Float64
    speedup::Float64        # serial / threaded
    control_dev::Float64
    reproducible::Bool      # two runs at this `nthreads`, bitwise
    maxdiff_real::Float64   # threaded result vs the real kernel — the race check
    maxdiff_ser::Float64    # threaded result vs `nthreads = 1`, at rounding level
    t_reduce::Float64       # µs spent in the fixed-order reduction
    t_dynamic::Float64      # atomic-counter pull loop, no work model
    dyn_speedup::Float64
    dyn_repro::Bool         # expected `false` — recorded, not required
    maxdiff_dyn::Float64    # dynamic result vs the real kernel — its race check
end

function thread_row(sym::Symbol, χ::Int, nthr::Int; reps = SUB_REPS, inner = SUB_INNER)
    a = _arms(sym, χ)
    Tn = scalartype(a.T)

    # WHAT "BITWISE" CAN AND CANNOT MEAN HERE — §2's hazard table asks for both
    # "`nthreads == 1` bitwise equal to serial" and "two runs at the same `nthreads`
    # bitwise equal to each other", and only the second of those is a property of the
    # *threading*. Partitioning a sum across `t` buffers and adding the buffers
    # regroups the additions, and floating-point addition is not associative, so a
    # `t`-thread result is **not** bitwise equal to a 1-thread one and no
    # implementation choice can make it so. Demanding it would either be unsatisfiable
    # or force one unit per thread.
    #
    # So three checks, each testing a different thing:
    #
    #   `reproducible`   two runs at this `nthreads`, bitwise — the determinism that
    #                    *is* achievable, and what a fixed unit→thread map plus a
    #                    fixed reduction order buys. This is the one to assert in a test.
    #   `maxdiff_real`   the threaded result against the real kernel. **This is the
    #                    race check** and it is the reason a speedup may be quoted at
    #                    all: disjoint bands are only disjoint if the plan says so, and
    #                    a wrong band would show up here rather than as a crash.
    #   `maxdiff_ser`    threaded against `nthreads = 1`; expected at rounding level,
    #                    and a value far above it means something worse than regrouping.
    b1 = UnitBuffers(Tn, a.plU, 1)
    chunks1 = unit_chunks(a.plU, 1)
    ref1 = unit_call_threaded!(a.plU, b1, a.T.data, a.msgdata, a.backend, a.allocator, 1, chunks1)
    serial_copy = [copy(o) for o in ref1]

    bT = UnitBuffers(Tn, a.plU, nthr)
    chunks = unit_chunks(a.plU, nthr)
    thr = unit_call_threaded!(a.plU, bT, a.T.data, a.msgdata, a.backend, a.allocator, nthr, chunks)
    thr_copy = [copy(o) for o in thr]
    thr2 = unit_call_threaded!(a.plU, bT, a.T.data, a.msgdata, a.backend, a.allocator, nthr, chunks)
    reproducible = all(thr_copy[i] == thr2[i] for i in eachindex(thr2))

    _rel(x, y) = maximum(norm(x[i] - y[i]) / max(norm(y[i]), eps()) for i in eachindex(y))
    maxdiff_real = _rel(thr_copy, [r.data for r in a.ref])
    maxdiff_ser = _rel(thr_copy, serial_copy)
    maxdiff_real < 1.0e-12 ||
        error("$sym χ=$χ t=$nthr: THREADED RESULT IS WRONG, rel diff $maxdiff_real vs the real kernel")

    # The dynamic arm gets its own buffers so the two never share a `tmp`, and its own
    # race check. `dyn_repro` is recorded rather than required — this arm exists because
    # bitwise reproducibility is *not* a requirement, so the interesting question about it
    # is only whether it schedules as well as LPT without needing a work model.
    bD = UnitBuffers(Tn, a.plU, nthr)
    dyn = unit_call_dynamic!(a.plU, bD, a.T.data, a.msgdata, a.backend, a.allocator, nthr)
    dyn_copy = [copy(o) for o in dyn]
    dyn2 = unit_call_dynamic!(a.plU, bD, a.T.data, a.msgdata, a.backend, a.allocator, nthr)
    dyn_repro = all(dyn_copy[i] == dyn2[i] for i in eachindex(dyn2))
    maxdiff_dyn = _rel(dyn_copy, [r.data for r in a.ref])
    maxdiff_dyn < 1.0e-12 ||
        error("$sym χ=$χ t=$nthr: DYNAMIC RESULT IS WRONG, rel diff $maxdiff_dyn vs the real kernel")

    serial() = unit_call_threaded!(a.plU, b1, a.T.data, a.msgdata, a.backend, a.allocator, 1, chunks1)
    threaded() = unit_call_threaded!(a.plU, bT, a.T.data, a.msgdata, a.backend, a.allocator, nthr, chunks)
    dynamic() = unit_call_dynamic!(a.plU, bD, a.T.data, a.msgdata, a.backend, a.allocator, nthr)
    reduce_only() = unit_reduce!(bT, nthr)
    serial(); threaded(); dynamic(); reduce_only()

    # LPT and dynamic are timed inside the *same* repetition so the pair is adjacent in
    # time — the whole point of `bench_backend_ab.jl`'s method, and the comparison the
    # implementation choice rests on.
    ts = Float64[]; tt = Float64[]; td = Float64[]; tc = Float64[]; trd = Float64[]
    for _ in 1:reps
        push!(ts, best(serial, inner))
        push!(tt, best(threaded, inner))
        push!(td, best(dynamic, inner))
        push!(tc, best(threaded, inner))
        push!(trd, best(reduce_only, inner))
    end
    ms, mt, md, mc = minimum(ts), minimum(tt), minimum(td), minimum(tc)
    return ThreadRow(
        sym, χ, a.plU.nunits, nthr, ms, mt, mc, ms / mt, abs(mc / mt - 1),
        reproducible, maxdiff_real, maxdiff_ser, minimum(trd),
        md, ms / md, dyn_repro, maxdiff_dyn,
    )
end

# §4.1 — BLAS threads on the *real* kernel. Independent of everything above, and
# checked first because it competes for the same cores.
struct BlasRow
    sym::Symbol
    chi::Int
    times::Vector{Float64}      # per BLAS thread count
    nblas::Vector{Int}
    control_dev::Float64
end

function blas_row(sym::Symbol, χ::Int, counts; reps = SUB_REPS, inner = SUB_INNER)
    state, msgs, edges, out = fixture(sym, χ)
    buf = Bumper.default_buffer(Bumper.ResizeBuffer)
    call() = compute_message!(out, msgs, state, edges, BlockedBackend(), buf)
    n0 = BLAS.get_num_threads()
    ts = [Float64[] for _ in counts]
    tc = Float64[]
    try
        BLAS.set_num_threads(first(counts)); call()
        for _ in 1:reps
            for (i, n) in enumerate(counts)
                BLAS.set_num_threads(n)
                push!(ts[i], best(call, inner))
            end
            BLAS.set_num_threads(first(counts))
            push!(tc, best(call, inner))
        end
    finally
        BLAS.set_num_threads(n0)
    end
    mins = map(minimum, ts)
    return BlasRow(sym, χ, mins, collect(counts), abs(minimum(tc) / mins[1] - 1))
end

# ---------------------------------------------------------------------------
function probe_main(args = ARGS)
    want(x) = isempty(args) || (x in args)
    load0 = _loadavg()
    nthr = Threads.nthreads()
    println("""
    ── subblock-outer-loop probe ────────────────────────────────────────────
      git sha           : $(_gitsha())
      julia             : $(VERSION)
      Threads.nthreads  : $nthr
      BLAS threads      : $(BLAS.get_num_threads())
      Sys.CPU_NAME      : $(Sys.CPU_NAME)
      hostname          : $(gethostname())
      loadavg at start  : $load0
      χ grid            : $(SUB_CHIS)
      reps × inner      : $(SUB_REPS) × $(SUB_INNER)
    ─────────────────────────────────────────────────────────────────────────""")

    census = CensusRow[]
    if want("--census")
        for χ in SUB_CHIS, sym in SUB_SYMS
            r = census_row(sym, χ)
            push!(census, r)
            @printf(
                "%-13s χ=%-4d  units %5d  gemms %d → %d  max/mean %.2f  dyn %s  static %s  lpt %s\n",
                r.sym, r.chi, r.nunits, r.ngemm_sector, r.ngemm_unit, r.imbalance,
                join((@sprintf("%.1f", x) for x in r.dyn_bound), "/"),
                join((@sprintf("%.1f", x) for x in r.stat_bound), "/"),
                join((@sprintf("%.1f", x) for x in r.lpt_bound), "/"),
            )
            flush(stdout)
            GC.gc()
        end
        println()
    end

    serials = SerialRow[]
    if want("--serial")
        for χ in SUB_CHIS, sym in SUB_SYMS
            r = serial_row(sym, χ)
            push!(serials, r)
            @printf(
                "%-13s χ=%-4d  real %9.1f  sector %9.1f  unit %9.1f µs   tax %6.3f  (ctl %.1f%%)\n",
                r.sym, r.chi, r.t_real, r.t_sector, r.t_unit, r.tax, 100 * r.control_dev
            )
            flush(stdout)
            GC.gc()
        end
        println()
    end

    threads = ThreadRow[]
    if want("--threaded")
        nthr > 1 || println("!! Threads.nthreads() == 1: run with --threads=N for the threaded arm")
        for χ in SUB_CHIS, sym in SUB_SYMS
            r = thread_row(sym, χ, nthr)
            push!(threads, r)
            @printf(
                "%-13s χ=%-4d t=%-2d  serial %9.1f  lpt %9.1f (%5.3f)  dynamic %9.1f (%5.3f)  repro %-5s/%-5s  vs_real %.1e/%.1e  reduce %.1f µs  (ctl %.1f%%)\n",
                r.sym, r.chi, r.nthreads, r.t_serial, r.t_threaded, r.speedup,
                r.t_dynamic, r.dyn_speedup, r.reproducible, r.dyn_repro,
                r.maxdiff_real, r.maxdiff_dyn, r.t_reduce, 100 * r.control_dev
            )
            flush(stdout)
            GC.gc()
        end
        println()
    end

    blas = BlasRow[]
    if want("--blas")
        counts = [n for n in (1, 2, 4, 8) if n ≤ Sys.CPU_THREADS]
        for χ in SUB_CHIS, sym in SUB_SYMS
            r = blas_row(sym, χ, counts)
            push!(blas, r)
            @printf("%-13s χ=%-4d  ", r.sym, r.chi)
            for (n, t) in zip(r.nblas, r.times)
                @printf("BLAS=%d %8.1f (%.2f×)  ", n, t, r.times[1] / t)
            end
            @printf(" (ctl %.1f%%)\n", 100 * r.control_dev)
            flush(stdout)
            GC.gc()
        end
        println()
    end

    load1 = _loadavg()
    println("loadavg at start: $load0")
    println("loadavg at end  : $load1")

    mkpath(PROBE_DIR)
    path = joinpath(PROBE_DIR, "subblock_probe_t$(nthr).md")
    open(path, "w") do io
        println(io, "# Subblock-outer-loop probe, `Threads.nthreads() = $nthr`")
        println(io)
        println(io, "Generated by `benchmark/subblock_probe.jl`; the design under test, the unit")
        println(io, "definition, the serial tax this exists to measure, and the **abelian-only**")
        println(io, "scope are all documented in that file's header. Read the scope note before")
        println(io, "quoting anything here.")
        println(io)
        println(io, "- host: `$(gethostname())`  ·  CPU: `$(Sys.CPU_NAME)`  ·  julia `$(VERSION)`")
        println(io, "- git sha: `$(_gitsha())`  ·  BLAS threads: $(BLAS.get_num_threads())  ·  `Threads.nthreads()` = $nthr")
        println(io, "- loadavg (1/5/15): `$load0` at start, `$load1` at end")
        println(io, "- $(SUB_REPS) reps × $(SUB_INNER) inner calls, arms interleaved per repetition")

        if !isempty(census)
            println(io, "\n## Unit census (structure only, machine-independent)\n")
            println(io, "`units` is the number of independent subblock chains — the achievable")
            println(io, "concurrency. `gemms` is calls per vertex call, per-coupled-sector →")
            println(io, "per-unit: the same total flops split finer.\n")
            println(io, "| sym | χ | d | units | dim | gemms/call (sector → unit) | copy min/med/max | flop min/med/max | work max/mean |")
            println(io, "|---|---|---|---|---|---|---|---|---|")
            for r in census
                @printf(
                    io, "| %s | %d | %d | **%d** | %d | %d → **%d** | %d / %.0f / %d | %d / %.0f / %d | **%.2f** |\n",
                    r.sym, r.chi, r.d, r.nunits, r.dim, r.ngemm_sector, r.ngemm_unit,
                    r.copy_min, r.copy_med, r.copy_max,
                    r.flop_min, r.flop_med, r.flop_max, r.imbalance,
                )
            end
            println(io, "\n### Scheduling bounds on the unit loop\n")
            println(io, "Speedup ceilings from the per-unit work vector at `nthreads` = $(join(SCHED_THREADS, " / ")).")
            println(io, "`dyn` is `total / max(longest unit, total/t)`, which no schedule can beat.")
            println(io, "`static` is contiguous chunking — the obvious first implementation. `lpt` is the")
            println(io, "deterministic longest-processing-time-first assignment, which is what")
            println(io, "`unit_call_threaded!` uses, and it is deterministic **on purpose**: work")
            println(io, "stealing would balance just as well but would put a unit in a different")
            println(io, "per-thread partial sum from run to run, which is incompatible with the bitwise")
            println(io, "reproducibility §2's hazard table asks for.")
            println(io)
            println(io, "Work is `copy + flop` weighted 1:1, which is a proxy — see `census_row`.\n")
            println(io, "| sym | χ | units | dyn bound | static (contiguous) | **lpt** |")
            println(io, "|---|---|---|---|---|---|")
            for r in census
                @printf(
                    io, "| %s | %d | %d | %s | %s | **%s** |\n", r.sym, r.chi, r.nunits,
                    join((@sprintf("%.2f", x) for x in r.dyn_bound), " / "),
                    join((@sprintf("%.2f", x) for x in r.stat_bound), " / "),
                    join((@sprintf("%.2f", x) for x in r.lpt_bound), " / "),
                )
            end
        end

        if !isempty(serials)
            println(io, "\n## Serial decomposition tax\n")
            println(io, "`sector` is `plancache_probe.jl`'s per-coupled-sector cheat kernel, which is")
            println(io, "bitwise identical to the real one; `unit` is the same kernel with the outer")
            println(io, "loop over subblocks. **`tax = unit / sector` is the factor threading has to")
            println(io, "beat before it breaks even.** `control` is the unit arm against itself.\n")
            println(io, "| sym | χ | units | real µs | sector µs | unit µs | **tax** | unit/real | control dev | maxdiff |")
            println(io, "|---|---|---|---|---|---|---|---|---|---|")
            for r in serials
                @printf(
                    io, "| %s | %d | %d | %.1f | %.1f | %.1f | **%.3f** | %.3f | %.1f%% | %.1e |\n",
                    r.sym, r.chi, r.nunits, r.t_real, r.t_sector, r.t_unit, r.tax,
                    r.vs_real, 100 * r.control_dev, max(r.maxdiff_sector, r.maxdiff_unit),
                )
            end
        end

        if !isempty(threads)
            println(io, "\n## Threaded unit loop\n")
            println(io, "`serial` is the *same call* at `nthreads = 1`, so `speedup` isolates the")
            println(io, "threading; `vs real` divides by the real kernel's time from the `--serial`")
            println(io, "table and is the figure that matters. `repro` — two runs at this `nthreads`,")
            println(io, "bitwise — is the determinism a fixed unit→thread map plus a fixed reduction")
            println(io, "order actually buys. `vs_real` is the **race check**: disjoint bands are only")
            println(io, "disjoint if the plan says so, and the probe refuses to report a speedup for a")
            println(io, "row where it exceeds 1e-12.")
            println(io)
            println(io, "`vs_ser` (threaded against `nthreads = 1`) is at rounding level *by")
            println(io, "construction* and is **not** a defect: partitioning a sum across `t` buffers")
            println(io, "regroups the additions, and floating-point addition is not associative. §2's")
            println(io, "hazard table asks for `nthreads == 1` bitwise equal to `t`-thread, which no")
            println(io, "implementation can give; `repro` is the achievable half of that request.\n")
            println(io, "`lpt` is the deterministic longest-processing-time-first assignment; `dynamic`")
            println(io, "is `nthreads` tasks pulling unit indices off one atomic counter, which needs")
            println(io, "**no work model at all**. If the two match, the plan's work estimate can be")
            println(io, "deleted. `dynamic` is not reproducible run to run by construction, which is")
            println(io, "fine — it is recorded, not required.\n")
            println(io, "| sym | χ | units | threads | serial µs | lpt µs | **lpt vs real** | dynamic µs | **dyn vs real** | dyn/lpt | repro lpt/dyn | vs_real lpt/dyn | reduce µs | control dev |")
            println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in threads
                s = findfirst(x -> x.sym === r.sym && x.chi == r.chi, serials)
                treal = isnothing(s) ? NaN : serials[s].t_real
                @printf(
                    io, "| %s | %d | %d | %d | %.1f | %.1f | **%.3f** | %.1f | **%.3f** | %.3f | %s/%s | %.1e/%.1e | %.1f | %.1f%% |\n",
                    r.sym, r.chi, r.nunits, r.nthreads, r.t_serial,
                    r.t_threaded, treal / r.t_threaded,
                    r.t_dynamic, treal / r.t_dynamic, r.t_dynamic / r.t_threaded,
                    r.reproducible, r.dyn_repro, r.maxdiff_real, r.maxdiff_dyn,
                    r.t_reduce, 100 * r.control_dev,
                )
            end
        end

        if !isempty(blas)
            println(io, "\n## BLAS threads on the real kernel (§4.1)\n")
            println(io, "Every other measurement in this project pinned `BLAS.set_num_threads(1)`.")
            println(io, "This is the free-parallelism check, and it competes with the §2 work for the")
            println(io, "same cores.\n")
            ns = blas[1].nblas
            println(io, "| sym | χ | ", join(("BLAS=$n µs" for n in ns), " | "), " | best gain | control dev |")
            println(io, "|---|---|", join(("---" for _ in ns), "|"), "|---|---|")
            for r in blas
                @printf(io, "| %s | %d | ", r.sym, r.chi)
                print(io, join((@sprintf("%.1f", t) for t in r.times), " | "))
                @printf(io, " | **%.2f×** | %.1f%% |\n", r.times[1] / minimum(r.times), 100 * r.control_dev)
            end
        end
    end
    println("wrote $path")
    return (; census, serials, threads, blas)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && probe_main()
