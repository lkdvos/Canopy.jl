#!/usr/bin/env julia
#
# Per-step timing trace of the real-time free-fermion quench (BP + bond truncation).
#
# Times each phase of every Trotter step (single → hop → single → BP reconverge) and writes
# one wide CSV per χ to `<outdir>/<prefix>_chi<χ>.csv`. Mirrors the circuit of
# `examples/realtime/main.jl`, but runs BP for a *fixed* `--bp-iters` sweeps (`tol=0`) so the
# work per step is constant and comparable across libraries.
#
#   ./run_timings.jl --prefix canopy --outdir data --nsteps 30 --chi 4 8 16 32
#   julia --project=benchmark/realtime_timing benchmark/realtime_timing/run_timings.jl --help
#
# Set JULIA_NUM_THREADS to a fixed value — `apply!` threads over gates and the thread count is
# recorded in every CSV.

using Pkg
Pkg.activate(@__DIR__; io=devnull)

using ArgParse
using Canopy: hexagonal_lattice, product_state, BPMessages, belief_propagation,
    LocalGate, CompositeGate, Circuit, apply!, edge_coloring, virtualspace
using TensorKit
using TensorKitTensors.FermionOperators: f_num, f_hopping, fermion_space
using MatrixAlgebraKit: truncrank
using Dictionaries
using DataFrames
using CSV
using Printf

using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

function parse_cli(args)
    s = ArgParseSettings(description="Per-step timing trace of the real-time free-fermion quench.")
    @add_arg_table! s begin
        "--prefix"
        help = "output filename prefix; CSVs go to <outdir>/<prefix>_chi<χ>.csv"
        default = "canopy"
        "--outdir"
        help = "directory for output CSVs"
        default = joinpath(@__DIR__, "data")
        "--nsteps"
        help = "number of Trotter steps to time"
        arg_type = Int
        default = 30
        "--chi"
        help = "bond dimensions to sweep (one CSV per value)"
        arg_type = Int
        nargs = '+'
        default = [4, 8, 16, 32]
        "--bp-iters"
        help = "fixed BP sweeps per step (run with tol=0)"
        arg_type = Int
        default = 30
    end
    return parse_args(args, s)
end

const M, N = 4, 6              # m × n unit cells → 2·m·n = 48 sites
const T_HOP = -1.0
const MU = 1.0
const DT = 0.01

const ES = hexagonal_lattice(M, N)
const VERTS = sort(unique(Iterators.flatten((e.src, e.dst) for e in ES)))
const NSITES = length(VERTS)
const HOPOP = f_hopping(ComplexF64, Trivial)

μ_of(v) = isodd(v[3]) ? MU : -MU
occ_of(v) = (sum(v) % 4 == 0) ? 0 : 1

function initial_state()
    P = fermion_space(Trivial)
    ps = Dictionary(VERTS, fill(P, length(VERTS)))
    ls = Dictionary(VERTS, [fℤ₂(occ_of(v)) => [1.0] for v in VERTS])
    return product_state(ComplexF64, ES, ps, ls)
end

function build_layers()
    n = f_num(ComplexF64, Trivial)
    g_plus = exp(-im * MU * DT * n)
    g_minus = exp(-im * (-MU) * DT * n)
    g_hop = exp(-im * (T_HOP * DT) * HOPOP)
    single = CompositeGate([LocalGate((v,), μ_of(v) > 0 ? g_plus : g_minus) for v in VERTS])
    hop = Circuit(
        [
        CompositeGate([LocalGate((e.src, e.dst), g_hop) for e in class])
        for class in edge_coloring(ES)
    ]
    )
    return single, hop
end

maxdim(state) = maximum(dim(virtualspace(state, e)) for e in ES)

# One timed trajectory. Returns a wide DataFrame; row `step=0` is the initial BP convergence
# (gate columns 0.0), rows `step ≥ 1` are the Trotter steps. No warmup: step 1 carries Julia's
# JIT-compilation cost, which is itself informative.
function run_chi_timed(χ, nsteps, bp_iters)
    state = initial_state()
    msgs = BPMessages(state)
    single, hop = build_layers()
    trunc = truncrank(χ)
    nt = Threads.nthreads()

    # `@timed` records both wall-clock and *heap* bytes per phase. Contraction temporaries are
    # bump-allocated off-heap (Bumper `ResizeBuffer`), so `*_bytes` capture the persistent
    # allocation churn that reaches the GC, not the full temporary working set.
    rows = NamedTuple[]
    r = @timed belief_propagation(msgs, state; maxiter=bp_iters, tol=0)
    msgs = r.value
    push!(
        rows, (;
            chi=χ, nthreads=nt, nsites=NSITES, dt=DT, step=0,
            single1=0.0, hop=0.0, single2=0.0, bp=r.time,
            single1_bytes=0, hop_bytes=0, single2_bytes=0, bp_bytes=r.bytes,
            maxdim=maxdim(state),
        )
    )

    for step in 1:nsteps
        r1 = @timed apply!(state, msgs, single)
        state, msgs, _ = r1.value
        rh = @timed apply!(state, msgs, hop; trunc)
        state, msgs, _ = rh.value
        r2 = @timed apply!(state, msgs, single)
        state, msgs, _ = r2.value
        rb = @timed belief_propagation(msgs, state; maxiter=bp_iters, tol=0)
        msgs = rb.value
        push!(
            rows, (;
                chi=χ, nthreads=nt, nsites=NSITES, dt=DT, step=step,
                single1=r1.time, hop=rh.time, single2=r2.time, bp=rb.time,
                single1_bytes=r1.bytes, hop_bytes=rh.bytes, single2_bytes=r2.bytes, bp_bytes=rb.bytes,
                maxdim=maxdim(state),
            )
        )
    end
    return DataFrame(rows)
end

function (@main)(args)
    opts = parse_cli(args)
    prefix, outdir = opts["prefix"], opts["outdir"]
    nsteps, chis, bp_iters = opts["nsteps"], opts["chi"], opts["bp-iters"]

    @printf("Real-time timing on a %d-site hexagonal lattice (Canopy.jl)\n", NSITES)
    @printf(
        "prefix=%s  dt=%.3g  nsteps=%d  bp_iters=%d  χ=%s  threads=%d\n\n",
        prefix, DT, nsteps, bp_iters, chis, Threads.nthreads()
    )

    mkpath(outdir)
    for χ in chis
        df = run_chi_timed(χ, nsteps, bp_iters)
        outfile = joinpath(outdir, "$(prefix)_chi$(χ).csv")
        CSV.write(outfile, df)
        loop = df[df.step.>=1, :]
        med = sort(loop.single1 .+ loop.hop .+ loop.single2 .+ loop.bp)[cld(nrow(loop), 2)]
        medb = sort(loop.single1_bytes .+ loop.hop_bytes .+ loop.single2_bytes .+ loop.bp_bytes)[cld(nrow(loop), 2)]
        @printf(
            "χ=%2d  median step %.4fs  (bp %.4fs)  heap %.1f MiB/step  maxdim=%d  → %s\n",
            χ, med, sort(loop.bp)[cld(nrow(loop), 2)], medb / 2^20, maximum(loop.maxdim), basename(outfile)
        )
        flush(stdout)
    end
    return 0
end
