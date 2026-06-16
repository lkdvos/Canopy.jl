#!/usr/bin/env julia
#
# Profiling driver for the real-time free-fermion quench (BP + bond truncation).
#
# Where `run_timings.jl` records *which phase* is slow, this records *which lines*: it runs the
# χ=16 and χ=32 cases under the sampling profiler and writes interactive pprof artifacts so the
# hotspots inside BP (and hop, and the heap-allocation sites) can be drilled into later.
#
#   ./profile_runs.jl --chi 16 32
#   julia --project=benchmark/realtime_timing benchmark/realtime_timing/profile_runs.jl --help
#
# Per χ it writes to `<outdir>/`:
#   chi<χ>_cpu_bp.pb.gz    — CPU profile of belief_propagation only (the dominant phase)
#   chi<χ>_cpu_hop.pb.gz   — CPU profile of the hopping layer only
#   chi<χ>_cpu_step.pb.gz  — CPU profile of the full Trotter step (single → hop → single → bp)
#   chi<χ>_allocs.pb.gz    — allocation profile of the full step (heap hotspots)
#   chi<χ>_meta.txt        — χ, threads, bp_iters, repeats, reached maxdim
#
# View a saved file with the bundled pprof tool:
#   julia --project=. -e 'using PProf; PProf.refresh(file="profiles/chi32_cpu_bp.pb.gz")'
#
# The model/circuit is reused verbatim from `run_timings.jl` (its `@main` does not run on
# `include`), so this stays in lock-step with the benchmark. BP runs with `tol=0` for a fixed
# `bp_iters` sweeps, so repeated calls on a converged state do identical, representative work.

include(joinpath(@__DIR__, "run_timings.jl"))

using Profile
using PProf

function parse_profile_cli(args)
    s = ArgParseSettings(description="Profile the real-time free-fermion quench and write pprof artifacts.")
    @add_arg_table! s begin
        "--chi"
        help = "bond dimensions to profile (one set of artifacts per value)"
        arg_type = Int
        nargs = '+'
        default = [16, 32]
        "--outdir"
        help = "directory for output .pb.gz profiles"
        default = joinpath(@__DIR__, "profiles")
        "--bp-iters"
        help = "fixed BP sweeps per call (run with tol=0)"
        arg_type = Int
        default = 30
        "--warmup"
        help = "full Trotter steps to run before profiling (grows bonds to the χ plateau, pays JIT)"
        arg_type = Int
        default = 4
        "--repeats"
        help = "sampling iterations per CPU segment"
        arg_type = Int
        default = 8
        "--alloc-repeats"
        help = "full steps sampled for the allocation profile"
        arg_type = Int
        default = 3
        "--alloc-rate"
        help = "allocation sampling rate (1.0 = every allocation; higher = slower pprof analysis)"
        arg_type = Float64
        default = 0.01
        "--delay"
        help = "CPU sampling interval in seconds"
        arg_type = Float64
        default = 0.001
        "--model"
        help = "physics model (passed to run_timings.jl): free-fermion, free-fermion-u1, or tfim"
        default = "free-fermion"
        range_tester = m -> m in ("free-fermion", "free-fermion-u1", "tfim")
    end
    return parse_args(args, s)
end

# Warm up to the bond-dimension plateau (and pay JIT) before any samples are taken.
function warmup_state(χ, warmup, bp_iters, model)
    state = initial_state(model)
    msgs = BPMessages(state)
    single, hop = build_layers(model)
    trunc = truncrank(χ)
    msgs = belief_propagation(msgs, state; maxiter=bp_iters, tol=0)
    for _ in 1:warmup
        state, msgs, _ = apply!(state, msgs, single)
        state, msgs, _ = apply!(state, msgs, hop; trunc)
        state, msgs, _ = apply!(state, msgs, single)
        msgs = belief_propagation(msgs, state; maxiter=bp_iters, tol=0)
    end
    return state, msgs, single, hop, trunc
end

# Run `f` under the CPU sampler, write a pprof file, and print a short top-frames summary.
function cpu_segment(f, outfile, label)
    Profile.clear()
    Profile.@profile f()
    PProf.pprof(; web=false, out=outfile)
    @printf("  %-5s → %s\n", label, basename(outfile))
    Profile.print(; format=:flat, sortedby=:count, mincount=10, maxdepth=1)
    flush(stdout)
    return nothing
end

function profile_chi(χ, opts)
    bp_iters, repeats = opts["bp-iters"], opts["repeats"]
    outdir = opts["outdir"]
    @printf("χ=%2d  warmup=%d  repeats=%d  alloc-repeats=%d\n", χ, opts["warmup"], repeats, opts["alloc-repeats"])
    flush(stdout)

    model = opts["model"]
    state, msgs, single, hop, trunc = warmup_state(χ, opts["warmup"], bp_iters, model)
    reached = maxdim(state)
    reached == χ || @warn "bond dimension did not reach χ; increase --warmup" χ reached

    tag = get(Dict("tfim" => "tfim_", "free-fermion-u1" => "u1_"), model, "")
    chi_out(suffix) = joinpath(outdir, "chi$(χ)_$(tag)$(suffix)")

    # CPU, BP only — fixed-work (tol=0), no state drift, so every call is identical work.
    cpu_segment(chi_out("cpu_bp.pb.gz"), "bp") do
        for _ in 1:repeats
            belief_propagation(msgs, state; maxiter=bp_iters, tol=0)
        end
    end

    # CPU, hop only — bonds stay capped at χ by `trunc`; the QR+SVD work is representative.
    cpu_segment(chi_out("cpu_hop.pb.gz"), "hop") do
        for _ in 1:repeats
            state, msgs, _ = apply!(state, msgs, hop; trunc)
        end
    end

    # CPU, whole step.
    cpu_segment(chi_out("cpu_step.pb.gz"), "step") do
        for _ in 1:repeats
            state, msgs, _ = apply!(state, msgs, single)
            state, msgs, _ = apply!(state, msgs, hop; trunc)
            state, msgs, _ = apply!(state, msgs, single)
            msgs = belief_propagation(msgs, state; maxiter=bp_iters, tol=0)
        end
    end

    # Allocations over the full step (heap hotspots; off-heap Bumper temporaries are invisible).
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = opts["alloc-rate"] for _ in 1:opts["alloc-repeats"]
        state, msgs, _ = apply!(state, msgs, single)
        state, msgs, _ = apply!(state, msgs, hop; trunc)
        state, msgs, _ = apply!(state, msgs, single)
        msgs = belief_propagation(msgs, state; maxiter=bp_iters, tol=0)
    end
    PProf.Allocs.pprof(; web=false, out=chi_out("allocs.pb.gz"))
    @printf("  alloc → %s\n", basename(chi_out("allocs.pb.gz")))

    open(chi_out("meta.txt"), "w") do io
        @printf(io, "chi=%d\nmodel=%s\nnthreads=%d\nbp_iters=%d\nwarmup=%d\nrepeats=%d\nalloc_repeats=%d\nalloc_rate=%g\nmaxdim=%d\n",
            χ, model, Threads.nthreads(), bp_iters, opts["warmup"], repeats, opts["alloc-repeats"], opts["alloc-rate"], reached)
    end
    flush(stdout)
    return nothing
end

function (@main)(args)
    opts = parse_profile_cli(args)
    mkpath(opts["outdir"])
    Profile.init(; n=10^8, delay=opts["delay"])   # large buffer — χ=32 BP runs are long

    @printf("Profiling real-time quench on a %d-site hexagonal lattice (Canopy.jl)\n", NSITES)
    @printf("model=%s  χ=%s  threads=%d  delay=%gs  → %s\n\n", opts["model"], opts["chi"], Threads.nthreads(), opts["delay"], opts["outdir"])

    for χ in opts["chi"]
        profile_chi(χ, opts)
        println()
    end
    println("done — view with: julia --project=. -e 'using PProf; PProf.refresh(file=\"...\")'")
    return 0
end
