#!/usr/bin/env julia
#
# Real-time Hubbard quench on Canopy.jl — CLI driver
# ==================================================
#
# One parameter point per invocation. Writes a self-describing result folder whose name
# encodes the physics, plus per-step CSVs and a full log so performance can be tracked
# alongside the physics.
#
#   ./run.jl --lattice hex --size 6 6 --U 4 --chi 16 --tfinal 1.0 \
#            --particle-symmetry u1 --spin-symmetry u1
#   julia --project=scripts/hubbard_quench scripts/hubbard_quench/run.jl --help
#
# Canopy's `apply!` is single-threaded (there is no `Threads` in `src/`), so on a cluster run
# one process per parameter point with BLAS pinned to one thread — see `make_sweep.jl`.

using Pkg
Pkg.activate(@__DIR__; io = devnull)

include(joinpath(@__DIR__, "HubbardQuench.jl"))
using .HubbardQuench

using ArgParse
using Canopy
using Canopy: belief_propagation
using TensorKit
using CSV
using DataFrames
using Dates: Dates, now
using Logging
using LoggingExtras
using Printf
using Statistics: median
using TOML
using TimerOutputs
using LinearAlgebra: BLAS

# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

function parse_cli(args)
    s = ArgParseSettings(
        description = "Real-time Hubbard quench (BP + bond truncation) on Canopy.jl.",
        autofix_names = true,
    )
    @add_arg_table! s begin
        "--lattice"
        help = "lattice geometry: hex (2mn sites) or square (mn sites)"
        default = "hex"
        range_tester = x -> x in HubbardQuench.LATTICE_NAMES
        "--size"
        help = "lattice extent M N (unit cells for hex)"
        arg_type = Int
        nargs = 2
        default = [6, 6]
        "--periodic"
        help = "boundary conditions: none | x | y | xy"
        default = "none"
        range_tester = x -> x in ("none", "x", "y", "xy")
        "--t"
        help = "hopping amplitude"
        arg_type = Float64
        default = -1.0
        "--U"
        help = "on-site interaction"
        arg_type = Float64
        default = 0.0
        "--interaction"
        help = "interaction form: ud (U n↑n↓) or half-ud (U (n↑-½)(n↓-½))"
        default = "ud"
        range_tester = x -> x in ("ud", "half-ud")
        "--quench"
        help = "initial product state: cdw (global AFM) or doublon (doublon+hole in an AFM)"
        default = "cdw"
        range_tester = x -> x in HubbardQuench.QUENCH_NAMES
        "--dt"
        help = "Trotter step"
        arg_type = Float64
        default = 0.01
        "--tfinal"
        help = "final time (ignored when --nsteps is given)"
        arg_type = Float64
        default = 1.0
        "--nsteps"
        help = "number of Trotter steps; overrides --tfinal"
        arg_type = Int
        default = 0
        "--chi"
        help = "maximum bond dimension"
        arg_type = Int
        default = 16
        "--cutoff"
        help = "discarded-weight threshold; MUST stay well above Canopy's gauge tolerance \
                (~1.8e-12) or large-χ runs go silently wrong. 0 disables (unsafe)."
        arg_type = Float64
        default = default_cutoff()
        "--particle-symmetry"
        help = "particle-number symmetry: trivial | u1"
        default = "trivial"
        range_tester = x -> x in HubbardQuench.SECTOR_NAMES
        "--spin-symmetry"
        help = "spin (Sz) symmetry: trivial | u1"
        default = "trivial"
        range_tester = x -> x in HubbardQuench.SECTOR_NAMES
        "--bp-maxiter"
        help = "maximum BP sweeps per step (warm-started from the previous step)"
        arg_type = Int
        default = 30
        "--bp-tol"
        help = "BP convergence tolerance; 0 runs a fixed --bp-maxiter sweeps"
        arg_type = Float64
        default = 1.0e-10
        "--bp-schedule"
        help = "BP schedule: sync | spanningtree | residual | splash"
        default = "sync"
        range_tester = x -> x in HubbardQuench.SCHEDULE_NAMES
        "--bp-batchsize"
        help = "batch size for the residual schedule"
        arg_type = Int
        default = 8
        "--no-bp-diag"
        help = "skip the per-measurement BP residual (saves one BP sweep)"
        action = :store_true
        "--measure-every"
        help = "measure every N steps (step 0 and the final step are always measured)"
        arg_type = Int
        default = 10
        "--bulk-depth"
        help = "peel this many boundary layers for the bulk order parameter"
        arg_type = Int
        default = 1
        "--site-resolved"
        help = "also write per-site occupations to site_observables.csv"
        action = :store_true
        "--no-energy"
        help = "skip energies (saves one 2-site expectation value per bond)"
        action = :store_true
        "--outroot"
        help = "root directory for result folders"
        default = joinpath(@__DIR__, "results")
        "--tag"
        help = "suffix appended to the result folder name"
        default = ""
        "--force"
        help = "overwrite a non-empty existing result folder"
        action = :store_true
        "--blas-threads"
        help = "BLAS thread count (1 keeps per-process timings comparable)"
        arg_type = Int
        default = 1
        "--log-level"
        help = "log verbosity: debug | info | warn. `debug` also enables Canopy's \
                per-message allocator tracing, which is very noisy."
        default = "info"
        range_tester = x -> x in ("debug", "info", "warn")
    end
    return parse_args(args, s)
end

# ---------------------------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------------------------

# `%g` so that -1.0 and -1 cannot yield two folders for one physics point.
fmt(x::Real) = @sprintf("%g", x)

function run_slug(o)
    m, n = o["size"]
    bc = o["periodic"] == "none" ? "" : "p" * o["periodic"]
    parts = [
        "$(o["lattice"])$(m)x$(n)$(bc)",
        o["quench"],
        "U$(fmt(o["U"]))",
        "t$(fmt(o["t"]))",
        "dt$(fmt(o["dt"]))",
        "T$(fmt(o["nsteps"] * o["dt"]))",
        "chi$(o["chi"])",
        "sym-$(o["particle_symmetry"])$(o["spin_symmetry"])",
        "bp$(o["bp_maxiter"])-$(fmt(o["bp_tol"]))",
    ]
    o["interaction"] == "ud" || push!(parts, o["interaction"])
    o["bp_schedule"] == "sync" || push!(parts, o["bp_schedule"])
    isempty(o["tag"]) || push!(parts, o["tag"])
    return join(parts, "_")
end

function prepare_outdir(o)
    dir = joinpath(o["outroot"], run_slug(o))
    if isdir(dir) && !isempty(readdir(dir)) && !o["force"]
        error(
            "result folder already exists and is not empty:\n  $dir\n" *
                "pass --force to overwrite, or --tag to write a distinct folder"
        )
    end
    mkpath(dir)
    return dir
end

# Tee `@info` to the console and to run.log, so a cluster job's log file is self-contained.
#
# The `MinLevelLogger` wrappers are not optional: a bare `FormatLogger` enables *all* levels,
# and Canopy emits a `@debug` line per message per BP sweep for allocator hygiene, which
# buries the run at several thousand lines per step.
function setup_logging(dir, level::AbstractString)
    minlevel =
        level == "debug" ? Logging.Debug :
        level == "info" ? Logging.Info :
        level == "warn" ? Logging.Warn :
        throw(ArgumentError("unknown log level `$level`"))
    io = open(joinpath(dir, "run.log"), "w")
    function fmtline(io, args)
        print(io, "[", Dates.format(now(), "HH:MM:SS"), " ", uppercase(string(args.level)), "] ")
        print(io, args.message)
        for (k, v) in args.kwargs
            print(io, "  ", k, "=", v)
        end
        println(io)
        return nothing
    end
    logger = TeeLogger(
        MinLevelLogger(FormatLogger(fmtline, io), minlevel),
        MinLevelLogger(FormatLogger(fmtline, stderr), minlevel),
    )
    return logger, io
end

# ---------------------------------------------------------------------------------------
# Incremental CSV writing
# ---------------------------------------------------------------------------------------
#
# Each row is appended and flushed as it is produced: cluster jobs are killed at the wall
# clock, and a buffered write-at-the-end would lose the whole trajectory.

mutable struct RowWriter
    path::String
    wrote_header::Bool
end
RowWriter(path) = RowWriter(path, false)

function Base.push!(w::RowWriter, row::NamedTuple)
    CSV.write(w.path, DataFrame([row]); append = w.wrote_header, writeheader = !w.wrote_header)
    w.wrote_header = true
    return w
end

function push_rows!(w::RowWriter, rows::Vector{<:NamedTuple})
    isempty(rows) && return w
    CSV.write(w.path, DataFrame(rows); append = w.wrote_header, writeheader = !w.wrote_header)
    w.wrote_header = true
    return w
end

# ---------------------------------------------------------------------------------------
# params.toml
# ---------------------------------------------------------------------------------------

function canopy_sha()
    try
        return readchomp(
            Cmd(`git -C $(pkgdir(Canopy)) rev-parse HEAD`; ignorestatus = true)
        )
    catch
        return "unknown"
    end
end

function write_params(dir, o, derived)
    versions = Dict{String, String}()
    for (uuid, dep) in Pkg.dependencies()
        dep.is_direct_dep || continue
        versions[dep.name] = string(something(dep.version, "unversioned"))
    end
    doc = Dict(
        "parameters" => Dict{String, Any}(k => v for (k, v) in o),
        "derived" => derived,
        "environment" => Dict(
            "julia" => string(VERSION),
            "nthreads" => Threads.nthreads(),
            "blas_threads" => BLAS.get_num_threads(),
            "hostname" => gethostname(),
            "canopy_sha" => canopy_sha(),
            "started" => string(now()),
            "versions" => versions,
        ),
    )
    open(joinpath(dir, "params.toml"), "w") do io
        TOML.print(io, doc; sorted = true)
    end
    return nothing
end

# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------

function (@main)(args)
    o = parse_cli(args)
    BLAS.set_num_threads(o["blas_threads"])

    o["nsteps"] > 0 || (o["nsteps"] = round(Int, o["tfinal"] / o["dt"]))
    nsteps = o["nsteps"]
    nsteps > 0 || error("nsteps resolved to 0; raise --tfinal or lower --dt")
    # Keep params.toml self-consistent when --nsteps overrode --tfinal.
    o["tfinal"] = nsteps * o["dt"]
    dt, t, U, χ = o["dt"], o["t"], o["U"], o["chi"]
    periodic = (occursin("x", o["periodic"]), occursin("y", o["periodic"]))
    measure_every = max(1, o["measure_every"])

    lat = lattice(o["lattice"], o["size"][1], o["size"][2]; periodic)
    pat = quench_pattern(o["quench"], lat)
    region = bulk_region(lat, o["bulk_depth"])
    P, S = sectortypes(o["particle_symmetry"], o["spin_symmetry"])
    ops = operators(P, S; interaction = o["interaction"] == "ud" ? :ud : :half_ud)

    dir = prepare_outdir(o)
    logger, logio = setup_logging(dir, o["log_level"])

    try
        with_logger(logger) do
            @info "Hubbard quench (Canopy.jl)" outdir = dir
            @info "lattice" kind = o["lattice"] size = Tuple(o["size"]) periodic nsites = length(lat) nedges = length(lat.edges)
            @info "model" t U interaction = o["interaction"] quench = o["quench"]
            @info "evolution" dt nsteps tfinal = nsteps * dt trotter = "2nd-order Strang"
            @info "truncation" chi = χ cutoff = o["cutoff"]
            @info "symmetry" particle = o["particle_symmetry"] spin = o["spin_symmetry"]
            @info "bp" schedule = o["bp_schedule"] maxiter = o["bp_maxiter"] tol = o["bp_tol"] diag = !o["no_bp_diag"]
            @info "region" bulk_depth = o["bulk_depth"] bulk_sites = length(region)

            # --- initial state -----------------------------------------------------------
            state, total_charge, bath = build_state(ComplexF64, lat, pat, P, S)
            msgs = BPMessages(state)
            schedule = bp_schedule(o["bp_schedule"]; batchsize = o["bp_batchsize"])
            bp(m, st) = belief_propagation(
                m, st; maxiter = o["bp_maxiter"], tol = o["bp_tol"], schedule
            )

            t_bp0 = @timed bp(msgs, state)
            msgs = t_bp0.value
            @info "initial state" total_charge = string(total_charge) charge_bath = bath bp_time = round(t_bp0.time; digits = 3)
            if bath
                @info "a charge-bath vertex was attached; all gates and observables range \
                       over the physical lattice only"
            end

            checks = check_initial_state(state, msgs, lat, pat, ops)
            @info "self-check passed" N_up = checks.nup_total N_dn = checks.ndn_total

            single, hoplayers, ncolors = build_layers(lat, ops, U, t, dt)
            trunc = truncation(χ, o["cutoff"])
            @info "layers" ncolors nhoplayers = length(hoplayers) interaction_layer = !isnothing(single)

            # --- reference ---------------------------------------------------------------
            ref_circuit, ref_cont = if iszero(U)
                r = @timed free_fermion_reference(lat, pat, t, dt, nsteps)
                @info "free-fermion reference computed" time = round(r.time; digits = 3) trotter_gap = maximum(abs, r.value[1] .- r.value[2])
                r.value
            else
                @info "U ≠ 0: no exact reference exists; E_tot drift is the validation signal"
                nothing, nothing
            end

            derived = Dict{String, Any}(
                "nsites" => length(lat), "nedges" => length(lat.edges),
                "ncolors" => ncolors, "nhoplayers" => length(hoplayers),
                "max_coordination" => maximum(length(lat.adj[v]) for v in lat.verts),
                "total_charge" => string(total_charge), "charge_bath" => bath,
                "bulk_sites" => length(region),
                "has_reference" => iszero(U),
                "gauge_tol" => Canopy.default_gauge_tol(ComplexF64[]),
            )
            write_params(dir, o, derived)

            obs_w = RowWriter(joinpath(dir, "observables.csv"))
            tim_w = RowWriter(joinpath(dir, "timings.csv"))
            site_w = o["site_resolved"] ? RowWriter(joinpath(dir, "site_observables.csv")) : nothing

            to = TimerOutput()
            eps_cum = 0.0
            logl_cum = 0.0

            function do_measure(step, eps_max, eps_sum)
                r = @timed measure(
                    state, msgs, lat, ops, region;
                    t, U, energy = !o["no_energy"]
                )
                sc, site = r.value
                resid = o["no_bp_diag"] ? NaN : bp_residual(msgs, state, lat)
                nsec, maxsec = bond_sector_structure(state, lat)
                push!(
                    obs_w, (;
                        step, time = step * dt,
                        sc.m_s_all, sc.m_s_bulk, sc.n_mean, sc.docc_mean,
                        sc.E_kin, sc.E_int, sc.E_tot,
                        ref_circuit = isnothing(ref_circuit) ? missing : ref_circuit[step + 1],
                        ref_cont = isnothing(ref_cont) ? missing : ref_cont[step + 1],
                        maxdim = maxvirtualdim(state, lat),
                        nsectors_max = nsec, maxsecdim = maxsec,
                        eps_max, eps_sum, eps_cum, logl_cum, bp_resid = resid,
                    )
                )
                if !isnothing(site_w)
                    push_rows!(
                        site_w,
                        [
                            (;
                                step, time = step * dt, site = string(v),
                                i = v[1], j = v[2], s = length(v) >= 3 ? v[3] : 0,
                                n_up = site.nup[k], n_dn = site.ndn[k], docc = site.docc[k],
                            )
                            for (k, v) in enumerate(site.verts)
                        ]
                    )
                end
                err = isnothing(ref_circuit) ? NaN : abs(sc.m_s_all - ref_circuit[step + 1])
                @info "measured" step time = round(step * dt; digits = 6) m_s = round(sc.m_s_all; digits = 8) err_vs_ref = err E_tot = sc.E_tot D = maxvirtualdim(state, lat) bp_resid = resid meas_time = round(r.time; digits = 3)
                return r.time
            end

            tm = do_measure(0, 0.0, 0.0)
            push!(
                tim_w, (;
                    step = 0, t_single1 = 0.0, t_hop = 0.0, t_single2 = 0.0,
                    t_bp = t_bp0.time, t_measure = tm, t_step = t_bp0.time + tm,
                    b_single1 = 0, b_hop = 0, b_single2 = 0, b_bp = t_bp0.bytes,
                    maxdim = maxvirtualdim(state, lat),
                )
            )

            # --- evolution ---------------------------------------------------------------
            @info "starting evolution" nsteps
            for step in 1:nsteps
                r1 = @timed if !isnothing(single)
                    state, msgs, _ = @timeit to "single" apply!(state, msgs, single)
                end

                eps_max = 0.0
                eps_sum = 0.0
                rh = @timed for layer in hoplayers
                    state, msgs, info = @timeit to "hop" apply!(state, msgs, layer; trunc)
                    eps_max = max(eps_max, info.ϵ)
                    eps_sum += info.ϵ
                    logl_cum += info.logλ
                end
                eps_cum += eps_sum

                r2 = @timed if !isnothing(single)
                    state, msgs, _ = @timeit to "single" apply!(state, msgs, single)
                end

                rb = @timed @timeit to "bp" bp(msgs, state)
                msgs = rb.value

                tmeas = 0.0
                if step % measure_every == 0 || step == nsteps
                    tmeas = @timeit to "measure" do_measure(step, eps_max, eps_sum)
                end

                push!(
                    tim_w, (;
                        step, t_single1 = r1.time, t_hop = rh.time, t_single2 = r2.time,
                        t_bp = rb.time, t_measure = tmeas,
                        t_step = r1.time + rh.time + r2.time + rb.time + tmeas,
                        b_single1 = r1.bytes, b_hop = rh.bytes, b_single2 = r2.bytes,
                        b_bp = rb.bytes, maxdim = maxvirtualdim(state, lat),
                    )
                )
            end

            # --- summary -----------------------------------------------------------------
            tim = CSV.read(joinpath(dir, "timings.csv"), DataFrame)
            loop = tim[tim.step .>= 1, :]
            # step 1 carries Julia's JIT cost, so it is excluded from the medians.
            body = nrow(loop) > 1 ? loop[2:end, :] : loop
            @info "done" median_step_s = round(median(body.t_step); digits = 4) median_bp_s = round(median(body.t_bp); digits = 4) total_s = round(sum(loop.t_step); digits = 2) final_maxdim = maximum(loop.maxdim) eps_cum

            open(joinpath(dir, "timer.txt"), "w") do io
                show(io, to; allocations = true, sortby = :firstexec)
                println(io)
            end
            @info "wrote results" dir
        end
    finally
        close(logio)
    end
    return 0
end
