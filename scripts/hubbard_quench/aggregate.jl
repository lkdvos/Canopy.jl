#!/usr/bin/env julia
#
# Collect Hubbard-quench result folders into one tidy table plus comparison figures.
# =================================================================================
#
#   ./aggregate.jl --outroot results
#   ./aggregate.jl --outroot results --figdir figs --summary summary.csv
#
# Reads every `<outroot>/*/` that has both `params.toml` and `observables.csv`, joins the
# parameters onto each observation row, and writes `summary.csv` plus four figures.
#
# Colour choices are not free-hand: χ is an *ordered* series, so it gets a single-hue ordinal
# ramp (light→dark, monotone lightness); symmetry is a genuine categorical dimension, so it
# gets fixed categorical slots assigned in a stable order. Both sets were checked with the
# data-viz palette validator (ordinal: monotone L, light end 2.06:1 on the light surface;
# categorical: worst adjacent CVD ΔE 9.1, normal-vision ΔE 22.9). Every panel carries a
# legend, and `summary.csv` is the table view backing the low-contrast slots.

using Pkg
Pkg.activate(@__DIR__; io = devnull)

using ArgParse
using CSV
using DataFrames
using Printf
using Statistics: median
using TOML
using CairoMakie

# --- palette (see header) --------------------------------------------------------------
# Ordinal blue ramp, light→dark; sampled for however many χ values are present.
const CHI_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# Ordinal rule: on a light surface do not start lighter than #86b6ef (index 4).
const CHI_RAMP_MIN = 4
const CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
const REF_COLOR = "#52514e"      # text-secondary: the reference is not a series
const GRID_COLOR = (:black, 0.08)

function chi_colors(n)
    n <= 1 && return [CHI_RAMP[end]]
    lo, hi = CHI_RAMP_MIN, length(CHI_RAMP)
    return [CHI_RAMP[round(Int, lo + (hi - lo) * (k - 1) / (n - 1))] for k in 1:n]
end

function cat_colors(n)
    n <= length(CATEGORICAL) && return CATEGORICAL[1:n]
    # never invent or cycle hues: fold the tail into one "Other" slot
    return vcat(CATEGORICAL, fill("#8a8a85", n - length(CATEGORICAL)))
end

# ---------------------------------------------------------------------------------------

function parse_cli(args)
    s = ArgParseSettings(
        description = "Aggregate Hubbard-quench result folders.", autofix_names = true
    )
    @add_arg_table! s begin
        "--outroot"
        help = "root directory containing result folders"
        default = joinpath(@__DIR__, "results")
        "--summary"
        help = "path of the tidy output CSV"
        default = ""
        "--figdir"
        help = "directory for figures"
        default = ""
        "--no-figures"
        help = "write only summary.csv"
        action = :store_true
    end
    return parse_args(args, s)
end

# Parameters we lift onto every observation row.
const PARAM_KEYS = [
    "lattice", "quench", "t", "U", "interaction", "dt", "nsteps", "chi", "cutoff",
    "particle_symmetry", "spin_symmetry", "bp_maxiter", "bp_tol", "bp_schedule",
    "bulk_depth", "tag",
]
const DERIVED_KEYS = ["nsites", "nedges", "ncolors", "charge_bath", "bulk_sites", "has_reference"]

function load_run(dir)
    pfile = joinpath(dir, "params.toml")
    ofile = joinpath(dir, "observables.csv")
    (isfile(pfile) && isfile(ofile)) || return nothing
    params = TOML.parsefile(pfile)
    obs = CSV.read(ofile, DataFrame)
    nrow(obs) == 0 && return nothing

    p = get(params, "parameters", Dict())
    d = get(params, "derived", Dict())
    env = get(params, "environment", Dict())

    n = nrow(obs)
    setcol!(name, v) = (obs[!, name] = fill(v, n))

    setcol!(:run, basename(dir))
    for k in PARAM_KEYS
        v = get(p, k, missing)
        setcol!(Symbol(k), v isa AbstractVector ? join(v, "x") : v)
    end
    for k in DERIVED_KEYS
        setcol!(Symbol(k), get(d, k, missing))
    end
    setcol!(:size, haskey(p, "size") ? join(p["size"], "x") : missing)
    setcol!(:symmetry, string(get(p, "particle_symmetry", "?"), "/", get(p, "spin_symmetry", "?")))
    setcol!(:hostname, get(env, "hostname", missing))
    setcol!(:canopy_sha, get(env, "canopy_sha", missing))

    # per-step timings live in a separate file; summarize rather than join row-by-row
    tfile = joinpath(dir, "timings.csv")
    med_step, med_bp, med_evolve, med_hop = missing, missing, missing, missing
    nsat = 0
    maxdim_reached = 0
    if isfile(tfile)
        tim = CSV.read(tfile, DataFrame)
        # step 1 carries Julia's JIT cost, step 0 is the initial BP; exclude both.
        body = tim[tim.step .>= 2, :]

        # Only steps whose bonds have actually reached χ measure the cost *at* χ.
        #
        # This is not a detail. Starting from a product state the bond dimension is
        # entanglement-limited, not rank-limited: with a discard threshold in play it climbs
        # gradually (measured on hex 4x4 at χ=128: 22, 29, 39, 44, 53, 63, 77, 85, 99, 107,
        # 118, 128 over twelve steps). A median over all steps therefore reports the cost at
        # some intermediate bond dimension, not at χ — which makes large-χ points look far
        # cheaper than they are and can flatten the cost curve entirely.
        chi = something(get(p, "chi", missing), 0)
        maxdim_reached = nrow(body) > 0 ? maximum(body.maxdim) : 0
        sat = chi > 0 ? body[body.maxdim .>= chi, :] : body
        nsat = nrow(sat)
        use = nsat > 0 ? sat : body

        if nrow(use) > 0
            med_step = median(use.t_step)
            med_bp = median(use.t_bp)
            med_hop = median(use.t_hop)
            # `t_step` includes measurement on the steps that measure, which is not part of
            # the evolution cost. For timing use gates + BP explicitly.
            med_evolve = median(
                use.t_single1 .+ use.t_hop .+ use.t_single2 .+ use.t_bp
            )
        end
    end
    setcol!(:median_step_s, med_step)
    setcol!(:median_bp_s, med_bp)
    setcol!(:median_hop_s, med_hop)
    setcol!(:median_evolve_s, med_evolve)
    setcol!(:nsaturated, nsat)
    # Largest bond dimension seen in the *timing* trace. `observables.csv` only samples every
    # `--measure-every` steps, so its maxdim can still read 1 for a run that is well underway —
    # reporting that alongside `nsat` (which comes from timings.csv) looks contradictory.
    setcol!(:maxdim_reached, maxdim_reached)
    return obs
end

# --- figures ---------------------------------------------------------------------------

axis_kw(; kwargs...) = (;
    xgridcolor = GRID_COLOR, ygridcolor = GRID_COLOR,
    xgridwidth = 1, ygridwidth = 1, topspinevisible = false, rightspinevisible = false,
    kwargs...
)

# `sub` is one lattice/quench/U/symmetry group; one line per χ, plus the references.
function panel_trajectory!(ax, sub, chis, colors)
    for (k, χ) in enumerate(chis)
        d = sort(sub[sub.chi .== χ, :], :step)
        lines!(ax, d.time, d.m_s_all; color = colors[k], linewidth = 2, label = "χ=$χ")
    end
    d0 = sort(sub[sub.chi .== last(chis), :], :step)
    if hasproperty(d0, :ref_circuit) && any(!ismissing, d0.ref_circuit)
        ok = .!ismissing.(d0.ref_circuit)
        lines!(
            ax, d0.time[ok], Float64.(d0.ref_circuit[ok]);
            color = REF_COLOR, linewidth = 2, linestyle = :dot, label = "exact"
        )
    end
    return ax
end

function panel_error!(ax, sub, chis, colors)
    drew = false
    for (k, χ) in enumerate(chis)
        d = sort(sub[sub.chi .== χ, :], :step)
        (hasproperty(d, :ref_circuit) && any(!ismissing, d.ref_circuit)) || continue
        ok = .!ismissing.(d.ref_circuit)
        err = max.(abs.(d.m_s_all[ok] .- Float64.(d.ref_circuit[ok])), 1.0e-16)
        lines!(ax, d.time[ok], err; color = colors[k], linewidth = 2, label = "χ=$χ")
        drew = true
    end
    return drew
end

# `--no-energy` records NaN rather than `missing`, so an `ismissing`-only guard silently draws
# invisible NaN lines and leaves an empty panel. Treat both as absent, and report which.
_absent(x) = ismissing(x) || (x isa Real && isnan(x))

function panel_energy!(ax, sub, chis, colors)
    drew = false
    for (k, χ) in enumerate(chis)
        d = sort(sub[sub.chi .== χ, :], :step)
        all(_absent, d.E_tot) && continue
        lines!(ax, d.time, d.E_tot; color = colors[k], linewidth = 2, label = "χ=$χ")
        drew = true
    end
    return drew
end

# Walltime vs χ, one line per (symmetry, U). Log-log, because the question is the scaling
# exponent — not a dual axis, a separate panel.
#
# U must be part of the series identity, not averaged over: at U = 0 the interaction layer is
# the identity and is skipped entirely, so U = 0 and U ≠ 0 do genuinely different work per
# step. Collapsing them would put two points at one χ and draw a spurious vertical jump.
function walltime_series(df)
    rows = NamedTuple[]
    for g in groupby(df, [:symmetry, :U, :chi]; sort = true)
        k = first(g)
        # evolution only — `median_step_s` would fold in measurement on measuring steps
        ms = collect(skipmissing(g.median_evolve_s))
        isempty(ms) && (ms = collect(skipmissing(g.median_step_s)))
        isempty(ms) && continue
        push!(rows, (; symmetry = k.symmetry, U = k.U, chi = k.chi, ms = median(ms)))
    end
    return rows
end

# Cost of evolution alone (gates + BP, no measurement), plus the sector structure that
# explains it. This is the table for "where does symmetry start winning".
function timing_table(df)
    rows = NamedTuple[]
    for g in groupby(df, [:U, :chi, :symmetry]; sort = true)
        k = first(g)
        ev = collect(skipmissing(g.median_evolve_s))
        isempty(ev) && continue
        rows = push!(
            rows, (;
                U = k.U, chi = k.chi, symmetry = k.symmetry,
                evolve = median(ev),
                bp = let b = collect(skipmissing(g.median_bp_s))
                    isempty(b) ? NaN : median(b)
                end,
                maxdim = maximum(skipmissing(g.maxdim_reached); init = 0),
                nsec = maximum(skipmissing(g.nsectors_max); init = 0),
                maxsec = maximum(skipmissing(g.maxsecdim); init = 0),
                nsat = maximum(skipmissing(g.nsaturated); init = 0),
            )
        )
    end
    return rows
end

function panel_walltime!(ax, series, keys, colors)
    for (k, key) in enumerate(keys)
        pts = sort(
            [r for r in series if (r.symmetry, r.U) == key], by = r -> r.chi
        )
        isempty(pts) && continue
        scatterlines!(
            ax, Float64[r.chi for r in pts], Float64[r.ms for r in pts];
            color = colors[k], linewidth = 2, markersize = 9,
            label = "$(key[1])  U=$(key[2])"
        )
    end
    return ax
end

function make_figures(df, figdir)
    mkpath(figdir)
    written = String[]

    # One figure per (lattice, size, quench, U, symmetry) group, χ as the series.
    groups = groupby(df, [:lattice, :size, :quench, :U, :symmetry]; sort = true)
    for g in groups
        chis = sort(unique(g.chi))
        colors = chi_colors(length(chis))
        key = first(g)
        title = "$(key.lattice) $(key.size)  $(key.quench)  U=$(key.U)  sym=$(key.symmetry)"

        fig = Figure(size = (1250, 400), backgroundcolor = "#fcfcfb")
        ax1 = Axis(
            fig[1, 1]; axis_kw(
                xlabel = "time", ylabel = "staggered order  mₛ",
                title = "Order parameter", backgroundcolor = "#fcfcfb"
            )...
        )
        panel_trajectory!(ax1, g, chis, colors)
        axislegend(ax1; position = :lb, framevisible = false, nbanks = 2, labelsize = 11)

        ax2 = Axis(
            fig[1, 2]; axis_kw(
                xlabel = "time", ylabel = "|mₛ − exact|", yscale = log10,
                title = "Truncation error vs exact", backgroundcolor = "#fcfcfb"
            )...
        )
        if panel_error!(ax2, g, chis, colors)
            axislegend(ax2; position = :rb, framevisible = false, labelsize = 11)
        else
            # No data on a log axis has no finite autolimits, so pin them before annotating.
            limits!(ax2, 0, 1, 1.0e-16, 1.0)
            text!(
                ax2, 0.5, 0.5; text = "no exact reference (U ≠ 0)", align = (:center, :center),
                space = :relative, color = REF_COLOR, fontsize = 12
            )
            hidedecorations!(ax2; label = false)
        end

        # For these quenches the exact total energy is identically 0 (a product start has no
        # inter-site coherence and no double occupancy), so this panel reads as pure drift.
        ax3 = Axis(
            fig[1, 3]; axis_kw(
                xlabel = "time", ylabel = "total energy",
                title = "Energy drift (exact ≡ 0)", backgroundcolor = "#fcfcfb"
            )...
        )
        if panel_energy!(ax3, g, chis, colors)
            axislegend(ax3; position = :lt, framevisible = false, labelsize = 11)
        else
            limits!(ax3, 0, 1, -1, 1)
            text!(
                ax3, 0.5, 0.5;
                text = "energy not measured\n(run used --no-energy)",
                align = (:center, :center), space = :relative, color = REF_COLOR,
                fontsize = 12,
            )
            hidedecorations!(ax3; label = false)
        end

        Label(fig[0, :], title; fontsize = 14, font = :bold, color = "#0b0b0b")
        slug = replace("$(key.lattice)$(key.size)_$(key.quench)_U$(key.U)_$(key.symmetry)", "/" => "-")
        out = joinpath(figdir, "trajectory_$(slug).svg")
        save(out, fig)
        push!(written, out)
    end

    # One cross-cutting figure: walltime scaling per (symmetry, U).
    series = walltime_series(df)
    if !isempty(series)
        keys = sort(unique([(r.symmetry, r.U) for r in series]))
        chis = sort(unique([r.chi for r in series]))
        fig = Figure(size = (620, 420), backgroundcolor = "#fcfcfb")
        ax = Axis(
            fig[1, 1]; axis_kw(
                xlabel = "χ", ylabel = "median step time (s)",
                xscale = log2, yscale = log10, title = "Cost per Trotter step",
                # explicit ticks: log2 otherwise labels the axis 2^2.0, 2^2.5, …
                xticks = (Float64.(chis), string.(chis)),
                backgroundcolor = "#fcfcfb"
            )...
        )
        panel_walltime!(ax, series, keys, cat_colors(length(keys)))
        axislegend(ax; position = :lt, framevisible = false, labelsize = 11)
        out = joinpath(figdir, "walltime_vs_chi.svg")
        save(out, fig)
        push!(written, out)
    end
    return written
end

function (@main)(args)
    o = parse_cli(args)
    root = o["outroot"]
    isdir(root) || error("no such directory: $root")

    # Skip our own output directories, so re-aggregating an already-aggregated root works.
    # `_run` holds the per-job Slurm/disBatch logs written by slurm/disbatch.sbatch.
    skip = ("figs", "logs", "_run")
    dirs = sort(
        [
            joinpath(root, d) for d in readdir(root)
                if isdir(joinpath(root, d)) && !(d in skip)
        ]
    )
    # Narrow the element type: `filter(!isnothing, ...)` leaves `Union{Nothing,DataFrame}`,
    # which sends `reduce(vcat, ...; cols=)` to the Base method instead of DataFrames'.
    frames = DataFrame[f for f in map(load_run, dirs) if !isnothing(f)]
    isempty(frames) && error(
        "found no result folders with both params.toml and observables.csv under $root"
    )
    df = reduce(vcat, frames; cols = :union)

    summary = isempty(o["summary"]) ? joinpath(root, "summary.csv") : o["summary"]
    CSV.write(summary, df)
    @printf("aggregated %d runs, %d rows → %s\n", length(frames), nrow(df), summary)

    for g in groupby(df, [:symmetry, :chi]; sort = true)
        k = first(g)
        # `init = NaN` would poison this: `max(NaN, x)` is NaN in Julia.
        errs = if hasproperty(g, :ref_circuit) && any(!ismissing, g.ref_circuit)
            ok = .!ismissing.(g.ref_circuit)
            maximum(abs.(g.m_s_all[ok] .- Float64.(g.ref_circuit[ok])); init = 0.0)
        else
            NaN
        end
        @printf(
            "  sym=%-16s χ=%-4d  max|err|=%9.3e  median step=%s\n",
            k.symmetry, k.chi, errs,
            ismissing(first(g.median_step_s)) ? "n/a" :
                @sprintf("%.4fs", first(g.median_step_s))
        )
    end

    # --- timing / crossover report ----------------------------------------------------------
    tt = timing_table(df)
    if !isempty(tt)
        base = "trivial/trivial"
        println("\nEvolution cost per step (gates + BP, measurement excluded)")
        println("  speedup > 1 means the symmetric run is FASTER than $base at the same χ")
        for u in sort(unique([r.U for r in tt]))
            println("\n  U = $u")
            @printf(
                "  %-5s %-18s %10s %10s %7s %7s %8s %6s %9s\n",
                "χ", "symmetry", "evolve/s", "bp/s", "maxdim", "sectors", "max/sec",
                "nsat", "speedup"
            )
            for chi in sort(unique([r.chi for r in tt if r.U == u]))
                here = [r for r in tt if r.U == u && r.chi == chi]
                b = findfirst(r -> r.symmetry == base, here)
                bt = isnothing(b) ? NaN : here[b].evolve
                for r in sort(here, by = r -> r.symmetry)
                    # `nsat` is the number of steps whose bonds had actually reached χ. Zero
                    # means this row does not measure the cost at χ at all; a small count means
                    # the median rests on very few samples. Either way, say so.
                    flag = r.nsat == 0 ? "  <-- NEVER reached χ; not a cost at this χ" :
                        r.nsat < 4 ? "  <-- only $(r.nsat) saturated step(s)" : ""
                    @printf(
                        "  %-5d %-18s %10.4f %10.4f %7d %7d %8d %6d %9s%s\n",
                        chi, r.symmetry, r.evolve, r.bp, r.maxdim, r.nsec, r.maxsec, r.nsat,
                        isnan(bt) ? "n/a" : @sprintf("%.2fx", bt / r.evolve), flag
                    )
                end
            end
        end
        # Where does each symmetry overtake the baseline?
        # Only χ values where BOTH rows have a usable number of saturated steps can support a
        # crossover claim; anything else is comparing intermediate bond dimensions.
        const_minsat = 4
        println("\n  Crossover (smallest χ at which the symmetric run is faster):")
        usable = sort(
            unique(
                [
                    r.chi for r in tt
                        if r.nsat >= const_minsat &&
                        any(q -> q.symmetry == base && q.chi == r.chi && q.nsat >= const_minsat, tt)
                ]
            )
        )
        if isempty(usable)
            println("    no χ has ≥$(const_minsat) saturated steps in both runs — inconclusive.")
            println("    Raise --nsteps: from a product state the bond dimension climbs")
            println("    gradually, so short runs never measure the cost at large χ.")
        else
            for sym in sort(unique([r.symmetry for r in tt if r.symmetry != base]))
                hit = nothing
                for chi in usable
                    s = findfirst(r -> r.symmetry == sym && r.chi == chi, tt)
                    b = findfirst(r -> r.symmetry == base && r.chi == chi, tt)
                    if !isnothing(s) && !isnothing(b) && tt[s].evolve < tt[b].evolve
                        hit = chi
                        break
                    end
                end
                @printf(
                    "    %-18s %s\n", sym,
                    isnothing(hit) ? "not faster at any usable χ ≤ $(maximum(usable))" :
                        "χ = $hit"
                )
            end
            println("    (usable χ, ≥$(const_minsat) saturated steps: $(join(usable, ", ")))")
        end
        println()
    end

    if !o["no_figures"]
        figdir = isempty(o["figdir"]) ? joinpath(root, "figs") : o["figdir"]
        for f in make_figures(df, figdir)
            println("wrote $f")
        end
    end
    return 0
end
