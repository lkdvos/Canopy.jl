#!/usr/bin/env julia
#
# Read per-step timing CSVs (one set per library) and render comparison figures.
#
#   ./plot_timings.jl --datadir data --outdir figs
#   julia --project=benchmark/realtime_timing benchmark/realtime_timing/plot_timings.jl --help
#
# Every `*.csv` under `--datadir` matching the schema written by `run_timings.jl`
# (chi, nthreads, nsites, dt, step, single1, hop, single2, bp, maxdim) is loaded. The run
# `label` for grouping/legends is taken from the filename prefix (`<label>_chi<χ>.csv`).
# Headline numbers use the median over the *saturated* steps (where the bond dimension has
# reached its plateau), excluding the bond-growth ramp and GC spikes.

using Pkg
Pkg.activate(@__DIR__; io=devnull)

using ArgParse
using CSV
using DataFrames
using Statistics: median
using CairoMakie

function parse_cli(args)
    s = ArgParseSettings(description="Render comparison figures from run_timings.jl CSVs.")
    @add_arg_table! s begin
        "--datadir"
        help = "directory of input CSVs"
        default = joinpath(@__DIR__, "data")
        "--outdir"
        help = "directory for output figures"
        default = joinpath(@__DIR__, "figs")
    end
    return parse_args(args, s)
end

# Load every CSV, tagging each with a `label` derived from its filename prefix.
function load_all(dir)
    files = filter(f -> endswith(f, ".csv"), readdir(dir; join=true))
    isempty(files) && error("no CSV files found in $dir")
    dfs = map(files) do f
        d = CSV.read(f, DataFrame)
        name = splitext(basename(f))[1]                 # e.g. "canopy_chi16"
        m = match(r"^(.*)_chi\d+$", name)
        d.label = fill(m === nothing ? name : String(m.captures[1]), nrow(d))
        d
    end
    df = reduce(vcat, dfs)
    df.total = df.single1 .+ df.hop .+ df.single2 .+ df.bp
    df.total_bytes = df.single1_bytes .+ df.hop_bytes .+ df.single2_bytes .+ df.bp_bytes
    return df
end

# Median of `col` over the saturated steps (step ≥ 1 with maxdim at its plateau value).
function plateau_median(g, col)
    loop = g[g.step.>=1, :]
    isempty(loop) && return NaN
    plat = loop[loop.maxdim.==maximum(loop.maxdim), :]
    return median(plat[!, col])
end

function (@main)(args)
    opts = parse_cli(args)
    df = load_all(opts["datadir"])
    outdir = opts["outdir"]
    labels = sort(unique(df.label))
    chis = sort(unique(df.chi))
    palette = Makie.wong_colors()
    lcolor = Dict(l => palette[mod1(i, length(palette))] for (i, l) in enumerate(labels))

    # Per-(label, χ) plateau medians.
    summ = combine(groupby(df, [:label, :chi]),
        sdf -> (; total=plateau_median(sdf, :total), bp=plateau_median(sdf, :bp),
            single=plateau_median(sdf, :single1) + plateau_median(sdf, :single2),
            hop=plateau_median(sdf, :hop), bytes=plateau_median(sdf, :total_bytes)))

    mkpath(outdir)

    # --- Fig 1: scaling — median step time vs χ ---------------------------------------------
    fig1 = Figure(size=(560, 440))
    ax = Axis(fig1[1, 1]; xlabel="χ", ylabel="median step time [s]",
        xscale=log2, yscale=log10, title="Per-step time vs bond dimension")
    for l in labels
        s = sort(summ[summ.label.==l, :], :chi)
        scatterlines!(ax, s.chi, s.total; label=l, color=lcolor[l])
    end
    axislegend(ax; position=:lt)
    save(joinpath(outdir, "scaling.svg"), fig1)

    # --- Fig 2: phase breakdown — stacked (phase) × dodged (label) bars per χ ----------------
    phases = [:single, :hop, :bp]
    pcolor = palette[1:3]
    xs = Int[]
    ys = Float64[]
    stk = Int[]
    dge = Int[]
    for (ci, χ) in enumerate(chis), (li, l) in enumerate(labels)
        row = summ[(summ.label.==l).&(summ.chi.==χ), :]
        isempty(row) && continue
        for (pi, p) in enumerate(phases)
            push!(xs, ci)
            push!(ys, row[1, p])
            push!(stk, pi)
            push!(dge, li)
        end
    end
    fig2 = Figure(size=(640, 440))
    ax2 = Axis(fig2[1, 1]; xlabel="χ", ylabel="median time [s]",
        xticks=(1:length(chis), string.(chis)), title="Phase breakdown (dodge = label)")
    barplot!(ax2, xs, ys; stack=stk, dodge=dge, color=pcolor[stk])
    Legend(fig2[1, 2],
        [PolyElement(color=c) for c in pcolor], string.(phases), "phase";
        framevisible=false)
    save(joinpath(outdir, "phase_breakdown.svg"), fig2)

    # --- Fig 3: per-step time series at the largest χ ---------------------------------------
    χmax = maximum(chis)
    fig3 = Figure(size=(720, 420))
    ax3 = Axis(fig3[1, 1]; xlabel="Trotter step", ylabel="step time [s]",
        title="Per-step time trace (χ=$χmax)")
    for l in labels
        @show l
        s = sort(df[(df.label.==l).&(df.chi.==χmax).&(df.step.>=1), :], :step)
        isempty(s) && continue
        lines!(ax3, s.step, s.total; label=l, color=lcolor[l])
    end
    axislegend(ax3; position=:rt)
    save(joinpath(outdir, "timeseries_chi$(χmax).svg"), fig3)

    # --- Fig 4: heap allocation per step vs χ -----------------------------------------------
    fig4 = Figure(size=(560, 440))
    ax4 = Axis(fig4[1, 1]; xlabel="χ", ylabel="median heap [MiB / step]",
        xscale=log2, title="Per-step heap allocation vs bond dimension")
    for l in labels
        s = sort(summ[summ.label.==l, :], :chi)
        scatterlines!(ax4, s.chi, s.bytes ./ 2^20; label=l, color=lcolor[l])
    end
    axislegend(ax4; position=:lt)
    save(joinpath(outdir, "allocations.svg"), fig4)

    println("wrote scaling.svg, phase_breakdown.svg, timeseries_chi$(χmax).svg, allocations.svg to $outdir")
    show(summ; allrows=true)
    println()
    return 0
end
