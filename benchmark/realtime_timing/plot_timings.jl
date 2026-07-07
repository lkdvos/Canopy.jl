#!/usr/bin/env julia
#
# Read per-run timing CSVs, write the combined CSV, and render comparison figures.
#
#   ./plot_timings.jl --datadir data --outdir figs
#   julia --project=benchmark/realtime_timing benchmark/realtime_timing/plot_timings.jl --help
#
# Every `*.csv` under `--datadir` written by `run_timings.jl` (one file per model/symmetry run,
# all χ stacked) is loaded; each carries `library` and `model` columns. They are concatenated
# into `<datadir>/combined.csv` (the single file for downstream plotting), then figures are
# faceted by `model` with one line per `library`. Headline numbers use the median over the
# *saturated* steps (where the bond dimension has reached its plateau), excluding the
# bond-growth ramp and GC spikes.

using Pkg
Pkg.activate(@__DIR__; io=devnull)

using ArgParse
using CSV
using DataFrames
using Statistics: median
using CairoMakie

const COMBINED = "combined.csv"

function parse_cli(args)
    s = ArgParseSettings(description="Render comparison figures from run_timings.jl CSVs.")
    @add_arg_table! s begin
        "--datadir"
        help = "directory of input CSVs (one per run); combined.csv is written here too"
        default = joinpath(@__DIR__, "data")
        "--outdir"
        help = "directory for output figures"
        default = joinpath(@__DIR__, "figs")
    end
    return parse_args(args, s)
end

# Load every per-run CSV (skipping the combined file itself), concatenate, and persist
# combined.csv. Files lacking the `library`/`model` columns (e.g. an old-schema CSV) are skipped.
function load_all(dir)
    files = filter(readdir(dir; join=true)) do f
        endswith(f, ".csv") && basename(f) != COMBINED
    end
    isempty(files) && error("no CSV files found in $dir")
    dfs = DataFrame[]
    for f in files
        d = CSV.read(f, DataFrame)
        if !all(in(names(d)), ("library", "model", "chi"))
            @warn "skipping $(basename(f)): missing library/model/chi columns"
            continue
        end
        push!(dfs, d)
    end
    isempty(dfs) && error("no conforming CSV files in $dir")
    df = reduce(vcat, dfs; cols=:union)
    df.total = df.single1 .+ df.hop .+ df.single2 .+ df.bp
    df.total_bytes = df.single1_bytes .+ df.hop_bytes .+ df.single2_bytes .+ df.bp_bytes
    CSV.write(joinpath(dir, COMBINED), df)
    return df
end

# Median of `col` over the saturated steps (step ≥ 1 with maxdim at its plateau value).
function plateau_median(g, col)
    loop = g[g.step.>=1, :]
    isempty(loop) && return NaN
    plat = loop[loop.maxdim.==maximum(loop.maxdim), :]
    return median(plat[!, col])
end

# Arrange `n` model facets into a roughly square grid of (row, col) positions.
function grid_positions(n)
    ncols = ceil(Int, sqrt(n))
    return [((i - 1) ÷ ncols + 1, (i - 1) % ncols + 1) for i in 1:n]
end

function (@main)(args)
    opts = parse_cli(args)
    df = load_all(opts["datadir"])
    outdir = opts["outdir"]
    libraries = sort(unique(df.library))
    models = sort(unique(df.model))
    chis = sort(unique(df.chi))
    pos = Dict(m => p for (m, p) in zip(models, grid_positions(length(models))))
    palette = Makie.wong_colors()
    modelcolor = Dict(m => palette[mod1(i, length(palette))] for (i, m) in enumerate(models))
    libstyle = Dict(l => [:solid, :dash, :dot, :dashdot][mod1(i, 4)] for (i, l) in enumerate(libraries))

    # Per-(library, model, χ) plateau medians.
    summ = combine(groupby(df, [:library, :model, :chi]),
        sdf -> (; total=plateau_median(sdf, :total), bp=plateau_median(sdf, :bp),
            single=plateau_median(sdf, :single1) + plateau_median(sdf, :single2),
            hop=plateau_median(sdf, :hop), bytes=plateau_median(sdf, :total_bytes)))

    mkpath(outdir)

    # All comparison figures share one axis: colour encodes the model/symmetry, linestyle the
    # library, so every model/symmetry overlays directly. The library legend is added only when
    # more than one library is present (e.g. when overlaying a comparison library).
    function add_legend!(ax)
        axislegend(ax, [LineElement(color=modelcolor[m]) for m in models], models, "model";
            position=:lt, framevisible=false)
        if length(libraries) > 1
            axislegend(ax, [LineElement(color=:gray, linestyle=libstyle[l]) for l in libraries],
                libraries, "library"; position=:rb, framevisible=false)
        end
    end

    # --- Fig 1: BP time vs χ, all models/symmetries on one axis --------------------------------
    fig1 = Figure(size=(640, 470))
    ax1 = Axis(fig1[1, 1]; xlabel="χ", ylabel="median BP time [s]",
        xscale=log2, yscale=log10, xticks=(chis, string.(chis)),
        title="Belief-propagation time vs bond dimension")
    for m in models, l in libraries
        s = sort(summ[(summ.model.==m).&(summ.library.==l), :], :chi)
        isempty(s) && continue
        scatterlines!(ax1, s.chi, s.bp; color=modelcolor[m], linestyle=libstyle[l])
    end
    add_legend!(ax1)
    save(joinpath(outdir, "scaling.svg"), fig1)

    # --- Fig 2: phase breakdown — stacked (phase) × dodged (library), faceted by model --------
    phases = [:single, :hop, :bp]
    pcolor = palette[1:3]
    fig2 = Figure(size=(720, 560))
    for m in models
        r, c = pos[m]
        ax = Axis(fig2[r, c]; xlabel="χ", ylabel="median time [s]",
            xticks=(1:length(chis), string.(chis)), title=m)
        xs, ys, stk, dge = Int[], Float64[], Int[], Int[]
        for (ci, χ) in enumerate(chis), (li, l) in enumerate(libraries)
            row = summ[(summ.library.==l).&(summ.model.==m).&(summ.chi.==χ), :]
            isempty(row) && continue
            for (pi, p) in enumerate(phases)
                push!(xs, ci); push!(ys, row[1, p]); push!(stk, pi); push!(dge, li)
            end
        end
        isempty(xs) || barplot!(ax, xs, ys; stack=stk, dodge=dge, color=pcolor[stk])
    end
    Legend(fig2[end+1, :], [PolyElement(color=c) for c in pcolor], string.(phases), "phase";
        orientation=:horizontal, framevisible=false)
    save(joinpath(outdir, "phase_breakdown.svg"), fig2)

    # --- Fig 3: per-step time series, all models on one axis at the largest *common* χ --------
    # (Using each model's own max χ would mix bond dimensions; the largest χ present for every
    # model keeps the comparison fair and avoids empty series for models that crashed early.)
    common = reduce(intersect, (Set(df[df.model.==m, :chi]) for m in models))
    χts = isempty(common) ? maximum(chis) : maximum(common)
    fig3 = Figure(size=(720, 440))
    ax3 = Axis(fig3[1, 1]; xlabel="Trotter step", ylabel="step time [s]",
        title="Per-step time trace (χ=$χts)")
    for m in models, l in libraries
        s = sort(df[(df.model.==m).&(df.library.==l).&(df.chi.==χts).&(df.step.>=1), :], :step)
        isempty(s) && continue
        lines!(ax3, s.step, s.total; color=modelcolor[m], linestyle=libstyle[l])
    end
    add_legend!(ax3)
    save(joinpath(outdir, "timeseries.svg"), fig3)

    # --- Fig 4: heap allocation per step vs χ, all models/symmetries on one axis --------------
    fig4 = Figure(size=(640, 470))
    ax4 = Axis(fig4[1, 1]; xlabel="χ", ylabel="median heap [MiB / step]",
        xscale=log2, xticks=(chis, string.(chis)),
        title="Per-step heap allocation vs bond dimension")
    for m in models, l in libraries
        s = sort(summ[(summ.model.==m).&(summ.library.==l), :], :chi)
        isempty(s) && continue
        scatterlines!(ax4, s.chi, s.bytes ./ 2^20; color=modelcolor[m], linestyle=libstyle[l])
    end
    add_legend!(ax4)
    save(joinpath(outdir, "allocations.svg"), fig4)

    println("wrote $COMBINED to $(opts["datadir"]); figures scaling/phase_breakdown/" *
            "timeseries/allocations.svg to $outdir")
    show(summ; allrows=true)
    println()
    return 0
end
