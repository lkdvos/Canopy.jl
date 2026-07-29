#!/usr/bin/env julia
#
# Figures for the symmetry timing benchmark.
# =========================================
#
#   ./plot_benchmark.jl --outroot results/bench
#
# Reads the `summary.csv` written by aggregate.jl and renders two figures:
#
#   benchmark_cost.svg       total cost vs χ, and speedup against no symmetry
#   benchmark_breakdown.svg  the same cost split into gates / BP, plus BP's share
#
# Design notes (so this stays consistent if extended):
#
# - Symmetry is a categorical dimension, so it gets fixed hues assigned in a stable order,
#   never cycled. The four slots used here were checked with the data-viz palette validator on
#   the light surface: worst adjacent CVD ΔE 9.1 (protan), worst adjacent normal-vision ΔE 22.9,
#   all inside the lightness band and above the chroma floor.
# - Colour follows the symmetry, never its rank, so a panel that drops a series does not
#   repaint the others.
# - Every panel carries a legend, and no panel uses two y-scales.
# - Points whose bonds never fully saturated (`nsat < 4`) are drawn as hollow markers: they are
#   provisional, and hiding that would overstate the large-χ numbers.
# - Light surface only, by intent — these are figures for a paper/repo, not a themed web page.

using Pkg
Pkg.activate(@__DIR__; io = devnull)

using ArgParse
using CSV
using DataFrames
using Printf
using Statistics: median
using CairoMakie

# --- palette -------------------------------------------------------------------------------
const SURFACE = "#fcfcfb"
const INK = "#0b0b0b"
const INK2 = "#52514e"
const GRID = (:black, 0.08)
# categorical slots 1-4, in fixed order
const HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# stable symmetry -> slot mapping, so colour never depends on which series are present
const SYM_ORDER = ["trivial/trivial", "u1/trivial", "trivial/u1", "u1/u1"]
const SYM_LABEL = Dict(
    "trivial/trivial" => "none",
    "u1/trivial" => "U(1) charge",
    "trivial/u1" => "U(1) spin",
    "u1/u1" => "U(1)×U(1)",
)
color_of(sym) = HUES[something(findfirst(==(sym), SYM_ORDER), length(HUES))]
const BASE = "trivial/trivial"
const MINSAT = 4

function parse_cli(args)
    s = ArgParseSettings(
        description = "Plot the symmetry timing benchmark.", autofix_names = true
    )
    @add_arg_table! s begin
        "--outroot"
        help = "benchmark result root (must contain summary.csv)"
        default = joinpath(@__DIR__, "results", "bench")
        "--summary"
        help = "explicit path to summary.csv"
        default = ""
        "--figdir"
        help = "output directory for figures"
        default = ""
        "--min-saturated"
        help = "drop (χ, symmetry) points with fewer than this many steps at χ. Such points \
                are not merely noisy but biased LOW: the first steps after a bond saturates \
                are the cheapest, because warm-started BP needs fewer sweeps early on. \
                Plotting them would understate large-χ cost."
        arg_type = Int
        default = 4
    end
    return parse_args(args, s)
end

# One row per (chi, symmetry): total evolve cost, BP cost, and saturated-step count.
function load_points(path)
    df = CSV.read(path, DataFrame)
    pts = NamedTuple[]
    for g in groupby(df, [:chi, :symmetry]; sort = true)
        k = first(g)
        ev = collect(skipmissing(g.median_evolve_s))
        bp = collect(skipmissing(g.median_bp_s))
        (isempty(ev) || isempty(bp)) && continue
        nsat = hasproperty(g, :nsaturated) ? maximum(skipmissing(g.nsaturated); init = 0) : MINSAT
        reached = hasproperty(g, :maxdim_reached) ?
            maximum(skipmissing(g.maxdim_reached); init = 0) : Int(k.chi)
        e, b = median(ev), median(bp)
        push!(
            pts, (;
                chi = Int(k.chi), symmetry = String(k.symmetry),
                evolve = e, bp = b, gates = max(e - b, eps()), share = 100b / e,
                nsat = nsat, reached = reached,
            )
        )
    end
    return pts
end

axis_kw(; kw...) = (;
    backgroundcolor = SURFACE, xgridcolor = GRID, ygridcolor = GRID,
    xgridwidth = 1, ygridwidth = 1, topspinevisible = false, rightspinevisible = false,
    titlecolor = INK, xlabelcolor = INK2, ylabelcolor = INK2, kw...
)

chi_ticks(chis) = (Float64.(chis), string.(chis))

# One series per symmetry. Provisional points (few saturated steps) get hollow markers so the
# data-quality caveat is visible in the figure rather than only in the caption.
function series!(ax, pts, syms, value)
    for sym in syms
        p = sort([r for r in pts if r.symmetry == sym], by = r -> r.chi)
        isempty(p) && continue
        c = color_of(sym)
        x = Float64[r.chi for r in p]
        y = Float64[value(r) for r in p]
        lines!(ax, x, y; color = c, linewidth = 2, label = SYM_LABEL[sym])
        solid = [r.nsat >= MINSAT for r in p]
        any(solid) && scatter!(ax, x[solid], y[solid]; color = c, markersize = 9)
        # hollow = provisional: white fill, coloured stroke
        any(.!solid) && scatter!(
            ax, x[.!solid], y[.!solid];
            color = SURFACE, strokecolor = c, strokewidth = 2, markersize = 9,
        )
    end
    return ax
end

function figure_cost(pts, chis, syms, figdir)
    fig = Figure(size = (1080, 430), backgroundcolor = SURFACE)

    ax1 = Axis(
        fig[1, 1]; axis_kw(
            xlabel = "bond dimension χ", ylabel = "time per Trotter step (s)",
            title = "Cost per step", xscale = log2, yscale = log10,
            xticks = chi_ticks(chis),
            # explicit ticks: log10 otherwise labels the axis 10^-0.5, 10^0.5, …
            yticks = let t = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
                hi = maximum(r.evolve for r in pts)
                lo = minimum(r.evolve for r in pts)
                k = filter(v -> v >= lo / 5 && v <= hi * 5, t)
                (k, [v < 1 ? string(v) : string(Int(v)) for v in k])
            end,
        )...
    )
    series!(ax1, pts, syms, r -> r.evolve)
    axislegend(ax1; position = :lt, framevisible = false, labelsize = 11, labelcolor = INK2)

    # Speedup against no symmetry. Ratios are multiplicative, so a log axis puts "2x faster"
    # and "2x slower" the same distance from the break-even rule.
    ax2 = Axis(
        fig[1, 2]; axis_kw(
            xlabel = "bond dimension χ", ylabel = "speedup vs no symmetry (×)",
            title = "", xscale = log2, yscale = log10,
            xticks = chi_ticks(chis),
            yticks = ([0.1, 0.25, 0.5, 1, 2, 3], ["0.1", "0.25", "0.5", "1", "2", "3"]),
        )...
    )
    base = Dict(r.chi => r.evolve for r in pts if r.symmetry == BASE)
    ratio = [
        (; r..., evolve = get(base, r.chi, NaN) / r.evolve)
            for r in pts if r.symmetry != BASE && haskey(base, r.chi)
    ]
    # Title from the data, so it cannot go stale when the benchmark is re-run or extended.
    cross = filter(!isnothing, [
        let hits = sort([r.chi for r in ratio if r.symmetry == s && r.evolve > 1])
                isempty(hits) ? nothing : first(hits)
            end
            for s in unique(r.symmetry for r in ratio)
    ])
    ax2.title = isempty(cross) ? "No symmetry is faster in this χ range" :
        length(unique(cross)) == 1 ? "Symmetry overtakes at χ = $(first(cross))" :
        "Symmetry overtakes at χ ≈ $(minimum(cross))–$(maximum(cross))"
    hlines!(ax2, [1.0]; color = INK2, linestyle = :dash, linewidth = 1.5)
    # data coordinates, not `space = :relative`: mixing the two silently places the label
    # outside the visible area.
    text!(
        ax2, Float64(last(chis)), 1.0; text = "break-even", offset = (-2, 5),
        align = (:right, :bottom), color = INK2, fontsize = 10,
    )
    series!(ax2, ratio, filter(!=(BASE), syms), r -> r.evolve)
    axislegend(ax2; position = :lt, framevisible = false, labelsize = 11, labelcolor = INK2)

    Label(
        fig[2, :],
        "hex 4×4 (32 sites), U=4, one core, icelake.  Cost is the median over steps whose \
         bonds had reached χ; points with too few such steps are omitted, so a series may \
         stop short of the widest χ shown.";
        fontsize = 10, color = INK2, halign = :left, padding = (10, 0, 0, 0),
    )
    mkpath(figdir)
    out = joinpath(figdir, "benchmark_cost.svg")
    save(out, fig)
    return out
end

function figure_breakdown(pts, chis, syms, figdir)
    fig = Figure(size = (1420, 430), backgroundcolor = SURFACE)

    ax1 = Axis(
        fig[1, 1]; axis_kw(
            xlabel = "bond dimension χ", ylabel = "time per step (s)",
            title = "Gates (simple update)", xscale = log2, yscale = log10,
            xticks = chi_ticks(chis),
        )...
    )
    series!(ax1, pts, syms, r -> r.gates)
    axislegend(ax1; position = :lt, framevisible = false, labelsize = 11, labelcolor = INK2)

    ax2 = Axis(
        fig[1, 2]; axis_kw(
            xlabel = "bond dimension χ", ylabel = "time per step (s)",
            title = "Belief propagation", xscale = log2, yscale = log10,
            xticks = chi_ticks(chis),
        )...
    )
    series!(ax2, pts, syms, r -> r.bp)
    axislegend(ax2; position = :lt, framevisible = false, labelsize = 11, labelcolor = INK2)

    # Share one y-scale across the gate and BP panels: the whole point is that BP overtakes the
    # gates for the symmetric runs, and independent autoscaled axes would hide the crossing by
    # rescaling each panel to fill its own frame.
    linkyaxes!(ax1, ax2)
    ax2.ylabel = ""

    ax3 = Axis(
        fig[1, 3]; axis_kw(
            xlabel = "bond dimension χ", ylabel = "BP share of step (%)",
            title = "Where the time goes", xscale = log2,
            xticks = chi_ticks(chis), yticks = (0:20:100, string.(0:20:100)),
        )...
    )
    ylims!(ax3, 0, 100)
    series!(ax3, pts, syms, r -> r.share)
    axislegend(ax3; position = :lt, framevisible = false, labelsize = 11, labelcolor = INK2)

    # Descriptive, not a hardcoded conclusion: the balance between these two panels shifts with
    # χ, lattice and BP settings, and a baked-in claim goes stale on the next run.
    Label(
        fig[2, :],
        "Gate and BP panels share a y-scale, so the two are directly comparable. Note that BP \
         cost here also reflects how many sweeps convergence needed, not only the cost per \
         sweep — run with --bp-tol 0 to hold the sweep count fixed.";
        fontsize = 10, color = INK2, halign = :left, padding = (10, 0, 0, 0),
    )
    mkpath(figdir)
    out = joinpath(figdir, "benchmark_breakdown.svg")
    save(out, fig)
    return out
end

function (@main)(args)
    o = parse_cli(args)
    path = isempty(o["summary"]) ? joinpath(o["outroot"], "summary.csv") : o["summary"]
    isfile(path) || error("no summary.csv at $path — run aggregate.jl first")
    figdir = isempty(o["figdir"]) ? joinpath(dirname(path), "figs") : o["figdir"]

    all_pts = load_points(path)
    isempty(all_pts) && error("no timing rows in $path")

    # Keep only points that actually measure the cost at their χ, for enough steps to be
    # unbiased. Report what was dropped instead of silently thinning the curves.
    minsat = o["min_saturated"]
    pts = [r for r in all_pts if r.nsat >= minsat]
    dropped = [r for r in all_pts if r.nsat < minsat]
    if !isempty(dropped)
        println("dropped $(length(dropped)) incomplete point(s) (<$(minsat) steps at χ):")
        for r in sort(dropped, by = r -> (r.chi, r.symmetry))
            @printf(
                "  χ=%-4d %-16s nsat=%-3d reached χ=%d\n",
                r.chi, r.symmetry, r.nsat, r.reached
            )
        end
    end
    isempty(pts) && error("no point has ≥$(minsat) saturated steps")
    chis = sort(unique(r.chi for r in pts))
    syms = [s for s in SYM_ORDER if any(r -> r.symmetry == s, pts)]
    extra = setdiff(unique(r.symmetry for r in pts), SYM_ORDER)
    isempty(extra) || @warn "symmetries with no assigned colour slot are omitted" extra

    for f in (figure_cost(pts, chis, syms, figdir), figure_breakdown(pts, chis, syms, figdir))
        println("wrote $f")
    end
    return 0
end
