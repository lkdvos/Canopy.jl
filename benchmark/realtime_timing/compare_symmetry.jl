#!/usr/bin/env julia
#
# Trivial vs U(1) free-fermion real-time quench: accuracy *and* walltime.
#
# Runs the same 48-site hexagonal quench as `run_timings.jl` / `examples/realtime/main.jl` under
# both `Trivial` and `U1Irrep` (particle-number conserving) symmetry. The U(1) run carries an extra
# charge-bath "dummy" site so the global charge is trivial (see `run_timings.jl`). For each (symmetry,
# χ) it tracks the central-bond hopping ⟨C†ᵢCⱼ + h.c.⟩ trajectory and the per-step walltime.
#
# The U(1) sweep is several× slower than trivial, so each symmetry is run and *persisted* separately
# (one CSV per symmetry), then a `plot` pass renders the comparison figure:
#
#   - accuracy : |⟨C†C⟩ − exact| vs time, per χ, trivial (solid) vs U(1) (dashed)
#   - walltime : median step time vs χ, trivial vs U(1)
#
# The exact reference is the truncation-free single-particle correlation matrix (symmetry-independent),
# so at equal χ both symmetries keep the same total bond rank and should match in accuracy — the figure
# isolates the symmetry-bookkeeping cost in walltime.
#
# Usage (run the two sweeps first, then plot):
#   julia --project=benchmark/realtime_timing benchmark/realtime_timing/compare_symmetry.jl trivial
#   julia --project=benchmark/realtime_timing benchmark/realtime_timing/compare_symmetry.jl u1
#   julia --project=benchmark/realtime_timing benchmark/realtime_timing/compare_symmetry.jl plot
# With no argument it does all three in one process (only practical for small T_FINAL/χ).
#
# Set JULIA_NUM_THREADS to a fixed value (apply! threads over gates) and run the two sweeps
# sequentially (not in parallel) so the walltime comparison is fair.

using Pkg
Pkg.activate(@__DIR__; io=devnull)

using Canopy: hexagonal_lattice, product_state, BPMessages, belief_propagation,
    UndirectedEdge, LocalGate, CompositeGate, Circuit, apply!, edge_coloring, expect, virtualspace
using TensorKit
using TensorKitTensors.FermionOperators: f_num, f_hopping, fermion_space
using MatrixAlgebraKit: truncrank
using Dictionaries
using LinearAlgebra: Diagonal, transpose, BLAS
using Statistics: median
using DataFrames
using CSV
using Printf
using CairoMakie

BLAS.set_num_threads(1)

const M, N = 4, 6              # m × n unit cells → 2·m·n = 48 sites
const T_HOP = -1.0
const MU = 1.0
const DT = 0.01
const T_FINAL = 1.0
const NSTEPS = round(Int, T_FINAL / DT)
const CHIS = (4, 8, 16, 32)
const BP_ITERS = 30           # max per-step BP sweeps (warm-started; converges to tol below)
const BP_TOL = 1e-10
const DATADIR = joinpath(@__DIR__, "data")
const FIGDIR = joinpath(@__DIR__, "figs")

const ES = hexagonal_lattice(M, N)
const VERTS = sort(unique(Iterators.flatten((e.src, e.dst) for e in ES)))
const DUMMY = (0, 0, 0)       # charge-bath site for the U(1) model (distinct from (i,j,s), i,j≥1)

μ_of(v) = isodd(v[3]) ? MU : -MU
occ_of(v) = (sum(v) % 4 == 0) ? 0 : 1

# Measure on the occupied/empty-straddling edge nearest the lattice centre (coherence only builds on
# a bond whose two endpoints start with different occupation).
const _CENTER = ((M + 1) / 2, (N + 1) / 2)
const BOND = argmin(
    e -> sum(abs2, ((e.src[1], e.src[2]) .+ (e.dst[1], e.dst[2])) ./ 2 .- _CENTER),
    filter(e -> occ_of(e.src) != occ_of(e.dst), ES),
)

initial_state(::Type{Trivial}) = let P = fermion_space(Trivial)
    ps = Dictionary(VERTS, fill(P, length(VERTS)))
    ls = Dictionary(VERTS, [fℤ₂(occ_of(v)) => [1.0] for v in VERTS])
    product_state(ComplexF64, ES, ps, ls)
end

function initial_state(::Type{U1Irrep})
    Q = sum(occ_of(v) for v in VERTS)
    dsec = fℤ₂(mod(Q, 2)) ⊠ U1Irrep(-Q)
    P = fermion_space(U1Irrep)
    I = sectortype(P)
    verts = vcat(VERTS, [DUMMY])
    es = vcat(ES, [UndirectedEdge(DUMMY, first(VERTS))])
    ps = Dictionary(verts, vcat(fill(P, length(VERTS)), [Vect[I](dsec => 1)]))
    ls = Dictionary(
        verts,
        vcat([(fℤ₂(occ_of(v)) ⊠ U1Irrep(occ_of(v))) => [1.0] for v in VERTS], [dsec => [1.0]]),
    )
    return product_state(ComplexF64, es, ps, ls)
end

function build_layers(sym)
    n = f_num(ComplexF64, sym)
    g_plus = exp(-im * MU * DT * n)
    g_minus = exp(-im * (-MU) * DT * n)
    g_hop = exp(-im * (T_HOP * DT) * f_hopping(ComplexF64, sym))
    single = CompositeGate([LocalGate((v,), μ_of(v) > 0 ? g_plus : g_minus) for v in VERTS])
    hop = Circuit([
        CompositeGate([LocalGate((e.src, e.dst), g_hop) for e in class])
        for class in edge_coloring(ES)
    ])
    return single, hop
end

# (number of charge sectors, largest per-sector dimension) on the measured bond.
# At equal total χ, symmetry only pays off once the per-sector dimension is large
# enough that block `gemm` dominates the per-block permutation/bookkeeping.
function bond_sector_structure(state)
    V = virtualspace(state, BOND)
    secs = collect(sectors(V))
    return length(secs), maximum(dim(V, c) for c in secs; init=0)
end

# One trajectory; returns (⟨C†C⟩ over time, per-step walltime, (nsectors, maxsecdim)).
# Step 1 carries JIT cost.
function run_chi(sym, χ)
    state = initial_state(sym)
    msgs = belief_propagation(BPMessages(state), state; maxiter=BP_ITERS, tol=BP_TOL)
    single, hop = build_layers(sym)
    op = f_hopping(ComplexF64, sym)
    trunc = truncrank(χ)

    traj = zeros(Float64, NSTEPS + 1)
    steptimes = zeros(Float64, NSTEPS + 1)   # steptimes[1] (step 0) is unused (0.0)
    traj[1] = real(expect(state, msgs, op, BOND))
    for step in 1:NSTEPS
        t1 = time()
        state, msgs, _ = apply!(state, msgs, single)
        state, msgs, _ = apply!(state, msgs, hop; trunc)
        state, msgs, _ = apply!(state, msgs, single)
        msgs = belief_propagation(msgs, state; maxiter=BP_ITERS, tol=BP_TOL)
        steptimes[step+1] = time() - t1
        traj[step+1] = real(expect(state, msgs, op, BOND))
    end
    return traj, steptimes, bond_sector_structure(state)
end

# Truncation-free reference: C[i,j] = ⟨c†_i c_j⟩ evolves as C → ū C uᵀ under the same circuit.
apply_one_mode!(C, a, ph) = (@views(C[a, :] .*= conj(ph)); @views(C[:, a] .*= ph); C)
function apply_two_mode!(C, a, b, w)
    blk = [a, b]
    @views C[blk, :] .= conj(w) * C[blk, :]
    @views C[:, blk] .= C[:, blk] * transpose(w)
    return C
end

function exact_traj()
    idx = Dict(v => k for (k, v) in enumerate(VERTS))
    occ = Float64[occ_of(v) for v in VERTS]
    w = exp(-im * (T_HOP * DT) * ComplexF64[0 1; 1 0])
    ph = Dict(v => exp(-im * μ_of(v) * DT) for v in VERTS)
    hop_order = collect(Iterators.flatten(edge_coloring(ES)))
    i1, i2 = idx[BOND.src], idx[BOND.dst]
    obs(C) = real(C[i1, i2] + C[i2, i1])

    traj = zeros(Float64, NSTEPS + 1)
    C = complex(Matrix(Diagonal(occ)))
    traj[1] = obs(C)
    for step in 1:NSTEPS
        for v in VERTS
            apply_one_mode!(C, idx[v], ph[v])
        end
        for e in hop_order
            apply_two_mode!(C, idx[e.src], idx[e.dst], w)
        end
        for v in VERTS
            apply_one_mode!(C, idx[v], ph[v])
        end
        traj[step+1] = obs(C)
    end
    return traj
end

symtype(name) = name == "trivial" ? Trivial : U1Irrep

# Run one symmetry's χ-sweep and persist a long-format CSV: (chi, step, obs, steptime).
function run_sym(name)
    sym = symtype(name)
    nt = Threads.nthreads()
    @printf("Sweep %-8s on a %d-site hexagonal lattice — dt=%.3g steps=%d χ=%s threads=%d\n",
        name, 2 * M * N, DT, NSTEPS, CHIS, nt)
    rows = NamedTuple[]
    for χ in CHIS
        traj, st, (nsec, maxsec) = run_chi(sym, χ)
        for step in 0:NSTEPS
            push!(rows, (; chi=χ, step=step, obs=traj[step+1], steptime=st[step+1]))
        end
        med = median(@view st[3:end])   # drop step 0 (unused) and step 1 (JIT)
        @printf("  χ=%3d  median step %.4fs   bond: %d sectors, max %d/sector\n", χ, med, nsec, maxsec)
        flush(stdout)
    end
    mkpath(DATADIR)
    outfile = joinpath(DATADIR, "compare_$(name).csv")
    CSV.write(outfile, DataFrame(rows))
    println("wrote $outfile")
    return outfile
end

function make_plot()
    times = collect(0:NSTEPS) .* DT
    ex = exact_traj()
    dfs = Dict(name => CSV.read(joinpath(DATADIR, "compare_$(name).csv"), DataFrame)
               for name in ("trivial", "u1"))

    traj(name, χ) = sort(dfs[name][dfs[name].chi.==χ, :], :step).obs
    medstep(name, χ) = let s = sort(dfs[name][dfs[name].chi.==χ, :], :step).steptime
        median(@view s[3:end])     # drop step 0 and step 1 (JIT)
    end

    colors = [:silver, :lightblue, :royalblue, :navy]
    fig = Figure(size=(1350, 420))

    ax1 = Axis(fig[1, 1]; xlabel="Time", ylabel="⟨C†ᵢCⱼ + h.c.⟩",
        title="Central-bond hopping ($(2 * M * N) sites)")
    for (k, χ) in enumerate(CHIS)
        lines!(ax1, times, traj("trivial", χ); color=colors[mod1(k, length(colors))], label="χ=$χ")
    end
    lines!(ax1, times, ex; color=:red, linestyle=:dot, linewidth=2, label="exact")
    axislegend(ax1; position=:rt, nbanks=2)

    ax2 = Axis(fig[1, 2]; xlabel="Time", ylabel="Abs. error", yscale=log10,
        title="Accuracy vs exact  (solid: trivial, dashed: U(1))")
    for (k, χ) in enumerate(CHIS)
        c = colors[mod1(k, length(colors))]
        lines!(ax2, times, max.(abs.(traj("trivial", χ) .- ex), 1e-16); color=c, label="χ=$χ")
        lines!(ax2, times, max.(abs.(traj("u1", χ) .- ex), 1e-16); color=c, linestyle=:dash)
    end
    axislegend(ax2; position=:rb)

    ax3 = Axis(fig[1, 3]; xlabel="χ", ylabel="median step time (s)", xscale=log2, yscale=log10,
        xticks=(collect(CHIS), [string(χ) for χ in CHIS]), title="Walltime per step")
    for (name, lbl) in (("trivial", "trivial"), ("u1", "U(1)"))
        scatterlines!(ax3, collect(CHIS), [medstep(name, χ) for χ in CHIS]; label=lbl)
    end
    axislegend(ax3; position=:lt)

    mkpath(FIGDIR)
    outfile = joinpath(FIGDIR, "compare_symmetry.svg")
    save(outfile, fig)
    println("wrote $outfile")

    # Console summary.
    println("\nχ    trivial(s)   U(1)(s)   U(1)/triv   max|err| triv   max|err| U(1)")
    for χ in CHIS
        et = maximum(abs.(traj("trivial", χ) .- ex))
        eu = maximum(abs.(traj("u1", χ) .- ex))
        mt, mu = medstep("trivial", χ), medstep("u1", χ)
        @printf("%-4d %.4f      %.4f    %5.2f×      %.2e        %.2e\n", χ, mt, mu, mu / mt, et, eu)
    end
    return outfile
end

function (@main)(args)
    mode = isempty(args) ? "all" : args[1]
    if mode == "trivial" || mode == "u1"
        run_sym(mode)
    elseif mode == "plot"
        make_plot()
    elseif mode == "all"
        run_sym("trivial")
        run_sym("u1")
        make_plot()
    else
        error("unknown mode $mode (expected: trivial | u1 | plot)")
    end
    return 0
end
