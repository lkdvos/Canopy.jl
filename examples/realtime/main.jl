using Markdown #hide

md"""
# Real-time evolution of free fermions

Real-time evolution of spinless free fermions on a finite honeycomb lattice,

```math
H = -\sum_{\langle ij \rangle} \left( c^\dagger_i c_j + \text{h.c.} \right) + \mu \sum_i (\pm 1)_i\, n_i,
\qquad |\psi_t\rangle = e^{-iHt} |\psi_0\rangle.
```

A quench from a product state in a staggered (sublattice-alternating) field, evolved in real
time by belief propagation + bond truncation, benchmarked against the exact single-particle
solution. This reproduces panel (b) of `FreeFermionBenchmark.pdf` (48-site hexagonal
lattice): the central-bond hopping $\langle C^\dagger_i C_j + \text{h.c.} \rangle$ vs. time
for BP bond dimensions $\chi \in \{4, 8, 16, 32\}$, overlaid on the truncation-free reference.

This is a Canopy port of `examples/realtime/example.jl` (which uses
TensorNetworkQuantumSimulator + NamedGraphs). Two representation notes:

- NamedGraphs labels honeycomb vertices `(i, j)` and the bipartite sublattice is
  `isodd(i+j)`; Canopy's `hexagonal_lattice` labels them `(i, j, s)` with the sublattice
  carried explicitly by `s ∈ {1, 2}`. The staggered field is therefore set per sublattice
  `s` here — the physical image of `example.jl`'s `isodd(sum(v))` field.
- The initial-occupation rule `sum(v) % 4 == 0 ? empty : occupied` is ported literally; on
  48 sites it leaves 36 occupied (¾-filled, even parity — the PDF's "quarter filling" counts
  the ¼ holes). Flip to `... ? occupied : empty` for literal ¼ occupation.

Real time uses the same first-order split as `example.jl`: each `dt` step applies the on-site
number layer, the hopping layer, then the on-site layer again. The exact reference applies the
identical circuit to the single-particle correlation matrix, so it differs from the TN run
only by bond truncation.

This example can be run from the command line with:

```
julia --project=examples examples/realtime/main.jl
```
"""

using Canopy: hexagonal_lattice, product_state, BPMessages, belief_propagation,
    UndirectedEdge, LocalGate, CompositeGate, Circuit, apply!, edge_coloring,
    expect, virtualspace
using TensorKit
using TensorKitTensors.FermionOperators: f_num, f_hopping, fermion_space
using MatrixAlgebraKit: truncrank
using Dictionaries
using LinearAlgebra: Diagonal, transpose
using Printf
using TimerOutputs
using CairoMakie

md"""
## Model and run parameters

The defaults run in a few minutes and already show BP tracking the exact curve through the
first oscillations. For the full `FreeFermionBenchmark.pdf` panel (b) x-axis, set
`T_FINAL = 5.0` (≈ 10× longer; cost ≈ `length(CHIS) · NSTEPS · BP`).
"""

const M, N = 4, 6              # m × n unit cells → 2·m·n = 48 sites
const T_HOP = -1.0             # hopping coefficient t  (H = -Σ c†c + h.c. + …)
const MU = 1.0                # staggered-field amplitude
const DT = 0.01
const T_FINAL = 5.0
const NSTEPS = round(Int, T_FINAL / DT)
const CHIS = (4, 8, 16, 32)
const BP_ITERS = 30           # per-step BP sweeps (warm-started from the previous step)

const ES = hexagonal_lattice(M, N)                       # open-boundary honeycomb
const VERTS = sort(unique(Iterators.flatten((e.src, e.dst) for e in ES)))
const HOPOP = f_hopping(ComplexF64, Trivial)             # c†_i c_j + c†_j c_i

md"""
The staggered $\pm\mu$ field alternates on the two sublattices ($s = 1 \to +\mu$,
$s = 2 \to -\mu$), and the initial occupation is empty iff `sum(v) % 4 == 0` (the
`example.jl` rule):
"""

μ_of(v) = isodd(v[3]) ? MU : -MU
occ_of(v) = (sum(v) % 4 == 0) ? 0 : 1

md"""
We measure on the occupied/empty-straddling edge nearest the lattice centre. Coherence
$\langle C^\dagger C \rangle$ only builds on a bond whose two sites start with *different*
occupation, so a bond inside a uniformly-filled region would stay $\approx 0$.
"""

const _CENTER = ((M + 1) / 2, (N + 1) / 2)
const BOND = argmin(
    e -> sum(abs2, ((e.src[1], e.src[2]) .+ (e.dst[1], e.dst[2])) ./ 2 .- _CENTER),
    filter(e -> occ_of(e.src) != occ_of(e.dst), ES),
)
const VC, VN = BOND.src, BOND.dst

md"""
## Initial product state
"""

function initial_state()
    P = fermion_space(Trivial)
    ps = Dictionary(VERTS, fill(P, length(VERTS)))
    ls = Dictionary(VERTS, [fℤ₂(occ_of(v)) => [1.0] for v in VERTS])
    return product_state(ComplexF64, ES, ps, ls)
end

md"""
## Trotter layers

One `dt` step is single-site, hopping, single-site.
"""

function build_layers()
    n = f_num(ComplexF64, Trivial)
    g_plus = exp(-im * MU * DT * n)
    g_minus = exp(-im * (-MU) * DT * n)
    g_hop = exp(-im * (T_HOP * DT) * HOPOP)
    single = CompositeGate([LocalGate((v,), μ_of(v) > 0 ? g_plus : g_minus) for v in VERTS])
    hop = Circuit([
        CompositeGate([LocalGate((e.src, e.dst), g_hop) for e in class])
        for class in edge_coloring(ES)
    ])
    return single, hop
end

md"""
## Single-χ real-time trajectory
"""

function run_chi(χ; verbose=true)
    state = initial_state()
    msgs = BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter=BP_ITERS)
    single, hop = build_layers()
    trunc = truncrank(χ)
    to = TimerOutput()

    traj = zeros(Float64, NSTEPS + 1)
    traj[1] = real(expect(state, msgs, HOPOP, BOND))
    t_gate = 0.0
    t_bp = 0.0
    for step in 1:NSTEPS
        t1 = time()
        state, msgs, _ = apply!(state, msgs, single; timer=to)
        state, msgs, _ = apply!(state, msgs, hop; trunc, timer=to)
        state, msgs, _ = apply!(state, msgs, single; timer=to)
        t2 = time()
        ## each dt step perturbs the state only slightly, so BP reconverges quickly
        ## from the previous messages — the tol lets it stop early.
        msgs = belief_propagation(msgs, state; maxiter=BP_ITERS, tol=1e-10, timer=to)
        t3 = time()
        t_gate += t2 - t1
        t_bp += t3 - t2
        traj[step+1] = real(expect(state, msgs, HOPOP, BOND))
        if verbose && step % 25 == 0
            bd = maximum(dim(virtualspace(state, e)) for e in ES)
            @printf("χ=%2d  t=%.2f  meas=% .5f  step=%.3fs (gate %.3f / bp %.3f)  D=%d\n",
                χ, step * DT, traj[step+1], t3 - t1, t2 - t1, t3 - t2, bd)
            flush(stdout)
        end
    end
    verbose && @printf("χ=%2d done — gate %.2fs, bp %.2fs total\n", χ, t_gate, t_bp)
    return traj, to
end

md"""
## Exact single-particle reference

The truncation-free reference uses the same circuit:
$C[i,j] = \langle c^\dagger_i c_j \rangle$ evolves as
$C \to \overline{u}\, C\, u^{\mathsf{T}}$ under each gate's single-particle unitary $u$,
applied in the same order as the TN circuit.
"""

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
    i1, i2 = idx[VC], idx[VN]
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

md"""
## Run sweep and plot

Run the exact reference and the BP trajectory at each $\chi$, then plot the central-bond
hopping vs. time alongside the absolute error from the exact curve.
"""

function main()
    println("Real-time free-fermion quench on a $(2 * M * N)-site hexagonal lattice (Canopy.jl)")
    println("dt=$DT  steps=$NSTEPS  (t_final=$T_FINAL)  χ=$(CHIS)  bond=$(VC)–$(VN)")
    println("filling = $(sum(occ_of, VERTS))/$(length(VERTS)) occupied\n")

    times = collect(0:NSTEPS) .* DT
    ex = exact_traj()
    trajs = Dict{Int,Vector{Float64}}()
    tos = Dict{Int,TimerOutput}()
    for χ in CHIS
        trajs[χ], tos[χ] = run_chi(χ)
    end

    println("\nTimerOutputs breakdown (χ=$(last(CHIS))):")
    show(tos[last(CHIS)]; allocations=false, compact=true, sortby=:firstexec)
    println()

    colors = [:silver, :lightblue, :royalblue, :navy]
    fig = Figure(size=(1000, 420))
    ax1 = Axis(fig[1, 1]; xlabel="Time", ylabel="⟨C†ᵢCⱼ + h.c.⟩",
        title="$(2 * M * N)-site hexagonal lattice")
    for (k, χ) in enumerate(CHIS)
        lines!(ax1, times, trajs[χ]; label="BP χ=$χ", color=colors[mod1(k, length(colors))])
    end
    lines!(ax1, times, ex; label="Exact", color=:red, linestyle=:dot, linewidth=2)
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[1, 2]; xlabel="Time", ylabel="Abs. Error", yscale=log10,
        title="BP truncation error")
    for (k, χ) in enumerate(CHIS)
        err = max.(abs.(trajs[χ] .- ex), 1e-16)   # floor for log scale
        lines!(ax2, times, err; label="χ=$χ", color=colors[mod1(k, length(colors))])
    end
    axislegend(ax2; position=:rb)

    outdir = joinpath(@__DIR__, "figs")
    mkpath(outdir)
    outfile = joinpath(outdir, "realtime_hexagonal.png")
    save(outfile, fig)
    println("wrote $outfile")
    return fig
end

main()
