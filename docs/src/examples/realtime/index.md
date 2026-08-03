```@meta
EditURL = "../../../../examples/realtime/main.jl"
```


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

````julia
using Canopy: hexagonal_lattice, product_state, vertices, BPMessages, belief_propagation,
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
````

## Model and run parameters

The defaults reproduce the full `FreeFermionBenchmark.pdf` panel (b) x-axis, evolving to
`T_FINAL = 5.0` (cost ≈ `length(CHIS) · NSTEPS · BP`). For a quicker preview that still shows
BP tracking the exact curve through the first oscillations, lower `T_FINAL` (e.g. `1.5`).

````julia
const M, N = 4, 6              # m × n unit cells → 2·m·n = 48 sites
const T_HOP = -1.0             # hopping coefficient t  (H = -Σ c†c + h.c. + …)
const MU = 1.0                # staggered-field amplitude
const DT = 0.01
const T_FINAL = 5.0
const NSTEPS = round(Int, T_FINAL / DT)
const CHIS = (4, 8, 16, 32)
const BP_ITERS = 30           # per-step BP sweeps (warm-started from the previous step)

const ES = hexagonal_lattice(M, N)                       # open-boundary honeycomb
const VERTS = sort(vertices(ES))
const HOPOP = f_hopping(ComplexF64, Trivial)             # c†_i c_j + c†_j c_i
````

````
2×2←2×2 TensorMap{ComplexF64, Vect[FermionParity], 2, 2, Vector{ComplexF64}}:
 codomain: (Vect[FermionParity](0 => 1, 1 => 1) ⊗ Vect[FermionParity](0 => 1, 1 => 1))
 domain: (Vect[FermionParity](0 => 1, 1 => 1) ⊗ Vect[FermionParity](0 => 1, 1 => 1))
 blocks: 
 * FermionParity(0) => 2×2 reshape(view(::Vector{ComplexF64}, 1:4), 2, 2) with eltype ComplexF64:
 0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im

 * FermionParity(1) => 2×2 reshape(view(::Vector{ComplexF64}, 5:8), 2, 2) with eltype ComplexF64:
 0.0+0.0im  1.0+0.0im
 1.0+0.0im  0.0+0.0im
````

The staggered $\pm\mu$ field alternates on the two sublattices ($s = 1 \to +\mu$,
$s = 2 \to -\mu$), and the initial occupation is empty iff `sum(v) % 4 == 0` (the
`example.jl` rule):

````julia
μ_of(v) = isodd(v[3]) ? MU : -MU
occ_of(v) = (sum(v) % 4 == 0) ? 0 : 1
````

````
occ_of (generic function with 1 method)
````

We measure on the occupied/empty-straddling edge nearest the lattice centre. Coherence
$\langle C^\dagger C \rangle$ only builds on a bond whose two sites start with *different*
occupation, so a bond inside a uniformly-filled region would stay $\approx 0$.

````julia
const _CENTER = ((M + 1) / 2, (N + 1) / 2)
const BOND = argmin(
    e -> sum(abs2, ((e.src[1], e.src[2]) .+ (e.dst[1], e.dst[2])) ./ 2 .- _CENTER),
    filter(e -> occ_of(e.src) != occ_of(e.dst), ES),
)
const VC, VN = BOND.src, BOND.dst
````

````
((2, 3, 1), (3, 3, 2))
````

## Initial product state

````julia
function initial_state()
    # The physical space is uniform, so it is passed as a single value; each local state is a
    # bare occupation sector, which is 1-dimensional in that space.
    ls = Dictionary(VERTS, [fℤ₂(occ_of(v)) for v in VERTS])
    return product_state(ComplexF64, ES, fermion_space(Trivial), ls)
end
````

````
initial_state (generic function with 1 method)
````

## Trotter layers

One `dt` step is single-site, hopping, single-site.

````julia
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
````

````
build_layers (generic function with 1 method)
````

## Single-χ real-time trajectory

````julia
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
        # `apply!` parallelizes its gates with `Threads.@threads`, so we time it from this
        # (sequential) level rather than threading a shared `TimerOutput` into the gate loop.
        state, msgs, _ = @timeit to "single" apply!(state, msgs, single)
        state, msgs, _ = @timeit to "hop" apply!(state, msgs, hop; trunc)
        state, msgs, _ = @timeit to "single" apply!(state, msgs, single)
        t2 = time()
        # each dt step perturbs the state only slightly, so BP reconverges quickly
        # from the previous messages — the tol lets it stop early.
        msgs = @timeit to "bp" belief_propagation(msgs, state; maxiter=BP_ITERS, tol=1e-10)
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
````

````
run_chi (generic function with 1 method)
````

## Exact single-particle reference

The truncation-free reference uses the same circuit:
$C[i,j] = \langle c^\dagger_i c_j \rangle$ evolves as
$C \to \overline{u}\, C\, u^{\mathsf{T}}$ under each gate's single-particle unitary $u$,
applied in the same order as the TN circuit.

````julia
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
````

````
exact_traj (generic function with 1 method)
````

## Run sweep and plot

Run the exact reference and the BP trajectory at each $\chi$, then plot the central-bond
hopping vs. time alongside the absolute error from the exact curve.

````julia
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
    outfile = joinpath(outdir, "realtime_hexagonal.svg")
    save(outfile, fig)
    println("wrote $outfile")
    return fig
end

main()
````

````
Real-time free-fermion quench on a 48-site hexagonal lattice (Canopy.jl)
dt=0.01  steps=500  (t_final=5.0)  χ=(4, 8, 16, 32)  bond=(2, 3, 1)–(3, 3, 2)
filling = 36/48 occupied

χ= 4  t=0.25  meas=-0.20909  step=0.058s (gate 0.042 / bp 0.017)  D=4
χ= 4  t=0.50  meas=-0.46999  step=0.059s (gate 0.039 / bp 0.020)  D=4
χ= 4  t=0.75  meas=-0.33713  step=0.111s (gate 0.085 / bp 0.026)  D=4
χ= 4  t=1.00  meas=-0.04522  step=0.071s (gate 0.040 / bp 0.031)  D=4
χ= 4  t=1.25  meas=-0.03061  step=0.073s (gate 0.041 / bp 0.032)  D=4
χ= 4  t=1.50  meas=-0.25369  step=0.069s (gate 0.039 / bp 0.030)  D=4
χ= 4  t=1.75  meas=-0.32351  step=0.070s (gate 0.040 / bp 0.031)  D=4
χ= 4  t=2.00  meas=-0.14215  step=0.071s (gate 0.040 / bp 0.032)  D=4
χ= 4  t=2.25  meas= 0.00314  step=0.076s (gate 0.040 / bp 0.036)  D=4
χ= 4  t=2.50  meas=-0.04556  step=0.085s (gate 0.041 / bp 0.045)  D=4
χ= 4  t=2.75  meas=-0.13201  step=0.084s (gate 0.042 / bp 0.042)  D=4
χ= 4  t=3.00  meas=-0.09819  step=0.082s (gate 0.040 / bp 0.042)  D=4
χ= 4  t=3.25  meas=-0.01264  step=0.087s (gate 0.040 / bp 0.046)  D=4
χ= 4  t=3.50  meas= 0.00952  step=0.093s (gate 0.040 / bp 0.053)  D=4
χ= 4  t=3.75  meas=-0.01103  step=0.090s (gate 0.039 / bp 0.050)  D=4
χ= 4  t=4.00  meas= 0.00139  step=0.094s (gate 0.041 / bp 0.053)  D=4
χ= 4  t=4.25  meas= 0.04083  step=0.099s (gate 0.040 / bp 0.059)  D=4
χ= 4  t=4.50  meas= 0.05384  step=0.098s (gate 0.041 / bp 0.057)  D=4
χ= 4  t=4.75  meas= 0.03570  step=0.152s (gate 0.043 / bp 0.109)  D=4
χ= 4  t=5.00  meas= 0.02263  step=0.105s (gate 0.041 / bp 0.063)  D=4
χ= 4 done — gate 22.46s, bp 22.82s total
χ= 8  t=0.25  meas=-0.20909  step=0.068s (gate 0.054 / bp 0.014)  D=8
χ= 8  t=0.50  meas=-0.47002  step=0.063s (gate 0.050 / bp 0.012)  D=8
χ= 8  t=0.75  meas=-0.33799  step=0.126s (gate 0.100 / bp 0.026)  D=8
χ= 8  t=1.00  meas=-0.05482  step=0.076s (gate 0.051 / bp 0.025)  D=8
χ= 8  t=1.25  meas=-0.06801  step=0.131s (gate 0.050 / bp 0.081)  D=8
χ= 8  t=1.50  meas=-0.26204  step=0.085s (gate 0.052 / bp 0.033)  D=8
χ= 8  t=1.75  meas=-0.25985  step=0.087s (gate 0.054 / bp 0.033)  D=8
χ= 8  t=2.00  meas=-0.09849  step=0.094s (gate 0.054 / bp 0.040)  D=8
χ= 8  t=2.25  meas=-0.05302  step=0.094s (gate 0.053 / bp 0.041)  D=8
χ= 8  t=2.50  meas=-0.10394  step=0.096s (gate 0.052 / bp 0.044)  D=8
χ= 8  t=2.75  meas=-0.06890  step=0.151s (gate 0.056 / bp 0.096)  D=8
χ= 8  t=3.00  meas= 0.00403  step=0.150s (gate 0.101 / bp 0.050)  D=8
χ= 8  t=3.25  meas=-0.03427  step=0.118s (gate 0.056 / bp 0.061)  D=8
χ= 8  t=3.50  meas=-0.09876  step=0.172s (gate 0.112 / bp 0.060)  D=8
χ= 8  t=3.75  meas=-0.02830  step=0.119s (gate 0.053 / bp 0.066)  D=8
χ= 8  t=4.00  meas= 0.09861  step=0.115s (gate 0.052 / bp 0.064)  D=8
χ= 8  t=4.25  meas= 0.08935  step=0.171s (gate 0.102 / bp 0.070)  D=8
χ= 8  t=4.50  meas=-0.04412  step=0.178s (gate 0.053 / bp 0.125)  D=8
χ= 8  t=4.75  meas=-0.09758  step=0.176s (gate 0.101 / bp 0.076)  D=8
χ= 8  t=5.00  meas= 0.00345  step=0.181s (gate 0.051 / bp 0.130)  D=8
χ= 8 done — gate 30.99s, bp 26.55s total
χ=16  t=0.25  meas=-0.20909  step=0.170s (gate 0.143 / bp 0.027)  D=16
χ=16  t=0.50  meas=-0.47002  step=0.159s (gate 0.144 / bp 0.014)  D=16
χ=16  t=0.75  meas=-0.33799  step=0.137s (gate 0.109 / bp 0.027)  D=16
χ=16  t=1.00  meas=-0.05482  step=0.191s (gate 0.149 / bp 0.042)  D=16
χ=16  t=1.25  meas=-0.06802  step=0.207s (gate 0.152 / bp 0.055)  D=16
χ=16  t=1.50  meas=-0.26226  step=0.206s (gate 0.151 / bp 0.055)  D=16
χ=16  t=1.75  meas=-0.26173  step=0.224s (gate 0.154 / bp 0.070)  D=16
χ=16  t=2.00  meas=-0.10140  step=0.235s (gate 0.152 / bp 0.082)  D=16
χ=16  t=2.25  meas=-0.05055  step=0.235s (gate 0.161 / bp 0.074)  D=16
χ=16  t=2.50  meas=-0.09570  step=0.189s (gate 0.117 / bp 0.072)  D=16
χ=16  t=2.75  meas=-0.07166  step=0.235s (gate 0.154 / bp 0.081)  D=16
χ=16  t=3.00  meas=-0.02282  step=0.246s (gate 0.160 / bp 0.086)  D=16
χ=16  t=3.25  meas=-0.05186  step=0.264s (gate 0.161 / bp 0.103)  D=16
χ=16  t=3.50  meas=-0.06705  step=0.284s (gate 0.169 / bp 0.114)  D=16
χ=16  t=3.75  meas= 0.01376  step=0.278s (gate 0.160 / bp 0.117)  D=16
χ=16  t=4.00  meas= 0.06479  step=0.288s (gate 0.159 / bp 0.129)  D=16
χ=16  t=4.25  meas=-0.00382  step=0.375s (gate 0.170 / bp 0.205)  D=16
χ=16  t=4.50  meas=-0.06190  step=0.306s (gate 0.163 / bp 0.143)  D=16
χ=16  t=4.75  meas= 0.00019  step=0.303s (gate 0.161 / bp 0.142)  D=16
χ=16  t=5.00  meas= 0.06551  step=0.314s (gate 0.160 / bp 0.154)  D=16
χ=16 done — gate 75.60s, bp 43.95s total
χ=32  t=0.25  meas=-0.20909  step=0.558s (gate 0.460 / bp 0.098)  D=32
χ=32  t=0.50  meas=-0.47002  step=0.877s (gate 0.678 / bp 0.200)  D=32
χ=32  t=0.75  meas=-0.33799  step=0.642s (gate 0.547 / bp 0.095)  D=32
χ=32  t=1.00  meas=-0.05482  step=0.713s (gate 0.519 / bp 0.194)  D=32
χ=32  t=1.25  meas=-0.06802  step=0.758s (gate 0.518 / bp 0.240)  D=32
χ=32  t=1.50  meas=-0.26227  step=0.820s (gate 0.538 / bp 0.282)  D=32
χ=32  t=1.75  meas=-0.26172  step=0.836s (gate 0.556 / bp 0.280)  D=32
χ=32  t=2.00  meas=-0.10130  step=0.952s (gate 0.538 / bp 0.414)  D=32
χ=32  t=2.25  meas=-0.05030  step=0.862s (gate 0.494 / bp 0.368)  D=32
χ=32  t=2.50  meas=-0.09579  step=1.120s (gate 0.608 / bp 0.512)  D=32
χ=32  t=2.75  meas=-0.07238  step=1.014s (gate 0.552 / bp 0.462)  D=32
χ=32  t=3.00  meas=-0.02291  step=1.482s (gate 0.926 / bp 0.556)  D=32
χ=32  t=3.25  meas=-0.04890  step=1.216s (gate 0.632 / bp 0.584)  D=32
χ=32  t=3.50  meas=-0.06384  step=1.187s (gate 0.596 / bp 0.591)  D=32
χ=32  t=3.75  meas= 0.00954  step=1.250s (gate 0.606 / bp 0.644)  D=32
χ=32  t=4.00  meas= 0.05584  step=1.327s (gate 0.613 / bp 0.714)  D=32
χ=32  t=4.25  meas=-0.00036  step=1.444s (gate 0.638 / bp 0.805)  D=32
χ=32  t=4.50  meas=-0.04204  step=1.619s (gate 0.838 / bp 0.780)  D=32
χ=32  t=4.75  meas= 0.00267  step=1.519s (gate 0.625 / bp 0.893)  D=32
χ=32  t=5.00  meas= 0.03208  step=1.541s (gate 0.628 / bp 0.912)  D=32
χ=32 done — gate 325.55s, bp 226.74s total

TimerOutputs breakdown (χ=32):
─────────────────────────────────
 Section  ncalls    time    %tot
─────────────────────────────────
 single    1.00k   12.9s    2.3%
 hop         500    313s   56.6%
 bp          500    227s   41.1%
─────────────────────────────────

wrote /mnt/home/ldevos/Projects/Canopy.jl/hubbard/docs/src/examples/realtime/figs/realtime_hexagonal.svg

````

![Real-time evolution of free fermions](figs/realtime_hexagonal.svg)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

