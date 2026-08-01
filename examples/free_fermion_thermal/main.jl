using Markdown #hide

md"""
# Free fermions at finite temperature

Imaginary-time evolution of a *density matrix* for spinless free fermions,

```math
H = -t \sum_{\langle ij \rangle} \left( c^\dagger_i c_j + c^\dagger_j c_i \right) - \mu \sum_i n_i,
```

starting from the infinite-temperature state ``\rho(0) = \mathbb{1}`` and sweeping the inverse
temperature ``\beta``. The thermal averages ``\langle n \rangle`` and ``E/L`` are compared
against the exact single-particle (Fermi–Dirac) result at the same finite ``L``, on two
geometries:

- an **open chain**, which is a tree — belief propagation is *exact* there, so any deviation is
  bond truncation or the Trotter step;
- a **ring**, which has one loop — belief propagation is approximate, and the gap between the
  two curves is the loop correction.

This is the finite-temperature companion to [Free fermions on a ring](@ref), which reaches the
ground state by imaginary-time evolution of a *state*.

This example can be run from the command line with:

```
julia --project=examples examples/free_fermion_thermal/main.jl
```
"""

using Canopy: identity_operator, BPMessages, belief_propagation, UndirectedEdge,
              LeftAction, apply!, expect, Strang, trotterize, edge_coloring
using TensorKit
using TensorKitTensors.FermionOperators: f_num, f_hopping, fermion_space
using MatrixAlgebraKit: truncrank, trunctol
using Statistics: mean
using Printf
using CairoMakie

md"""
## The purification picture, and where the factor of two lives

`identity_operator` is ``\beta = 0``. Evolving it **one-sided** (`action = LeftAction`) with
``\exp(-d\tau\,h_e)`` builds

```math
X = e^{-\tau H}, \qquad \tau = n_\text{steps}\, d\tau,
```

and `expect` on a `TensorNetworkOperator` measures against ``X X^\dagger / \operatorname{tr} X
X^\dagger``, which is the thermal ensemble at

```math
\beta = 2\tau .
```

So one one-sided step of size ``d\tau`` advances ``\beta`` by ``2 d\tau``. (A `SandwichAction`
step applies the gate on *both* sides and advances it by ``4 d\tau``; the one-sided route is
used here because it is also the cheaper one — the untouched physical leg stays in the QR
environment.)
"""

md"""
## Exact single-particle reference

Both geometries are quadratic, so ``\langle n \rangle`` and ``E`` follow from the single-particle
spectrum alone:

```math
\langle n \rangle = \frac{1}{L} \sum_k f(\varepsilon_k), \qquad
E = \sum_k \varepsilon_k f(\varepsilon_k), \qquad
f(\varepsilon) = \frac{1}{e^{\beta \varepsilon} + 1},
```

with ``\varepsilon_k = -2t\cos(\pi k/(L+1)) - \mu`` for ``k = 1 \dots L`` on the open chain
(hard-wall modes) and ``\varepsilon_n = -2t\cos(2\pi n/L) - \mu`` for ``n = 0 \dots L-1`` on the
ring.
"""

chain_spectrum(L, μ; t = 1.0) = [-2t * cos(π * k / (L + 1)) - μ for k in 1:L]
ring_spectrum(L, μ; t = 1.0) = [-2t * cos(2π * n / L) - μ for n in 0:(L - 1)]

function exact_thermal(ε, β)
    f = @. 1 / (exp(β * ε) + 1)
    return sum(f) / length(ε), sum(ε .* f) / length(ε)
end

md"""
## Free-fermion bond operators

The same edge decomposition as the ground-state example: the chemical-potential term is split
over the endpoints of each bond by their degree, so that ``\sum_e h_e = H`` exactly.
"""

function bond_hamiltonians(edges, degrees; t = 1.0, μ = 0.0)
    P = fermion_space(Trivial)
    hop = f_hopping(ComplexF64, Trivial)
    nop = f_num(ComplexF64, Trivial)
    I1 = id(P)
    return Dict(
        e => -t * hop - (μ / degrees[first(e)]) * (nop ⊗ I1) -
            (μ / degrees[last(e)]) * (I1 ⊗ nop)
            for e in edges
    )
end

md"""
## Sweeping β

The evolution is incremental: the network is carried forward through the whole `βs` grid and
measured at each point, rather than restarted from ``\mathbb{1}`` for every temperature. BP is
re-converged before each measurement (and periodically during the evolution) so the Bethe
environments are current.
"""

function beta_sweep(edges, degrees, verts, βs; t = 1.0, μ = 0.0, dτ = 0.02, Dmax = 64)
    P = fermion_space(Trivial)
    hams = bond_hamiltonians(edges, degrees; t, μ)
    circuit = trotterize(hams, dτ, Strang(edge_coloring(edges)))
    trunc = truncrank(Dmax) & trunctol(; atol = 1.0e-12)

    ρ = identity_operator(ComplexF64, edges, P)
    msgs = BPMessages(ρ)
    nop = f_num(ComplexF64, Trivial)

    L = length(verts)
    n̄ = Float64[]
    Ē = Float64[]
    τ = 0.0                                   # accumulated one-sided time; β = 2τ
    ϵmax = 0.0                                # worst truncation error over the whole sweep
    nsteps = 0
    for β in βs
        target = β / 2
        while τ + dτ / 2 < target
            _, _, info = apply!(ρ, msgs, circuit; action = LeftAction, trunc)
            ϵmax = max(ϵmax, info.ϵ)
            τ += dτ
            nsteps += 1
            iszero(mod(nsteps, 10)) &&
                (msgs = belief_propagation(msgs, ρ; maxiter = 200, tol = 1.0e-11))
        end
        msgs = belief_propagation(msgs, ρ; maxiter = 400, tol = 1.0e-12)
        push!(n̄, mean(real(expect(ρ, msgs, nop, v)) for v in verts))
        push!(Ē, sum(real(expect(ρ, msgs, h, (first(e), last(e)))) for (e, h) in hams) / L)
    end
    return n̄, Ē, ϵmax
end

md"""
## Geometries
"""

chain_edges(L) = [UndirectedEdge(j, j + 1) for j in 1:(L - 1)]
ring_edges(L) = [UndirectedEdge(min(j, mod1(j + 1, L)), max(j, mod1(j + 1, L))) for j in 1:L]

chain_degrees(L) = Dict(j => (j == 1 || j == L) ? 1 : 2 for j in 1:L)
ring_degrees(L) = Dict(j => 2 for j in 1:L)

md"""
## Run and plot
"""

function main(; L::Int = 8, Dmax::Int = 64, t::Real = 1.0, μ::Real = 0.3,
              βs = 0.0:0.25:4.0, dτ::Real = 0.02)
    βs = collect(βs)
    verts = 1:L

    println("Free fermions at finite temperature  L=$L  Dmax=$Dmax  t=$t  μ=$μ  dτ=$dτ")

    exact_chain = [exact_thermal(chain_spectrum(L, μ; t), β) for β in βs]
    exact_ring = [exact_thermal(ring_spectrum(L, μ; t), β) for β in βs]

    println("  open chain (BP exact on a tree)")
    n_chain, E_chain, ϵ_chain = beta_sweep(chain_edges(L), chain_degrees(L), verts, βs; t, μ, dτ, Dmax)
    println("  ring (one loop: BP approximate)")
    n_ring, E_ring, ϵ_ring = beta_sweep(ring_edges(L), ring_degrees(L), verts, βs; t, μ, dτ, Dmax)
    @printf "  max truncation error: chain %.3e, ring %.3e\n" ϵ_chain ϵ_ring

    @printf "  %-6s  %-12s %-12s  %-12s %-12s\n" "β" "⟨n⟩ chain" "exact" "⟨n⟩ ring" "exact"
    for (i, β) in enumerate(βs)
        @printf "  %-6.2f  %-12.8f %-12.8f  %-12.8f %-12.8f\n" β n_chain[i] exact_chain[i][1] n_ring[i] exact_ring[i][1]
    end

    fig = Figure(size = (760, 760))
    ax1 = Axis(fig[1, 1]; xlabel = "β t", ylabel = "⟨n⟩",
               title = "Free fermions, L=$L, μ=$μ, Dmax=$Dmax")
    lines!(ax1, βs, first.(exact_chain); label = "Exact (open chain)")
    scatter!(ax1, βs, n_chain; label = "TN, open chain", color = :red)
    lines!(ax1, βs, first.(exact_ring); label = "Exact (ring)", linestyle = :dash)
    scatter!(ax1, βs, n_ring; label = "TN, ring", color = :blue, marker = :cross)
    axislegend(ax1; position = :rt)

    ax2 = Axis(fig[2, 1]; xlabel = "β t", ylabel = "E / L")
    lines!(ax2, βs, last.(exact_chain); label = "Exact (open chain)")
    scatter!(ax2, βs, E_chain; label = "TN, open chain", color = :red)
    lines!(ax2, βs, last.(exact_ring); label = "Exact (ring)", linestyle = :dash)
    scatter!(ax2, βs, E_ring; label = "TN, ring", color = :blue, marker = :cross)
    axislegend(ax2; position = :rt)

    outdir = joinpath(@__DIR__, "figs")
    mkpath(outdir)
    outfile = joinpath(outdir, "free_fermion_thermal.svg")
    save(outfile, fig)
    println("wrote $outfile")
    return fig
end

main()
nothing #hide

md"""
![Free fermions at finite temperature](figs/free_fermion_thermal.svg)

The open-chain points sit on the exact curve to plotting accuracy: belief propagation is exact
on a tree, so the only error is the Trotter step and the bond truncation. The ring points track
the exact ring curve but drift off it as ``\beta`` grows and correlations wrap the loop — that
gap is the Bethe (loop) approximation, not a truncation artefact.
"""
