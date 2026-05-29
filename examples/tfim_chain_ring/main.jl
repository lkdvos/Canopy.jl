# Imaginary-time evolution of the 1D transverse-field Ising model
#
#     H = -J Σ_⟨ij⟩ σz_i σz_j - h Σ_i σx_i
#
# on a ring (cycle graph, PBC), via simple-update on top of belief-propagation
# messages. Sweeps h, computes ground-state energy per site E/L and transverse
# magnetization ⟨σx⟩, and compares against the exact Jordan–Wigner solution at
# the same finite L.
#
# Run:  julia --project=examples examples/tfim_chain_ring.jl

using Canopy: TensorNetworkState, BPMessages, belief_propagation,
              UndirectedEdge, LocalGate, apply!, reduced_density_matrix, expect,
              Strang, trotterize, edge_coloring
using TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
using MatrixAlgebraKit: truncrank, trunctol
using Dictionaries
using Graphs: cycle_graph, edges, src, dst, nv
using Random
using Statistics: mean
using Printf
using CairoMakie

# --- Jordan-Wigner reference --------------------------------------------------
# Finite-L, even-parity (ground-state) sector of the TFIM ring at J=1:
# antiperiodic momenta k_n = (2n-1)π/L, single-particle dispersion
# ε(k) = 2√(J² + h² - 2Jh cos k), ground-state energy E₀ = -½ Σ_k ε(k).

_ε(J, h, k) = sqrt(J^2 + h^2 - 2J*h*cos(k))

function jw_energy_per_site(L::Int, h::Real; J::Real=1.0)
    return -(1/L) * sum(_ε(J, h, (2n-1)*π/L) for n in 1:L)
end

function jw_mx_per_site(L::Int, h::Real; J::Real=1.0)
    return (1/L) * sum((h - J*cos((2n-1)*π/L)) / _ε(J, h, (2n-1)*π/L) for n in 1:L)
end

# --- random ring state --------------------------------------------------------

function ring_state(L::Int, Dmax::Int; T::Type=ComplexF64, S::Type=ComplexSpace)
    g = cycle_graph(L)
    @assert nv(g) == L
    pspaces = Dictionary{Int, S}(1:L, [ComplexSpace(2) for _ in 1:L])
    ekeys = [UndirectedEdge(src(e), dst(e)) for e in edges(g)]
    vspaces = Dictionary{UndirectedEdge{Int}, S}(ekeys, [ComplexSpace(Dmax) for _ in ekeys])
    st = TensorNetworkState{T}(undef, pspaces, vspaces)
    Random.randn!(st)
    return st, ekeys
end

# --- TFIM bond operators ------------------------------------------------------
# Bond Hamiltonian distributed across edges as
#   h_e = -J σz⊗σz - (h/deg(u)) σx⊗I - (h/deg(v)) I⊗σx
# so Σ_e h_e = H exactly. For a ring every vertex has degree 2.

function tfim_bond_hamiltonian(J::Real, h::Real, deg_u::Int, deg_v::Int; T::Type=ComplexF64)
    sx, sz = σˣ(T), σᶻ(T)
    iI = id(ComplexSpace(2))
    return -J*(sz ⊗ sz) - (h/deg_u)*(sx ⊗ iI) - (h/deg_v)*(iI ⊗ sx)
end

# --- run a single (L, h, Dmax) point ------------------------------------------

const SCHEDULE = ((0.1, 60), (0.01, 60), (0.001, 40))

function run_one(L::Int, h::Real, Dmax::Int; J::Real=1.0, seed::UInt=hash((L, h, Dmax)))
    Random.seed!(seed)
    T = ComplexF64
    state, ekeys = ring_state(L, Dmax; T)
    msgs = BPMessages(state)
    msgs = belief_propagation(msgs, state; maxiter=300)

    h_e = tfim_bond_hamiltonian(J, h, 2, 2; T)
    bond_hams = Dict(e => h_e for e in ekeys)
    alg = Strang(edge_coloring(keys(bond_hams)))
    circuits = Dict(dτ => trotterize(bond_hams, dτ, alg) for (dτ, _) in SCHEDULE)
    trunc = truncrank(Dmax) & trunctol(; atol=0.0)

    for (dτ, nsteps) in SCHEDULE
        circuit = circuits[dτ]
        for _ in 1:nsteps
            apply!(state, msgs, circuit; trunc)
            msgs = belief_propagation(msgs, state; maxiter=200)
        end
    end

    E = sum(real(expect(state, msgs, h_e, e)) for e in ekeys)
    mx = mean(real(expect(state, msgs, σˣ(T), v)) for v in 1:L)
    return E / L, mx
end

# --- scan + plot --------------------------------------------------------------

function main(; L::Int=16, Dmax::Int=8, hs=range(0.2, 1.8; length=9), J::Real=1.0)
    E_su = similar(hs, Float64); mx_su = similar(hs, Float64)
    E_jw = similar(hs, Float64); mx_jw = similar(hs, Float64)

    println("TFIM ring  L=$L  Dmax=$Dmax  J=$J")
    @printf "  %-6s  %-12s  %-12s  %-12s  %-12s\n" "h" "E/L (SU)" "E/L (JW)" "mx (SU)" "mx (JW)"
    for (i, h) in enumerate(hs)
        E_jw[i]  = jw_energy_per_site(L, h; J)
        mx_jw[i] = jw_mx_per_site(L, h; J)
        E_su[i], mx_su[i] = run_one(L, h, Dmax; J)
        @printf "  %-6.3f  %-12.8f  %-12.8f  %-12.8f  %-12.8f\n" h E_su[i] E_jw[i] mx_su[i] mx_jw[i]
        flush(stdout)
    end

    fig = Figure(size=(720, 720))
    ax1 = Axis(fig[1, 1]; xlabel="h / J", ylabel="E / L",
               title="TFIM ring, L=$L, Dmax=$Dmax")
    lines!(ax1, collect(hs), E_jw; label="Jordan–Wigner (finite L)")
    scatter!(ax1, collect(hs), E_su; label="Simple update + BP", color=:red)
    vlines!(ax1, [1.0]; color=(:gray, 0.5), linestyle=:dash)
    axislegend(ax1; position=:lb)

    ax2 = Axis(fig[2, 1]; xlabel="h / J", ylabel="⟨σx⟩")
    lines!(ax2, collect(hs), mx_jw; label="Jordan–Wigner (finite L)")
    scatter!(ax2, collect(hs), mx_su; label="Simple update + BP", color=:red)
    vlines!(ax2, [1.0]; color=(:gray, 0.5), linestyle=:dash)
    axislegend(ax2; position=:rb)

    outdir = joinpath(@__DIR__, "figs"); mkpath(outdir)
    outfile = joinpath(outdir, "tfim_chain_ring.png")
    save(outfile, fig)
    println("wrote $outfile")
    return fig
end

main()
