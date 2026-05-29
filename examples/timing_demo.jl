# Profiling a simple-update run with TimerOutputs.jl
#
# Mirrors the inner loop of `tfim_chain_ring.jl` on a small ring and threads a
# `TimerOutput` through `belief_propagation` and `apply!`. Prints a nested
# breakdown afterwards: `belief_propagation` → `bp_iteration` → `compute_message`,
# and `apply! Circuit` → `apply! CompositeGate` → `apply! 2-site` → the five
# sub-regions (gauge in / QR / gate+SVD / reconstruct / message update).
#
# Run:  julia --project=examples examples/timing_demo.jl

using Canopy: TensorNetworkState, BPMessages, belief_propagation,
    UndirectedEdge, apply!, Strang, trotterize, edge_coloring
using TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
using MatrixAlgebraKit: truncrank, trunctol
using Dictionaries
using Graphs: cycle_graph, edges, src, dst, nv
using Random
using TimerOutputs

function ring_state(L::Int, Dmax::Int; T::Type=ComplexF64, S::Type=ComplexSpace)
    g = cycle_graph(L)
    pspaces = Dictionary{Int,S}(1:L, [ComplexSpace(2) for _ in 1:L])
    ekeys = [UndirectedEdge(src(e), dst(e)) for e in edges(g)]
    vspaces = Dictionary{UndirectedEdge{Int},S}(ekeys, [ComplexSpace(Dmax) for _ in ekeys])
    st = TensorNetworkState{T}(undef, pspaces, vspaces)
    Random.randn!(st)
    return st, ekeys
end

function tfim_bond_hamiltonian(J::Real, h::Real, deg_u::Int, deg_v::Int; T::Type=ComplexF64)
    sx, sz = σˣ(T), σᶻ(T)
    iI = id(ComplexSpace(2))
    return -J * (sz ⊗ sz) - (h / deg_u) * (sx ⊗ iI) - (h / deg_v) * (iI ⊗ sx)
end

L = 10
Dmax = 20
h = 1.0
J = 1.0
schedule = ((0.1, 40), (0.01, 40))
T = ComplexF64


state, ekeys = ring_state(L, Dmax; T)
msgs = BPMessages(state)

msgs = belief_propagation(msgs, state; maxiter=3000, timer=to);

h_e = tfim_bond_hamiltonian(J, h, 2, 2; T)
bond_hams = Dict(e => h_e for e in ekeys)
alg = Strang(edge_coloring(keys(bond_hams)))
circuits = Dict(dτ => trotterize(bond_hams, dτ, alg) for (dτ, _) in schedule)
trunc = truncrank(Dmax)

# dτ, nsteps = first(schedule)
# circuit = circuits[dτ]
# @profview for _ in 1:nsteps
#     apply!(state, msgs, circuit; trunc, timer=to)
# end

@profview let to = TimerOutput(), msgs = deepcopy(msgs), state = state
    msgs = belief_propagation(msgs, state; maxiter=200, timer=to, tol=1e-8)
    for (dτ, nsteps) in schedule
        circuit = circuits[dτ]
        for i in 1:nsteps
            apply!(state, msgs, circuit; trunc, timer=to)
            normalize!.(state.vertices)
            mod(i, 20) == 0 && (msgs = belief_propagation(msgs, state; maxiter=200, timer=to, tol=1e-8))
        end
    end
    to
end


println("TFIM ring  L=$L  Dmax=$Dmax  h=$h  J=$J")
println("schedule=$schedule")
println()
show(to; allocations=false, compact=true, sortby=:firstexec)
to
println()


