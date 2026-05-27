module Canopy

export physicalspace, virtualspace
export reduced_density_matrix
export TensorNetworkState, BPMessages

using Dictionaries
using Graphs: Graphs
using Random: Random

using AlgorithmsInterface: AlgorithmsInterface as AI
using AlgorithmsInterface: StopAfter

using TensorKit
using TensorKit: TupleTools
using MatrixAlgebraKit: diagview, eigh_full, eigh_vals!, qr_compact, svd_trunc, svd_vals!, truncrank, trunctol
using VectorInterface

include("edges.jl")

include("states.jl")
include("messages.jl")
include("utility.jl")

include("expect.jl")

include("operators/abstract_gate.jl")
export AbstractGate

include("operators/local_gate.jl")
export LocalGate, apply!

include("simple_update.jl")

include("operators/composite_gate.jl")
export CompositeGate, Circuit

include("operators/trotterize.jl")
export TrotterScheme, Strang, trotterize, edge_coloring

include("evolve.jl")

export trotter_step!, imaginary_time_evolve!

include("models.jl")


end
