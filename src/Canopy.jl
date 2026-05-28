module Canopy

export physicalspace, virtualspace
export reduced_density_matrix
export TensorNetworkState, BPMessages

using Dictionaries
using Graphs: Graphs
using Graphs: src, dst, nv, ne
import Graphs: vertices, edges, neighbors, degree, has_vertex, has_edge
using Random: Random

using AlgorithmsInterface: AlgorithmsInterface as AI
using AlgorithmsInterface: StopAfter

using TensorKit
using TensorKit: TupleTools
using MatrixAlgebraKit: diagview, eigh_full, eigh_vals!, qr_compact, svd_trunc, svd_vals!, truncrank, trunctol
using TimerOutputs: TimerOutput, @timeit
using VectorInterface

include("edges.jl")

include("states.jl")
include("utility.jl")
include("messages.jl")

include("expect.jl")
export expect

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
