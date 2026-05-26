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
using MatrixAlgebraKit: eigh_vals!, svd_vals!
using VectorInterface

include("edges.jl")

include("states.jl")
include("messages.jl")
include("utility.jl")

include("expect.jl")

include("simple_update.jl")

export apply_gate!

include("evolve.jl")

export edge_coloring, trotter_step!, imaginary_time_evolve!

include("models.jl")


end
