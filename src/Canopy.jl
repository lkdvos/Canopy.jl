module Canopy

export physicalspace, virtualspace
export reduced_density_matrix
export TensorNetworkState, BPMessages
export BPSchedule, SynchronousSchedule, SpanningTreeSchedule, ResidualSchedule, ResidualSplashSchedule
export BPSampler, GreedySampler, WeightedSampler

using Dictionaries
using Graphs: Graphs
using Graphs: src, dst, nv, ne
import Graphs: vertices, edges, neighbors, degree, has_vertex, has_edge
using Random: Random

using AlgorithmsInterface: AlgorithmsInterface as AI
using AlgorithmsInterface: StopAfter

using TensorKit
using TensorKit: TupleTools
using TensorKit.TO: tensorcontract, tensorcontract!, tensoralloc, tensoralloc_contract, tensorfree!, promote_contract
using TensorKit.TO: tensoradd!, tensoralloc_add
using TensorKit.TO: DefaultBackend, DefaultAllocator, allocator_checkpoint!, allocator_reset!
using MatrixAlgebraKit: diagview, eigh_full, eigh_vals!, qr_compact, svd_trunc, svd_vals!, truncrank, trunctol
using TimerOutputs: TimerOutput, @timeit
using VectorInterface
using Bumper: Bumper
using Adapt: Adapt, adapt

using TensorKitTensors: QuantumGates as QG
using OpenQASM: OpenQASM
using OpenQASM.RBNF: Token
using OpenQASM.Types: RegDecl, Include, Gate, Instruction, UGate, CXGate, Barrier,
    Measure, Reset, IfStmt, Opaque, Bit, Call, Neg

include("bumper.jl")
include("backends.jl")
export PairwiseBackend

include("edges.jl")

include("states/charges.jl")
include("states/abstract_tensornetwork.jl")
include("states/tensornetworkstate.jl")
include("states/lattices.jl")
include("states/productstate.jl")
include("states/tensornetworkoperator.jl")
export AbstractTensorNetwork, TensorNetworkOperator
export square_lattice, triangular_lattice, hexagonal_lattice
export product_state, randn_state, rand_state
export identity_operator, randn_operator, rand_operator
export vertices

include("utility.jl")
include("messages.jl")
include("beliefpropagation.jl")
include("schedules/synchronous.jl")
include("schedules/spanning_tree.jl")
include("schedules/residual.jl")
include("schedules/residual_splash.jl")

include("expect.jl")
export expect

include("operators/abstract_gate.jl")
export AbstractGate

include("operators/local_gate.jl")
export LocalGate, apply!

include("operators/composite_gate.jl")
export CompositeGate, Circuit

# `simple_update.jl`'s kernel dispatches on the gate action, so the enum comes first
include("operators/gate_action.jl")
export GateAction, LeftAction, RightAction, SandwichAction

include("simple_update.jl")

include("operators/trotterize.jl")
export TrotterScheme, Strang, trotterize, edge_coloring

include("qasm/ir.jl")
include("qasm/parse.jl")
include("qasm/lowering.jl")
export QASMCircuit, QASMStatement, QASMGate, read_qasm, parse_qasm, qasm_lattice

end
