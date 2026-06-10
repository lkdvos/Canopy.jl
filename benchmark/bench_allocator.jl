# Allocator comparison benchmarks.
#
# Each kernel runs under both allocators in `BENCH_ALLOCATORS`:
#   :default — plain heap allocation (`TO.DefaultAllocator`)
#   :bumper  — a Bumper bump buffer that reclaims contraction temporaries
#
# This is the explicit before/after for the allocation-reduction work: compare
# the `:default` and `:bumper` keys of each kernel, both for time and for the
# allocation counts BenchmarkTools records automatically (`memory` / `allocs`).
# The win grows with site degree and bond dimension, since those set the size
# and number of intermediate tensors the bump buffer reclaims.
#
# Run just this group and inspect allocations:
#   include("benchmark/benchmarks.jl");
#   res = run(SUITE["allocator"]; verbose = true);
#   display(median(res))   # `memory`/`allocs` columns show the reduction

using Canopy: compute_message, LocalGate, apply!
using TensorKit: id, ⊗
using TensorKit.TO: DefaultBackend

# The center site of a 3×3 grid has degree 4 — the high-degree case where the
# message-absorption chain allocates the most intermediates.
_center_edge(msgs) = first(Iterators.filter(e -> first(e) == 5, keys(msgs.messages)))

SUITE["allocator"]["compute_message"] = BenchmarkGroup()
SUITE["allocator"]["sweep"] = BenchmarkGroup()
SUITE["allocator"]["apply_gate"] = BenchmarkGroup()

# Single-edge BP message kernel at a degree-4 site.
for Dmax in (4, 8, 16)
    for (tag, alloc) in BENCH_ALLOCATORS
        SUITE["allocator"]["compute_message"]["square_3x3_D$Dmax", tag] = @benchmarkable(
            compute_message(msgs, state, edge, DefaultBackend(), $alloc),
            setup = (
                state = square_state(3, 3, $Dmax);
                msgs = warm_messages(state; maxiter = 5);
                edge = _center_edge(msgs)
            ),
        )
    end
end

# One full synchronous BP sweep (`AI.step!` mutates `bp_state`, so `evals = 1`).
for Dmax in (4, 8)
    for (tag, alloc) in BENCH_ALLOCATORS
        SUITE["allocator"]["sweep"]["square_3x3_D$Dmax", tag] = @benchmarkable(
            AI.step!(problem, alg, bp_state),
            setup = (
                (problem, alg, bp_state) = bp_kernel_setup(square_state(3, 3, $Dmax); allocator = $alloc);
                AI.step!(problem, alg, bp_state)
            ),
            evals = 1,
        )
    end
end

# Two-site simple-update gate application (mutates `state`/`msgs`, `evals = 1`).
for Dmax in (4, 8, 16)
    for (tag, alloc) in BENCH_ALLOCATORS
        SUITE["allocator"]["apply_gate"]["square_3x3_D$Dmax", tag] = @benchmarkable(
            apply!(state, msgs, gate; allocator = $alloc),
            setup = (
                base = square_state(3, 3, $Dmax);
                bmsgs = warm_messages(base; maxiter = 5);
                gate = LocalGate((5, 6), id(physicalspace(base, 5) ⊗ physicalspace(base, 5)));
                state = deepcopy(base);
                msgs = deepcopy(bmsgs)
            ),
            evals = 1,
        )
    end
end
