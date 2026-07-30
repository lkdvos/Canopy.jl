# Single-edge and vertex-batched `compute_message` benchmarks.
#
# Times the BP message kernel, parametrised by geometry, symmetry and total bond
# dimension χ. Scaling with χ is the kernel-improvement signal to watch.
#
# `SUITE["message"]["hex_vertex", sym, χ]` is the **decision metric** for the
# `Layout(k)` reformulation and the blocked kernel: a degree-3 (production
# coordination) vertex, swept over every entry of `BENCH_SPACES` at equal total χ.
# Compare `:fz2_u1` against `:z2` / `:fz2` at fixed χ to separate "blocks
# multiplied" from "blocks grew"; `report_structure.jl` is the census that
# explains the ratios.
#
# The pair to read for a *selection threshold* on mean subblock size is
# `:fz2_u1` vs `:fz2_u1_flat` at equal χ. Same symmetry group, same total χ, but
# `:fz2_u1` moves block count and block size together (nsectors 7→13, meanblk
# 1.6→234.7 over the χ grid) while `:fz2_u1_flat` holds the count at 4 (ntrees 16)
# and moves only the size (meanblk 7.5→3840). A τ fitted on `:fz2_u1` alone is
# fitted on a mixed axis.
#
# Every entry here uses `cold_messages`: `compute_message!` never branches on
# tensor values, so warming is pure setup cost. See `setup.jl`.

using Canopy: compute_message, compute_message!, outgoing_edges

# χ = 64 on the honeycomb fixture is ~8 MB per on-site tensor and dominates the
# group; gate it behind `CANOPY_BENCH_FULL=1`.
const MSG_CHIS = BENCH_FULL ? (8, 16, 32, 64) : (8, 16, 32)

for (L, Dmax) in [(8, 4), (16, 8), (32, 16)]
    SUITE["message"]["ring", L, Dmax] = @benchmarkable(
        compute_message(msgs, state, edge),
        setup = (
            Random.seed!($BENCH_SEED);
            state = ring_state($L, $Dmax);
            msgs = cold_messages(state);
            edge = first(keys(msgs.messages))
        ),
    )
end

# Allocating vs in-place at the medium point — `compute_message!` should
# match `compute_message` modulo the output allocation.
SUITE["message"]["ring", 16, 8, :inplace] = @benchmarkable(
    compute_message!(out, msgs, state, edge),
    setup = (
        Random.seed!($BENCH_SEED);
        state = ring_state(16, 8);
        msgs = cold_messages(state);
        edge = first(keys(msgs.messages));
        out = similar(msgs[edge])
    ),
)

# Vertex-centric kernel at a degree-4 interior vertex. `grid([3, 3])` rather than
# `grid([5, 5])`: only the single interior degree-4 vertex is ever used, so the
# larger lattice was 25 sites of setup for one measured contraction.
for Dmax in [8, 16, 32]
    SUITE["message"]["square_vertex", Dmax] = @benchmarkable(
        compute_message!(out, msgs, state, edges),
        setup = (
            Random.seed!($BENCH_SEED);
            state = square_state(3, 3, $Dmax);
            msgs = cold_messages(state);
            v = 5;                           # interior (degree-4) vertex of grid([3, 3])
            edges = collect(outgoing_edges(state, v));
            out = compute_message(msgs, state, edges)
        ),
    )
end

# Vertex-centric kernel at a degree-3 honeycomb vertex, over every symmetry in
# `BENCH_SPACES` at equal total χ. The primary fixture.
for (sym, P, V) in BENCH_SPACES, χ in MSG_CHIS
    SUITE["message"]["hex_vertex", sym, χ] = @benchmarkable(
        compute_message!(out, msgs, state, edges),
        setup = (
            Random.seed!($BENCH_SEED);
            state = hex_state(2, 2, $χ; P = $P, V = $V);
            msgs = cold_messages(state);
            edges = collect(outgoing_edges(state, $HEX_VERTEX));
            out = compute_message(msgs, state, edges)
        ),
    )
end
