# Single-edge `compute_message` benchmarks.
#
# Times the BP message kernel on a single directed edge, parametrised by
# ring length `L` and virtual bond dimension `Dmax`. Scaling with `Dmax`
# is the kernel-improvement signal to watch.

using Canopy: compute_message, compute_message!

for (L, Dmax) in [(8, 4), (16, 8), (32, 16)]
    SUITE["message"]["ring", L, Dmax] = @benchmarkable(
        compute_message(msgs, state, edge),
        setup = (
            state = ring_state($L, $Dmax);
            msgs = warm_messages(state);
            edge = first(keys(msgs.messages))
        ),
    )
end

# Allocating vs in-place at the medium point — `compute_message!` should
# match `compute_message` modulo the output allocation.
SUITE["message"]["ring", 16, 8, :inplace] = @benchmarkable(
    compute_message!(out, msgs, state, edge),
    setup = (
        state = ring_state(16, 8);
        msgs = warm_messages(state);
        edge = first(keys(msgs.messages));
        out = similar(msgs[edge])
    ),
)
