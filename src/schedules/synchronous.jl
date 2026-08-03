"""
    SynchronousSchedule()

Jacobi-style schedule: every message is recomputed from the previous iterate and
the whole set is swapped in at once. No update within a sweep sees a
freshly-computed neighbor message, so information travels one edge per sweep and a
tree needs as many sweeps as its depth — the default
[`SpanningTreeSchedule`](@ref) is exact on a tree in one. Alone among the
schedules it leaves the input messages untouched, returning a fresh container
instead.
"""
struct SynchronousSchedule <: BPSchedule end

# Synchronous: recompute every message from the previous iterate and swap the
# whole set in at once.
function update_messages!(
        ::SynchronousSchedule, problem::BPProblem, alg::BeliefPropagation, state::BPState
    )
    old = state.iterate
    new_dict = Dictionary{keytype(old.messages), eltype(old)}()
    for edge in keys(old.messages)
        new_msg = recompute_message(old, problem.network, edge, alg.backend, alg.allocator; timer = alg.timer)
        insert!(new_dict, edge, new_msg)
        state.residuals[edge] = tr_distance(old[edge], new_msg; is_hermitian = true)
        @debug "between messages" edge = edge isempty = buffer_isempty(alg.allocator) stats = buffer_stats(alg.allocator)
    end
    state.iterate = BPMessages(new_dict)
    return state
end
