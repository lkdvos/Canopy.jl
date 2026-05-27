struct StopWhenStable <: AI.StoppingCriterion
    tol::Float64
end

mutable struct StopWhenStableState <: AI.StoppingCriterionState
    at_iteration::Int
    delta::Float64
end

function AI.initialize_state(::AI.Problem, ::AI.Algorithm, c::StopWhenStable; kwargs...)
    return StopWhenStableState(-1, NaN)
end

function AI.initialize_state!(
        ::AI.Problem, ::AI.Algorithm, stop_when::StopWhenStable, st::StopWhenStableState;
        kwargs...
    )
    st.at_iteration = -1
    st.delta = NaN
    return st
end

function AI.is_finished!(
        ::AI.Problem, ::AI.Algorithm, state::AI.State, c::StopWhenStable, st::StopWhenStableState
    )
    k = state.iteration
    k == 0 && return false

    st.delta = maximum(values(state.residuals))
    if st.delta < c.tol
        st.at_iteration = k
        return true
    end
    return false
end

function AI.is_finished(
        ::AI.Problem, ::AI.Algorithm, state::AI.State, c::StopWhenStable, ::StopWhenStableState
    )
    state.iteration == 0 && return false
    return maximum(values(state.residuals)) < c.tol
end

iterate_difference(previous_iterate, iterate) =
    iterate_difference!(previous_iterate, iterate)

"""
    @maybe_timeit timer name expr

Like `TimerOutputs.@timeit`, but a no-op when `timer === nothing` — lets
callers thread `timer = nothing` (the default) through `apply!` / `belief_propagation`
to disable timing entirely.
"""
macro maybe_timeit(timer, name, expr)
    return quote
        if $(esc(timer)) === nothing
            $(esc(expr))
        else
            @timeit $(esc(timer)) $(esc(name)) $(esc(expr))
        end
    end
end
