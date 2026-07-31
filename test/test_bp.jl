using Canopy
using Canopy: DirectedEdge, belief_propagation, check_consistency, tr_distance
using Canopy: SynchronousSchedule, SpanningTreeSchedule, ResidualSchedule, ResidualSplashSchedule,
              GreedySampler, WeightedSampler
using Canopy: random_bfs_order, recompute_message, update_messages_at!, project_hermitian!
using TensorKit
using TensorKit.TO: DefaultAllocator, DefaultBackend
using TensorKitTensors.FermionOperators: fermion_space
using Graphs
using Graphs: binary_tree, blockdiag, complete_graph, star_graph
using Random
using Test

const AI = Canopy.AI

# Largest trace distance between corresponding messages of two `BPMessages`.
max_msg_distance(a, b) =
    maximum(de -> tr_distance(a[de], b[de]; is_hermitian = true), keys(a.messages))

# The residual of `msgs` as a *fixed point*: recompute every message from the
# current set and compare. This is the quantity `tol` is supposed to bound, and it
# is what `HubbardQuench.bp_residual` reports; it is not the same object as
# `bp_state.residuals`, which measures the change made by the last sweep.
function fixed_point_residual(msgs, state)
    return maximum(keys(msgs.messages)) do de
        fresh = recompute_message(msgs, state, de, DefaultBackend(), DefaultAllocator())
        return tr_distance(msgs[de], fresh; is_hermitian = true)
    end
end

_state_on(g, P, V; seed) = (Random.seed!(seed); randn_state(ComplexF64, g, P, V))

# (physical, virtual) space pairs spanning trivial / U(1) / fermionic symmetry.
# Duplicated from `test/test_messages.jl:23-37` on purpose: every test file runs in
# its own worker process, so sharing would mean a new included file.
const _MSG_SPACES = (
    ("bosonic",   ComplexSpace(2),                        ComplexSpace(3)),
    ("U(1)",      Vect[U1Irrep](0 => 1, 1 => 1),          Vect[U1Irrep](-1 => 1, 0 => 2, 1 => 1)),
    ("fermionic", fermion_space(Trivial),                 Vect[fℤ₂](0 => 2, 1 => 2)),
)

# Geometries spanning coordination numbers 1..5 and dual/non-dual leg mixes.
const _MSG_GEOMETRIES = (
    ("chain L=4",  path_graph(4)),       # degrees 1,2
    ("cycle L=6",  cycle_graph(6)),      # degree 2
    ("star deg 4", star_graph(5)),       # central degree 4
    ("3x3 grid",   grid([3, 3])),        # degrees 2,3,4
    ("K5",         complete_graph(5)),   # degree 4, dense
    ("K6",         complete_graph(6)),   # degree 5
)

# Trees, where the two-pass schedule must be exact in one iteration.
const _TREE_GEOMETRIES = (
    ("path L=6", path_graph(6)),
    ("star deg 4", star_graph(5)),
    ("binary tree h=3", binary_tree(3)),
)

# Two vertices in different connected components, so the BFS order must restart.
const _DISCONNECTED = blockdiag(path_graph(3), cycle_graph(4))

# `(problem, alg, bp_state)` for direct `AI.step!` driving, under `schedule`.
function bp_setup(state; schedule, allocator = DefaultAllocator())
    problem = Canopy.BPProblem(state)
    alg = Canopy.BeliefPropagation(
        AI.StopAfterIteration(1); schedule = schedule, allocator = allocator,
    )
    bp_state = AI.initialize_state(problem, alg; messages = BPMessages(state))
    return problem, alg, bp_state
end


@testset "Identity-initialized messages are consistent" begin
    Random.seed!(0)
    for graph in (path_graph(4), cycle_graph(6))
        state = TensorNetworkState{ComplexF64}(undef, graph, ComplexSpace(2), ComplexSpace(3))
        randn!(state)
        msgs = BPMessages(state)
        @test check_consistency(state, msgs)
    end
end


@testset "BP smoke + idempotence on OBC chain" begin
    Random.seed!(1)
    L = 6
    state = TensorNetworkState{ComplexF64}(undef, path_graph(L), ComplexSpace(2), ComplexSpace(3))
    randn!(state)
    msgs0 = BPMessages(state)
    msgs1 = belief_propagation(msgs0, state; maxiter = L + 2, tol = 1.0e-12)
    @test check_consistency(state, msgs1)
    # One more iteration moves messages by < tol: BP is at its fixed point. The
    # extra sweep must be *synchronous*: the in-place default would hand back the
    # very container it was given, making every distance below trivially zero.
    msgs2 = belief_propagation(msgs1, state; maxiter = 1, schedule = SynchronousSchedule())
    @test msgs2 !== msgs1
    for de in keys(msgs1.messages)
        @test tr_distance(msgs1[de], msgs2[de]; is_hermitian = true) < 1.0e-10
    end
end


@testset "BP smoke + idempotence on even PBC ring" begin
    Random.seed!(2)
    state = TensorNetworkState{ComplexF64}(undef, cycle_graph(10), ComplexSpace(2), ComplexSpace(3))
    randn!(state)
    msgs0 = BPMessages(state)
    msgs1 = belief_propagation(msgs0, state; maxiter = 300, tol = 1.0e-10)
    @test check_consistency(state, msgs1)
    # Synchronous, for the same reason as the OBC chain above.
    msgs2 = belief_propagation(msgs1, state; maxiter = 1, schedule = SynchronousSchedule())
    @test msgs2 !== msgs1
    for de in keys(msgs1.messages)
        @test tr_distance(msgs1[de], msgs2[de]; is_hermitian = true) < 1.0e-8
    end
end


@testset "BP message spaces follow receiver-side convention" begin
    Random.seed!(4)
    state = TensorNetworkState{ComplexF64}(undef, cycle_graph(4), ComplexSpace(2), ComplexSpace(3))
    randn!(state)
    msgs = BPMessages(state)
    for de in keys(msgs.messages)
        V_recv = virtualspace(state, reverse(de))
        @test space(msgs[de]) == (V_recv ← V_recv)
    end
end


@testset "check_consistency rejects mismatched bond spaces" begin
    Random.seed!(5)
    state = TensorNetworkState{ComplexF64}(undef, path_graph(3), ComplexSpace(2), ComplexSpace(3))
    randn!(state)
    msgs = BPMessages(state)
    @test check_consistency(state, msgs)
    bad_de = first(keys(msgs.messages))
    msgs.messages[bad_de] = TensorKit.id(ComplexSpace(99))
    @test !check_consistency(state, msgs)
end


# `SpanningTreeSchedule` drives every vertex from `order` and prunes its inward
# pass with `pos`, so what it needs from `random_bfs_order` is (a) that `order`
# covers the network — including *every* connected component, since a missed
# component would keep its residuals at `Inf` forever — and (b) that every vertex
# other than a component root has an already-visited neighbour, i.e. a BFS parent
# for the inward pass to send to.
@testset "random_bfs_order is a valid BFS order" begin
    Random.seed!(6)
    for graph in (path_graph(6), cycle_graph(8), grid([3, 3]), _DISCONNECTED)
        state = TensorNetworkState{ComplexF64}(undef, graph, ComplexSpace(2), ComplexSpace(3))
        randn!(state)
        for seed in (0, 1, 2)
            order, pos = random_bfs_order(state, MersenneTwister(seed))
            @test sort(collect(order)) == sort(collect(vertices(state)))  # spans every vertex
            @test length(pos) == length(order)                            # no vertex twice
            @test all(i -> pos[order[i]] == i, eachindex(order))           # `pos` inverts `order`

            # Every vertex has a smaller-`pos` neighbour, except the first vertex
            # of each connected component.
            roots = [v for v in order if all(n -> pos[n] > pos[v], neighbors(state, v))]
            ncomp = length(connected_components(graph))
            @test length(roots) == ncomp
            @test first(order) in roots
        end
    end
end


# All schedules solve the same fixed-point equations, so from the identity start
# they should converge to the same messages (up to per-edge scale, which
# `tr_distance` quotients out) as the synchronous default.
@testset "BP schedules agree with the synchronous fixed point" begin
    schedules(ndir, nv) = (
        "spanning_tree" => SpanningTreeSchedule(; rng = MersenneTwister(0)),
        "residual_full" => ResidualSchedule(GreedySampler(ndir)),
        "residual_half" => ResidualSchedule(GreedySampler(cld(ndir, 2))),
        "residual_weighted" => ResidualSchedule(WeightedSampler(ndir; rng = MersenneTwister(0))),
        "residual_splash" => ResidualSplashSchedule(height = nv),
    )
    @testset "$name graph" for (name, graph) in
            ("ring" => cycle_graph(8), "tree" => path_graph(6), "grid" => grid([3, 3]))
        Random.seed!(7)
        state = TensorNetworkState{ComplexF64}(undef, graph, ComplexSpace(2), ComplexSpace(3))
        randn!(state)
        ndir = length(BPMessages(state).messages)
        nv = length(vertices(state))
        ref = belief_propagation(
            BPMessages(state), state; maxiter = 500, tol = 1.0e-12,
            schedule = SynchronousSchedule(),
        )
        @testset "$sname" for (sname, sched) in schedules(ndir, nv)
            msgs = belief_propagation(
                BPMessages(state), state; maxiter = 5000, tol = 1.0e-11, schedule = sched,
            )
            @test check_consistency(state, msgs)
            @test max_msg_distance(msgs, ref) < 1.0e-6
        end
    end
end


# On a tree, a single splash with height ≥ diameter reaches the exact fixed point
# for the root's inbound messages. With all residuals seeded to `Inf`, the root is
# the first vertex in `vertices(state)` order.
@testset "acyclic one-splash convergence" begin
    Random.seed!(8)
    state = TensorNetworkState{ComplexF64}(undef, path_graph(6), ComplexSpace(2), ComplexSpace(3))
    randn!(state)
    ref = belief_propagation(
        BPMessages(state), state; maxiter = 500, tol = 1.0e-12, schedule = SynchronousSchedule(),
    )
    root = first(vertices(state))
    msgs = belief_propagation(
        BPMessages(state), state; maxiter = 1, schedule = ResidualSplashSchedule(height = 6),
    )
    @test check_consistency(state, msgs)
    for k in neighbors(state, root)
        @test tr_distance(msgs[DirectedEdge(k, root)], ref[DirectedEdge(k, root)]; is_hermitian = true) < 1.0e-8
    end
end


# A single internal loop must still converge to the synchronous fixed point.
@testset "single internal loop convergence" begin
    Random.seed!(9)
    state = TensorNetworkState{ComplexF64}(undef, cycle_graph(4), ComplexSpace(2), ComplexSpace(3))
    randn!(state)
    ref = belief_propagation(
        BPMessages(state), state; maxiter = 500, tol = 1.0e-12, schedule = SynchronousSchedule(),
    )
    msgs = belief_propagation(
        BPMessages(state), state; maxiter = 5000, tol = 1.0e-11,
        schedule = ResidualSplashSchedule(height = 4),
    )
    @test check_consistency(state, msgs)
    @test max_msg_distance(msgs, ref) < 1.0e-6
end


# ── SpanningTreeSchedule (the default) ───────────────────────────────────────────

# The two invariants of the two-pass schedule, at their sharpest: on a tree the
# inward pass makes every `child → parent` exact and the outward pass then makes
# *all* of a vertex's outgoing messages exact, so one iteration reaches the fixed
# point from any starting messages. Fails if a pass is walked in the wrong
# direction, or if the batch at a vertex clobbers a sibling message that a later
# visit in the same pass still needs.
@testset "spanning-tree is exact on trees in one step" begin
    for (sname, P, V) in _MSG_SPACES, (gname, g) in _TREE_GEOMETRIES
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "tree")))
            ref = belief_propagation(
                BPMessages(state), state; maxiter = 200, tol = 1.0e-14,
                schedule = SynchronousSchedule(),
            )
            @testset "$start start" for start in ("identity", "perturbed")
                msgs = BPMessages(state)
                if start == "perturbed"
                    Random.seed!(hash((sname, gname, "perturb")))
                    for de in keys(msgs.messages)
                        pert = project_hermitian!(randn!(similar(msgs[de])))
                        msgs.messages[de] = msgs[de] + 0.1 * pert
                    end
                end
                one = belief_propagation(msgs, state; maxiter = 1)
                @test check_consistency(state, one)
                @test max_msg_distance(one, ref) < 1.0e-13
            end
        end
    end
end


# The outward pass writes *every* outgoing edge of *every* vertex, which is what
# makes the old third (cotree) pass unnecessary. Residuals start at `Inf`, so a
# skipped directed edge shows up as a non-finite entry — and would freeze
# `StopWhenStable` at `Inf` forever.
@testset "spanning-tree step updates every directed edge" begin
    for (sname, P, V) in _MSG_SPACES,
            (gname, g) in (("3x3 grid", grid([3, 3])), ("K5", complete_graph(5)),
                           ("disconnected", _DISCONNECTED))
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "cover")))
            problem, alg, bp_state = bp_setup(state; schedule = SpanningTreeSchedule())
            @test all(isinf, values(bp_state.residuals))
            AI.step!(problem, alg, bp_state)
            @test all(isfinite, values(bp_state.residuals))
            @test length(bp_state.residuals) == 2 * length(edges(state))
        end
    end
end


# `residuals[e]` must be the change over the *whole* iteration, not the change
# made by the outward pass alone: `tol` is calibrated against the full-sweep
# quantity the synchronous schedule reports, and a half-step measure would be
# systematically optimistic. Pins the `last_used` start-of-iteration snapshot.
@testset "spanning-tree residuals are the full-step change" begin
    # The full coordination sweep on the bosonic spaces, plus the graded spaces on
    # the geometries whose degrees the testsets above have already specialized the
    # kernel for. (Every new `(sectortype, degree)` pair costs 10-30 s of
    # compilation here, and the property under test does not depend on either.)
    cases = (
        ((sname, P, V), (gname, g)) for (sname, P, V) in _MSG_SPACES
            for (gname, g) in _MSG_GEOMETRIES
                if sname == "bosonic" || gname in ("3x3 grid", "K5", "cycle L=6")
    )
    for ((sname, P, V), (gname, g)) in cases
        @testset "$sname / $gname" begin
            state = _state_on(g, P, V; seed = hash((sname, gname, "res")))
            problem, alg, bp_state = bp_setup(state; schedule = SpanningTreeSchedule())
            AI.step!(problem, alg, bp_state)          # past the identity start
            snap = copy(bp_state.iterate)
            AI.step!(problem, alg, bp_state)
            for de in keys(bp_state.residuals)
                @test bp_state.residuals[de] ≈
                    tr_distance(bp_state.iterate[de], snap[de]; is_hermitian = true) atol =
                    1.0e-14 rtol = 1.0e-10
            end
        end
    end
end


# What `tol` buys the caller: once `max(residuals) < tol`, the messages really are
# a fixed point of the BP equations to comparable accuracy — the quantity
# `HubbardQuench.bp_residual` reports. A last-write-wins (half-step) residual
# would pass `tol` while sitting well above it.
@testset "spanning-tree tol is a true fixed-point residual" begin
    for (gname, g) in (("cycle L=6", cycle_graph(6)), ("3x3 grid", grid([3, 3])))
        @testset "$gname" begin
            state = _state_on(g, ComplexSpace(2), ComplexSpace(3); seed = hash((gname, "tol")))
            msgs = belief_propagation(BPMessages(state), state; maxiter = 500, tol = 1.0e-10)
            @test fixed_point_residual(msgs, state) < 1.0e-8
        end
    end
end


# The default `rng` is fixed-seed, so `belief_propagation` is a pure function of
# its inputs and does not consume the global RNG stream. Anything else makes every
# downstream `Random.seed!`-based draw depend on how much BP ran first.
@testset "default schedule is reproducible and leaves the global RNG alone" begin
    state = _state_on(cycle_graph(6), ComplexSpace(2), ComplexSpace(3); seed = 11)
    a = belief_propagation(BPMessages(state), state; maxiter = 4)
    b = belief_propagation(BPMessages(state), state; maxiter = 4)
    @test a !== b
    for de in keys(a.messages)
        @test a[de] == b[de]        # bit-identical, not just ≈
    end

    Random.seed!(12)
    plain = rand(3)
    Random.seed!(12)
    belief_propagation(BPMessages(state), state; maxiter = 4)
    @test rand(3) == plain

    # Opting into the global stream is still possible, and then it *does* draw.
    Random.seed!(12)
    belief_propagation(
        BPMessages(state), state; maxiter = 4,
        schedule = SpanningTreeSchedule(; rng = Random.default_rng()),
    )
    @test rand(3) != plain
end


# The pruned inward pass drops every write `v → n` with `pos[n] > pos[v]`. Those
# are dead (see the comment in `src/schedules/spanning_tree.jl`), so pruning must
# be *observationally* invisible: same messages and same residuals as the variant
# that recomputes all outgoing edges in both passes, given the same BFS order.
@testset "spanning-tree inward pruning is observationally equivalent" begin
    # The unpruned variant, spelled out: identical outward pass, inward pass over
    # all outgoing edges instead of the smaller-`pos` ones.
    function unpruned_step!(bp_state, state, rng)
        msgs = bp_state.iterate
        order, _ = random_bfs_order(state, rng)
        for v in Iterators.reverse(order)
            update_messages_at!(msgs, state, v, DefaultBackend(), DefaultAllocator())
        end
        for v in order
            update_messages_at!(
                msgs, state, v, DefaultBackend(), DefaultAllocator();
                residuals = bp_state.residuals, snapshot = bp_state.last_used,
            )
        end
        return bp_state
    end

    for (gname, g) in (("3x3 grid", grid([3, 3])), ("K5", complete_graph(5)))
        @testset "$gname" begin
            state = _state_on(g, ComplexSpace(2), ComplexSpace(3); seed = hash((gname, "prune")))
            seed = 4321
            problem, alg, pruned = bp_setup(
                state; schedule = SpanningTreeSchedule(; rng = Xoshiro(seed)),
            )
            _, _, unpruned = bp_setup(state; schedule = SpanningTreeSchedule())
            rng = Xoshiro(seed)                          # same order, drawn in lockstep
            for _ in 1:3
                AI.step!(problem, alg, pruned)
                unpruned_step!(unpruned, state, rng)
            end
            for de in keys(pruned.iterate.messages)
                @test tr_distance(pruned.iterate[de], unpruned.iterate[de]; is_hermitian = true) <
                    1.0e-14
                @test pruned.residuals[de] ≈ unpruned.residuals[de] atol = 1.0e-14 rtol = 1.0e-10
            end
        end
    end
end


# A single-rooted BFS would never visit the second component, leaving its
# residuals at `Inf` so that `tol` can never be met. `random_bfs_order` spans a
# forest, so both components converge.
@testset "spanning-tree converges on a disconnected network" begin
    state = _state_on(_DISCONNECTED, ComplexSpace(2), ComplexSpace(3); seed = 13)
    msgs = belief_propagation(BPMessages(state), state; maxiter = 200, tol = 1.0e-10)
    @test check_consistency(state, msgs)
    @test fixed_point_residual(msgs, state) < 1.0e-8
    ref = belief_propagation(
        BPMessages(state), state; maxiter = 500, tol = 1.0e-12, schedule = SynchronousSchedule(),
    )
    @test max_msg_distance(msgs, ref) < 1.0e-6
end
