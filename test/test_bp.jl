using Canopy
using Canopy: DirectedEdge, belief_propagation, check_consistency, tr_distance
using Canopy: SpanningTreeSchedule, ResidualSchedule, ResidualSplashSchedule, GreedySampler, WeightedSampler
using Canopy: random_spanning_tree
using TensorKit
using Graphs
using Random
using Test

# Largest trace distance between corresponding messages of two `BPMessages`.
max_msg_distance(a, b) =
    maximum(de -> tr_distance(a[de], b[de]; is_hermitian = true), keys(a.messages))


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
    # One more iteration moves messages by < tol: BP is at its fixed point.
    msgs2 = belief_propagation(msgs1, state; maxiter = 1)
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
    msgs2 = belief_propagation(msgs1, state; maxiter = 1)
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


@testset "random_spanning_tree is a valid spanning tree" begin
    Random.seed!(6)
    for graph in (path_graph(6), cycle_graph(8), grid([3, 3]))
        state = TensorNetworkState{ComplexF64}(undef, graph, ComplexSpace(2), ComplexSpace(3))
        randn!(state)
        nv = length(vertices(state))
        ne = length(edges(state))
        order, parent, cotree = random_spanning_tree(state, MersenneTwister(0))
        @test sort(collect(order)) == sort(collect(vertices(state)))  # spans every vertex
        @test length(parent) == nv - 1                                # tree has |V|-1 edges
        @test length(parent) + length(cotree) == ne                   # tree ⊔ cotree = all edges
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
        ref = belief_propagation(BPMessages(state), state; maxiter = 500, tol = 1.0e-12)
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
    ref = belief_propagation(BPMessages(state), state; maxiter = 500, tol = 1.0e-12)
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
    ref = belief_propagation(BPMessages(state), state; maxiter = 500, tol = 1.0e-12)
    msgs = belief_propagation(
        BPMessages(state), state; maxiter = 5000, tol = 1.0e-11,
        schedule = ResidualSplashSchedule(height = 4),
    )
    @test check_consistency(state, msgs)
    @test max_msg_distance(msgs, ref) < 1.0e-6
end
