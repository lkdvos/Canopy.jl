using Canopy
using Canopy: DirectedEdge, belief_propagation, check_consistency, tr_distance
using TensorKit
using Graphs
using Random
using Test


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
