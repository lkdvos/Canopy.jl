"""
    PairwiseBackend(inner = DefaultBackend())

Force the **pairwise** formulation of the vertex-batched BP message kernel
([`compute_message!`](@ref)), with `inner` as the underlying TensorOperations backend.

A testing and benchmarking handle, not a production one: an ordinary backend runs the
`Layout(k)` blocked formulation unconditionally. The pairwise formulation is retained
because it is the oracle the blocked kernel is differentially tested against, and the arm
the A/B in `benchmark/reports/backend_ab.md` measures. Only the vertex-batched message
kernel is specialized; every other kernel simply uses `inner`.

Both formulations, and the four conditions the old selection rule carried before it was
removed, are documented in `docs/src/design.md`.
"""
struct PairwiseBackend{B <: TensorKit.TO.AbstractBackend} <: TensorKit.TO.AbstractBackend
    inner::B
end
PairwiseBackend() = PairwiseBackend(DefaultBackend())

"""
    inner_backend(backend) -> TO.AbstractBackend

The real TensorOperations backend behind a Canopy kernel selector; the identity on anything
else. Called on the first line of every Canopy kernel that forwards its `backend` to
TensorOperations.

[`PairwiseBackend`](@ref) is a *selector*, not a backend: no `TO.tensoradd!` /
`TO.tensorcontract!` method accepts it, deliberately. Forwarding methods would risk
ambiguity with TensorOperations' own `(::AbstractArray, …, ::StridedBackend, …)` methods,
and would let a kernel that forgot to unwrap silently take the wrong path. Unwrapping here
instead means a missed site fails loudly with a `MethodError`.
"""
inner_backend(backend) = backend
inner_backend(backend::PairwiseBackend) = backend.inner
