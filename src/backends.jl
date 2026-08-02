# --- kernel-selection backends ------------------------------------------------
#
# Canopy's contraction kernels take a `backend` argument that is threaded through
# to TensorOperations. The two wrappers below ride on that argument to select
# *which Canopy kernel* runs, and carry the real TO backend as `inner`.
#
# They are **selectors, not backends**: no `TO.tensoradd!` / `TO.tensorcontract!`
# method accepts them, deliberately. Adding forwarding methods would risk
# ambiguity with TensorOperations' own `(::AbstractArray, …, ::StridedBackend, …)`
# methods, and would make a kernel that forgot to unwrap silently take the wrong
# path. Instead every kernel entry point unwraps with [`inner_backend`](@ref) on
# its first line, so a missed site fails loudly with a `MethodError`.

"""
    BlockedBackend(inner = DefaultBackend())

Force the `Layout(k)` formulation of the vertex-batched BP message kernel
([`compute_message!`](@ref)), with `inner` as the underlying TensorOperations
backend.

**This is not normally needed.** An ordinary backend already selects the blocked
kernel wherever [`uses_blocked_kernel`](@ref) holds; the wrapper exists so tests
and benchmarks can pin one kernel against the other on identical inputs.

The blocked kernel keeps the open (target) leg alone in the **domain** at every
step of both chains, so each chain step is a single transposition plus one `gemm`
per coupled sector, and the closing writes the output message in its natural
partition. Fermionic signs are folded into the (tiny) transposed messages and
the ket entry copy rather than paid as `twist!` passes over the chain links.

Falls back to the pairwise kernel whenever
`Canopy.uses_blocked_kernel(state[v])` is `false` — non-abelian or non-symmetric
braiding, non-`Array` storage, and `Trivial` sectors (for which the pairwise path
short-circuits to plain-array TensorOperations and one large BLAS call, which
this cannot beat).

Only the vertex-batched message kernel is specialized; every other kernel simply
uses `inner`. The single-edge `compute_message!` and the residual-driven
schedules that use it are unaffected by either wrapper. See also
[`PairwiseBackend`](@ref).
"""
struct BlockedBackend{B <: TensorKit.TO.AbstractBackend} <: TensorKit.TO.AbstractBackend
    inner::B
end
BlockedBackend() = BlockedBackend(DefaultBackend())

"""
    PairwiseBackend(inner = DefaultBackend())

Force the pairwise BP message kernel, with `inner` as the underlying
TensorOperations backend.

The pairwise kernel is no longer what an ordinary backend runs on symmetric
states — see [`uses_blocked_kernel`](@ref) — so this is the handle for pinning it
anyway: it is the oracle [`BlockedBackend`](@ref) is differentially tested
against, and the two can be run on identical inputs in one process.
"""
struct PairwiseBackend{B <: TensorKit.TO.AbstractBackend} <: TensorKit.TO.AbstractBackend
    inner::B
end
PairwiseBackend() = PairwiseBackend(DefaultBackend())

"""
    inner_backend(backend) -> TO.AbstractBackend

The real TensorOperations backend behind a Canopy kernel selector; the identity
on anything else. Called on the first line of every Canopy kernel that forwards
its `backend` to TensorOperations.
"""
inner_backend(backend) = backend
inner_backend(backend::Union{BlockedBackend, PairwiseBackend}) = backend.inner

"""
    uses_blocked_kernel(t) -> Bool
    uses_blocked_kernel(::Type{S}, ::Type{A}) -> Bool

Whether the blocked message kernel applies to a tensor with space type `S` and
storage type `A`. **This is the selection rule**: the vertex-batched
[`compute_message!`](@ref) runs the blocked kernel exactly when this holds and
the caller has not forced a kernel with [`BlockedBackend`](@ref) /
[`PairwiseBackend`](@ref).

Requires abelian fusion (`UniqueFusion`, so every coupled sector labels exactly
one fusion tree per uncoupled tuple and the twists are `±1`), symmetric braiding,
and plain CPU `Array` storage. `Trivial` is excluded on purpose — see
[`BlockedBackend`](@ref).

There is deliberately **no size threshold**. Interleaved same-fixture A/B on the
degree-3 honeycomb vertex (`benchmark/reports/backend_ab.md`) puts the blocked
kernel ahead at every measured `(symmetry, χ)` over χ ∈ 8…128, by 1.02-2.9×, with
a control ratio of 1.0 ± 0.022; it never loses, so there is nothing for a
threshold to protect. Both arguments are types, so this constant-folds at any
call site with a concrete state type.

The tensor method additionally requires **exactly one physical leg**
(`numout(t) == 1`). The `Layout(k)` chain addresses virtual leg `k` at tensor
slot `k + 1` and folds a single physical leg's twist into the ket entry, so a
[`TensorNetworkOperator`](@ref) site tensor (`numout == 2`) would silently be
contracted on the wrong slots. Those fall back to the pairwise kernel, which is
generic in the codomain arity. Generalizing the blocked chain to `numout > 1` is
follow-up work: `layout`, `dual_phys` and the entry braids all need the physical
block treated as a group rather than as slot 1.
"""
uses_blocked_kernel(t::AbstractTensorMap) =
    numout(t) == 1 && uses_blocked_kernel(spacetype(t), TensorKit.storagetype(t))
function uses_blocked_kernel(::Type{S}, ::Type{A}) where {S, A}
    I = sectortype(S)
    return I !== Trivial && FusionStyle(I) === UniqueFusion() &&
        BraidingStyle(I) isa SymmetricBraiding && A <: Array
end
