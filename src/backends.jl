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
`Canopy.uses_blocked_kernel(state[v])` is `false` — non-symmetric braiding,
non-`Array` storage, more than one physical leg, and `Trivial` sectors (for which
the pairwise path short-circuits to plain-array TensorOperations and one large
BLAS call, which this cannot beat). Non-abelian fusion is *not* excluded.

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

Requires symmetric braiding (the fermionic-sign derivation in `src/messages.jl`
uses `twist(σ)² = 1`, which is what `SymmetricBraiding` gives — see the note below
on why abelian fusion is *not* required) and plain CPU `Array` storage. `Trivial`
is excluded on purpose — see [`BlockedBackend`](@ref).

**Non-abelian fusion is supported.** The gate used to require `UniqueFusion` as
well; that was unnecessary. Every step of the blocked chain is a `TensorMap`-level
operation that handles a general fusion style on its own — the braids and
relayouts are `tensoradd!` with a permutation (TensorKit routes those through
`GenericTreeTransformer`, which is the correct basis change), the absorptions and
the closing are `mul!` / `adjoint` (composition is block-wise in the *coupled*
sector and never mixes fusion trees), and both twists are TensorKit's own or
provably equal to it (`_twist_message!`). No step ever has to recover all
uncoupled sectors from one coupled label: `twist(σⱼ)` is folded onto message `j`,
where leg `j`'s sector is manifest, and the physical factor onto the ket entry.

There is deliberately **no size threshold**. Interleaved same-fixture A/B on the
degree-3 honeycomb vertex (`benchmark/reports/backend_ab.md`) puts the blocked
kernel ahead at every measured `(symmetry, χ)`, and it never loses, so there is
nothing for a threshold to protect. The current report covers seven symmetries at
χ ∈ {64, 128}: ratios 1.04-2.42 excluding the `Trivial` control, which itself
lands at 0.985-1.024 and is this harness's measurement floor. Earlier runs
extended down to χ = 8 with the same verdict.

The two **non-abelian** rows are the largest ratios in the table — `fz2_su2`
2.42× / 1.90× and `su2` 1.54× / 1.41× at χ = 64 / 128, against `fz2_u1`'s 1.23× at
χ = 128 — because they have the smallest mean subblocks (91-122 entries against
1365), which is the regime this formulation targets, and because the *pairwise*
arm pays the same `GenericTreeTransformer` tax so non-abelian hands nothing back.

Both arguments are types, so this constant-folds at any call site with a concrete
state type.

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
    return I !== Trivial && BraidingStyle(I) isa SymmetricBraiding && A <: Array
end
