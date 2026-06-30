# TODO: remove once tensoroperations  supports this

# Bumper `ResizeBuffer` support for TensorOperations' allocator interface.
#
# TensorOperations' `TensorOperationsBumperExt` wires `SlabBuffer`/`AllocBuffer`
# into `tensoralloc` / `allocator_checkpoint!` / `allocator_reset!`, but predates
# Bumper's `ResizeBuffer` (new in Bumper 0.7.2). We add the same three methods
# here so a `ResizeBuffer` can back the contraction temporaries; without them a
# `ResizeBuffer` would silently fall through to the no-op generic checkpoint.
# Drop this shim once upstream covers `ResizeBuffer`.
function TensorKit.TO.tensoralloc(
        ::Type{A}, structure, ::Val{istemp}, buf::Bumper.ResizeBuffer,
    ) where {A <: AbstractArray, istemp}
    if istemp && ndims(A) > 0
        return Bumper.alloc!(buf, eltype(A), structure...)
    else
        return TensorKit.TO.tensoralloc(A, structure, Val(istemp))
    end
end
TensorKit.TO.allocator_checkpoint!(buf::Bumper.ResizeBuffer) = Bumper.checkpoint_save(buf)
TensorKit.TO.allocator_reset!(::Bumper.ResizeBuffer, cp) = Bumper.checkpoint_restore!(cp)

# Default allocator for the BP and simple-update kernels: a task-local Bumper
# `ResizeBuffer`. Unlike a `SlabBuffer` (whose oversized allocations spill to the
# heap on every call), it warms up to the peak intermediate size and reuses that
# buffer across the repeated contractions, growing dynamically as needed.
_default_allocator() = Bumper.default_buffer(Bumper.ResizeBuffer)

# Pick the allocator matching the storage of a state / messages: the task-local
# Bumper buffer for CPU `Array` storage, and the (heap/device) `DefaultAllocator`
# otherwise — a CPU Bumper buffer cannot back GPU (`CuArray`) temporaries. Dispatch
# is on the storage type only, so this needs no GPU dependency.
default_allocator(x) = default_allocator(TensorKit.storagetype(x))
default_allocator(::Type{<:Array}) = _default_allocator()
default_allocator(::Type{<:DenseVector}) = DefaultAllocator()

# --- buffer introspection (debug only) ---------------------------------------
# Used by the `@debug` memory-churn checks in `apply!` / `belief_propagation`. The
# generic fallbacks make these no-ops for allocators whose internals we don't track
# (e.g. the GPU `DefaultAllocator`), so the call sites stay allocator-agnostic.

# Allocation-state snapshot of a Bumper `ResizeBuffer`, in bytes. `nothing` otherwise.
buffer_stats(buf::Bumper.ResizeBuffer) =
    (; capacity = Int(buf.buf_len), used = Int(buf.offset),
        peak = Int(buf.max_offset), noverflow = length(buf.overflow))
buffer_stats(::Any) = nothing

# A `ResizeBuffer` is "empty" — all temporaries freed — when the bump offset is back to
# zero and no heap overflow blocks are live. Vacuously true for other allocators.
buffer_isempty(buf::Bumper.ResizeBuffer) = iszero(buf.offset) && isempty(buf.overflow)
buffer_isempty(::Any) = true
