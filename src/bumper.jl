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
