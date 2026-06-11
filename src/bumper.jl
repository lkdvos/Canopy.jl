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

# --- pooled allocator for multithreaded gate application ---------------------
#
# `CompositeGate` applies its gates concurrently (see `operators/composite_gate.jl`).
# Each worker needs its own scratch buffer, but a fresh task-local `ResizeBuffer`
# per task is malloc-backed and reclaimed only by a finalizer the GC rarely runs
# (the off-heap bytes are invisible to its heuristics), so under threading they
# accumulate and OOM. A `BufferPool` instead lends a fixed, reused set of
# `ResizeBuffer`s: a worker `acquire!`s one for the lifetime of its task and
# `release!`s (resets) it when done. Because a borrowed buffer is removed from the
# pool while held, no two tasks ever share one — correct under task migration
# without indexing by `threadid`.
struct BufferPool
    free::Vector{Bumper.ResizeBuffer}
    lock::Threads.SpinLock
end
BufferPool() = BufferPool(Bumper.ResizeBuffer[], Threads.SpinLock())

# Take a buffer out of the pool (creating one if the pool is empty); while held it
# is absent from `free`, so no other worker can take it.
acquire!(pool::BufferPool) = lock(pool.lock) do
    isempty(pool.free) ? Bumper.ResizeBuffer() : pop!(pool.free)
end

# Reset the buffer and return it to the pool for reuse.
function release!(pool::BufferPool, buf::Bumper.ResizeBuffer)
    Bumper.reset_buffer!(buf)
    lock(pool.lock) do
        push!(pool.free, buf)
    end
    return nothing
end

# Run `f(buf)` with a buffer borrowed from `pool`, returning it afterwards.
function withbuffer(f, pool::BufferPool)
    buf = acquire!(pool)
    try
        return f(buf)
    finally
        release!(pool, buf)
    end
end

# Task-local default pool, created lazily and reused across all calls in a task —
# not a global pool, and migration-safe (keyed by task, not thread).
const _bufferpool_key = gensym(:bufferpool)
_default_pool() = get!(BufferPool, task_local_storage(), _bufferpool_key)::BufferPool
