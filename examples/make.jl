if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path = joinpath(@__DIR__, "..")))
    Pkg.resolve()
    Pkg.instantiate()
end

using Canopy
using Literate
using TOML, SHA
using LinearAlgebra: BLAS

# Gate application parallelizes over non-overlapping gates (see `CompositeGate`); the
# per-gate contractions are tiny, so pin BLAS to one thread to avoid oversubscription
# against those workers. Run this script with `julia -t auto` to make it effective.
BLAS.set_num_threads(1)

const EXAMPLES_DIR = @__DIR__
const OUTPUT_DIR = joinpath(@__DIR__, "..", "docs", "src", "examples")
const CACHEFILE = joinpath(@__DIR__, "Cache.toml")

# --- caching ---------------------------------------------------------------

getcache() = isfile(CACHEFILE) ? TOML.parsefile(CACHEFILE) : Dict{String, Any}()

function checksum(name)
    path = joinpath(EXAMPLES_DIR, name, "main.jl")
    @assert isfile(path)
    return open(io -> bytes2hex(sha256(io)), path, "r")
end

iscached(name) = get(getcache(), name, nothing) == checksum(name)

function setcached(name)
    cache = getcache()
    cache[name] = checksum(name)
    return open(f -> TOML.print(f, cache), CACHEFILE, "w")
end

# --- build -----------------------------------------------------------------

function build_example(name)
    src_dir = joinpath(EXAMPLES_DIR, name)
    src_file = joinpath(src_dir, "main.jl")
    out_dir = joinpath(OUTPUT_DIR, name)
    isfile(src_file) || return
    iscached(name) && return
    Literate.markdown(
        src_file, out_dir;
        execute = true, name = "index",
        mdstrings = true, credits = false,
        repo_root_url = "https://github.com/lkdvos/Canopy"
    )
    for f in filter(!=("main.jl"), readdir(src_dir))
        cp(joinpath(src_dir, f), joinpath(out_dir, f); force = true)
    end
    return setcached(name)
end

for name in readdir(EXAMPLES_DIR)
    isdir(joinpath(EXAMPLES_DIR, name)) || continue
    build_example(name)
end
