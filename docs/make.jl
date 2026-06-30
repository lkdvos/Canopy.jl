if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.resolve()
    Pkg.instantiate()
end

using Canopy
using Documenter

DocMeta.setdocmeta!(Canopy, :DocTestSetup, :(using Canopy); recursive = true)

example_dir = joinpath(@__DIR__, "src", "examples")
example_pages = if isdir(example_dir)
    [
        joinpath("examples", d, "index.md")
            for d in readdir(example_dir)
            if isfile(joinpath(example_dir, d, "index.md"))
    ]
else
    String[]
end

makedocs(;
    sitename = "Canopy.jl",
    modules = [Canopy],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Design" => "design.md",
        "Examples" => ["examples/index.md"; example_pages],
    ],
    checkdocs = :none,
    doctest = false,
)

deploydocs(; repo = "github.com/lkdvos/Canopy.jl.git", push_preview = true)
