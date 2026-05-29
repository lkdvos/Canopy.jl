# Examples

```@contents
Pages = map(file -> joinpath(file, "index.md"),
            filter(d -> isdir(joinpath(@__DIR__, d)),
                   readdir(@__DIR__)))
Depth = 1
```
