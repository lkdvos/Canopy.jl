# Canopy examples

Each example lives in `examples/<name>/main.jl` and is rendered into the
documentation by Literate.jl. This folder is a workspace member of the root
`Project.toml`, so it pins its own visualization / analysis dependencies
(CairoMakie etc.) without bloating the main package.

## Build pipeline

Regenerate the rendered markdown (executes each example; SHA256-cached in
`Cache.toml`, so unchanged examples are skipped on rerun):

```
julia --project=examples -t auto examples/make.jl
```

`-t auto` enables the gate-level parallelism in `apply!` (BLAS is pinned to one thread
inside `make.jl` to avoid oversubscription) — this matters most for the `realtime` example.

Then build the docs site (does not re-execute):

```
julia --project=docs docs/make.jl
# open docs/build/index.html
```

Run a single example interactively:

```
julia --project=examples examples/<name>/main.jl
```

First-time precompilation (CairoMakie in particular) can take several minutes.

## Authoring convention

Examples follow the MPSKit/PEPSKit Literate style so the rendered pages typeset cleanly:

- Start the file with `using Markdown #hide`.
- Write prose in `md"""…"""` blocks (Literate is invoked with `mdstrings=true`). Use real
  markdown headers (`# Title` once, then `## Section`), inline math `$…$`, and fenced
  ` ```math ` blocks for displayed equations.
- Keep code as plain Julia between the `md` blocks — a blank line is enough to separate them.
- **Comments that must stay inside code (e.g. inside a function body) use `##` (double
  hash).** A single-`#` full-line comment is parsed as prose and will split the surrounding
  code chunk — inside a `for`/`function` this orphans the block and breaks the build.

## Available examples

- `free_fermion_ring` — spinless free fermions on a 1D PBC ring via simple-update + BP
- `free_fermion_honeycomb` — spinless free fermions on the honeycomb lattice
- `tfim_chain_ring` — transverse-field Ising model on a ring, compared to the exact Jordan–Wigner finite-`L` result
- `realtime` — real-time quench of spinless free fermions on a 48-site hexagonal lattice in a staggered field, BP + truncation vs. the exact single-particle reference (reproduces `FreeFermionBenchmark.pdf`)
