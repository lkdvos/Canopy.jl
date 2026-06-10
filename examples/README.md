# Canopy examples

Each example lives in `examples/<name>/main.jl` and is rendered into the
documentation by Literate.jl. This folder is a workspace member of the root
`Project.toml`, so it pins its own visualization / analysis dependencies
(CairoMakie etc.) without bloating the main package.

## Build pipeline

Regenerate the rendered markdown (executes each example; SHA256-cached in
`Cache.toml`, so unchanged examples are skipped on rerun):

```
julia --project=examples examples/make.jl
```

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

## Available examples

- `free_fermion_ring` — spinless free fermions on a 1D PBC ring via simple-update + BP
- `free_fermion_honeycomb` — spinless free fermions on the honeycomb lattice
- `tfim_chain_ring` — transverse-field Ising model on a ring, compared to the exact Jordan–Wigner finite-`L` result
- `realtime` — real-time quench of spinless free fermions on a 48-site hexagonal lattice in a staggered field, BP + truncation vs. the exact single-particle reference (reproduces `FreeFermionBenchmark.pdf`)
