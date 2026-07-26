#!/usr/bin/env julia
#
# Per-step timing trace of the real-time free-fermion quench (BP + bond truncation).
#
# Times each phase of every Trotter step (single → hop → single → BP reconverge) and writes
# one wide CSV per run (all χ stacked) to `<outdir>/<prefix>_<model>.csv`. Mirrors the circuit
# of `examples/realtime/main.jl`, but runs BP for a *fixed* `--bp-iters` sweeps (`tol=0`) so the
# work per step is constant and comparable across libraries.
#
#   ./run_timings.jl --prefix canopy --outdir data --model tfim-z2 --nsteps 15 --chi 4 8 16 32 64
#   julia --project=benchmark/realtime_timing benchmark/realtime_timing/run_timings.jl --help
#
# Set JULIA_NUM_THREADS to a fixed value — `apply!` threads over gates and the thread count is
# recorded in every CSV.

using Pkg
Pkg.activate(@__DIR__; io=devnull)

using ArgParse
using Canopy: hexagonal_lattice, product_state, vertices, BPMessages, belief_propagation,
    LocalGate, CompositeGate, Circuit, apply!, edge_coloring, virtualspace
using TensorKit
using TensorKitTensors.FermionOperators: f_num, f_hopping, fermion_space
using TensorKitTensors.SpinOperators: σˣ, σᶻ, S_z_S_z, spin_space
using MatrixAlgebraKit: truncrank
using Dictionaries
using DataFrames
using CSV
using Printf

using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

function parse_cli(args)
    s = ArgParseSettings(description="Per-step timing trace of the real-time free-fermion quench.")
    @add_arg_table! s begin
        "--prefix"
        help = "output filename prefix; CSVs go to <outdir>/<prefix>_chi<χ>.csv"
        default = "canopy"
        "--outdir"
        help = "directory for output CSVs"
        default = joinpath(@__DIR__, "data")
        "--nsteps"
        help = "number of Trotter steps to time"
        arg_type = Int
        default = 15
        "--chi"
        help = "bond dimensions to sweep (all stacked into one CSV)"
        arg_type = Int
        nargs = '+'
        default = [4, 8, 16, 32, 64]
        "--bp-iters"
        help = "fixed BP sweeps per step (run with tol=0)"
        arg_type = Int
        default = 30
        "--model"
        help = "physics model: free-fermion (default, fℤ₂ parity), free-fermion-u1 (parity + U(1) particle number), tfim (staggered transverse-field Ising), or tfim-z2 (the same TFIM with spin-flip Z2 symmetry)"
        default = "free-fermion"
        range_tester = m -> m in ("free-fermion", "free-fermion-u1", "tfim", "tfim-z2")
    end
    return parse_args(args, s)
end

const M, N = 4, 6              # m × n unit cells → 2·m·n = 48 sites
const T_HOP = -1.0
const MU = 1.0
const J_ISING = 1.0
const H_FIELD = 1.0
const DT = 0.01

const ES = hexagonal_lattice(M, N)
const VERTS = sort(vertices(ES))
const NSITES = length(VERTS)

μ_of(v) = isodd(v[3]) ? MU : -MU
occ_of(v) = (sum(v) % 4 == 0) ? 0 : 1
h_of(v) = isodd(v[3]) ? H_FIELD : -H_FIELD   # staggered transverse field, by sublattice
sublattice_A(v) = isodd(v[3])                 # σˣ-eigenstate pattern: A → |+⟩, B → |−⟩

function initial_state(model)
    if model == "tfim" || model == "tfim-z2"
        # σˣ eigenstates: |+⟩ (σˣ=+1) = Z2 charge 0, |−⟩ (σˣ=−1) = Z2 charge 1. The |+⟩/|−⟩ Néel
        # pattern puts equal counts of each charge on the bipartite lattice, so the total Z2 charge
        # is trivial — representable under Z2 with no charge-bath site. Both variants use the *same*
        # physical state, isolating the symmetry-bookkeeping cost (cf. free-fermion trivial vs U(1)).
        if model == "tfim-z2"
            P = spin_space(Z2Irrep)
            ps = Dictionary(VERTS, fill(P, length(VERTS)))
            ls = Dictionary(VERTS, [Z2Irrep(sublattice_A(v) ? 0 : 1) => [1.0] for v in VERTS])
        else
            P = ComplexSpace(2)
            plus, minus = ComplexF64[1, 1] / sqrt(2), ComplexF64[1, -1] / sqrt(2)
            ps = Dictionary(VERTS, fill(P, length(VERTS)))
            ls = Dictionary(VERTS, [Trivial() => (sublattice_A(v) ? plus : minus) for v in VERTS])
        end
        return product_state(ComplexF64, ES, ps, ls)
    elseif model == "free-fermion-u1"
        # The lattice carries total charge fℤ₂(Q mod 2) ⊠ U1Irrep(Q) with Q = Σ occ_of(v), which
        # `total_charge` realizes by anchoring a single charge-bath site carrying the compensating
        # charge. It stays idle (no gate, 1-dim bond) since the gate layers only range over VERTS.
        Q = sum(occ_of(v) for v in VERTS)
        P = fermion_space(U1Irrep)
        ps = Dictionary(VERTS, fill(P, length(VERTS)))
        ls = Dictionary(VERTS, [(fℤ₂(occ_of(v)) ⊠ U1Irrep(occ_of(v))) => [1.0] for v in VERTS])
        return product_state(
            ComplexF64, ES, ps, ls; total_charge = fℤ₂(mod(Q, 2)) ⊠ U1Irrep(Q)
        )
    end
    P = fermion_space(Trivial)
    ps = Dictionary(VERTS, fill(P, length(VERTS)))
    ls = Dictionary(VERTS, [fℤ₂(occ_of(v)) => [1.0] for v in VERTS])
    return product_state(ComplexF64, ES, ps, ls)
end

# Returns (single-site layer, two-site layer); the two-site layer is the `hop` phase in the CSV
# (free-fermion hopping, or the σᶻσᶻ Ising coupling for the TFIM).
function build_layers(model)
    if model == "tfim" || model == "tfim-z2"
        if model == "tfim-z2"
            sx = σˣ(ComplexF64, Z2Irrep)
            zz = 4 * S_z_S_z(ComplexF64, Z2Irrep)   # σᶻ⊗σᶻ = (2 S_z)⊗(2 S_z)
        else
            sx = σˣ(ComplexF64)
            zz = σᶻ(ComplexF64) ⊗ σᶻ(ComplexF64)
        end
        g_plus = exp(-im * H_FIELD * DT * sx)
        g_minus = exp(-im * (-H_FIELD) * DT * sx)
        g_zz = exp(-im * (-J_ISING * DT) * zz)
        single = CompositeGate([LocalGate((v,), h_of(v) > 0 ? g_plus : g_minus) for v in VERTS])
        two = Circuit(
            [
            CompositeGate([LocalGate((e.src, e.dst), g_zz) for e in class])
            for class in edge_coloring(ES)
        ]
        )
        return single, two
    end
    sym = model == "free-fermion-u1" ? U1Irrep : Trivial
    n = f_num(ComplexF64, sym)
    g_plus = exp(-im * MU * DT * n)
    g_minus = exp(-im * (-MU) * DT * n)
    g_hop = exp(-im * (T_HOP * DT) * f_hopping(ComplexF64, sym))
    single = CompositeGate([LocalGate((v,), μ_of(v) > 0 ? g_plus : g_minus) for v in VERTS])
    hop = Circuit(
        [
        CompositeGate([LocalGate((e.src, e.dst), g_hop) for e in class])
        for class in edge_coloring(ES)
    ]
    )
    return single, hop
end

maxdim(state) = maximum(dim(virtualspace(state, e)) for e in ES)

# One timed trajectory. Returns a wide DataFrame; row `step=0` is the initial BP convergence
# (gate columns 0.0), rows `step ≥ 1` are the Trotter steps. No warmup: step 1 carries Julia's
# JIT-compilation cost, which is itself informative.
function run_chi_timed(χ, nsteps, bp_iters, model, library)
    state = initial_state(model)
    msgs = BPMessages(state)
    single, hop = build_layers(model)
    trunc = truncrank(χ)
    nt = Threads.nthreads()

    # `@timed` records both wall-clock and *heap* bytes per phase. Contraction temporaries are
    # bump-allocated off-heap (Bumper `ResizeBuffer`), so `*_bytes` capture the persistent
    # allocation churn that reaches the GC, not the full temporary working set.
    rows = NamedTuple[]
    r = @timed belief_propagation(msgs, state; maxiter=bp_iters, tol=0)
    msgs = r.value
    push!(
        rows, (;
            library=library, model=model, chi=χ, nthreads=nt, nsites=NSITES, dt=DT, step=0,
            single1=0.0, hop=0.0, single2=0.0, bp=r.time,
            single1_bytes=0, hop_bytes=0, single2_bytes=0, bp_bytes=r.bytes,
            maxdim=maxdim(state),
        )
    )

    for step in 1:nsteps
        r1 = @timed apply!(state, msgs, single)
        state, msgs, _ = r1.value
        rh = @timed apply!(state, msgs, hop; trunc)
        state, msgs, _ = rh.value
        r2 = @timed apply!(state, msgs, single)
        state, msgs, _ = r2.value
        rb = @timed belief_propagation(msgs, state; maxiter=bp_iters, tol=0)
        msgs = rb.value
        push!(
            rows, (;
                library=library, model=model, chi=χ, nthreads=nt, nsites=NSITES, dt=DT, step=step,
                single1=r1.time, hop=rh.time, single2=r2.time, bp=rb.time,
                single1_bytes=r1.bytes, hop_bytes=rh.bytes, single2_bytes=r2.bytes, bp_bytes=rb.bytes,
                maxdim=maxdim(state),
            )
        )
    end
    return DataFrame(rows)
end

function (@main)(args)
    opts = parse_cli(args)
    prefix, outdir = opts["prefix"], opts["outdir"]
    nsteps, chis, bp_iters, model = opts["nsteps"], opts["chi"], opts["bp-iters"], opts["model"]
    outfile = joinpath(outdir, "$(prefix)_$(replace(model, "-" => "_")).csv")

    @printf("Real-time timing on a %d-site hexagonal lattice (Canopy.jl)\n", NSITES)
    @printf(
        "model=%s  prefix=%s  dt=%.3g  nsteps=%d  bp_iters=%d  χ=%s  threads=%d\n\n",
        model, prefix, DT, nsteps, bp_iters, chis, Threads.nthreads()
    )

    mkpath(outdir)
    # Write each χ as it completes (append after the first) so a crash at a large χ — the heaviest
    # ones can OOM — keeps the smaller-χ data already on disk rather than losing the whole run.
    for (i, χ) in enumerate(chis)
        df = run_chi_timed(χ, nsteps, bp_iters, model, prefix)
        CSV.write(outfile, df; append=i > 1, writeheader=i == 1)
        loop = df[df.step.>=1, :]
        med = sort(loop.single1 .+ loop.hop .+ loop.single2 .+ loop.bp)[cld(nrow(loop), 2)]
        medb = sort(loop.single1_bytes .+ loop.hop_bytes .+ loop.single2_bytes .+ loop.bp_bytes)[cld(nrow(loop), 2)]
        @printf(
            "χ=%2d  median step %.4fs  (bp %.4fs)  heap %.1f MiB/step  maxdim=%d\n",
            χ, med, sort(loop.bp)[cld(nrow(loop), 2)], medb / 2^20, maximum(loop.maxdim)
        )
        flush(stdout)
    end
    @printf("\nwrote %d steps × %d χ → %s\n", nsteps, length(chis), basename(outfile))
    return 0
end
