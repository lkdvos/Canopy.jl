#!/bin/bash -l
#
# Run a Hubbard-quench sweep under disBatch.
# =========================================
#
# Submit with the task file as the only argument, from the repo root:
#
#   sbatch -n 8 -t 30:00 scripts/hubbard_quench/slurm/disbatch.sbatch \
#          scripts/hubbard_quench/tryout.disbatch
#
# `-n` is the number of concurrent tasks, one core each — Canopy's `apply!` is
# single-threaded, so one process per parameter point is the efficient shape. Note the
# partitions here allocate whole nodes, so a small `-n` still bills a full node: check
# `AllocCPUS` from `sacct` and raise `-n` to fill what you are already paying for.
# `-t` must cover the *slowest single task*, not the sum.
#
# Optional flags worth adding at submit time:
#   -p ccq              your centre's partition (default is `gen`)
#   -C rome|icelake     pin a microarchitecture so precompilation is reused between runs
#   --mail-type=END
#
# Overrides, if the autodetection below guesses wrong:
#   JULIA_BIN=/path/to/julia   sbatch ... disbatch.sbatch <taskfile>
#   HQ_PROJECT=/path/to/scripts/hubbard_quench
#
# `bash -l` is for `module`, which is a shell function that only gets initialised by the login
# profile. It is NOT how Julia is found — see find_julia below.

#SBATCH --job-name=hubbard-quench
#SBATCH --cpus-per-task=1
#SBATCH --output=%x-%j.log

set -uo pipefail

TASKFILE="${1:?usage: sbatch [slurm opts] disbatch.sbatch <taskfile>}"

# sbatch starts us in the submit directory, so a relative task file works.
[[ -f "$TASKFILE" ]] || { echo "no such task file: $TASKFILE" >&2; exit 1; }
TASKFILE="$(readlink -f "$TASKFILE")"

# --- locate Julia --------------------------------------------------------------------------
#
# Do NOT rely on `julia` being on PATH here, even from a login shell. juliaup may be installed
# on a workstation's *local* disk (/home/$USER/.juliaup), which compute nodes cannot see; the
# guard in ~/.bashrc then correctly skips it and `julia` simply does not resolve. The actual
# toolchains live under ~/.julia/juliaup (GPFS) and are visible everywhere.
#
# The site `julia` module is 1.11.x, which is below this project's `julia = "1.12"` compat, so
# it deliberately is not used as a fallback.
find_julia() {
    if [[ -n "${JULIA_BIN:-}" ]]; then
        echo "$JULIA_BIN"; return
    fi
    # Newest 1.12.x from juliaup on shared home. Matching the minor version keeps us off 1.11,
    # which would fail to resolve this project.
    local cand
    cand="$(ls -d "$HOME"/.julia/juliaup/julia-1.12*/bin/julia 2>/dev/null | sort -V | tail -1)"
    if [[ -n "$cand" && -x "$cand" ]]; then
        echo "$cand"; return
    fi
    # Last resort: whatever is on PATH, but only if it actually exists.
    cand="$(command -v julia 2>/dev/null || true)"
    if [[ -n "$cand" && -x "$cand" ]]; then
        echo "$cand"; return
    fi
    echo ""
}

JULIA="$(find_julia)"
if [[ -z "$JULIA" ]]; then
    echo "ERROR: could not find a Julia >= 1.12 visible from this node." >&2
    echo "Looked for \$JULIA_BIN, $HOME/.julia/juliaup/julia-1.12*/bin/julia, then PATH." >&2
    echo "Available juliaup toolchains:" >&2
    ls -d "$HOME"/.julia/juliaup/julia-* 2>/dev/null >&2 || echo "  (none)" >&2
    exit 1
fi

# --- locate the project --------------------------------------------------------------------
#
# `${BASH_SOURCE[0]}` is useless here: sbatch copies the script to /var/spool/slurmd/..., so
# deriving paths from it yields the spool directory. Instead take the project straight out of
# the task file's `#DISBATCH PREFIX` line, which make_sweep.jl wrote as an absolute
# `--project=` path. That also guarantees the warmup precompiles exactly the environment the
# tasks will use.
if [[ -n "${HQ_PROJECT:-}" ]]; then
    PROJECT="$HQ_PROJECT"
else
    PROJECT="$(sed -n 's/^#DISBATCH PREFIX .*--project=\([^ ]*\).*/\1/p' "$TASKFILE" | head -1)"
    [[ -n "$PROJECT" ]] || PROJECT="${SLURM_SUBMIT_DIR:-$PWD}/scripts/hubbard_quench"
fi
if [[ ! -f "$PROJECT/Project.toml" ]]; then
    echo "ERROR: no Project.toml at $PROJECT" >&2
    echo "Set HQ_PROJECT to the scripts/hubbard_quench directory." >&2
    exit 1
fi

# One core per task, and keep BLAS from oversubscribing inside each task.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

module -q load modules/2.4-20250724
module -q load disBatch

echo "host          : $(hostname)"
echo "julia         : $JULIA ($("$JULIA" --version))"
echo "disBatch      : $(command -v disBatch)"
echo "project       : $PROJECT"
echo "task file     : $TASKFILE  ($(grep -vc '^#' "$TASKFILE") tasks)"
echo "ntasks        : ${SLURM_NTASKS:-?}  (alloc cpus: ${SLURM_CPUS_ON_NODE:-?})"
echo

# Precompile ONCE, serially, before fanning out.
#
# Without this every task precompiles the same ~50 packages at the same moment: they contend
# for the same cache files under ~/.julia/compiled on GPFS and waste minutes of allocation
# each. It also absorbs the case where the compute nodes' CPU differs from whatever last
# precompiled, which silently invalidates the cache.
echo "== precompiling (serial warmup) =="
"$JULIA" --project="$PROJECT" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' || {
    echo "precompilation failed" >&2
    exit 1
}
echo "== warmup done =="
echo

# The task file's PREFIX invokes `julia` by bare name, so make the one we resolved win.
export PATH="$(dirname "$JULIA"):$PATH"

# Keep disBatch's own log/status files beside the task file, not in the submit cwd.
PREFIX="${TASKFILE%.disbatch}_${SLURM_JOB_ID:-local}_"

started=$SECONDS

# `-e` makes disBatch exit non-zero if ANY task failed, so a sweep that finished with silent
# holes shows up as a failed job instead of a green one.
# `|| rc=$?` because a bare non-zero exit under `set -e` would skip the reporting below.
rc=0
disBatch -e --status-header -p "$PREFIX" "$TASKFILE" || rc=$?

echo
echo "disBatch finished in $((SECONDS - started))s (exit $rc)"
echo "logs and status : ${PREFIX}*"
exit $rc
