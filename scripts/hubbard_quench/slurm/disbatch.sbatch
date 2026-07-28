#!/bin/bash -l
#
# Run a Hubbard-quench sweep under disBatch.
# =========================================
#
# Submit with the task file as the only argument:
#
#   sbatch -n 8 -t 1:00:00 scripts/hubbard_quench/slurm/disbatch.sbatch path/to/sweep.disbatch
#
# `-n` is the number of concurrent tasks (one core each — Canopy's `apply!` is
# single-threaded, so one process per parameter point is the efficient shape). Pick it from
# the task count: `-n` equal to the number of tasks runs them all at once; fewer just queues
# them. `-t` must cover the *slowest single task*, not the sum.
#
# Optional flags worth adding at submit time:
#   -p ccq              your centre's partition (default is `gen`)
#   -C icelake          pin a microarchitecture so precompilation is reused across runs
#   --mail-type=END
#
# A login shell (`bash -l`) is deliberate: it puts juliaup's `julia` on PATH and makes
# `module` available on the compute node.

#SBATCH --job-name=hubbard-quench
#SBATCH --cpus-per-task=1
#SBATCH --output=%x-%j.log

set -euo pipefail

TASKFILE="${1:?usage: sbatch [slurm opts] disbatch.sbatch <taskfile>}"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f "$TASKFILE" ]] || { echo "no such task file: $TASKFILE" >&2; exit 1; }

# One core per task, and keep BLAS from oversubscribing inside each task.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

module -q load modules/2.4-20250724
module -q load disBatch

echo "host          : $(hostname)"
echo "julia         : $(command -v julia) $(julia --version)"
echo "disBatch      : $(command -v disBatch)"
echo "project       : $PROJECT"
echo "task file     : $TASKFILE  ($(grep -vc '^#' "$TASKFILE") tasks)"
echo "ntasks        : ${SLURM_NTASKS:-?}"
echo

# Precompile ONCE, serially, before fanning out.
#
# This is not optional. Without it, every task starts by precompiling the same ~50 packages
# at the same moment: they contend for the same cache files under ~/.julia/compiled on GPFS,
# waste minutes of allocation each, and can end up racing. It also absorbs the case where the
# compute nodes' CPU differs from the workstation that last precompiled, which silently
# invalidates the cache.
echo "== precompiling (serial warmup) =="
julia --project="$PROJECT" -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
' || { echo "precompilation failed" >&2; exit 1; }
echo "== warmup done =="
echo

# Keep disBatch's own log/status files beside the task file rather than in the submit cwd.
PREFIX="${TASKFILE%.disbatch}_${SLURM_JOB_ID:-local}_"

started=$SECONDS

# `-e` makes disBatch exit non-zero if ANY task failed, so a job that finished with silent
# holes in the sweep shows up as a failed job instead of a green one.
# `--status-header` labels the status file's columns for later reading.
# `|| rc=$?` is required: under `set -e` a bare non-zero exit would kill the script before the
# status line below ever runs.
rc=0
disBatch -e --status-header -p "$PREFIX" "$TASKFILE" || rc=$?

echo
echo "disBatch finished in $((SECONDS - started))s (exit $rc)"
echo "logs and status : ${PREFIX}*"
exit $rc
