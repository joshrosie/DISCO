#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=backfill_synth_csv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Backfill raw/train.csv for synthetic curation runs produced before the
# MP20Tokens-unification refactor (commit a3d8f53). CPU-only work; reads
# CIFs from structures/ and writes raw/train.csv per root. Existing
# samples.pt is renamed to samples.pre_unified.pt.
#
# Defaults to backfilling the three clean M0-source datasets needed for
# the v2 cascade and curation-ladder ablations. Override with:
#
#   sbatch --export=ALL,ROOTS="data/synthetic/foo:data/synthetic/bar" \
#     scripts/backfill_synthetic_csv.slurm

set -euo pipefail

module purge
module load 2024

export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/atom-reps"
mkdir -p "$HOME/atom-reps/logs/synthetic_aug_msun"

export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-$SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

ROOTS="${ROOTS:-data/synthetic/crystalite_round0_msun_27k:data/synthetic/crystalite_dedup_only_27k:data/synthetic/crystalite_raw_27k}"
IFS=':' read -r -a ROOT_LIST <<< "$ROOTS"

echo "[backfill] roots:"
for r in "${ROOT_LIST[@]}"; do
  echo "[backfill]   $r"
done

srun uv run python scripts/backfill_synthetic_csv.py "${ROOT_LIST[@]}"
