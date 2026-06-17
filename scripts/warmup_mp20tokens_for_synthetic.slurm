#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=warmup_mp20t_synth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Force MP20Tokens lazy preprocess on synthetic roots so any CIF-parsing /
# Niggli / tokenization failure surfaces here (~5 min CPU on rome) instead of
# at the start of a 72h GPU training. Builds the processed/ cache so the
# subsequent training job loads instantly.
#
# Defaults to warming the three M0-clean synthetic roots needed by the v2
# cascade (S0) and the curation-ladder ablations (dedup, raw). Override:
#
#   sbatch --export=ALL,ROOTS="data/synthetic/foo:data/synthetic/bar" \
#     scripts/warmup_mp20tokens_for_synthetic.slurm

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

echo "[warmup-slurm] roots:"
for r in "${ROOT_LIST[@]}"; do
  echo "[warmup-slurm]   $r"
done

srun uv run python scripts/warmup_mp20tokens_for_synthetic.py "${ROOT_LIST[@]}"
