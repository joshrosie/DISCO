#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synthetic_r1_msun_54k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Round 1 synthetic curation.
#
# Generates S1 from the Round 0 synthetic-augmented model (M1), filters with
# the same MP-hull msun_like rule, and deduplicates against MP20 + S0.
# The default target size doubles the previous training set:
#
#   M1 train: MP20 + S0 ~= 54k
#   S1 target:          ~= 54k
#   M2 train: MP20 + S0 + S1 ~= 108k

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
CHECKPOINT="${CHECKPOINT:-outputs/dng_synthetic_round0_msun_27k/checkpoints/best.pt}"
if [[ ! -f "$CHECKPOINT" ]]; then
  ALT="${ALT_CHECKPOINT:-outputs/dng_synthetic_round0_msun_27k/checkpoints/step_latest.pt}"
  if [[ -f "$ALT" ]]; then
    echo "[round1] CHECKPOINT not found, using ALT_CHECKPOINT=$ALT"
    CHECKPOINT="$ALT"
  else
    echo "[round1] checkpoint not found: $CHECKPOINT" >&2
    exit 2
  fi
fi

if [[ ! -d "$S0_DIR" ]]; then
  echo "[round1] S0 synthetic dir not found: $S0_DIR" >&2
  exit 2
fi

export CHECKPOINT
export NUM_GENERATE="${NUM_GENERATE:-200000}"
export MAX_SAMPLES="${MAX_SAMPLES:-54276}"
export OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_round1_msun_54k}"
export GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-data/mp20}"
export REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-data/mp20:$S0_DIR}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260510}"
export SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-150}"
export SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
export THERMO_BATCH="${THERMO_BATCH:-32}"

echo "[round1] checkpoint=$CHECKPOINT"
echo "[round1] s0=$S0_DIR"
echo "[round1] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[round1] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[round1] out_dir=$OUT_DIR"

exec bash scripts/make_synthetic_dataset_msun_25k.slurm
