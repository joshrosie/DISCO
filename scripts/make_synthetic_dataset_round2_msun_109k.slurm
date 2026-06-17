#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synthetic_r2_msun_109k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Round 2 synthetic curation.
#
# Generates S2 from M2, filters with the same MP-hull msun_like rule, and
# deduplicates against MP20 + S0 + S1.
#
# Defaults:
#   M2 train: MP20 + S0 + S1 ~= 108k
#   S2 target:              ~= 108k
#   M3 train: MP20 + S0 + S1 + S2 ~= 217k

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
S1_DIR="${S1_DIR:-data/synthetic/crystalite_round1_msun_54k}"
CHECKPOINT="${CHECKPOINT:-outputs/dng_synthetic_round1_msun_108k/checkpoints/best.pt}"
if [[ ! -f "$CHECKPOINT" ]]; then
  ALT="${ALT_CHECKPOINT:-outputs/dng_synthetic_round1_msun_108k/checkpoints/step_latest.pt}"
  if [[ -f "$ALT" ]]; then
    echo "[round2] CHECKPOINT not found, using ALT_CHECKPOINT=$ALT"
    CHECKPOINT="$ALT"
  else
    echo "[round2] checkpoint not found: $CHECKPOINT" >&2
    exit 2
  fi
fi

for d in "$S0_DIR" "$S1_DIR"; do
  if [[ ! -d "$d" ]]; then
    echo "[round2] synthetic dir not found: $d" >&2
    exit 2
  fi
done

export CHECKPOINT
export NUM_GENERATE="${NUM_GENERATE:-400000}"
export MAX_SAMPLES="${MAX_SAMPLES:-108552}"
export OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_round2_msun_109k}"
export GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-data/mp20}"
export REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-data/mp20:$S0_DIR:$S1_DIR}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260512}"
export SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-150}"
export SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
export THERMO_BATCH="${THERMO_BATCH:-32}"
export WRITE_CIFS="${WRITE_CIFS:-0}"

echo "[round2] checkpoint=$CHECKPOINT"
echo "[round2] s0=$S0_DIR"
echo "[round2] s1=$S1_DIR"
echo "[round2] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[round2] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[round2] out_dir=$OUT_DIR"
echo "[round2] write_cifs=$WRITE_CIFS"

exec bash scripts/make_synthetic_dataset_msun_25k.slurm
