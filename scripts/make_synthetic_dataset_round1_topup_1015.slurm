#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synthetic_r1_topup_1015
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# S1_v2 top-up. Closes the gap between curated S1_v2 (53,261 kept) and the
# matched target |S0| * 2 = 54,276. Generates from M1_v2, dedups against
# MP20 + S0 + existing S1_v2.
#
# Combined with the original S1_v2, the union is 54,276 — matches the
# round-1 doubling target exactly.

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
S1_DIR="${S1_DIR:-data/synthetic/crystalite_round1_msun_54k_v2}"
CHECKPOINT="${CHECKPOINT:-outputs/dng_synthetic_round0_msun_27k_v2/checkpoints/best.pt}"

for path in "$CHECKPOINT" "$S0_DIR/raw/train.csv" "$S1_DIR/raw/train.csv"; do
  if [[ ! -e "$path" ]]; then
    echo "[r1-topup] missing prerequisite: $path" >&2
    exit 2
  fi
done

export CHECKPOINT
export NUM_GENERATE="${NUM_GENERATE:-20000}"
export MAX_SAMPLES="${MAX_SAMPLES:-1015}"
export OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_round1_topup_1015}"
export GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-data/mp20}"
export REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-data/mp20:$S0_DIR:$S1_DIR}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260516}"
export SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-150}"
export SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
export THERMO_BATCH="${THERMO_BATCH:-32}"

echo "[r1-topup] checkpoint=$CHECKPOINT"
echo "[r1-topup] s0=$S0_DIR"
echo "[r1-topup] s1=$S1_DIR"
echo "[r1-topup] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[r1-topup] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[r1-topup] out_dir=$OUT_DIR"

exec bash scripts/make_synthetic_dataset_msun_25k.slurm
