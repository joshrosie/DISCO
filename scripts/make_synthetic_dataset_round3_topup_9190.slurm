#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synthetic_r3_topup_9190
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# S3 top-up. Closes the gap between curated S3 (207,914 kept) and the
# matched target |S2| * 2 = 217,104. Generates from M3, dedups against
# MP20 + S0 + S1_v2_full + S2_v2 + existing S3.

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
S1_DIR="${S1_DIR:-data/synthetic/crystalite_round1_msun_54k_v2_full}"
S2_DIR="${S2_DIR:-data/synthetic/crystalite_round2_msun_108k_v2}"
S3_DIR="${S3_DIR:-data/synthetic/crystalite_round3_msun_217k_v2}"
CHECKPOINT="${CHECKPOINT:-outputs/dng_synthetic_round2_msun_217k_v2/checkpoints/best.pt}"

for path in "$CHECKPOINT" "$S0_DIR/raw/train.csv" "$S1_DIR/raw/train.csv" \
            "$S2_DIR/raw/train.csv" "$S3_DIR/raw/train.csv"; do
  if [[ ! -e "$path" ]]; then
    echo "[r3-topup] missing prerequisite: $path" >&2
    exit 2
  fi
done

export CHECKPOINT
# Round 3 main run hit ~27% effective yield. Budget ~4.4× MAX to leave margin.
export NUM_GENERATE="${NUM_GENERATE:-40000}"
export MAX_SAMPLES="${MAX_SAMPLES:-9190}"
export OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_round3_topup_9190}"
export GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-data/mp20}"
export REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-data/mp20:$S0_DIR:$S1_DIR:$S2_DIR:$S3_DIR}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260608}"
export SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-150}"
export SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
export THERMO_BATCH="${THERMO_BATCH:-32}"

echo "[r3-topup] checkpoint=$CHECKPOINT"
echo "[r3-topup] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[r3-topup] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[r3-topup] out_dir=$OUT_DIR"

exec bash scripts/make_synthetic_dataset_msun_25k.slurm
