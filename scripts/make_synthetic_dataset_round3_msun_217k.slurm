#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synthetic_r3_msun_217k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Round 3 synthetic curation — continues the v2 cascade from M3.
#
#   M3 train:  MP20 + S0 + S1_v2_full + S2_v2  ~= 217k
#   S3 target:                                  ~= 217k   (doubles again)
#   M4 train:  MP20 + S0 + S1_v2_full + S2_v2 + S3 ~= 434k
#
# Inherits all v2 curator patches via src/data/synthetic_augmentation.py.

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
S1_DIR="${S1_DIR:-data/synthetic/crystalite_round1_msun_54k_v2_full}"
S2_DIR="${S2_DIR:-data/synthetic/crystalite_round2_msun_108k_v2}"
CHECKPOINT="${CHECKPOINT:-outputs/dng_synthetic_round2_msun_217k_v2/checkpoints/best.pt}"

if [[ ! -f "$CHECKPOINT" ]]; then
  ALT="${ALT_CHECKPOINT:-outputs/dng_synthetic_round2_msun_217k_v2/checkpoints/step_latest.pt}"
  if [[ -f "$ALT" ]]; then
    echo "[round3] CHECKPOINT not found, using ALT_CHECKPOINT=$ALT"
    CHECKPOINT="$ALT"
  else
    echo "[round3] checkpoint not found: $CHECKPOINT" >&2
    exit 2
  fi
fi

for path in "$S0_DIR/raw/train.csv" "$S1_DIR/raw/train.csv" "$S2_DIR/raw/train.csv"; do
  if [[ ! -e "$path" ]]; then
    echo "[round3] missing prerequisite: $path" >&2
    exit 2
  fi
done

export CHECKPOINT
# Round 2 used NUM_GENERATE=500000 for MAX_SAMPLES=108552 (~22% effective yield
# after dedup vs ~190k synth ref). Round 3's dedup ref is ~325k synth (S0 + S1
# + S2) and we want 217,104 accepted; budget ~5× MAX. If undershoot, fire a
# topup slurm modeled on scripts/make_synthetic_dataset_round1_topup_1015.slurm.
export NUM_GENERATE="${NUM_GENERATE:-1100000}"
export MAX_SAMPLES="${MAX_SAMPLES:-217104}"
export OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_round3_msun_217k_v2}"
export GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-data/mp20}"
export REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-data/mp20:$S0_DIR:$S1_DIR:$S2_DIR}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260603}"
export SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-150}"
export SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
export THERMO_BATCH="${THERMO_BATCH:-32}"
export WRITE_CIFS="${WRITE_CIFS:-0}"

echo "[round3] checkpoint=$CHECKPOINT  (M3, v2 lineage)"
echo "[round3] s0=$S0_DIR"
echo "[round3] s1=$S1_DIR"
echo "[round3] s2=$S2_DIR"
echo "[round3] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[round3] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[round3] out_dir=$OUT_DIR"

exec bash scripts/make_synthetic_dataset_msun_25k.slurm
