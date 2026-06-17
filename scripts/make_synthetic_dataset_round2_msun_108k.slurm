#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synthetic_r2_msun_108k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Round 2 synthetic curation. Supervisor-approved continuation: branches off
# the v1 lineage checkpoint (M2_v1) and applies the v2 curator corrections
# (MAX_SAMPLES quota enforcement, CIF round-trip validation) going forward.
#
#   M2_v1 train:  MP20 + S0 + S1_v1 ~= 108k
#   S2 target:                       ~= 108k   (doubles again)
#   M3 train:     MP20 + S0 + S1_v1 + S2 ~= 217k
#
# All v2 patches are inherited automatically since they live in
# src/data/synthetic_augmentation.py — this slurm just sets paths.

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
S1_DIR="${S1_DIR:-data/synthetic/crystalite_round1_msun_54k}"   # v1 lineage
CHECKPOINT="${CHECKPOINT:-outputs/dng_synthetic_round1_msun_108k/checkpoints/best.pt}"   # M2_v1

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

for path in "$S0_DIR/raw/train.csv" "$S1_DIR/raw/train.csv"; do
  if [[ ! -e "$path" ]]; then
    echo "[round2] missing prerequisite: $path" >&2
    exit 2
  fi
done

export CHECKPOINT
# Round 1 needed ~3.7× MAX_SAMPLES at ~27% effective yield. Round 2 ref set
# triples so per-structure dedup-survival drops; budget ~4.6× to leave ~25k
# of margin above target. If the curator still raises on undershoot, fire
# a topup slurm modeled on scripts/make_synthetic_dataset_round1_topup_1015.slurm.
export NUM_GENERATE="${NUM_GENERATE:-500000}"
export MAX_SAMPLES="${MAX_SAMPLES:-108552}"
export OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_round2_msun_108k}"
export GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-data/mp20}"
export REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-data/mp20:$S0_DIR:$S1_DIR}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260516}"
export SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-150}"
export SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
export THERMO_BATCH="${THERMO_BATCH:-32}"

echo "[round2] checkpoint=$CHECKPOINT  (M2_v1; v1 lineage per supervisor)"
echo "[round2] s0=$S0_DIR"
echo "[round2] s1=$S1_DIR  (v1)"
echo "[round2] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[round2] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[round2] out_dir=$OUT_DIR"

exec bash scripts/make_synthetic_dataset_msun_25k.slurm
