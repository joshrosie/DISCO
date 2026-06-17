#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synthetic_r3_topup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# S3 top-up. Closes the gap between the main Round 3 curation
# (scripts/make_synthetic_dataset_round3_msun_217k.slurm, job 23434282) and the
# S3 target |S0|+|S1|+|S2|+|S0| = 217,104. The main job is wall-clock limited
# (~96h budget undershoots the target at the observed ~30% yield), so this
# fills the remainder.
#
# Generates from M3 (same checkpoint as the main job) and dedups against
# MP20 + S0 + S1 + S2 AND the partial S3 already produced, so concatenating
# the two roots afterwards yields no cross-duplicates.
#
# MAX_SAMPLES is computed at runtime from the *final* partial-S3 count, so this
# can be queued as a dependency before the main job finishes:
#   sbatch --dependency=afterany:23434282 scripts/make_synthetic_dataset_round3_topup.slurm
#
# After it finishes, merge into one root:
#   python scripts/merge_synthetic_roots.py \
#     --inputs data/synthetic/crystalite_round3_msun_217k_v2 \
#              data/synthetic/crystalite_round3_topup \
#     --output data/synthetic/crystalite_round3_msun_217k_v2_merged

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
S1_DIR="${S1_DIR:-data/synthetic/crystalite_round1_msun_54k_v2_full}"
S2_DIR="${S2_DIR:-data/synthetic/crystalite_round2_msun_108k_v2}"
S3_DIR="${S3_DIR:-data/synthetic/crystalite_round3_msun_217k_v2}"
CHECKPOINT="${CHECKPOINT:-outputs/dng_synthetic_round2_msun_217k_v2/checkpoints/best.pt}"
S3_TARGET="${S3_TARGET:-217104}"

for path in "$CHECKPOINT" "$S0_DIR/raw/train.csv" "$S1_DIR/raw/train.csv" \
            "$S2_DIR/raw/train.csv" "$S3_DIR/raw/train.csv"; do
  if [[ ! -e "$path" ]]; then
    echo "[r3-topup] missing prerequisite: $path" >&2
    exit 2
  fi
done

# Auto-size to the remaining gap from whatever the main job actually produced.
EXISTING=$(grep -c '^synthetic_[0-9]*,' "$S3_DIR/raw/train.csv")
GAP=$(( S3_TARGET - EXISTING ))
if [[ "$GAP" -le 0 ]]; then
  echo "[r3-topup] S3 already at/over target (existing=$EXISTING target=$S3_TARGET); nothing to do." >&2
  exit 0
fi

export CHECKPOINT
# Budget generation for the gap at a conservative ~15% yield floor, capped so a
# pathological run can't sample forever; the job stops early once MAX_SAMPLES is
# reached.
export MAX_SAMPLES="${MAX_SAMPLES:-$GAP}"
export NUM_GENERATE="${NUM_GENERATE:-$(( GAP * 7 ))}"
export OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_round3_topup}"
export GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-data/mp20}"
export REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-data/mp20:$S0_DIR:$S1_DIR:$S2_DIR:$S3_DIR}"
# Distinct from the main job's seed (20260603) so we don't redraw identical samples.
export SAMPLE_SEED="${SAMPLE_SEED:-20260605}"
export SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-150}"
export SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
export THERMO_BATCH="${THERMO_BATCH:-32}"
export WRITE_CIFS="${WRITE_CIFS:-0}"

echo "[r3-topup] checkpoint=$CHECKPOINT  (M3, v2 lineage)"
echo "[r3-topup] s0=$S0_DIR"
echo "[r3-topup] s1=$S1_DIR"
echo "[r3-topup] s2=$S2_DIR"
echo "[r3-topup] s3(partial)=$S3_DIR  existing=$EXISTING"
echo "[r3-topup] target=$S3_TARGET  gap=$GAP"
echo "[r3-topup] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[r3-topup] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[r3-topup] out_dir=$OUT_DIR"

exec bash scripts/make_synthetic_dataset_msun_25k.slurm
