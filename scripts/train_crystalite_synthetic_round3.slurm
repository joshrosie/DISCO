#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dng_synth_r3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/err-%x.%A.err

# Train Round 3 synthetic augmentation (v2 cascade):
#
#   MP20 train + S0 + S1_v2_full + S2_v2 + S3_v2  ~= 434k
#
# Longer wall (120h) vs round 2 (96h) because effective train set ~2× larger.

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
S1_DIR="${S1_DIR:-data/synthetic/crystalite_round1_msun_54k_v2_full}"
S2_DIR="${S2_DIR:-data/synthetic/crystalite_round2_msun_108k_v2}"
S3_DIR="${S3_DIR:-data/synthetic/crystalite_round3_msun_217k_v2}"

for d in "$S0_DIR" "$S1_DIR" "$S2_DIR" "$S3_DIR"; do
  if [[ ! -e "$d/raw/train.csv" ]]; then
    echo "[round3-train] synthetic root missing raw/train.csv: $d/raw/train.csv" >&2
    exit 2
  fi
done

export AUGMENTATION="${AUGMENTATION:-synthetic_concat}"
export SYNTHETIC_DATA="${SYNTHETIC_DATA:-$S0_DIR:$S1_DIR:$S2_DIR:$S3_DIR}"
export OUT_DIR="${OUT_DIR:-outputs/dng_synthetic_round3_msun_434k_v2}"
export WANDB_NAME="${WANDB_NAME:-dng_synthetic_round3_msun_434k_v2}"

echo "[round3-train] synthetic_data=$SYNTHETIC_DATA"
echo "[round3-train] output_dir=$OUT_DIR"

exec bash scripts/train_crystalite_synthetic_round0.slurm
