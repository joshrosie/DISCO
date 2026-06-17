#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dng_synth_r2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/err-%x.%A.err

# Train Round 2 synthetic augmentation:
#
#   MP20 train + S0 + S1 + S2

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
S1_DIR="${S1_DIR:-data/synthetic/crystalite_round1_msun_54k}"
S2_DIR="${S2_DIR:-data/synthetic/crystalite_round2_msun_109k}"

for d in "$S0_DIR" "$S1_DIR" "$S2_DIR"; do
  if [[ ! -e "$d/raw/train.csv" ]]; then
    echo "[round2-train] synthetic root missing raw/train.csv: $d/raw/train.csv" >&2
    exit 2
  fi
done

export AUGMENTATION="${AUGMENTATION:-synthetic_concat}"
export SYNTHETIC_DATA="${SYNTHETIC_DATA:-$S0_DIR:$S1_DIR:$S2_DIR}"
export OUT_DIR="${OUT_DIR:-outputs/dng_synthetic_round2_msun_217k}"
export WANDB_NAME="${WANDB_NAME:-dng_synthetic_round2_msun_217k}"

echo "[round2-train] synthetic_data=$SYNTHETIC_DATA"
echo "[round2-train] output_dir=$OUT_DIR"

exec bash scripts/train_crystalite_synthetic_round0.slurm
