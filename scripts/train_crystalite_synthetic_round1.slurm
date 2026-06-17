#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dng_synth_r1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/err-%x.%A.err

# Train Round 1 synthetic augmentation:
#
#   MP20 train + S0 + S1
#
# Defaults assume:
#   S0 = data/synthetic/crystalite_round0_msun_27k
#   S1 = data/synthetic/crystalite_round1_msun_54k

set -euo pipefail

cd "$HOME/atom-reps"

S0_DIR="${S0_DIR:-data/synthetic/crystalite_round0_msun_27k}"
S1_DIR="${S1_DIR:-data/synthetic/crystalite_round1_msun_54k}"

if [[ ! -e "$S0_DIR/raw/train.csv" ]]; then
  echo "[round1-train] S0 not found (missing raw/train.csv): $S0_DIR/raw/train.csv" >&2
  exit 2
fi
if [[ ! -e "$S1_DIR/raw/train.csv" ]]; then
  echo "[round1-train] S1 not found (missing raw/train.csv): $S1_DIR/raw/train.csv" >&2
  exit 2
fi

export AUGMENTATION="${AUGMENTATION:-synthetic_concat}"
export SYNTHETIC_DATA="${SYNTHETIC_DATA:-$S0_DIR:$S1_DIR}"
export OUT_DIR="${OUT_DIR:-outputs/dng_synthetic_round1_msun_108k}"
export WANDB_NAME="${WANDB_NAME:-dng_synthetic_round1_msun_108k}"

echo "[round1-train] synthetic_data=$SYNTHETIC_DATA"
echo "[round1-train] output_dir=$OUT_DIR"

exec bash scripts/train_crystalite_synthetic_round0.slurm
