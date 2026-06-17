#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dng_synth_raw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/err-%x.%A.err

set -euo pipefail

cd "$HOME/atom-reps"

SYN_DIR="${SYN_DIR:-data/synthetic/crystalite_raw_27k}"

if [[ ! -e "$SYN_DIR/raw/train.csv" ]]; then
  echo "[raw-train] synthetic root missing raw/train.csv: $SYN_DIR/raw/train.csv" >&2
  exit 2
fi

export AUGMENTATION="${AUGMENTATION:-synthetic_concat}"
export SYNTHETIC_DATA="${SYNTHETIC_DATA:-$SYN_DIR}"
export OUT_DIR="${OUT_DIR:-outputs/dng_synthetic_raw_54k}"
export WANDB_NAME="${WANDB_NAME:-dng_synthetic_raw_54k}"

echo "[raw-train] synthetic_data=$SYNTHETIC_DATA"
echo "[raw-train] output_dir=$OUT_DIR"

exec bash scripts/train_crystalite_synthetic_round0.slurm
