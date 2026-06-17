#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dng_synth_sbig
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/err-%x.%A.err

# Train one-shot S_big control:
#
#   MP20 train + S_big
#
# S_big is mined from M0 at roughly |S0| + |S1| synthetic budget, so this
# controls "more curated data once" against iterative S0->M1->S1->M2.

set -euo pipefail

cd "$HOME/atom-reps"

S_BIG_DIR="${S_BIG_DIR:-data/synthetic/crystalite_S_big_81k}"

if [[ ! -e "$S_BIG_DIR/raw/train.csv" ]]; then
  echo "[s-big-train] S_big root missing raw/train.csv: $S_BIG_DIR/raw/train.csv" >&2
  exit 2
fi

export AUGMENTATION="${AUGMENTATION:-synthetic_concat}"
export SYNTHETIC_DATA="${SYNTHETIC_DATA:-$S_BIG_DIR}"
export OUT_DIR="${OUT_DIR:-outputs/dng_synthetic_S_big_108k}"
export WANDB_NAME="${WANDB_NAME:-dng_synthetic_S_big_108k}"

echo "[s-big-train] synthetic_data=$SYNTHETIC_DATA"
echo "[s-big-train] output_dir=$OUT_DIR"

exec bash scripts/train_crystalite_synthetic_round0.slurm
