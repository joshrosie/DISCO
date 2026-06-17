#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synth_raw_27k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Ablation: raw model-generated data.
#
# No MLIP, no metastability filter, no geometry-validity filter, no dedup.
# This is intentionally dirty; it tests whether the Flywheel gain requires
# curation rather than just appending generated samples.

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/atom-reps"
mkdir -p "$HOME/atom-reps/logs/synthetic_aug_msun"

export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-$SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

CHECKPOINT="${CHECKPOINT:-dng.pt}"
NUM_GENERATE="${NUM_GENERATE:-30000}"
MAX_SAMPLES="${MAX_SAMPLES:-27138}"
OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_raw_27k}"
GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-data/mp20}"
NMAX="${NMAX:-20}"

SAMPLE_SEED="${SAMPLE_SEED:-20260514}"
SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-150}"
SAMPLE_MODE="${SAMPLE_MODE:-ema}"

EXTRA_ARGS=()
if [[ -n "${SAMPLER:-}" ]]; then
  EXTRA_ARGS+=(--sampler "$SAMPLER")
fi
if [[ -n "${ATOM_COUNT_STRATEGY:-}" ]]; then
  EXTRA_ARGS+=(--atom_count_strategy "$ATOM_COUNT_STRATEGY")
fi
if [[ -n "${FIXED_NUM_ATOMS:-}" ]]; then
  EXTRA_ARGS+=(--fixed_num_atoms "$FIXED_NUM_ATOMS")
fi
if [[ "${BF16:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--bf16)
fi

echo "[raw-synth] checkpoint=$CHECKPOINT"
echo "[raw-synth] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[raw-synth] out_dir=$OUT_DIR"

srun uv run python scripts/make_synthetic_dataset.py \
  --checkpoint "$CHECKPOINT" \
  --num_generate "$NUM_GENERATE" \
  --output_dir "$OUT_DIR" \
  --generation_data_root "$GENERATION_DATA_ROOT" \
  --filter_level raw \
  --dedup_mode none \
  --max_samples "$MAX_SAMPLES" \
  --nmax "$NMAX" \
  --sample_device cuda \
  --sample_mode "$SAMPLE_MODE" \
  --sample_seed "$SAMPLE_SEED" \
  --sample_num_steps "$SAMPLE_NUM_STEPS" \
  --sample_chunk_size "$SAMPLE_CHUNK_SIZE" \
  "${EXTRA_ARGS[@]}"
