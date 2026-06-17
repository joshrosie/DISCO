#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synth_stable_27k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Ablation: strict-stable verifier curation.
#
# Same as S0 curation except the stability gate is e_hull <= 0.0 rather than
# e_hull <= 0.1. This tests whether training on strict-stable survivors recovers
# stable yield or whether the broader metastable band is the useful target.

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
NUM_GENERATE="${NUM_GENERATE:-500000}"
MAX_SAMPLES="${MAX_SAMPLES:-27138}"
OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_stable_only_27k}"
GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-${DATA_ROOT:-data/mp20}}"
REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-$GENERATION_DATA_ROOT}"
NMAX="${NMAX:-20}"

SAMPLE_SEED="${SAMPLE_SEED:-20260513}"
SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-100}"
SAMPLE_MODE="${SAMPLE_MODE:-ema}"

THERMO_BATCH="${THERMO_BATCH:-32}"
THERMO_RELAX_STEPS="${THERMO_RELAX_STEPS:-200}"
THERMO_PPD_MP="${THERMO_PPD_MP:-/home/jrosenthal/atom-reps/mp_02072023/2023-02-07-ppd-mp.pkl}"
NEQUIP_COMPILE_PATH="${NEQUIP_COMPILE_PATH:-/home/jrosenthal/atom-reps/mlips/aot_batch.nequip.pt2}"

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

echo "[stable-only] checkpoint=$CHECKPOINT"
echo "[stable-only] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[stable-only] out_dir=$OUT_DIR"
echo "[stable-only] generation_data_root=$GENERATION_DATA_ROOT"
echo "[stable-only] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[stable-only] thermo_ppd_mp=$THERMO_PPD_MP"
echo "[stable-only] thermo_batch=$THERMO_BATCH nequip=$NEQUIP_COMPILE_PATH"

srun uv run python scripts/make_synthetic_dataset.py \
  --checkpoint "$CHECKPOINT" \
  --num_generate "$NUM_GENERATE" \
  --output_dir "$OUT_DIR" \
  --generation_data_root "$GENERATION_DATA_ROOT" \
  --reference_data_root "$REFERENCE_DATA_ROOT" \
  --filter_level stable_like \
  --dedup_mode structure \
  --max_samples "$MAX_SAMPLES" \
  --nmax "$NMAX" \
  --sample_device cuda \
  --sample_mode "$SAMPLE_MODE" \
  --sample_seed "$SAMPLE_SEED" \
  --sample_num_steps "$SAMPLE_NUM_STEPS" \
  --sample_chunk_size "$SAMPLE_CHUNK_SIZE" \
  --thermo_mlip nequip \
  --thermo_ppd_mp "$THERMO_PPD_MP" \
  --thermo_stability_device cuda \
  --thermo_ehull_method mp2020_like \
  --thermo_relax_steps "$THERMO_RELAX_STEPS" \
  --thermo_stability_batch "$THERMO_BATCH" \
  --ehull_metastable_thresh 0.1 \
  --nequip_compile_path "$NEQUIP_COMPILE_PATH" \
  --nequip_relax_mode batch \
  --nequip_cell_filter frechet \
  --nequip_fmax 0.005 \
  "${EXTRA_ARGS[@]}"
