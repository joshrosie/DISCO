#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synth_S_big_topup_12115
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# S_big top-up. Closes the gap between curated S_big (69,299 kept) and the
# matched-budget target |S0| + |S1| = 81,414. Generates from M0 (same
# generator as original S_big curation), dedups against MP20 + existing
# S_big.
#
# Combined with the original S_big, the union is 81,414 — matched to
# (MP20 + S0 + S1) cumulative synthetic exactly.

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/atom-reps"
mkdir -p "$HOME/atom-reps/logs/synthetic_aug_msun"

export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-$SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

S_BIG_DIR="${S_BIG_DIR:-data/synthetic/crystalite_S_big_81k}"
CHECKPOINT="${CHECKPOINT:-dng.pt}"                  # M0 — same as original S_big
NUM_GENERATE="${NUM_GENERATE:-100000}"              # ~8.3× target at 23% yield (observed)
MAX_SAMPLES="${MAX_SAMPLES:-12115}"                 # 81414 - 69299
OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_S_big_topup_12115}"
GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-${DATA_ROOT:-data/mp20}}"
REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-$GENERATION_DATA_ROOT:$S_BIG_DIR}"
NMAX="${NMAX:-20}"

if [[ ! -e "$S_BIG_DIR/raw/train.csv" ]]; then
  echo "[S_big-topup] S_big root missing raw/train.csv: $S_BIG_DIR/raw/train.csv" >&2
  exit 2
fi

SAMPLE_SEED="${SAMPLE_SEED:-20260516}"              # fresh seed, independent of original S_big
SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-100}"
SAMPLE_MODE="${SAMPLE_MODE:-ema}"

THERMO_BATCH="${THERMO_BATCH:-32}"
THERMO_RELAX_STEPS="${THERMO_RELAX_STEPS:-200}"
EHULL_THRESH="${EHULL_THRESH:-0.1}"
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
if [[ "${WRITE_CIFS:-0}" != "1" ]]; then
  EXTRA_ARGS+=(--no_write_cifs)
fi

echo "[S_big-topup] checkpoint=$CHECKPOINT"
echo "[S_big-topup] existing S_big=$S_BIG_DIR"
echo "[S_big-topup] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[S_big-topup] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[S_big-topup] out_dir=$OUT_DIR"
echo "[S_big-topup] sample_seed=$SAMPLE_SEED"

srun uv run python scripts/make_synthetic_dataset.py \
  --checkpoint "$CHECKPOINT" \
  --num_generate "$NUM_GENERATE" \
  --output_dir "$OUT_DIR" \
  --generation_data_root "$GENERATION_DATA_ROOT" \
  --reference_data_root "$REFERENCE_DATA_ROOT" \
  --filter_level msun_like \
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
  --ehull_metastable_thresh "$EHULL_THRESH" \
  --nequip_compile_path "$NEQUIP_COMPILE_PATH" \
  --nequip_relax_mode batch \
  --nequip_cell_filter frechet \
  --nequip_fmax 0.005 \
  "${EXTRA_ARGS[@]}"
