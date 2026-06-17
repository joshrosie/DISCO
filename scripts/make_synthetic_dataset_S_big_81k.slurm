#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synth_S_big_81k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# S_big control: one-shot data mining from M0 (dng.pt) at matched total
# synthetic budget to (S0 + S1) ≈ 81k. Distinguishes "iteration compounds"
# (M2 > M_big) from "more curated data alone explains the gain"
# (M2 ≈ M_big) once Round 1 has landed.
#
# Same M0, same NequIP verifier, same MP DFT hull, same msun_like filter,
# same StructureMatcher dedup — only the source-of-candidates is one-shot
# rather than iterative.
#
# Submit:
#   sbatch scripts/make_synthetic_dataset_S_big_81k.slurm
#   sbatch --export=ALL,WRITE_CIFS=0 scripts/make_synthetic_dataset_S_big_81k.slurm

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/atom-reps"
mkdir -p "$HOME/atom-reps/logs/synthetic_aug_msun"

export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-$SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

CHECKPOINT="${CHECKPOINT:-dng.pt}"                  # M0 — base Crystalite
NUM_GENERATE="${NUM_GENERATE:-300000}"              # ~3.7× target for ~27% acceptance
MAX_SAMPLES="${MAX_SAMPLES:-81414}"                 # matches |S0| + |S1| ≈ 27138 + 54276
OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_S_big_81k}"
GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-${DATA_ROOT:-data/mp20}}"
REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-$GENERATION_DATA_ROOT}"
NMAX="${NMAX:-20}"

SAMPLE_SEED="${SAMPLE_SEED:-20260512}"              # fresh seed, independent of S0/S1
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

echo "[S_big] checkpoint=$CHECKPOINT"
echo "[S_big] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[S_big] out_dir=$OUT_DIR"
echo "[S_big] sample_seed=$SAMPLE_SEED"
echo "[S_big] write_cifs=${WRITE_CIFS:-0}"
echo "[S_big] thermo_ppd_mp=$THERMO_PPD_MP"
echo "[S_big] thermo_batch=$THERMO_BATCH nequip=$NEQUIP_COMPILE_PATH"

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
