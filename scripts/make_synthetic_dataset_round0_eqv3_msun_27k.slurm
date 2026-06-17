#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=synth_round0_eqv3_msun_27k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Round 0 with the EquiformerV3-OAM verifier (parallel ablation to NequIP Round 0).
#
# Same generator (M0 = dng.pt), same target budget (27k accepted), same filter
# (msun_like + structure dedup vs MP20), same MP2020 correction convention.
# Only the verifier changes — EquiformerV3-OAM via subprocess bridge to the
# inner equiformer_v3 venv, scoring against the eqv3-self-consistent hull.
#
# Prerequisites:
#   1. Inner venv set up:        bash scripts/setup_equiformer_v3.sh
#   2. EquiformerV3 hull built:  sbatch --export=ALL,RELAX=1,OUT=data/hull/equiformer_v3_oam_ppd_relaxed_v0.pkl \
#                                  scripts/build_equiformer_v3_hull.slurm
#                                → produces data/hull/equiformer_v3_oam_ppd_relaxed_v0.pkl
#
# Submit:
#   sbatch scripts/make_synthetic_dataset_round0_eqv3_msun_27k.slurm

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
NUM_GENERATE="${NUM_GENERATE:-100000}"
MAX_SAMPLES="${MAX_SAMPLES:-27138}"            # match NequIP Round 0 exactly
OUT_DIR="${OUT_DIR:-data/synthetic/crystalite_round0_eqv3_msun_27k}"
GENERATION_DATA_ROOT="${GENERATION_DATA_ROOT:-data/mp20}"
REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-$GENERATION_DATA_ROOT}"
NMAX="${NMAX:-20}"

SAMPLE_SEED="${SAMPLE_SEED:-2026}"               # fresh seed independent of NequIP Round 0
SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-100}"
SAMPLE_MODE="${SAMPLE_MODE:-ema}"

THERMO_BATCH="${THERMO_BATCH:-32}"
EHULL_THRESH="${EHULL_THRESH:-0.1}"

# EquiformerV3-specific paths.
EQV3_PPD="${EQV3_PPD:-data/hull/equiformer_v3_oam_ppd_relaxed_v0.pkl}"
EQV3_INNER_PY="${EQV3_INNER_PY:-external/equiformer_v3/.venv/bin/python}"
EQV3_WRAPPER="${EQV3_WRAPPER:-scripts/equiformer_v3_inference_wrapper.py}"
EQV3_CKPT="${EQV3_CKPT:-external/equiformer_v3/checkpoints/omat24-mptrj-salex_gradient.pt}"
EQV3_MAX_STEPS="${EQV3_MAX_STEPS:-500}"
EQV3_FMAX="${EQV3_FMAX:-0.02}"

# Preflight checks: bail early if any required artifact is missing.
for path in "$EQV3_PPD" "$EQV3_INNER_PY" "$EQV3_WRAPPER" "$EQV3_CKPT"; do
  if [[ ! -e "$path" ]]; then
    echo "[synth-eqv3] FATAL: missing $path" >&2
    echo "[synth-eqv3] make sure setup_equiformer_v3.sh + build_equiformer_v3_hull.slurm have run." >&2
    exit 2
  fi
done

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

echo "[synth-eqv3] checkpoint=$CHECKPOINT"
echo "[synth-eqv3] num_generate=$NUM_GENERATE max_samples=$MAX_SAMPLES"
echo "[synth-eqv3] out_dir=$OUT_DIR"
echo "[synth-eqv3] reference_data_root=$REFERENCE_DATA_ROOT"
echo "[synth-eqv3] eqv3 hull=$EQV3_PPD"
echo "[synth-eqv3] eqv3 inner python=$EQV3_INNER_PY"

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
  --thermo_mlip equiformer_v3 \
  --thermo_ppd_mp "$EQV3_PPD" \
  --thermo_stability_device cuda \
  --thermo_ehull_method mp2020_like \
  --thermo_stability_batch "$THERMO_BATCH" \
  --ehull_metastable_thresh "$EHULL_THRESH" \
  --equiformer_v3_inner_python "$EQV3_INNER_PY" \
  --equiformer_v3_wrapper "$EQV3_WRAPPER" \
  --equiformer_v3_checkpoint "$EQV3_CKPT" \
  --equiformer_v3_max_steps "$EQV3_MAX_STEPS" \
  --equiformer_v3_fmax "$EQV3_FMAX" \
  --equiformer_v3_cell_filter frechet \
  --equiformer_v3_optimizer FIRE \
  --equiformer_v3_device auto \
  "${EXTRA_ARGS[@]}"
