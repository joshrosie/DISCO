#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=export_cifs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/export_cifs/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/export_cifs/err-%x.%A.err

# Export Crystalite samples as CIFs for external evaluators such as
# LeMat-GenBench. Defaults to the synthetic-augmented Round 0 model.
# Set PRERELAX_NEQUIP=1 to additionally export NequIP-relaxed CIFs in
# $OUT_DIR/relaxed_cifs. Use that directory only for explicitly relaxed
# LeMat submissions.
#
# Example:
#   sbatch --export=ALL,CHECKPOINT=dng.pt,OUT_DIR=outputs/external_eval/base_n2500,NUM_SAMPLES=2500 \
#     scripts/export_crystalite_cifs.slurm

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/atom-reps"

mkdir -p "$HOME/atom-reps/logs/export_cifs"

export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-$SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

CHECKPOINT="${CHECKPOINT:-outputs/dng_synthetic_round0_msun_27k/checkpoints/best.pt}"
if [[ ! -f "$CHECKPOINT" ]]; then
  ALT="${ALT_CHECKPOINT:-outputs/dng_synthetic_round0_msun_27k/checkpoints/step_latest.pt}"
  if [[ -f "$ALT" ]]; then
    echo "[export-cifs] CHECKPOINT not found, using ALT_CHECKPOINT=$ALT"
    CHECKPOINT="$ALT"
  else
    echo "[export-cifs] checkpoint not found: $CHECKPOINT" >&2
    exit 2
  fi
fi

OUT_DIR="${OUT_DIR:-outputs/external_eval/synthetic_aug_n2500}"
NUM_SAMPLES="${NUM_SAMPLES:-2500}"
SAMPLE_SEED="${SAMPLE_SEED:-123}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-150}"
SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-512}"
SAMPLE_MODE="${SAMPLE_MODE:-ema}"
ATOM_COUNT_STRATEGY="${ATOM_COUNT_STRATEGY:-empirical}"
CIF_LIMIT="${CIF_LIMIT:-0}"
PRERELAX_NEQUIP="${PRERELAX_NEQUIP:-0}"
THERMO_PPD_MP="${THERMO_PPD_MP:-/home/jrosenthal/atom-reps/mp_02072023/2023-02-07-ppd-mp.pkl}"
NEQUIP_COMPILE_PATH="${NEQUIP_COMPILE_PATH:-/home/jrosenthal/atom-reps/mlips/aot_batch.nequip.pt2}"
THERMO_STABILITY_BATCH="${THERMO_STABILITY_BATCH:-32}"
THERMO_RELAX_STEPS="${THERMO_RELAX_STEPS:-200}"

EXTRA_ARGS=()
if [[ "${BF16:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--bf16)
fi
if [[ -n "${SAMPLER:-}" ]]; then
  EXTRA_ARGS+=(--sampler "$SAMPLER")
fi

echo "[export-cifs] checkpoint=$CHECKPOINT"
echo "[export-cifs] out_dir=$OUT_DIR"
echo "[export-cifs] num_samples=$NUM_SAMPLES seed=$SAMPLE_SEED steps=$SAMPLE_NUM_STEPS"
echo "[export-cifs] sample_chunk_size=$SAMPLE_CHUNK_SIZE mode=$SAMPLE_MODE"
echo "[export-cifs] prerelax_nequip=$PRERELAX_NEQUIP"

srun uv run python src/sample_crystalite_ckpt.py \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --sample_seed "$SAMPLE_SEED" \
  --sample_num_steps "$SAMPLE_NUM_STEPS" \
  --sample_chunk_size "$SAMPLE_CHUNK_SIZE" \
  --sample_mode "$SAMPLE_MODE" \
  --atom_count_strategy "$ATOM_COUNT_STRATEGY" \
  --device cuda \
  --save_cifs \
  --cif_limit "$CIF_LIMIT" \
  --no-save_extxyz \
  "${EXTRA_ARGS[@]}"

if [[ "$PRERELAX_NEQUIP" == "1" || "$PRERELAX_NEQUIP" == "true" ]]; then
  echo "[export-cifs] NequIP pre-relaxing samples into $OUT_DIR/relaxed_cifs"
  srun uv run python scripts/relax_samples_to_cifs.py \
    --samples "$OUT_DIR/samples.pt" \
    --output_dir "$OUT_DIR/relaxed_cifs" \
    --nequip_compile_path "$NEQUIP_COMPILE_PATH" \
    --thermo_ppd_mp "$THERMO_PPD_MP" \
    --thermo_stability_device cuda \
    --thermo_ehull_method mp2020_like \
    --thermo_relax_steps "$THERMO_RELAX_STEPS" \
    --thermo_stability_batch "$THERMO_STABILITY_BATCH" \
    --nequip_relax_mode batch \
    --nequip_cell_filter frechet \
    --nequip_fmax 0.005
fi
