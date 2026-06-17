#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eqv3_hull_mp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/equiformer_v3/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/equiformer_v3/err-%x.%A.err

# Full-MP EquiformerV3 hull build. Two-stage:
#   1. Fetch DFT-relaxed structures for every mp-id in the existing MP PPD
#      pickle via the Materials Project API (canonical source; scope-aligned
#      with the NequIP curation hull by construction).
#   2. Run scripts/build_equiformer_v3_hull.py against that CSV, preferably
#      with RELAX=1 so the reference hull and generated candidates share the
#      same EquiformerV3 relaxation convention.
#
# Replaces the MP-20-scoped hull. Scope identical to the existing NequIP DFT
# hull by construction (we fetch exactly the mp-ids in the source PPD).
#
# Auth: MP_API_KEY env var must be set (e.g., in ~/.bashrc). The key is
# never read from a CLI arg and never written to a tracked file.
#
# Submit:
#   sbatch --export=ALL,RELAX=1 scripts/build_equiformer_v3_hull_mp.slurm

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

cd "$HOME/atom-reps"
mkdir -p "$HOME/atom-reps/logs/equiformer_v3" "$HOME/atom-reps/data/hull" "$HOME/atom-reps/data/mp_full/raw"

CSV="${CSV:-data/mp_full/raw/relaxed.csv}"
OUT="${OUT:-data/hull/equiformer_v3_oam_ppd_relaxed_v0_mp.pkl}"
SOURCE_PPD="${SOURCE_PPD:-mp_02072023/2023-02-07-ppd-mp.pkl}"
LIMIT="${LIMIT:--1}"
RELAX="${RELAX:-1}"
RELAX_MAX_STEPS="${RELAX_MAX_STEPS:-500}"
RELAX_FMAX="${RELAX_FMAX:-0.02}"
RELAX_CELL_FILTER="${RELAX_CELL_FILTER:-frechet}"
RELAX_OPTIMIZER="${RELAX_OPTIMIZER:-FIRE}"
DEVICE="${DEVICE:-auto}"

INNER_PY="external/equiformer_v3/.venv/bin/python"
if [[ ! -x "$INNER_PY" ]]; then
  echo "[hull-mp] inner venv missing at $INNER_PY" >&2
  exit 2
fi

if [[ -z "${MP_API_KEY:-}" ]]; then
  echo "[hull-mp] FATAL: MP_API_KEY env var not set." >&2
  echo "[hull-mp] export MP_API_KEY='<your-key>' in your shell rc first." >&2
  exit 3
fi

# ---- Stage 1: prep ----
if [[ ! -f "$CSV" ]]; then
  echo "[hull-mp] preparing CSV via mp-api (~30 min - 2 hr)"
  srun uv run python scripts/prep_mp_api_for_hull.py \
    --source-ppd "$SOURCE_PPD" \
    --output "$CSV"
else
  echo "[hull-mp] CSV present at $CSV; skipping prep"
fi
echo "[hull-mp] CSV: $CSV ($(wc -l < "$CSV") lines)"

# ---- Stage 2: hull build ----
echo "[hull-mp] running eqv3 scoring/relaxation + PPD construction"
echo "[hull-mp] RELAX=$RELAX"
echo "[hull-mp] relax: max_steps=$RELAX_MAX_STEPS fmax=$RELAX_FMAX cell_filter=$RELAX_CELL_FILTER optimizer=$RELAX_OPTIMIZER"
echo "[hull-mp] inner python: $($INNER_PY --version)"
echo "[hull-mp] cuda available: $($INNER_PY -c 'import torch; print(torch.cuda.is_available())')"

EXTRA_ARGS=()
if [[ "$RELAX" == "1" ]]; then
  EXTRA_ARGS+=(--relax)
fi

srun "$INNER_PY" scripts/build_equiformer_v3_hull.py \
  --source-csv "$CSV" \
  --cif-column cif \
  --limit "$LIMIT" \
  --output "$OUT" \
  --device "$DEVICE" \
  --relax-max-steps "$RELAX_MAX_STEPS" \
  --relax-fmax "$RELAX_FMAX" \
  --relax-cell-filter "$RELAX_CELL_FILTER" \
  --relax-optimizer "$RELAX_OPTIMIZER" \
  "${EXTRA_ARGS[@]}"

echo "[hull-mp] round-trip scoring smoke"
srun "$INNER_PY" scripts/smoke_equiformer_v3_hull_roundtrip.py --ppd "$OUT"

echo "[hull-mp] done. pickle → $OUT"
