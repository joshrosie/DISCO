#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eqv3_hull_v0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/equiformer_v3/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/equiformer_v3/err-%x.%A.err

# Build the EquiformerV3-OAM PatchedPhaseDiagram from MP entries.
# Default protocol: single-point at DFT-min, MP2020 corrections applied.
# Set RELAX=1 to fully relax MP reference structures before building the hull.
#
# Submit:
#   sbatch scripts/build_equiformer_v3_hull.slurm
#
# Override limit (e.g., quick cluster smoke):
#   sbatch --export=ALL,LIMIT=1000,OUT=data/hull/equiformer_v3_oam_ppd_smoke.pkl \
#     scripts/build_equiformer_v3_hull.slurm
#
# Build the relaxed reference hull:
#   sbatch --export=ALL,RELAX=1,OUT=data/hull/equiformer_v3_oam_ppd_relaxed_v0.pkl \
#     scripts/build_equiformer_v3_hull.slurm

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

cd "$HOME/atom-reps"
mkdir -p "$HOME/atom-reps/logs/equiformer_v3" "$HOME/atom-reps/data/hull"

LIMIT="${LIMIT:--1}"                                           # -1 = all MP-20 entries (~45k)
OUT="${OUT:-data/hull/equiformer_v3_oam_ppd_v0.pkl}"
SOURCE_CSV="${SOURCE_CSV:-data/mp20/raw/all.csv}"
SEED="${SEED:-0}"
RELAX="${RELAX:-0}"
RELAX_MAX_STEPS="${RELAX_MAX_STEPS:-500}"
RELAX_FMAX="${RELAX_FMAX:-0.02}"
RELAX_CELL_FILTER="${RELAX_CELL_FILTER:-frechet}"
RELAX_OPTIMIZER="${RELAX_OPTIMIZER:-FIRE}"
DEVICE="${DEVICE:-auto}"

INNER_PY="external/equiformer_v3/.venv/bin/python"
if [[ ! -x "$INNER_PY" ]]; then
  echo "[hull] inner venv missing at $INNER_PY" >&2
  echo "[hull] run scripts/setup_equiformer_v3.sh first." >&2
  exit 2
fi

echo "[hull] LIMIT=$LIMIT"
echo "[hull] OUT=$OUT"
echo "[hull] SOURCE_CSV=$SOURCE_CSV"
echo "[hull] RELAX=$RELAX"
echo "[hull] relax: max_steps=$RELAX_MAX_STEPS fmax=$RELAX_FMAX cell_filter=$RELAX_CELL_FILTER optimizer=$RELAX_OPTIMIZER"
echo "[hull] inner python: $($INNER_PY --version)"
echo "[hull] cuda available: $($INNER_PY -c 'import torch; print(torch.cuda.is_available())')"

EXTRA_ARGS=()
if [[ "$RELAX" == "1" ]]; then
  EXTRA_ARGS+=(--relax)
fi

srun "$INNER_PY" scripts/build_equiformer_v3_hull.py \
  --source-csv "$SOURCE_CSV" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --output "$OUT" \
  --device "$DEVICE" \
  --relax-max-steps "$RELAX_MAX_STEPS" \
  --relax-fmax "$RELAX_FMAX" \
  --relax-cell-filter "$RELAX_CELL_FILTER" \
  --relax-optimizer "$RELAX_OPTIMIZER" \
  "${EXTRA_ARGS[@]}"

echo "[hull] round-trip scoring smoke"
srun "$INNER_PY" scripts/smoke_equiformer_v3_hull_roundtrip.py --ppd "$OUT"

echo "[hull] done. pickle → $OUT"
