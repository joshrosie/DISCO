#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eqv3_vs_mp20_ehull
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/equiformer_v3/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/equiformer_v3/err-%x.%A.err

# Run EquiformerV3 single-points across all of MP-20 and compare to stored
# e_above_hull. Writes a CSV per-entry with mp20_e_hull, eqv3_e_total,
# eqv3_e_hull_raw, eqv3_e_hull_mp2020. Plotting is downstream and local.
#
# Submit:
#   sbatch scripts/equiformer_v3_vs_mp20_ehull.slurm
# Override defaults:
#   sbatch --export=ALL,LIMIT=5000,OUT=results/eqv3_vs_mp20_n5k.csv \
#     scripts/equiformer_v3_vs_mp20_ehull.slurm

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

cd "$HOME/atom-reps"
mkdir -p "$HOME/atom-reps/logs/equiformer_v3"

LIMIT="${LIMIT:--1}"                                           # -1 = all of MP-20
OUT="${OUT:-results/equiformer_v3_vs_mp20_all.csv}"
SEED="${SEED:-0}"
CIF_COLUMN="${CIF_COLUMN:-cif.conv}"

INNER_PY="external/equiformer_v3/.venv/bin/python"
if [[ ! -x "$INNER_PY" ]]; then
  echo "[eqv3-vs-mp20] inner venv missing at $INNER_PY" >&2
  echo "[eqv3-vs-mp20] run scripts/setup_equiformer_v3.sh first." >&2
  exit 2
fi

echo "[eqv3-vs-mp20] LIMIT=$LIMIT"
echo "[eqv3-vs-mp20] OUT=$OUT"
echo "[eqv3-vs-mp20] CIF_COLUMN=$CIF_COLUMN"
echo "[eqv3-vs-mp20] inner python: $($INNER_PY --version)"
echo "[eqv3-vs-mp20] cuda available: $($INNER_PY -c 'import torch; print(torch.cuda.is_available())')"

srun "$INNER_PY" scripts/equiformer_v3_vs_mp20_ehull.py \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --cif-column "$CIF_COLUMN" \
  --output "$OUT"

echo "[eqv3-vs-mp20] done. CSV → $OUT"
