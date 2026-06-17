#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=bench_relax
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/bench_relax/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/bench_relax/err-%x.%A.err

# Time the Flywheel sampling + curation-relaxation costs on the same H100, so
# they can sit side by side in the thesis. Samples N raw structures from a
# Crystalite checkpoint, then relaxes+scores them exactly as the curator does
# (NequIP-OAM-L, 200 FIRE steps, frechet cell filter, batch=32, fmax=0.005).
#
# Self-contained: needs only the checkpoint + compiled NequIP model + MP PPD.
#
# Submit:
#   sbatch scripts/benchmark_relaxation.slurm
#   sbatch --export=ALL,N=1000,CHECKPOINT=dng.pt scripts/benchmark_relaxation.slurm

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/atom-reps"

mkdir -p "$HOME/atom-reps/logs/bench_relax"

export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-$SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

N="${N:-1000}"
CHECKPOINT="${CHECKPOINT:-dng.pt}"
NL_BACKEND="${NL_BACKEND:-vesin}"
THERMO_PPD_MP="${THERMO_PPD_MP:-/home/jrosenthal/atom-reps/mp_02072023/2023-02-07-ppd-mp.pkl}"
NEQUIP_COMPILE_PATH="${NEQUIP_COMPILE_PATH:-/home/jrosenthal/atom-reps/mlips/aot_batch.nequip.pt2}"

echo "[bench-relax] n=$N checkpoint=$CHECKPOINT nl_backend=$NL_BACKEND"
echo "[bench-relax] ppd=$THERMO_PPD_MP nequip=$NEQUIP_COMPILE_PATH"

srun uv run python scripts/benchmark_relaxation.py \
  --checkpoint "$CHECKPOINT" \
  --n "$N" \
  --device cuda \
  --neighborlist_backend "$NL_BACKEND" \
  --nequip_compile_path "$NEQUIP_COMPILE_PATH" \
  --ppd_path "$THERMO_PPD_MP"
