#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=lemat_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/lemat_genbench/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/lemat_genbench/err-%x.%A.err

# Run LeMat-GenBench on a directory of CIFs.
#
# Required:
#   CIFS=/path/to/cifs
#   NAME=crystalite_synthetic_aug_n2500_nequip_relaxed
#
# Optional:
#   LEMAT_ROOT=$HOME/lemat-genbench
#   CONFIG=comprehensive_multi_mlip_hull
#   FAMILIES="validity novelty uniqueness diversity hhi"
#   EXTRA_ARGS="..."
#   HF_TOKEN=... or HUGGINGFACE_HUB_TOKEN=...
#   HF_TOKEN_FILE=$HOME/.cache/huggingface/token
#   REQUIRE_HF_AUTH=1
#   HF_LOGIN_IN_JOB=1
#
# Examples:
#   sbatch --export=ALL,CIFS=$HOME/atom-reps/outputs/external_eval/synthetic_aug_n2500_nequip_relaxed/relaxed_cifs,NAME=crystalite_synthetic_aug_n2500_nequip_relaxed \
#     scripts/run_lemat_genbench.slurm
#
#   sbatch --export=ALL,CIFS=$HOME/atom-reps/outputs/external_eval/synthetic_aug_n2500_nequip_relaxed/cifs,NAME=crystalite_synthetic_aug_n2500_raw,FAMILIES="validity novelty uniqueness diversity hhi" \
#     scripts/run_lemat_genbench.slurm

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

export PATH="$HOME/.local/bin:$PATH"

ATOM_REPS_ROOT="${ATOM_REPS_ROOT:-$HOME/atom-reps}"
LEMAT_ROOT="${LEMAT_ROOT:-$HOME/lemat-genbench}"
CONFIG="${CONFIG:-comprehensive_multi_mlip_hull}"
CIFS="${CIFS:-}"
NAME="${NAME:-}"
FAMILIES="${FAMILIES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REQUIRE_HF_AUTH="${REQUIRE_HF_AUTH:-1}"
HF_LOGIN_IN_JOB="${HF_LOGIN_IN_JOB:-1}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

mkdir -p "$ATOM_REPS_ROOT/logs/lemat_genbench"

if [[ -z "$CIFS" ]]; then
  echo "[lemat] ERROR: set CIFS=/path/to/cif_directory" >&2
  exit 2
fi
if [[ -z "$NAME" ]]; then
  echo "[lemat] ERROR: set NAME=run_name" >&2
  exit 2
fi
if [[ ! -d "$CIFS" && ! -f "$CIFS" ]]; then
  echo "[lemat] ERROR: CIFS path not found: $CIFS" >&2
  exit 2
fi
if [[ ! -d "$LEMAT_ROOT" ]]; then
  echo "[lemat] ERROR: LeMat-GenBench checkout not found: $LEMAT_ROOT" >&2
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
if [[ "$REQUIRE_HF_AUTH" == "1" && -z "${HF_TOKEN:-}" ]]; then
  echo "[lemat] ERROR: HF auth is required but no token was found." >&2
  echo "[lemat] Set HF_TOKEN, HUGGINGFACE_HUB_TOKEN, or HF_TOKEN_FILE=$HF_TOKEN_FILE." >&2
  exit 2
fi

export HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

JOB_ID="${SLURM_JOB_ID:-manual}"
export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-$JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

cd "$LEMAT_ROOT"

uv run python - <<'PY'
import os
import sys

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
require = os.environ.get("REQUIRE_HF_AUTH", "1") == "1"
if not token:
    print("[lemat] HF auth: no token configured")
    sys.exit(2 if require else 0)

try:
    from huggingface_hub import HfApi, login

    if os.environ.get("HF_LOGIN_IN_JOB", "1") == "1":
        login(token=token, add_to_git_credential=False)
    user = HfApi(token=token).whoami().get("name", "<unknown>")
    print(f"[lemat] HF auth: authenticated as {user}")
    print(f"[lemat] HF auth: token written under {os.environ.get('HF_HOME')}")
except Exception as exc:
    print(f"[lemat] HF auth check failed: {exc}", file=sys.stderr)
    sys.exit(2 if require else 0)
PY

FAMILY_ARGS=()
if [[ -n "$FAMILIES" ]]; then
  # shellcheck disable=SC2206
  FAMILY_LIST=($FAMILIES)
  FAMILY_ARGS+=(--families "${FAMILY_LIST[@]}")
fi

EXTRA_ARG_ARRAY=()
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARG_ARRAY=($EXTRA_ARGS)
fi

echo "[lemat] root=$LEMAT_ROOT"
echo "[lemat] cifs=$CIFS"
echo "[lemat] config=$CONFIG"
echo "[lemat] name=$NAME"
echo "[lemat] families=${FAMILIES:-<all>}"
echo "[lemat] extra_args=${EXTRA_ARGS:-<none>}"
echo "[lemat] hf_home=$HF_HOME"
echo "[lemat] hf_datasets_cache=$HF_DATASETS_CACHE"
echo "[lemat] hf_hub_cache=$HF_HUB_CACHE"

srun uv run scripts/run_benchmarks.py \
  --cifs "$CIFS" \
  --config "$CONFIG" \
  --name "$NAME" \
  "${FAMILY_ARGS[@]}" \
  "${EXTRA_ARG_ARRAY[@]}"
