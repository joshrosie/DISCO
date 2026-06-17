#!/usr/bin/env bash
# Provision the external EquiformerV3 env under external/equiformer_v3/.
#
# Idempotent: safe to re-run. Skips clone, install, and checkpoint download
# if already done. Pinned to a specific upstream commit; update COMMIT_HASH
# when intentionally bumping the EquiformerV3 version.
#
# Usage:
#   bash scripts/setup_equiformer_v3.sh             # auto-detect: CUDA on Linux, CPU on macOS
#   bash scripts/setup_equiformer_v3.sh --cpu       # force CPU install (e.g., on Linux box w/o GPU)
#   bash scripts/setup_equiformer_v3.sh --cuda      # force CUDA install
#
# CPU mode is fine for smoke tests and small inference workloads. Full hull
# build (60–80k MP entries) or large curation runs want the CUDA build on the
# cluster — EquiformerV3-OAM is 30M params with lmax=4, slow on CPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="$REPO_ROOT/external"
EQV3_DIR="$EXT_DIR/equiformer_v3"
VENV_DIR="$EQV3_DIR/.venv"
CKPT_DIR="$EQV3_DIR/checkpoints"
CKPT_FILE="$CKPT_DIR/omat24-mptrj-salex_gradient.pt"

COMMIT_HASH="a7300c58df683dc99cb48027d5bfd4c887486c48"
CKPT_URL="https://huggingface.co/mirror-physics/equiformer_v3/resolve/main/checkpoint/omat24-mptrj-salex_gradient.pt"

log() { echo "[setup-equiformer-v3] $*"; }

# 1. Clone (or pull) the upstream repo and pin to COMMIT_HASH.
mkdir -p "$EXT_DIR"
if [[ ! -d "$EQV3_DIR/.git" ]]; then
  log "cloning atomicarchitects/equiformer_v3 → $EQV3_DIR"
  git clone https://github.com/atomicarchitects/equiformer_v3 "$EQV3_DIR"
else
  log "repo present; fetching to ensure pinned commit is available"
  git -C "$EQV3_DIR" fetch --quiet
fi
log "checking out pinned commit $COMMIT_HASH"
git -C "$EQV3_DIR" checkout --quiet "$COMMIT_HASH"

# 2. Pick the torch / PyG wheel variant.
#    Default: CUDA 12.8 on Linux, CPU on macOS (auto). Override with --cpu / --cuda.
TORCH_VARIANT="auto"
for arg in "$@"; do
  case "$arg" in
    --cpu)  TORCH_VARIANT="cpu" ;;
    --cuda) TORCH_VARIANT="cuda" ;;
    *)
      log "unknown argument: $arg (expected --cpu or --cuda)" >&2
      exit 1
      ;;
  esac
done
if [[ "$TORCH_VARIANT" == "auto" ]]; then
  if [[ "$(uname -s)" == "Linux" ]]; then
    TORCH_VARIANT="cuda"
  else
    TORCH_VARIANT="cpu"
  fi
fi

case "$TORCH_VARIANT" in
  cuda)
    TORCH_INDEX_ARGS=(--index-url https://download.pytorch.org/whl/cu128)
    PYG_FIND_LINKS="https://data.pyg.org/whl/torch-2.7.0+cu128.html"
    ;;
  cpu)
    # CPU wheels ship from PyPI by default; specifying explicitly keeps the array
    # non-empty so `"${TORCH_INDEX_ARGS[@]}"` is safe under `set -u`.
    TORCH_INDEX_ARGS=(--index-url https://pypi.org/simple)
    PYG_FIND_LINKS="https://data.pyg.org/whl/torch-2.7.0+cpu.html"
    ;;
esac
log "torch variant: $TORCH_VARIANT"

# 3. Create the venv if missing.
if [[ ! -d "$VENV_DIR" ]]; then
  log "creating uv venv at $VENV_DIR (Python 3.11)"
  ( cd "$EQV3_DIR" && uv venv --python 3.11 )
else
  log "venv exists at $VENV_DIR"
fi

# 4. Install torch + PyG + their requirements + vendored fairchem.
#    Each `uv pip install` is idempotent — re-running is a no-op if everything
#    is already present at the requested versions.
#    IMPORTANT: pass --python explicitly. Without it, uv discovery walks up
#    looking for a project and lands on the outer atom-reps .venv (Python
#    3.12), polluting the wrong env.
VENV_PY="$VENV_DIR/bin/python"

log "installing torch 2.7.1 ($TORCH_VARIANT wheels)"
( cd "$EQV3_DIR" && uv pip install --quiet --python "$VENV_PY" \
  torch==2.7.1 torchvision==0.22.1 "${TORCH_INDEX_ARGS[@]}" )

log "installing PyG ops (pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv)"
( cd "$EQV3_DIR" && uv pip install --quiet --python "$VENV_PY" \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  --find-links "$PYG_FIND_LINKS" )

log "installing torch_geometric"
( cd "$EQV3_DIR" && uv pip install --quiet --python "$VENV_PY" torch_geometric )

log "installing experimental/env/conda_requirements.txt (pip-format despite the name)"
# `triton` is Linux-only (no macOS wheels exist). Filter it out on darwin.
# Triton is only used by torch.compile()'s GPU kernels — not needed for CPU inference.
if [[ "$(uname -s)" == "Darwin" ]]; then
  REQ_FILE_FILTERED="$EQV3_DIR/experimental/env/conda_requirements_no_triton.txt"
  grep -vE "^triton(==|$|\s)" "$EQV3_DIR/experimental/env/conda_requirements.txt" > "$REQ_FILE_FILTERED"
  ( cd "$EQV3_DIR" && uv pip install --quiet --python "$VENV_PY" -r "$REQ_FILE_FILTERED" )
else
  ( cd "$EQV3_DIR" && uv pip install --quiet --python "$VENV_PY" -r experimental/env/conda_requirements.txt )
fi

log "installing vendored fairchem-core in editable mode"
( cd "$EQV3_DIR" && uv pip install --quiet --python "$VENV_PY" -e packages/fairchem-core )

# 5. Download the OAM checkpoint (skip if already present and non-trivially-sized).
mkdir -p "$CKPT_DIR"
if [[ -f "$CKPT_FILE" && $(stat -c%s "$CKPT_FILE" 2>/dev/null || stat -f%z "$CKPT_FILE") -gt 1000000 ]]; then
  log "checkpoint present at $CKPT_FILE ($(du -h "$CKPT_FILE" | cut -f1))"
else
  log "downloading OAM checkpoint → $CKPT_FILE"
  curl -L --fail --progress-bar -o "$CKPT_FILE" "$CKPT_URL"
fi

# 6. Freeze the resolved env so future re-provisioning is bit-stable.
log "freezing env lockfile"
( cd "$EQV3_DIR" && uv pip freeze --python "$VENV_PY" > equiformer_v3_requirements_lock.txt )

log "done."
log "  repo:        $EQV3_DIR"
log "  python:      $VENV_DIR/bin/python ($("$VENV_DIR/bin/python" --version 2>&1))"
log "  checkpoint:  $CKPT_FILE"
log "  lockfile:    $EQV3_DIR/equiformer_v3_requirements_lock.txt"
