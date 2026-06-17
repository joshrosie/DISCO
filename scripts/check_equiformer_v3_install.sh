#!/usr/bin/env bash
# Verify that:
#   1. The outer atom-reps .venv is intact (torch 2.8.0, no equiformer pollution).
#   2. The inner external/equiformer_v3/.venv has the right packages.
#   3. The OAM checkpoint is present.
# Prints [OK] / [FAIL] per check. Exits 0 if all pass, 1 otherwise.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PASS=0
FAIL=0

ok()   { echo "  [OK]   $*"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }

echo "=== outer venv (atom-reps) ==="

OUTER_PY=".venv/bin/python"
if [[ ! -x "$OUTER_PY" ]]; then
  fail "outer venv not found at $OUTER_PY"
else
  outer_torch=$("$OUTER_PY" -c "import torch; print(torch.__version__)" 2>&1 || echo "IMPORT_FAIL")
  if [[ "$outer_torch" == 2.8.* ]]; then
    ok "outer torch: $outer_torch"
  else
    fail "outer torch: $outer_torch (expected 2.8.x; equiformer install may have downgraded it)"
  fi

  # Check for equiformer-pollution stragglers in the outer venv.
  stragglers=$(
    ls .venv/lib/python*/site-packages/ 2>/dev/null \
      | grep -iE "^(fairchem|pyg_lib|torch_geometric|torch_scatter|torch_sparse|torch_cluster|torch_spline_conv)" \
      || true
  )
  if [[ -z "$stragglers" ]]; then
    ok "outer venv clean (no equiformer stragglers)"
  else
    fail "outer venv contaminated; run: uv pip uninstall <pkg>:"
    echo "$stragglers" | sed 's/^/         /'
  fi

  # Sanity: nequip still importable from outer venv.
  if "$OUTER_PY" -c "import nequip" 2>/dev/null; then
    ok "outer venv: nequip importable"
  else
    fail "outer venv: nequip not importable (main pipeline broken)"
  fi
fi

echo ""
echo "=== inner venv (equiformer_v3) ==="

INNER_PY="external/equiformer_v3/.venv/bin/python"
if [[ ! -x "$INNER_PY" ]]; then
  fail "inner venv not found at $INNER_PY (run setup_equiformer_v3.sh)"
else
  inner_pyver=$("$INNER_PY" --version 2>&1)
  if [[ "$inner_pyver" == *"3.11"* ]]; then
    ok "inner python: $inner_pyver"
  else
    fail "inner python: $inner_pyver (expected 3.11.x)"
  fi

  inner_torch=$("$INNER_PY" -c "import torch; print(torch.__version__)" 2>&1 || echo "IMPORT_FAIL")
  if [[ "$inner_torch" == 2.7.* ]]; then
    ok "inner torch: $inner_torch"
  else
    fail "inner torch: $inner_torch (expected 2.7.x)"
  fi

  # Distinguish "CUDA build" (torch.version.cuda set) from "CUDA visible right now"
  # (is_available()). Login nodes have the right build but no GPU visible — that's
  # fine and not a real failure.
  cuda_build=$("$INNER_PY" -c "import torch; print(torch.version.cuda)" 2>&1 || echo "IMPORT_FAIL")
  cuda_status=$("$INNER_PY" -c "import torch; print(torch.cuda.is_available())" 2>&1 || echo "IMPORT_FAIL")
  if [[ "$(uname -s)" == "Linux" ]]; then
    if [[ "$cuda_build" == "None" || "$cuda_build" == "" ]]; then
      fail "inner torch built without CUDA on Linux (got CPU wheels; re-run setup with --cuda)"
    elif [[ "$cuda_status" == "True" ]]; then
      ok "inner cuda: build=$cuda_build, available=True (on GPU node)"
    else
      ok "inner cuda: build=$cuda_build, available=False (no GPU visible — likely login node; install is fine)"
    fi
  else
    ok "inner cuda: build=$cuda_build, available=$cuda_status (CPU expected on darwin)"
  fi

  for mod in e3nn fairchem torch_geometric ase pymatgen; do
    if "$INNER_PY" -c "import $mod" 2>/dev/null; then
      ok "inner: $mod importable"
    else
      fail "inner: $mod not importable"
    fi
  done
fi

echo ""
echo "=== checkpoint ==="

CKPT="external/equiformer_v3/checkpoints/omat24-mptrj-salex_gradient.pt"
if [[ -f "$CKPT" ]]; then
  size=$(du -h "$CKPT" | cut -f1)
  size_bytes=$(stat -c%s "$CKPT" 2>/dev/null || stat -f%z "$CKPT" 2>/dev/null || echo 0)
  if [[ $size_bytes -gt 100000000 ]]; then  # > 100 MB
    ok "checkpoint present: $CKPT ($size)"
  else
    fail "checkpoint suspiciously small: $CKPT ($size); possibly truncated download"
  fi
else
  fail "checkpoint missing: $CKPT"
fi

echo ""
echo "=== summary ==="
echo "passed: $PASS"
echo "failed: $FAIL"
if [[ $FAIL -gt 0 ]]; then
  echo "result: NOT OK — see [FAIL] lines above"
  exit 1
fi
echo "result: ALL CHECKS PASSED"
