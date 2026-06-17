#!/usr/bin/env python
"""Local smoke test: load OAM checkpoint, single-point a few structures.

Run via the inner equiformer_v3 venv (NOT the main repo's venv):

    external/equiformer_v3/.venv/bin/python scripts/smoke_equiformer_v3.py

Passes if every test structure returns a finite, sane-magnitude energy + force.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ase.build import bulk
from fairchem.core import OCPCalculator


REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = REPO_ROOT / "external/equiformer_v3/checkpoints/omat24-mptrj-salex_gradient.pt"


def expect_finite(name: str, value: float, abs_max: float = 1e6) -> bool:
    import math
    ok = math.isfinite(value) and abs(value) < abs_max
    status = "OK " if ok else "BAD"
    print(f"   [{status}] {name}: {value:+.6f}")
    return ok


def smoke_one(calc: OCPCalculator, label: str, atoms) -> bool:
    print(f"\n--- {label} ({len(atoms)} atoms, {atoms.symbols}) ---")
    atoms.calc = calc
    t0 = time.perf_counter()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    elapsed = time.perf_counter() - t0

    ok_e = expect_finite("energy (eV)", float(energy))
    ok_ea = expect_finite("energy/atom (eV/atom)", float(energy) / len(atoms))
    ok_fmax = expect_finite(
        "max force component (eV/Å)",
        float(forces.max()),
        abs_max=1e3,
    )
    print(f"   timing: {elapsed:.2f}s ({elapsed / len(atoms):.2f}s/atom)")
    return ok_e and ok_ea and ok_fmax


def main() -> int:
    if not CKPT_PATH.exists():
        print(f"FAIL: checkpoint not found at {CKPT_PATH}", file=sys.stderr)
        print("Run `bash scripts/setup_equiformer_v3.sh` first.", file=sys.stderr)
        return 1

    print(f"loading OAM checkpoint: {CKPT_PATH.name}")
    t0 = time.perf_counter()
    import torch
    cpu = not torch.cuda.is_available()
    calc = OCPCalculator(checkpoint_path=str(CKPT_PATH), cpu=cpu, seed=0)
    print(f"loaded in {time.perf_counter() - t0:.1f}s; cpu={cpu}")

    # Diverse small structures from ASE's bulk builder.
    cases = [
        ("NaCl rocksalt",  bulk("NaCl", "rocksalt", a=5.64)),
        ("Si diamond",     bulk("Si",   "diamond",  a=5.43)),
        ("Fe BCC",         bulk("Fe",   "bcc",      a=2.87)),
        ("Cu FCC",         bulk("Cu",   "fcc",      a=3.61)),
        ("MgO rocksalt",   bulk("MgO",  "rocksalt", a=4.21)),
    ]

    all_ok = True
    for label, atoms in cases:
        try:
            ok = smoke_one(calc, label, atoms)
        except Exception as exc:
            print(f"   [BAD] exception: {type(exc).__name__}: {exc}")
            ok = False
        all_ok = all_ok and ok

    print("\n" + ("=" * 40))
    if all_ok:
        print("SMOKE PASS: every structure returned finite energy + forces.")
        return 0
    print("SMOKE FAIL: see [BAD] markers above.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
