#!/usr/bin/env python
"""EquiformerV3 inference wrapper — relax a directory of CIFs.

Runs inside the inner equiformer_v3 venv. The main repo's curator invokes
this as a subprocess. File-based IPC keeps the two envs cleanly isolated
(no shared imports, no daemon lifecycle).

Reads:
    --input-dir/*.cif                   (any CIF files)

Writes:
    --output-dir/<basename>.cif         (relaxed structure for each input,
                                         only if relaxation succeeded)
    --output-dir/manifest.jsonl         (one JSON line per input)

Manifest line schema:

    {
        "sample_id":       "sample_0000",
        "input_path":      "input/sample_0000.cif",
        "output_path":     "output/sample_0000.cif" | null,
        "success":         true | false,
        "err":             null | "exception_class_name",
        "initial_formula": "Sr2NbInO6",
        "relaxed_formula": "Sr2NbInO6" | null,
        "num_sites":       40,
        "e_total":         -284.628 | null,
        "nsteps":          73 | null,
        "max_force_init":  0.117 | null,        // initial max-force (before relax)
        "max_force_final": 0.018 | null,        // final max-force (after relax)
        "relax_time_s":    12.4 | null
    }

Hyperparameters default to the matbench-discovery IS2RE-SR protocol:
FIRE optimizer, FrechetCellFilter, fmax=0.02 eV/Å, max_steps=500,
graph_construction_radius=6 Å. Override via CLI flags if needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.optimize import FIRE, LBFGS

FILTER_CLS = {"frechet": FrechetCellFilter, "unit": UnitCellFilter, "none": None}
OPTIM_CLS = {"FIRE": FIRE, "LBFGS": LBFGS}


def _max_force(forces) -> float:
    a = np.asarray(forces, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    return float(np.abs(a).max())


def _load_calculator(checkpoint: Path, device: str):
    import torch
    from fairchem.core import OCPCalculator

    if device == "auto":
        cpu = not torch.cuda.is_available()
    elif device == "cpu":
        cpu = True
    elif device in ("cuda", "gpu"):
        cpu = False
    else:
        raise ValueError(f"unknown device: {device!r}")
    print(f"[wrapper] loading checkpoint {checkpoint.name}; cpu={cpu}", flush=True)
    t0 = time.perf_counter()
    calc = OCPCalculator(checkpoint_path=str(checkpoint), cpu=cpu, seed=0)
    print(f"[wrapper] checkpoint loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    return calc


def relax_one(atoms, calc, *, optim_cls, filter_cls, fmax: float, max_steps: int) -> dict:
    """Relax one ASE Atoms object. Returns a partially-populated manifest row.

    Records the initial max-force, runs FIRE+FrechetCellFilter, records final
    max-force + e_total + steps + timing. Failures land in the err field;
    the caller decides how to handle them.
    """
    row: dict = {
        "success": False,
        "err": None,
        "initial_formula": str(atoms.symbols),
        "relaxed_formula": None,
        "num_sites": len(atoms),
        "e_total": None,
        "nsteps": None,
        "max_force_init": None,
        "max_force_final": None,
        "relax_time_s": None,
    }
    atoms.calc = calc
    t0 = time.perf_counter()
    try:
        forces_init = atoms.get_forces()
        row["max_force_init"] = _max_force(forces_init)

        wrapped = filter_cls(atoms) if filter_cls is not None else atoms
        opt = optim_cls(wrapped, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        row["nsteps"] = int(opt.nsteps)

        e_total = float(atoms.get_potential_energy())
        forces_final = atoms.get_forces()
        row["e_total"] = e_total
        row["max_force_final"] = _max_force(forces_final)
        row["relaxed_formula"] = str(atoms.symbols)
        row["success"] = True
    except Exception as exc:
        row["err"] = f"{type(exc).__name__}: {exc}"
        print(
            f"[wrapper] relax failed: {row['err']}\n"
            + traceback.format_exc(limit=2),
            flush=True,
            file=sys.stderr,
        )
    row["relax_time_s"] = time.perf_counter() - t0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--cell-filter", default="frechet", choices=list(FILTER_CLS.keys()))
    parser.add_argument("--optimizer", default="FIRE", choices=list(OPTIM_CLS.keys()))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "gpu"])
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"[wrapper] FATAL: input dir missing: {args.input_dir}", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"[wrapper] FATAL: checkpoint missing: {args.checkpoint}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cif_paths = sorted(args.input_dir.glob("*.cif"))
    print(f"[wrapper] {len(cif_paths)} input CIFs", flush=True)
    if not cif_paths:
        (args.output_dir / "manifest.jsonl").write_text("")
        return 0

    calc = _load_calculator(args.checkpoint, args.device)
    filter_cls = FILTER_CLS[args.cell_filter]
    optim_cls = OPTIM_CLS[args.optimizer]

    manifest_path = args.output_dir / "manifest.jsonl"
    n_success = 0
    n_fail = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for cif_path in cif_paths:
            sample_id = cif_path.stem
            row: dict = {
                "sample_id": sample_id,
                "input_path": str(cif_path.relative_to(args.input_dir.parent)),
                "output_path": None,
            }
            try:
                atoms = ase_read(str(cif_path))
            except Exception as exc:
                row.update(
                    success=False,
                    err=f"{type(exc).__name__}: {exc}",
                    initial_formula=None,
                    relaxed_formula=None,
                    num_sites=None,
                    e_total=None,
                    nsteps=None,
                    max_force_init=None,
                    max_force_final=None,
                    relax_time_s=None,
                )
                manifest_f.write(json.dumps(row) + "\n")
                manifest_f.flush()
                n_fail += 1
                print(
                    f"[wrapper] {sample_id}: parse_fail ({row['err']})",
                    flush=True,
                    file=sys.stderr,
                )
                continue

            relax_row = relax_one(
                atoms,
                calc,
                optim_cls=optim_cls,
                filter_cls=filter_cls,
                fmax=args.fmax,
                max_steps=args.max_steps,
            )
            row.update(relax_row)

            if row["success"]:
                out_path = args.output_dir / f"{sample_id}.cif"
                ase_write(str(out_path), atoms, format="cif")
                row["output_path"] = str(out_path.relative_to(args.output_dir.parent))
                n_success += 1
            else:
                n_fail += 1

            manifest_f.write(json.dumps(row) + "\n")
            manifest_f.flush()
            print(
                f"[wrapper] {sample_id}: "
                f"{'ok' if row['success'] else 'FAIL'} "
                f"n={row.get('num_sites')} "
                f"e={row.get('e_total')} "
                f"nsteps={row.get('nsteps')} "
                f"t={row.get('relax_time_s'):.2f}s"
                if row.get("relax_time_s") is not None
                else f"[wrapper] {sample_id}: parse_fail",
                flush=True,
            )

    print(f"[wrapper] done. n_success={n_success} n_fail={n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
