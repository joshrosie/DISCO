#!/usr/bin/env python
"""Relax token/CIF samples with NequIP and export relaxed CIFs.

This is for external-eval variants where relaxation is considered part of the
generation pipeline. Keep the output directory label explicit, e.g.
``relaxed_cifs`` or ``nequip_relaxed_cifs``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.mp20_tokens import tokens_to_structure
from src.eval.oracles import NequipFormEnergyOracle


def _load_samples(path: Path) -> list[Structure]:
    if path.is_dir():
        samples_pt = path / "samples.pt"
        if samples_pt.exists():
            path = samples_pt
        else:
            cif_dir = path / "cifs"
            if not cif_dir.exists():
                cif_dir = path
            structs: list[Structure] = []
            for cif_path in sorted(cif_dir.glob("*.cif")):
                structs.append(Structure.from_file(cif_path))
            return structs

    if path.suffix == ".pt":
        records = torch.load(path, map_location="cpu", weights_only=False)
        return [tokens_to_structure(rec) for rec in records]
    if path.suffix == ".cif":
        return [Structure.from_file(path)]
    raise ValueError(f"Unsupported sample input: {path}")


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relax samples with NequIP and export relaxed CIFs."
    )
    parser.add_argument("--samples", required=True, help="samples.pt, output dir, CIF, or CIF dir.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nequip_compile_path", required=True)
    parser.add_argument("--thermo_ppd_mp", required=True)
    parser.add_argument("--thermo_stability_device", default="cuda")
    parser.add_argument("--thermo_ehull_method", choices=["uncorrected", "mp2020_like"], default="mp2020_like")
    parser.add_argument("--thermo_relax_steps", type=int, default=200)
    parser.add_argument("--thermo_stability_batch", type=int, default=32)
    parser.add_argument("--nequip_relax_mode", choices=["sequential", "batch"], default="batch")
    parser.add_argument("--nequip_optimizer", default="FIRE")
    parser.add_argument("--nequip_cell_filter", choices=["none", "frechet", "exp"], default="frechet")
    parser.add_argument("--nequip_fmax", type=float, default=0.005)
    parser.add_argument("--nequip_max_force_abort", type=float, default=1e6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sample_path = Path(args.samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    structs = _load_samples(sample_path)
    if int(args.limit) > 0:
        structs = structs[: int(args.limit)]
    if not structs:
        raise SystemExit(f"No structures loaded from {sample_path}")

    oracle = NequipFormEnergyOracle(
        nequip_compile_path=str(args.nequip_compile_path),
        ppd_path=str(args.thermo_ppd_mp),
        stability_device=str(args.thermo_stability_device),
        optimizer=str(args.nequip_optimizer),
        cell_filter=str(args.nequip_cell_filter),
        fmax=float(args.nequip_fmax),
        max_force_abort=float(args.nequip_max_force_abort),
        relax_steps=int(args.thermo_relax_steps),
        apply_mp2020=str(args.thermo_ehull_method) == "mp2020_like",
        relax_mode=str(args.nequip_relax_mode),
        batch_size=int(args.thermo_stability_batch),
    )

    manifest = []
    n_success = 0
    n_failed = 0
    for start in tqdm(range(0, len(structs), int(args.thermo_stability_batch)), desc="relax"):
        batch = structs[start : start + int(args.thermo_stability_batch)]
        results = oracle.call_many(batch)
        for offset, (initial, result) in enumerate(zip(batch, results, strict=True)):
            idx = start + offset
            final = result.get("final_structure")
            err = result.get("err") or ""
            success = final is not None and not str(err).startswith("relax_exc")
            row = {
                "sample_idx": idx,
                "success": bool(success),
                "initial_formula": initial.composition.reduced_formula,
                "relaxed_formula": final.composition.reduced_formula if final is not None else None,
                "num_sites": int(len(final)) if final is not None else None,
                "e_form": _finite_float(result.get("e_form")),
                "e_above_hull": _finite_float(result.get("e_above_hull")),
                "e_total": _finite_float(result.get("e_total")),
                "nsteps": int(result.get("nsteps", -1)),
                "err": err or None,
            }
            if success:
                cif_name = f"sample_{idx:05d}.cif"
                (output_dir / cif_name).write_text(str(CifWriter(final)), encoding="utf-8")
                row["file"] = cif_name
                n_success += 1
            else:
                n_failed += 1
            manifest.append(row)

    summary = {
        "input": str(sample_path),
        "output_dir": str(output_dir),
        "num_input": len(structs),
        "num_success": int(n_success),
        "num_failed": int(n_failed),
        "nequip_compile_path": str(args.nequip_compile_path),
        "thermo_ppd_mp": str(args.thermo_ppd_mp),
        "thermo_relax_steps": int(args.thermo_relax_steps),
        "thermo_stability_batch": int(args.thermo_stability_batch),
        "nequip_relax_mode": str(args.nequip_relax_mode),
        "nequip_cell_filter": str(args.nequip_cell_filter),
        "nequip_fmax": float(args.nequip_fmax),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
