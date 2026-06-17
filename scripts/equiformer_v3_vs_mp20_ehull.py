#!/usr/bin/env python
"""Compare EquiformerV3-predicted e_above_hull to MP-20's stored labels.

For N random MP-20 entries:
  1. Read the CIF and stored mp20_e_hull (DFT ground truth)
  2. Single-point with EquiformerV3 (OAM checkpoint) → eqv3_e_total
  3. Score against the existing MP DFT hull pickle under TWO conditions:
       A: raw eqv3 energy, no corrections        → eqv3_e_hull_raw
       B: eqv3 energy + MP2020 corrections        → eqv3_e_hull_mp2020
  4. Write a per-entry CSV: sample_id, formula, num_atoms,
     mp20_e_hull, eqv3_e_total, eqv3_e_hull_raw, eqv3_e_hull_mp2020

The output CSV is the durable artifact. Plotting/analysis is downstream and
can be re-run cheaply against the CSV without re-running inference.

This script MUST be run via the inner equiformer_v3 venv so fairchem is
importable. The outer atom-reps venv won't have fairchem:

    external/equiformer_v3/.venv/bin/python \
        scripts/equiformer_v3_vs_mp20_ehull.py \
        --limit 500 \
        --output results/equiformer_v3_vs_mp20_n500.csv

Local CPU at n=500 takes ~30 min on Apple Silicon. Cluster CUDA at n=45000
(`--limit -1`) takes ~4 hours.
"""
from __future__ import annotations

import argparse
import csv
import math
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data/mp20/raw/all.csv"
DEFAULT_PPD = REPO_ROOT / "mp_02072023/2023-02-07-ppd-mp.pkl"
DEFAULT_CKPT = REPO_ROOT / "external/equiformer_v3/checkpoints/omat24-mptrj-salex_gradient.pt"
DEFAULT_OUT = REPO_ROOT / "results/equiformer_v3_vs_mp20.csv"


def _build_mp2020_parameters(composition: Any, mp2020_compat: Any) -> dict[str, Any]:
    """Synthetic metadata MP2020 compatibility needs to apply corrections.

    Mirrored from src/utils/sample_stats.py:_build_mp2020_parameters so this
    script is self-contained inside the inner venv.
    """
    sorted_elements = sorted(
        (el for el in composition.elements if composition[el] > 0),
        key=lambda el: el.X,
    )
    most_electroneg = sorted_elements[-1].symbol if sorted_elements else None
    u_settings = mp2020_compat.u_settings.get(most_electroneg, {})
    hubbards = {
        el.symbol: float(u_settings.get(el.symbol, 0.0))
        for el in composition.elements
        if float(u_settings.get(el.symbol, 0.0)) != 0.0
    }
    run_type = "GGA+U" if hubbards else "GGA"
    return {"run_type": run_type, "hubbards": hubbards, "software": "vasp"}


def score_uncorrected(ppd, structure: Structure, e_total: float) -> float | None:
    if structure.num_sites <= 0 or not math.isfinite(e_total):
        return None
    try:
        e_hull_per_atom = ppd.get_hull_energy_per_atom(structure.composition)
    except Exception:
        return None
    if not (math.isfinite(e_hull_per_atom) and math.isfinite(e_total)):
        return None
    e_above = (e_total / structure.num_sites) - e_hull_per_atom
    return float(e_above) if math.isfinite(e_above) else None


def score_mp2020(
    ppd, structure: Structure, e_total: float, mp2020_compat: Any
) -> float | None:
    if not math.isfinite(e_total):
        return None
    try:
        params = _build_mp2020_parameters(structure.composition, mp2020_compat)
        entry = ComputedStructureEntry(
            composition=structure.composition,
            energy=float(e_total),
            structure=structure,
            parameters=params,
        )
        corrected = mp2020_compat.process_entry(entry.copy(), on_error="raise")
        if corrected is None:
            return None
        e_above = float(ppd.get_e_above_hull(corrected, allow_negative=True))
    except Exception:
        return None
    return e_above if math.isfinite(e_above) else None


def load_calculator(ckpt_path: Path):
    """Load OCPCalculator (fairchem v1 API the vendored fork exposes)."""
    import torch
    from fairchem.core import OCPCalculator

    cpu = not torch.cuda.is_available()
    print(f"loading EquiformerV3 OAM ({ckpt_path.name}), cpu={cpu}", flush=True)
    t0 = time.perf_counter()
    calc = OCPCalculator(checkpoint_path=str(ckpt_path), cpu=cpu, seed=0)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    return calc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--ppd", type=Path, default=DEFAULT_PPD)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of MP-20 entries to evaluate (-1 = all).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument(
        "--cif-column",
        default="cif.conv",
        choices=["cif.conv", "cif"],
        help="Which CIF column to use (cif.conv = conventional cell, what Crystalite saw).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"FATAL: --csv not found: {args.csv}", file=sys.stderr)
        return 1
    if not args.ppd.exists():
        print(f"FATAL: --ppd not found: {args.ppd}", file=sys.stderr)
        return 1
    if not args.checkpoint.exists():
        print(f"FATAL: --checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading MP-20 entries from {args.csv}", flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(
        args.csv,
        usecols=["material_id", "pretty_formula", "e_above_hull", args.cif_column],
    )
    # Rename the CIF column so it's a valid Python identifier (itertuples is
    # finicky with dotted names).
    df = df.rename(columns={args.cif_column: "cif_str"})
    print(f"  loaded {len(df)} rows in {time.perf_counter() - t0:.1f}s", flush=True)

    if args.limit > 0 and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
        print(f"  subsampled to {len(df)} entries (seed={args.seed})", flush=True)

    print(f"loading PPD pickle from {args.ppd}", flush=True)
    t0 = time.perf_counter()
    with args.ppd.open("rb") as f:
        ppd = pickle.load(f)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    mp2020_compat = MaterialsProject2020Compatibility(check_potcar=False)
    calc = load_calculator(args.checkpoint)
    adaptor = AseAtomsAdaptor()

    rows: list[dict] = []
    n_eqv3_fail = 0
    n_parse_fail = 0
    n_score_fail = 0
    print(f"running EquiformerV3 single-points on {len(df)} structures...", flush=True)

    bar = tqdm(df.itertuples(index=False), total=len(df), dynamic_ncols=True)
    for record in bar:
        sample_id = getattr(record, "material_id", None)
        formula = getattr(record, "pretty_formula", None)
        mp20_e_hull = getattr(record, "e_above_hull", float("nan"))
        cif_str = record.cif_str

        try:
            structure = Structure.from_str(cif_str, fmt="cif")
        except Exception as exc:
            n_parse_fail += 1
            bar.set_postfix(parse_fail=n_parse_fail, eqv3_fail=n_eqv3_fail)
            continue

        try:
            atoms = adaptor.get_atoms(structure)
            atoms.calc = calc
            e_total = float(atoms.get_potential_energy())
        except Exception as exc:
            n_eqv3_fail += 1
            bar.set_postfix(parse_fail=n_parse_fail, eqv3_fail=n_eqv3_fail)
            continue

        e_hull_raw = score_uncorrected(ppd, structure, e_total)
        e_hull_mp2020 = score_mp2020(ppd, structure, e_total, mp2020_compat)
        if e_hull_raw is None and e_hull_mp2020 is None:
            n_score_fail += 1

        rows.append(
            {
                "sample_id": sample_id,
                "formula": formula,
                "num_atoms": int(structure.num_sites),
                "mp20_e_hull": float(mp20_e_hull) if mp20_e_hull is not None else float("nan"),
                "eqv3_e_total": e_total,
                "eqv3_e_per_atom": e_total / max(1, structure.num_sites),
                "eqv3_e_hull_raw": e_hull_raw if e_hull_raw is not None else float("nan"),
                "eqv3_e_hull_mp2020": e_hull_mp2020 if e_hull_mp2020 is not None else float("nan"),
            }
        )
        bar.set_postfix(parse_fail=n_parse_fail, eqv3_fail=n_eqv3_fail, score_fail=n_score_fail)

    bar.close()

    print(
        f"\ndone. parse_fail={n_parse_fail} eqv3_fail={n_eqv3_fail} "
        f"score_fail={n_score_fail} written={len(rows)}",
        flush=True,
    )
    if not rows:
        print("FATAL: no successful rows; refusing to write empty CSV.", file=sys.stderr)
        return 2

    keys = list(rows[0].keys())
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {args.output} ({len(rows)} rows)", flush=True)

    arr_mp20 = np.array([r["mp20_e_hull"] for r in rows])
    arr_raw = np.array([r["eqv3_e_hull_raw"] for r in rows])
    arr_mp2020 = np.array([r["eqv3_e_hull_mp2020"] for r in rows])
    mask_raw = np.isfinite(arr_mp20) & np.isfinite(arr_raw)
    mask_mp2020 = np.isfinite(arr_mp20) & np.isfinite(arr_mp2020)
    if mask_raw.sum() > 1:
        rmse_raw = float(np.sqrt(np.mean((arr_raw[mask_raw] - arr_mp20[mask_raw]) ** 2)))
        bias_raw = float(np.mean(arr_raw[mask_raw] - arr_mp20[mask_raw]))
        print(f"  raw    : n={mask_raw.sum()} RMSE={rmse_raw:.4f}  bias={bias_raw:+.4f}")
    if mask_mp2020.sum() > 1:
        rmse_mp = float(
            np.sqrt(np.mean((arr_mp2020[mask_mp2020] - arr_mp20[mask_mp2020]) ** 2))
        )
        bias_mp = float(np.mean(arr_mp2020[mask_mp2020] - arr_mp20[mask_mp2020]))
        print(f"  mp2020 : n={mask_mp2020.sum()} RMSE={rmse_mp:.4f}  bias={bias_mp:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
