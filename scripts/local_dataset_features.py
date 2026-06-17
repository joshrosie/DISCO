#!/usr/bin/env python3
"""Extract per-structure features from the synthetic-augmentation CSVs.

Reads raw/train.csv files for {MP20, S_raw, S_dedup, S0, S1_v2, S1_v2_topup},
parses each CIF with pymatgen, and writes a single combined parquet at
local_data/features.parquet with one row per structure:

    dataset, material_id, n_atoms, n_ary, volume, density, volume_per_atom,
    a, b, c, alpha, beta, gamma, e_above_hull, formation_energy_per_atom,
    elements (semicolon-joined sorted element list)

Slow first run (CIF parse), fast subsequent runs (read parquet).

Usage:
    python scripts/local_dataset_features.py
    python scripts/local_dataset_features.py --rebuild    # force re-parse
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

LOCAL = Path(__file__).resolve().parent.parent / "local_data"
DATASETS = {
    "MP20":          "MP20.csv",
    "S_raw":         "S_raw.csv",
    "S_dedup":       "S_dedup.csv",
    "S0":            "S0.csv",
    "S1_v2":         "S1_v2.csv",
    "S1_v2_topup":   "S1_v2_topup.csv",
    "S2_v2":         "S2_v2.csv",
    "S3_v2_full":    "S3_v2_full.csv",
    "S_big_merged":  "S_big_merged.csv",
}


def _features_from_struct(s: Structure) -> dict:
    lat = s.lattice
    elements_sorted = sorted({e.symbol for e in s.composition.elements})
    n_atoms = len(s)
    return {
        "n_atoms": n_atoms,
        "n_ary": len(elements_sorted),
        "volume": lat.volume,
        "density": s.density,
        "volume_per_atom": lat.volume / max(1, n_atoms),
        "a": lat.a,
        "b": lat.b,
        "c": lat.c,
        "alpha": lat.alpha,
        "beta": lat.beta,
        "gamma": lat.gamma,
        "elements": ";".join(elements_sorted),
    }


def extract_one_csv(csv_path: Path, dataset_label: str, progress_every: int = 2000) -> list[dict]:
    rows: list[dict] = []
    n_total = 0
    n_failed = 0
    t0 = time.time()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            try:
                s = Structure.from_str(row["cif"], fmt="cif")
                feat = _features_from_struct(s)
            except Exception:
                n_failed += 1
                continue
            feat["dataset"] = dataset_label
            feat["material_id"] = row.get("material_id", "")
            for key in ("e_above_hull", "formation_energy_per_atom"):
                val = row.get(key, "")
                feat[key] = float(val) if val not in ("", None) else None
            rows.append(feat)
            if progress_every and n_total % progress_every == 0:
                rate = n_total / max(1e-6, time.time() - t0)
                print(
                    f"  [{dataset_label}] {n_total} processed, "
                    f"{len(rows)} kept, {n_failed} failed ({rate:.0f}/s)",
                    file=sys.stderr,
                )
    elapsed = time.time() - t0
    print(
        f"[{dataset_label}] DONE: {len(rows)}/{n_total} kept "
        f"({elapsed:.1f}s, {n_total/max(1e-6, elapsed):.0f}/s)",
        file=sys.stderr,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Re-parse from CSVs even if cache exists.")
    ap.add_argument("--out", type=Path, default=LOCAL / "features.parquet")
    args = ap.parse_args()

    if args.out.exists() and not args.rebuild:
        df = pd.read_parquet(args.out)
        print(f"[cached] loaded {len(df)} rows from {args.out}", file=sys.stderr)
        print(df["dataset"].value_counts().sort_index())
        return

    all_rows: list[dict] = []
    for label, fname in DATASETS.items():
        path = LOCAL / fname
        if not path.exists():
            print(f"[warn] missing {path}, skipping {label}", file=sys.stderr)
            continue
        all_rows.extend(extract_one_csv(path, label))

    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out, index=False)
    print(f"[ok] wrote {len(df)} rows to {args.out}")
    print(df["dataset"].value_counts().sort_index())


if __name__ == "__main__":
    main()
