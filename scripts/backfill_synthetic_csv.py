"""Backfill raw/train.csv for synthetic curation runs produced before the
MP20Tokens-unification refactor.

Walks each provided synthetic root, reads metadata.jsonl for kept samples,
reads each CIF from structures/<sample_id>.cif, and writes raw/train.csv with
the columns MP20Tokens expects.

The existing samples.pt (if present) is renamed to samples.pre_unified.pt so
the buggy tokenization (no Niggli reduction, no canonical lattice) is
preserved on disk for provenance but no longer the source of truth.

Idempotent: skips a root if raw/train.csv already exists with the expected
row count.

Usage:
    uv run python scripts/backfill_synthetic_csv.py \\
        data/synthetic/crystalite_round0_msun_27k \\
        data/synthetic/crystalite_round1_msun_54k
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


_CSV_FIELDS = ["material_id", "cif", "formation_energy_per_atom", "e_above_hull"]


def _iter_kept_metadata(metadata_path: Path):
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("filter_status") != "kept":
                continue
            yield rec


def _expected_kept(summary_path: Path) -> int | None:
    if not summary_path.exists():
        return None
    try:
        return int(json.loads(summary_path.read_text()).get("num_kept", -1))
    except Exception:
        return None


def backfill_root(root: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f"Root not found: {root}")
    metadata_path = root / "metadata.jsonl"
    structures_dir = root / "structures"
    raw_dir = root / "raw"
    csv_path = raw_dir / "train.csv"
    summary_path = root / "summary.json"
    samples_pt = root / "samples.pt"
    preserved_pt = root / "samples.pre_unified.pt"

    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} missing")
    if not structures_dir.is_dir():
        raise FileNotFoundError(f"{structures_dir} missing")

    expected_kept = _expected_kept(summary_path)

    # Idempotent check: if raw/train.csv already exists with the expected row count, skip.
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            existing_rows = sum(1 for _ in f) - 1  # minus header
        if expected_kept is not None and existing_rows == expected_kept:
            print(f"[backfill] {root}: raw/train.csv already has {existing_rows} rows — skipping.")
            return {"root": str(root), "skipped": True, "rows": existing_rows}

    raw_dir.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    missing_cifs = 0
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for rec in _iter_kept_metadata(metadata_path):
            sample_id = rec.get("sample_id")
            if not sample_id:
                continue
            cif_path = structures_dir / f"{sample_id}.cif"
            if not cif_path.exists():
                missing_cifs += 1
                continue
            cif_str = cif_path.read_text(encoding="utf-8")
            writer.writerow(
                {
                    "material_id": sample_id,
                    "cif": cif_str,
                    "formation_energy_per_atom": rec.get("formation_energy_per_atom"),
                    "e_above_hull": rec.get("e_above_hull"),
                }
            )
            rows_written += 1

    # Preserve the legacy tokenized cache (buggy preprocessing) under a rename.
    moved_samples_pt = False
    if samples_pt.exists() and not preserved_pt.exists():
        samples_pt.rename(preserved_pt)
        moved_samples_pt = True

    result = {
        "root": str(root),
        "skipped": False,
        "rows": rows_written,
        "missing_cifs": missing_cifs,
        "expected_kept": expected_kept,
        "moved_samples_pt": moved_samples_pt,
        "csv_path": str(csv_path),
    }
    print(
        f"[backfill] {root}: wrote {rows_written} rows to {csv_path.name} "
        f"(expected {expected_kept}, missing_cifs={missing_cifs}, "
        f"moved samples.pt={moved_samples_pt})"
    )
    if expected_kept is not None and rows_written != expected_kept:
        print(
            f"[backfill] WARNING: row count {rows_written} != expected_kept {expected_kept}",
            file=sys.stderr,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Synthetic curation roots")
    args = parser.parse_args()

    results = []
    for root in args.roots:
        results.append(backfill_root(root))

    print()
    print("[backfill] summary:")
    for r in results:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
