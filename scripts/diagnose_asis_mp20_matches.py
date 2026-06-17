#!/usr/bin/env python3
"""Characterise the 'as-is MP-20 match' MSUN structures (the M1 3.4 pp question).

In the nested MSUN partition, some MSUN structures match an MP-20 structure
*as-is* (element-sensitive, ltol=0.1) even though LeMat called them novel vs
LeMat-Bulk. M1 has ~3.4 pp of these; M2 has ~0. This script characterises them:
are they near-exact MP-20 duplicates that LeMat missed (-> not framework-novel),
or matcher-boundary cases (-> legitimately novel)?

For each model it reports, among the MSUN set, the structures matching MP-20
as-is: count, the StructureMatcher RMS displacement of each match (small => a
genuine duplicate; near the tolerance => borderline), and the lanthanide split.

    python scripts/diagnose_asis_mp20_matches.py \\
        --lemat_json <m1.json> --query_cif_dir <m1 relaxed_cifs>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10**7)


def _is_lanthanide(struct) -> bool:
    return any(el.is_lanthanoid for el in struct.composition.elements)


def _load_csv(csv_path: Path):
    from pymatgen.core import Structure

    out = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cif = row.get("cif")
            if not cif:
                continue
            try:
                out.append(Structure.from_str(cif, fmt="cif"))
            except Exception:
                continue
    return out


def _msun_indices(lemat_json: Path, label_value: float = 0.5):
    d = json.load(open(lemat_json))
    vsi = d["validity_filtering"]["valid_structure_ids"]
    s = d["results"]["sun"]
    vals = [float(x) for x in re.search(r"individual_values=\[([0-9.,\s]+)\]", s).group(1).split(",") if x.strip()]
    return [vsi[i] for i, v in enumerate(vals) if abs(v - label_value) < 1e-6]


def _load_cif_indices(cif_dir: Path, indices, pattern="sample_{:05d}.cif"):
    from pymatgen.core import Structure

    out = []
    for idx in indices:
        p = cif_dir / pattern.format(idx)
        if p.exists():
            try:
                out.append(Structure.from_file(p))
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemat_json", type=Path, required=True)
    ap.add_argument("--query_cif_dir", type=Path, required=True)
    ap.add_argument("--mp20_root", type=Path, default=Path("data/mp20"))
    ap.add_argument("--ltol", type=float, default=0.1)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    from pymatgen.analysis.structure_matcher import StructureMatcher

    msun = _load_cif_indices(args.query_cif_dir, _msun_indices(args.lemat_json))
    print(f"[{args.label}] {len(msun)} MSUN structures")
    mp20 = _load_csv(args.mp20_root / "raw" / "train.csv")
    buckets = defaultdict(list)
    for s in mp20:
        buckets[s.composition.reduced_formula].append(s)

    m = StructureMatcher(ltol=args.ltol, stol=0.3, angle_tol=5)
    rms_vals, lanth, nonlanth, formulas = [], 0, 0, []
    for q in msun:
        best = None
        for ref in buckets.get(q.composition.reduced_formula, ()):
            try:
                rd = m.get_rms_dist(q, ref)  # (rms, max) or None
            except Exception:
                rd = None
            if rd is not None and (best is None or rd[0] < best):
                best = rd[0]
        if best is not None:  # matched MP-20 as-is
            rms_vals.append(best)
            formulas.append(q.composition.reduced_formula)
            if _is_lanthanide(q):
                lanth += 1
            else:
                nonlanth += 1

    n = len(rms_vals)
    print(f"[{args.label}] MSUN structures matching MP-20 as-is (ltol={args.ltol}): "
          f"{n}  ({100*n/max(len(msun),1):.2f}% of MSUN)")
    if n:
        print(f"  lanthanide-bearing: {lanth}   non-lanthanide: {nonlanth}")
        print(f"  RMS displacement of match  min={min(rms_vals):.4f}  "
              f"median={statistics.median(rms_vals):.4f}  max={max(rms_vals):.4f}")
        near_exact = sum(1 for r in rms_vals if r < 0.02)
        print(f"  near-exact duplicates (RMS<0.02): {near_exact}/{n} "
              f"({100*near_exact/n:.0f}%)")
        print(f"  example formulas: {formulas[:12]}")


if __name__ == "__main__":
    main()
