#!/usr/bin/env python3
"""Test whether the lanthanide enrichment is substitutional near-duplication.

Collaborator hypothesis (Ivor): lanthanides are more mutually similar than
typical ionic compounds (lanthanide contraction -> near-identical radii/
chemistry), so the generator finds a combinatorially "easy" region where
swapping La<->Nd<->Eu yields many technically-distinct metastable structures.
If so, much of the lanthanide novelty is chemically-trivial substitution.

Two tests on a curated synthetic set (default S0):

  TEST 1 — tolerance sweep (Ivor's literal suggestion).
    Re-dedup within the set at increasing StructureMatcher ltol, stratified by
    lanthanide-bearing vs rest. If lanthanides collapse faster as tolerance
    loosens, their structures sit closer together. Same-composition only
    (StructureMatcher default comparator), so this catches geometric
    near-duplicates, not cross-element substitution.

  TEST 2 — lanthanide-anonymized dedup (the substitution test).
    Map every lanthanide species to a single canonical Ln (La), so LaFeO3 and
    NdFeO3 become the same composition and get compared. Report:
      2a composition-collapse: distinct reduced formulas after anonymization
      2b structure-collapse:   distinct structures (StructureMatcher) after
                               anonymization
    The gap between the anonymized lanthanide collapse and the non-lanthanide
    baseline is the substitutional near-duplication.

Local, pymatgen-only — no MLIP, no hull, no GPU.

    python scripts/diagnose_lanthanide_dedup.py \\
        --synthetic_root data/synthetic/crystalite_round0_msun_27k --limit 8000
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10**7)


def _is_lanthanide_bearing(struct) -> bool:
    return any(el.is_lanthanoid for el in struct.composition.elements)


def _anonymize_lanthanides(struct):
    """Return a copy with every lanthanide site species replaced by La."""
    from pymatgen.core import Element, Structure

    new_species = []
    for site in struct:
        sym = site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie)
        try:
            is_lanth = Element(sym).is_lanthanoid
        except Exception:
            is_lanth = False
        new_species.append("La" if is_lanth else sym)
    return Structure(struct.lattice, new_species, struct.frac_coords,
                     coords_are_cartesian=False)


def _load(csv_path: Path, limit: int):
    from pymatgen.core import Structure

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cif = row.get("cif")
            if not cif:
                continue
            try:
                s = Structure.from_str(cif, fmt="cif")
            except Exception:
                continue
            rows.append((row.get("material_id", "?"), s, _is_lanthanide_bearing(s)))
            if limit and len(rows) >= limit:
                break
    return rows


def _n_unique(structs, matcher) -> int:
    """Count distinct structures: bucket by reduced formula, group within bucket."""
    buckets: dict[str, list] = defaultdict(list)
    for s in structs:
        buckets[s.composition.reduced_formula].append(s)
    total_groups = 0
    for group in buckets.values():
        if len(group) == 1:
            total_groups += 1
        else:
            total_groups += len(matcher.group_structures(group))
    return total_groups


def _collapse_pct(n_struct: int, n_unique: int) -> float:
    return 100.0 * (1.0 - n_unique / n_struct) if n_struct else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_root", type=Path,
                    default=Path("data/synthetic/crystalite_round0_msun_27k"))
    ap.add_argument("--limit", type=int, default=8000,
                    help="cap structures loaded (0 = all of the set).")
    ap.add_argument("--tolerances", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--anon_tol", type=float, default=0.2,
                    help="ltol for the anonymized structure-collapse test.")
    args = ap.parse_args()

    from pymatgen.analysis.structure_matcher import StructureMatcher

    csv_path = args.synthetic_root / "raw" / "train.csv"
    if not csv_path.exists():
        sys.exit(f"missing {csv_path}")

    print(f"[dedup] loading structures from {csv_path} (limit={args.limit or 'all'})...")
    rows = _load(csv_path, args.limit)
    lanth = [s for _, s, isl in rows if isl]
    rest = [s for _, s, isl in rows if not isl]
    print(f"[dedup] loaded {len(rows)}: {len(lanth)} lanthanide-bearing, "
          f"{len(rest)} other\n")

    # --- TEST 1: tolerance sweep, same-composition ---
    print("TEST 1 — same-composition dedup collapse vs tolerance "
          "(stol=0.3, angle_tol=5 fixed):")
    print(f"{'ltol':>6} {'lanth collapse%':>16} {'other collapse%':>16}")
    print("-" * 42)
    for tol in args.tolerances:
        m = StructureMatcher(ltol=tol, stol=0.3, angle_tol=5)
        cl = _collapse_pct(len(lanth), _n_unique(lanth, m))
        cr = _collapse_pct(len(rest), _n_unique(rest, m))
        print(f"{tol:>6.2f} {cl:>15.2f}% {cr:>15.2f}%")

    # --- TEST 2: lanthanide-anonymized dedup ---
    print("\nTEST 2 — lanthanide-anonymized (all Ln -> La) substitution test:")
    anon_lanth = [_anonymize_lanthanides(s) for s in lanth]

    # 2a composition-collapse
    raw_formulas = {s.composition.reduced_formula for s in lanth}
    anon_formulas = {s.composition.reduced_formula for s in anon_lanth}
    print(f"  lanthanide structures:                {len(lanth)}")
    print(f"  distinct compositions (as-is):        {len(raw_formulas)}")
    print(f"  distinct compositions (Ln->La):       {len(anon_formulas)}  "
          f"=> composition-collapse {_collapse_pct(len(raw_formulas), len(anon_formulas)):.1f}% "
          f"of compositions merge under Ln substitution")

    # 2b structure-collapse at anon_tol, vs the non-lanthanide baseline
    m = StructureMatcher(ltol=args.anon_tol, stol=0.3, angle_tol=5)
    anon_unique = _n_unique(anon_lanth, m)
    lanth_unique_raw = _n_unique(lanth, m)
    rest_unique = _n_unique(rest, m)
    print(f"\n  structure-collapse at ltol={args.anon_tol} "
          f"(distinct structures / total):")
    print(f"    lanthanide, as-is:        {lanth_unique_raw}/{len(lanth)}  "
          f"({_collapse_pct(len(lanth), lanth_unique_raw):.1f}% collapse)")
    print(f"    lanthanide, Ln->La:       {anon_unique}/{len(lanth)}  "
          f"({_collapse_pct(len(lanth), anon_unique):.1f}% collapse)")
    print(f"    non-lanthanide baseline:  {rest_unique}/{len(rest)}  "
          f"({_collapse_pct(len(rest), rest_unique):.1f}% collapse)")

    subst = _collapse_pct(len(lanth), anon_unique) - _collapse_pct(len(lanth), lanth_unique_raw)
    print(f"\n  => substitutional near-duplication among lanthanides: "
          f"~{subst:.1f} pp")
    print("     (extra collapse from treating lanthanides as interchangeable, "
          "beyond same-composition duplicates)")
    print("     High => lanthanide novelty is largely La<->Nd<->Eu substitution; "
          "family-aware dedup would arrest the drift and tighten the novelty claim.")
    print("     Low  => lanthanide structures are genuinely distinct; the "
          "enrichment is real new chemistry.")


if __name__ == "__main__":
    main()
