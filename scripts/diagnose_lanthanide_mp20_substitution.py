#!/usr/bin/env python3
"""S0-vs-MP20 lanthanide-anonymised cross-match.

Stronger version of the within-S0 substitution test: is the generator
substituting lanthanides onto *MP-20* scaffolds? For each S0 lanthanide-bearing
structure, check whether it matches an MP-20 structure under lanthanide-element
anonymisation (all Ln -> La). A match = the generated structure is a known MP-20
framework with a different f-block element swapped in (substitutional, not a new
framework).

Baseline is the as-is (element-sensitive) match rate, which should be ~0 because
curation already deduped S0 against MP-20 with a standard StructureMatcher — so
the gap opened by anonymisation is the substitution-onto-MP-20 signal.

  S0 lanthanide  -> MP20, as-is match %        (expect ~0; curation deduped)
  S0 lanthanide  -> MP20, anonymised match %   (the substitution number)
  S0 non-lanth   -> MP20, anonymised match %   (control; expect ~0)

Local, pymatgen-only.

    python scripts/diagnose_lanthanide_mp20_substitution.py \\
        --query_root data/synthetic/crystalite_round0_msun_27k \\
        --reference_root data/mp20 --query_limit 4000
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
    from pymatgen.core import Element, Structure

    species = []
    for site in struct:
        sym = site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie)
        try:
            is_lanth = Element(sym).is_lanthanoid
        except Exception:
            is_lanth = False
        species.append("La" if is_lanth else sym)
    return Structure(struct.lattice, species, struct.frac_coords,
                     coords_are_cartesian=False)


def _load(csv_path: Path, limit: int):
    from pymatgen.core import Structure

    out = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cif = row.get("cif")
            if not cif:
                continue
            try:
                s = Structure.from_str(cif, fmt="cif")
            except Exception:
                continue
            out.append(s)
            if limit and len(out) >= limit:
                break
    return out


def _load_cif_dir(cif_dir: Path, limit: int):
    """Load individual *.cif files from a directory (e.g. external_eval relaxed_cifs)."""
    from pymatgen.core import Structure

    out = []
    for p in sorted(cif_dir.glob("*.cif")):
        try:
            out.append(Structure.from_file(p))
        except Exception:
            continue
        if limit and len(out) >= limit:
            break
    return out


def _msun_indices(lemat_json: Path, label_value: float):
    """Original sample indices whose SUN label == label_value (0.5=MSUN, 1.0=SUN).

    Maps the SUN evaluator's per-valid-structure individual_values back to
    original sample indices via validity_filtering.valid_structure_ids.
    """
    import json
    import re

    d = json.load(open(lemat_json))
    vsi = d["validity_filtering"]["valid_structure_ids"]
    s = d["results"]["sun"]
    m = re.search(r"individual_values=\[([0-9.,\s]+)\]", s)
    if not m:
        raise RuntimeError("could not parse SUN individual_values from lemat_json")
    vals = [float(x) for x in m.group(1).split(",") if x.strip()]
    if len(vals) != len(vsi):
        raise RuntimeError(
            f"length mismatch: {len(vals)} individual_values vs {len(vsi)} valid ids")
    return [vsi[i] for i, v in enumerate(vals) if abs(v - label_value) < 1e-6]


def _load_cif_indices(cif_dir: Path, indices, pattern: str = "sample_{:05d}.cif"):
    """Load specific sample_{idx:05d}.cif files by original index."""
    from pymatgen.core import Structure

    out = []
    for idx in indices:
        p = cif_dir / pattern.format(idx)
        if not p.exists():
            continue
        try:
            out.append(Structure.from_file(p))
        except Exception:
            continue
    return out


def _build_buckets(structs, anonymize: bool):
    """reduced_formula -> [structures] (optionally anonymising lanthanides first)."""
    buckets: dict[str, list] = defaultdict(list)
    for s in structs:
        key_struct = _anonymize_lanthanides(s) if anonymize else s
        buckets[key_struct.composition.reduced_formula].append(key_struct)
    return buckets


def _has_match(struct, buckets, matcher, anonymize: bool) -> bool:
    key_struct = _anonymize_lanthanides(struct) if anonymize else struct
    candidates = buckets.get(key_struct.composition.reduced_formula)
    if not candidates:
        return False
    for ref in candidates:
        try:
            if matcher.fit(key_struct, ref):
                return True
        except Exception:
            continue
    return False


def _match_rate(queries, buckets, matcher, anonymize: bool) -> tuple[int, int]:
    n_match = sum(_has_match(q, buckets, matcher, anonymize) for q in queries)
    return n_match, len(queries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_root", type=Path,
                    default=Path("data/synthetic/crystalite_round0_msun_27k"))
    ap.add_argument("--query_cif_dir", type=Path, default=None,
                    help="load query structures from a dir of *.cif (e.g. "
                         "external_eval/<run>/relaxed_cifs) instead of query_root/raw/train.csv.")
    ap.add_argument("--lemat_json", type=Path, default=None,
                    help="if set with --query_cif_dir, load only the MSUN-flagged "
                         "CIFs (SUN label == --label_value) — answers 'how much of "
                         "the reported MSUN is substitutional'.")
    ap.add_argument("--label_value", type=float, default=0.5,
                    help="SUN individual_value to keep: 0.5=MSUN (default), 1.0=SUN.")
    ap.add_argument("--reference_root", type=Path, default=Path("data/mp20"))
    ap.add_argument("--query_limit", type=int, default=4000,
                    help="cap query structures loaded (0 = all).")
    ap.add_argument("--ref_limit", type=int, default=0,
                    help="cap MP20 reference structures (0 = all).")
    ap.add_argument("--ltol", type=float, default=0.2)
    args = ap.parse_args()

    from pymatgen.analysis.structure_matcher import StructureMatcher

    r_csv = args.reference_root / "raw" / "train.csv"
    if not r_csv.exists():
        sys.exit(f"missing {r_csv}")

    if args.query_cif_dir is not None:
        if not args.query_cif_dir.is_dir():
            sys.exit(f"missing query_cif_dir {args.query_cif_dir}")
        if args.lemat_json is not None:
            idx = _msun_indices(args.lemat_json, args.label_value)
            tag = {0.5: "MSUN", 1.0: "SUN"}.get(args.label_value, str(args.label_value))
            print(f"[xmatch] loading {len(idx)} {tag}-flagged CIFs from "
                  f"{args.query_cif_dir} (label_value={args.label_value})...")
            q_all = _load_cif_indices(args.query_cif_dir, idx)
        else:
            print(f"[xmatch] loading query from CIF dir {args.query_cif_dir} "
                  f"(limit={args.query_limit or 'all'})...")
            q_all = _load_cif_dir(args.query_cif_dir, args.query_limit)
    else:
        q_csv = args.query_root / "raw" / "train.csv"
        if not q_csv.exists():
            sys.exit(f"missing {q_csv}")
        print(f"[xmatch] loading query from {q_csv} (limit={args.query_limit or 'all'})...")
        q_all = _load(q_csv, args.query_limit)
    q_lanth = [s for s in q_all if _is_lanthanide_bearing(s)]
    q_rest = [s for s in q_all if not _is_lanthanide_bearing(s)]

    print(f"[xmatch] loading reference (MP20) from {r_csv} (limit={args.ref_limit or 'all'})...")
    ref = _load(r_csv, args.ref_limit)
    print(f"[xmatch] query: {len(q_all)} ({len(q_lanth)} lanthanide, {len(q_rest)} other); "
          f"reference: {len(ref)} MP20\n")

    matcher = StructureMatcher(ltol=args.ltol, stol=0.3, angle_tol=5)

    # MP20 reference buckets, as-is and anonymised
    ref_asis = _build_buckets(ref, anonymize=False)
    ref_anon = _build_buckets(ref, anonymize=True)

    print(f"S0 lanthanide -> MP20  (n={len(q_lanth)}), StructureMatcher ltol={args.ltol}:")
    m_asis, n = _match_rate(q_lanth, ref_asis, matcher, anonymize=False)
    print(f"  as-is (element-sensitive) match:  {m_asis}/{n}  ({100*m_asis/max(n,1):.1f}%)"
          "   [baseline; expect ~0 — curation deduped]")
    m_anon, n = _match_rate(q_lanth, ref_anon, matcher, anonymize=True)
    print(f"  anonymised (Ln->La) match:        {m_anon}/{n}  ({100*m_anon/max(n,1):.1f}%)"
          "   [substitution onto MP-20 scaffold]")

    subst = 100 * m_anon / max(n, 1) - 100 * m_asis / max(n, 1)
    print(f"  => substitution-onto-MP-20 signal: ~{subst:.1f} pp\n")

    print(f"S0 non-lanthanide -> MP20  (n={len(q_rest)})  [control]:")
    c_anon, nc = _match_rate(q_rest, ref_anon, matcher, anonymize=True)
    print(f"  anonymised match:                 {c_anon}/{nc}  ({100*c_anon/max(nc,1):.1f}%)"
          "   [expect ~0; anonymisation is a no-op for non-lanthanide]")

    print("\nInterpretation:")
    print("  High anonymised rate => the generator is substituting f-block elements")
    print("  onto known MP-20 frameworks (stronger claim than within-S0 redundancy).")
    print("  Low => the lanthanide frameworks are not simply MP-20 scaffolds with a")
    print("  swapped element — they are new frameworks (even if self-redundant within S0).")


if __name__ == "__main__":
    main()
