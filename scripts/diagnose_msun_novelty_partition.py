#!/usr/bin/env python3
"""Nested partition of a model's MSUN into replay / substitution / framework-novel.

Avoids any disjointness assumption by nesting the two corrections:

  external MSUN
    − replay (matches synthetic augmentation S0[/S1], element-sensitive)   → train-novel
        − substitution AMONG train-novel (matches MP20 under f-block          → framework-novel
          anonymisation; "both" structures already removed as replay)

So a structure that is both a replay and an MP20-substitution is counted once
(as replay). The framework-novel floor is MSUN that is neither a replay of the
synthetic augmentation nor an f-block swap of an MP20 scaffold.

All StructureMatcher calls at ltol=0.1 (LeMat's novelty tolerance). MP20
reference makes substitution a lower bound (LeMat novelty is vs LeMat-Bulk).

    python scripts/diagnose_msun_novelty_partition.py \\
        --lemat_json m2v2.json \\
        --query_cif_dir external_eval/synthetic_round1_v2_n2500_nequip_relaxed/relaxed_cifs \\
        --synthetic_roots data/synthetic/crystalite_round0_msun_27k \\
                          data/synthetic/crystalite_round1_msun_54k_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10**7)


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
    return Structure(struct.lattice, species, struct.frac_coords, coords_are_cartesian=False)


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
    m = re.search(r"individual_values=\[([0-9.,\s]+)\]", s)
    vals = [float(x) for x in m.group(1).split(",") if x.strip()]
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


def _buckets(structs, anonymize):
    b = defaultdict(list)
    for s in structs:
        ks = _anonymize_lanthanides(s) if anonymize else s
        b[ks.composition.reduced_formula].append(ks)
    return b


def _matches(struct, buckets, matcher, anonymize):
    ks = _anonymize_lanthanides(struct) if anonymize else struct
    for ref in buckets.get(ks.composition.reduced_formula, ()):
        try:
            if matcher.fit(ks, ref):
                return True
        except Exception:
            continue
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemat_json", type=Path, required=True)
    ap.add_argument("--query_cif_dir", type=Path, required=True)
    ap.add_argument("--synthetic_roots", type=Path, nargs="+", required=True,
                    help="synthetic augmentation roots for the replay check (S0[, S1]).")
    ap.add_argument("--mp20_root", type=Path, default=Path("data/mp20"))
    ap.add_argument("--ltol", type=float, default=0.1)
    ap.add_argument("--label_value", type=float, default=0.5)
    args = ap.parse_args()

    from pymatgen.analysis.structure_matcher import StructureMatcher

    idx = _msun_indices(args.lemat_json, args.label_value)
    msun = _load_cif_indices(args.query_cif_dir, idx)
    print(f"[partition] {len(msun)} MSUN structures loaded")

    synth = []
    for root in args.synthetic_roots:
        p = root / "raw" / "train.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        s = _load_csv(p)
        synth.extend(s)
        print(f"[partition]   synthetic ref {root.name}: {len(s)}")
    mp20 = _load_csv(args.mp20_root / "raw" / "train.csv")
    print(f"[partition]   MP20 ref: {len(mp20)}")

    matcher = StructureMatcher(ltol=args.ltol, stol=0.3, angle_tol=5)
    synth_buckets = _buckets(synth, anonymize=False)        # replay: element-sensitive
    mp20_asis_buckets = _buckets(mp20, anonymize=False)      # MP20 as-is (matcher baseline)
    mp20_anon_buckets = _buckets(mp20, anonymize=True)        # MP20 anonymised (swap)

    n = len(msun)
    n_replay = n_subst = n_asis_baseline = n_framework = 0
    for s in msun:
        if _matches(s, synth_buckets, matcher, anonymize=False):
            n_replay += 1
            continue
        # train-novel from here; split MP20-relationship
        asis = _matches(s, mp20_asis_buckets, matcher, anonymize=False)
        anon = _matches(s, mp20_anon_buckets, matcher, anonymize=True)
        if anon and not asis:
            n_subst += 1          # true f-block swap: matches only after anonymisation
        elif asis:
            n_asis_baseline += 1  # matches MP20 as-is (LeMat-vs-our-MP20 matcher discrepancy)
        else:
            n_framework += 1

    nv = json.load(open(args.lemat_json))["validity_filtering"]["valid_structures"]
    pct = lambda c: 100.0 * c / nv
    tn = n - n_replay
    print(f"\nMSUN partition (n_valid={nv}, counts -> % of valid):")
    print(f"  external MSUN:          {n:>4}   {pct(n):.2f}%")
    print(f"  replay:                 {n_replay:>4}   {pct(n_replay):.2f}%")
    print(f"  -> train-novel:         {tn:>4}   {pct(tn):.2f}%")
    print(f"  substitution (swap):    {n_subst:>4}   {pct(n_subst):.2f}%   (anon-only MP20 match)")
    print(f"  MP20 as-is baseline:    {n_asis_baseline:>4}   {pct(n_asis_baseline):.2f}%   "
          f"(matcher discrepancy; left in framework-novel)")
    fw = n_framework + n_asis_baseline   # baseline left in framework-novel (LeMat called it novel)
    print(f"  -> framework-novel:     {fw:>4}   {pct(fw):.2f}%   (train-novel − swap)")
    print(f"\nFor the scaling plot ROWS (rates as % of valid):")
    print(f"  external_msun={pct(n):.2f}, train_novel_msun={pct(tn):.2f}, "
          f"framework_novel_msun={pct(fw):.2f}  (substitution gap = {pct(n_subst):.2f} pp)")


if __name__ == "__main__":
    main()
