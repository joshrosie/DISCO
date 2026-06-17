#!/usr/bin/env python3
"""Smoke test: does PET-OAM-XL agree with NequIP on lanthanide-bearing S0 structures?

The flywheel curates with a single MLIP (NequIP-OAM-L) scored against the MP
hull. If NequIP systematically over-stabilises lanthanide-bearing compositions,
the round-on-round lanthanide surge could be a verifier artifact rather than
genuine discovery. This script re-scores already-curated S0 structures with a
second MP-consistent-PBE potential (PET-OAM-XL) on the *same* geometry and
reports where the two models disagree on metastability.

Both energies are scored against the SAME MP phase diagram with MP2020
corrections, so any disagreement is purely the energy model.

Run on the cluster (needs the MP PPD pickle + ideally a GPU):

    uv run python scripts/smoke_pet_oam_ensemble.py \\
        --synthetic_root data/synthetic/crystalite_round0_msun_27k \\
        --ppd_path /home/jrosenthal/atom-reps/mp_02072023/2023-02-07-ppd-mp.pkl \\
        --n_samples 30 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# CSV cells (embedded CIFs) can exceed the default field-size limit.
csv.field_size_limit(10**7)

METASTABLE_THRESH = 0.1  # eV/atom; msun_like keeps 0 < e_hull <= 0.1


def _is_lanthanide_bearing(structure) -> bool:
    return any(el.is_lanthanoid for el in structure.composition.elements)


def _iter_rows(csv_path: Path, n_target: int, include: str = "lanthanide"):
    """Yield (material_id, structure, nequip_e_hull, is_lanthanide).

    include="lanthanide": only lanthanide-bearing rows.
    include="all":        every row (stratify by is_lanthanide in the summary).
    """
    from pymatgen.core import Structure

    found = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cif = row.get("cif")
            if not cif:
                continue
            try:
                struct = Structure.from_str(cif, fmt="cif")
            except Exception:
                continue
            is_lanth = _is_lanthanide_bearing(struct)
            if include == "lanthanide" and not is_lanth:
                continue
            e_hull_raw = row.get("e_above_hull")
            try:
                nequip_e_hull = float(e_hull_raw) if e_hull_raw not in (None, "") else None
            except ValueError:
                nequip_e_hull = None
            yield row.get("material_id", "?"), struct, nequip_e_hull, is_lanth
            found += 1
            if n_target and found >= n_target:
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_root", type=Path,
                    default=Path("data/synthetic/crystalite_round0_msun_27k"))
    ap.add_argument("--ppd_path", type=str,
                    default="/home/jrosenthal/atom-reps/mp_02072023/2023-02-07-ppd-mp.pkl")
    ap.add_argument("--pet_model", default="pet-oam-xl")
    ap.add_argument("--device", default=None, help="cuda / cpu; None auto-selects.")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--include", choices=["lanthanide", "all"], default="lanthanide",
                    help="'lanthanide' (default) or 'all' chemistry (stratified summary).")
    args = ap.parse_args()

    csv_path = args.synthetic_root / "raw" / "train.csv"
    if not csv_path.exists():
        sys.exit(f"missing {csv_path}")

    print(f"[smoke] scanning {csv_path} (include={args.include})...")
    rows = list(_iter_rows(csv_path, args.n_samples, include=args.include))
    if not rows:
        sys.exit("no structures found")
    n_lanth = sum(1 for *_, isl in rows if isl)
    print(f"[smoke] collected {len(rows)} structures "
          f"({n_lanth} lanthanide, {len(rows) - n_lanth} other)")

    # --- load PET-OAM-XL, the MP PPD, and the MP2020 compat object ---
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from upet.calculator import UPETCalculator

    from src.eval.stability import load_phase_diagram
    from src.utils.sample_stats import compute_e_above_hull_mp2020_like

    print(f"[smoke] loading PET model {args.pet_model} (device={args.device})...")
    calc = UPETCalculator(model=args.pet_model, device=args.device)
    print(f"[smoke] loading MP PPD from {args.ppd_path}...")
    ppd = load_phase_diagram(args.ppd_path)
    mp2020 = MaterialsProject2020Compatibility(check_potcar=False)
    adaptor = AseAtomsAdaptor()

    # --- score each structure with PET-OAM-XL on the same geometry ---
    # per-class accumulators: key in {"lanthanide","non-lanthanide","all"}
    import statistics

    stats = {k: {"scored": 0, "both": 0, "nequip_only": 0, "pet_only": 0,
                 "neither": 0, "deltas": []}
             for k in ("lanthanide", "non-lanthanide", "all")}

    for material_id, struct, nequip_e_hull, is_lanth in rows:
        atoms = adaptor.get_atoms(struct)
        atoms.calc = calc
        try:
            e_total_pet = float(atoms.get_potential_energy())
        except Exception:
            continue
        pet_e_hull, _ = compute_e_above_hull_mp2020_like(
            ppd, struct, e_total_pet, mp2020_compat=mp2020,
        )
        if pet_e_hull is None or nequip_e_hull is None:
            continue

        nq_pass = nequip_e_hull <= METASTABLE_THRESH
        pet_pass = pet_e_hull <= METASTABLE_THRESH
        cls = "lanthanide" if is_lanth else "non-lanthanide"
        for key in (cls, "all"):
            s = stats[key]
            s["scored"] += 1
            s["deltas"].append(pet_e_hull - nequip_e_hull)
            if nq_pass and pet_pass:
                s["both"] += 1
            elif nq_pass and not pet_pass:
                s["nequip_only"] += 1
            elif pet_pass and not nq_pass:
                s["pet_only"] += 1
            else:
                s["neither"] += 1

    # --- stratified summary ---
    print("\n" + "=" * 72)
    print(f"{'class':<16} {'n':>6} {'agree%':>8} {'Nq-only':>8} {'PET-only':>9} "
          f"{'meanΔ(meV)':>11} {'medΔ(meV)':>10}")
    print("-" * 72)
    for key in ("all", "lanthanide", "non-lanthanide"):
        s = stats[key]
        n = s["scored"]
        if n == 0:
            continue
        agree = 100.0 * (s["both"] + s["neither"]) / n
        mean_d = 1000.0 * statistics.mean(s["deltas"])
        med_d = 1000.0 * statistics.median(s["deltas"])
        print(f"{key:<16} {n:>6} {agree:>7.1f}% {s['nequip_only']:>8} "
              f"{s['pet_only']:>9} {mean_d:>+11.2f} {med_d:>+10.2f}")

    print("\nBlind-spot check: compare lanthanide vs non-lanthanide agreement.")
    print("  If non-lanthanide agreement is comparably high (~98%) and meanΔ small,")
    print("  NequIP is not an outlier anywhere — lanthanides are not a special blind spot.")
    print("  If non-lanthanide agreement is markedly worse, the disagreement lives")
    print("  elsewhere and the lanthanide-only check missed it.")


if __name__ == "__main__":
    main()
