#!/usr/bin/env python
"""Smoke-test an EquiformerV3 PatchedPhaseDiagram artifact.

Checks that the saved PPD can be loaded and used for e_above_hull scoring with
the same MP2020-like candidate-entry convention used by the curator.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.computed_entries import ComputedStructureEntry as CSE


def _build_mp2020_parameters(composition, mp2020_compat) -> dict:
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
    return {
        "run_type": "GGA+U" if hubbards else "GGA",
        "hubbards": hubbards,
        "software": "vasp",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppd", type=Path, required=True)
    parser.add_argument(
        "--entries-jsonl",
        type=Path,
        default=None,
        help="Defaults to <ppd>.entries.jsonl.",
    )
    args = parser.parse_args()

    entries_jsonl = args.entries_jsonl or args.ppd.with_suffix(".entries.jsonl")
    if not args.ppd.exists():
        raise FileNotFoundError(args.ppd)
    if not entries_jsonl.exists():
        raise FileNotFoundError(entries_jsonl)

    with args.ppd.open("rb") as f:
        ppd = pickle.load(f)

    first = None
    with entries_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                first = json.loads(line)
                break
    if first is None:
        raise ValueError(f"No entries found in {entries_jsonl}")

    source_entry = CSE.from_dict(first)
    structure = source_entry.structure
    if hasattr(structure, "calc"):
        raise AssertionError("round-tripped structure unexpectedly has .calc")

    mp2020 = MaterialsProject2020Compatibility(check_potcar=False)
    raw_candidate = ComputedStructureEntry(
        composition=structure.composition,
        energy=float(source_entry.uncorrected_energy),
        structure=structure,
        entry_id="roundtrip_candidate",
        parameters=_build_mp2020_parameters(structure.composition, mp2020),
    )
    candidate = mp2020.process_entry(raw_candidate, on_error="raise")
    if candidate is None:
        raise RuntimeError("MP2020 dropped candidate")

    e_above = float(ppd.get_e_above_hull(candidate, allow_negative=True))
    if not math.isfinite(e_above):
        raise RuntimeError(f"Non-finite e_above_hull: {e_above}")

    e_hull = float(ppd.get_hull_energy_per_atom(candidate.composition))
    print(
        json.dumps(
            {
                "ppd": str(args.ppd),
                "entries_jsonl": str(entries_jsonl),
                "ppd_type": f"{type(ppd).__module__}.{type(ppd).__name__}",
                "n_all_entries": len(getattr(ppd, "all_entries", [])),
                "candidate_formula": candidate.composition.reduced_formula,
                "candidate_energy_per_atom": float(candidate.energy_per_atom),
                "hull_energy_per_atom": e_hull,
                "e_above_hull": e_above,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
