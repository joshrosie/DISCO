#!/usr/bin/env python3
"""Quantify the element-class drift across the Flywheel curation rounds.

Reads local_data/features.parquet (per-structure compositions for MP20, S0,
S1_v2, ...) and reports, per dataset, the fraction of structures containing:

  - any lanthanide (La-Lu)
  - any rare-earth element (lanthanides + Sc + Y)
  - any radioactive / synthetic-impractical element (Tc, Pm, actinoids, Z>=84)

Plus the largest per-element frequency increases vs MP20. This puts concrete
element chemistry behind the HHI-production drift seen in the LeMat eval.

    python scripts/diagnose_element_drift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pymatgen.core import Element

LOCAL = Path(__file__).resolve().parent.parent / "local_data"
PARQUET = LOCAL / "features.parquet"

# Series in round order; S1_v2 + its topup are merged into one Round-1 series.
# S_big_merged is the one-shot M0 control at the M2 budget — listed separately
# so iterative cascade drift can be compared to one-shot drift.
SERIES = ["MP20", "S0", "S1_v2_full", "S2_v2", "S_big_merged"]


def _classify(symbol: str) -> tuple[bool, bool, bool]:
    """Return (is_lanthanide, is_rare_earth, is_radioactive_impractical)."""
    el = Element(symbol)
    z = el.Z
    is_lanth = el.is_lanthanoid
    is_ree = is_lanth or z in (21, 39)  # + Sc, Y
    # Tc and Pm have no stable isotopes; actinoids + Z>=84 are radioactive.
    is_radio = (z in (43, 61)) or el.is_actinoid or z >= 84
    return is_lanth, is_ree, is_radio


def _element_set(elements_field: str) -> list[str]:
    return [e for e in str(elements_field).split(";") if e]


def _row_flags(elements_field: str) -> tuple[bool, bool, bool]:
    has_lanth = has_ree = has_radio = False
    for sym in _element_set(elements_field):
        try:
            lanth, ree, radio = _classify(sym)
        except Exception:
            continue
        has_lanth |= lanth
        has_ree |= ree
        has_radio |= radio
    return has_lanth, has_ree, has_radio


def main() -> None:
    if not PARQUET.exists():
        sys.exit(f"missing {PARQUET}; run scripts/local_dataset_features.py first")
    df = pd.read_parquet(PARQUET)

    # merge S1_v2 + topup into a single Round-1 series
    df = df.copy()
    df.loc[df["dataset"].isin(["S1_v2", "S1_v2_topup"]), "dataset"] = "S1_v2_full"

    flags = df["elements"].apply(_row_flags)
    df["has_lanthanide"] = flags.apply(lambda t: t[0])
    df["has_rare_earth"] = flags.apply(lambda t: t[1])
    df["has_radioactive"] = flags.apply(lambda t: t[2])

    print(f"{'dataset':<12} {'n':>7} {'%lanthanide':>12} {'%rare_earth':>12} {'%radioactive':>13}")
    print("-" * 60)
    for name in SERIES:
        sub = df[df["dataset"] == name]
        if len(sub) == 0:
            print(f"{name:<12} {'(absent)':>7}")
            continue
        n = len(sub)
        pl = 100.0 * sub["has_lanthanide"].mean()
        pr = 100.0 * sub["has_rare_earth"].mean()
        pa = 100.0 * sub["has_radioactive"].mean()
        print(f"{name:<12} {n:>7} {pl:>11.1f}% {pr:>11.1f}% {pa:>12.1f}%")

    # --- largest per-element frequency increases vs MP20 ---
    def elem_freq(name: str) -> pd.Series:
        sub = df[df["dataset"] == name]
        counts: dict[str, int] = {}
        for field in sub["elements"]:
            for sym in set(_element_set(field)):
                counts[sym] = counts.get(sym, 0) + 1
        return pd.Series(counts, dtype=float) / max(len(sub), 1)

    base = elem_freq("MP20")
    comparators = [name for name in SERIES if name != "MP20"]
    # Show per-element frequencies side-by-side for every comparator series.
    # Sort by max increase vs MP20 across comparators so the most-shifted
    # elements lead.
    all_elems: set[str] = set()
    freqs: dict[str, pd.Series] = {}
    for name in comparators:
        f = elem_freq(name)
        if len(f) == 0:
            continue
        freqs[name] = f
        all_elems.update(f.index)
    max_delta = pd.Series(
        {
            sym: max(
                (freqs[name].get(sym, 0.0) - base.get(sym, 0.0)) for name in freqs
            )
            for sym in all_elems
        }
    ).sort_values(ascending=False)

    print("\nPer-element frequency by series (% of structures containing the "
          "element), sorted by max Δ vs MP20:")
    header_cols = ["element", "MP20%"] + [f"{n}%" for n in freqs] + ["max Δpp", "class"]
    widths = [8, 8] + [12] * len(freqs) + [9, 14]
    print(" ".join(f"{c:>{w}}" for c, w in zip(header_cols, widths)))
    print("-" * sum(widths))
    for sym, d in max_delta.head(20).items():
        try:
            lanth, ree, radio = _classify(sym)
            cls = "radioactive" if radio else ("lanthanide" if lanth else ("rare-earth" if ree else ""))
        except Exception:
            cls = ""
        cells = [f"{sym:>8}", f"{100*base.get(sym, 0.0):>7.1f}%"]
        for name in freqs:
            cells.append(f"{100*freqs[name].get(sym, 0.0):>11.1f}%")
        cells.append(f"{100*d:>+8.1f}")
        cells.append(f"{cls:>14}")
        print(" ".join(cells))


if __name__ == "__main__":
    main()
