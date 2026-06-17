#!/usr/bin/env python3
"""How much of each curated set survives a practicality filter?

Retrospective yield analysis of two proposed curation constraints, run on the
existing curated datasets (MP20 / S0 / S1) from local_data/features.parquet:

  1. Hard denylist — reject any structure containing a radioactive / synthetic
     element (Tc, Pm, actinoids, Z >= 84). These are indefensible regardless of
     DFT-metastability.

  2. Cost-aware HHI cap — reject any structure whose worst constituent element
     exceeds an HHI-production threshold (supply-risk; smact element data,
     0-10000 scale, higher = more supply-concentrated). Swept over thresholds.

Reports the kept fraction per dataset under denylist alone, HHI cap alone (swept),
and both combined — so we can see the yield/practicality tradeoff before
committing to a filter and re-curating.

    python scripts/diagnose_practicality_yield.py
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pymatgen.core import Element as PmgElement

LOCAL = Path(__file__).resolve().parent.parent / "local_data"
PARQUET = LOCAL / "features.parquet"

SERIES = ["MP20", "S0", "S1_v2_full"]
HHI_THRESHOLDS = [4000, 5000, 6000, 7000, 8000]


@lru_cache(maxsize=256)
def _elem_info(sym: str):
    """Return (hhi_production_or_None, is_radioactive_denylisted)."""
    # HHI production from smact
    hhi_p = None
    try:
        from smact import Element as SmactElement
        hhi_p = getattr(SmactElement(sym), "HHI_p", None)
    except Exception:
        hhi_p = None
    # radioactive / synthetic-impractical denylist
    try:
        z = PmgElement(sym).Z
        is_denied = (z in (43, 61)) or PmgElement(sym).is_actinoid or z >= 84
    except Exception:
        is_denied = False
    return hhi_p, is_denied


def _row_metrics(elements_field: str):
    """Return (max_hhi_production, has_denylisted_element)."""
    max_hhi = 0.0
    denied = False
    for sym in str(elements_field).split(";"):
        if not sym:
            continue
        hhi_p, is_denied = _elem_info(sym)
        denied |= is_denied
        if hhi_p is not None:
            max_hhi = max(max_hhi, float(hhi_p))
    return max_hhi, denied


def main() -> None:
    if not PARQUET.exists():
        sys.exit(f"missing {PARQUET}; run scripts/local_dataset_features.py first")
    df = pd.read_parquet(PARQUET).copy()
    df.loc[df["dataset"].isin(["S1_v2", "S1_v2_topup"]), "dataset"] = "S1_v2_full"

    metrics = df["elements"].apply(_row_metrics)
    df["max_hhi_p"] = metrics.apply(lambda t: t[0])
    df["denylisted"] = metrics.apply(lambda t: t[1])

    # --- denylist-only survival ---
    print("HARD DENYLIST (Tc, Pm, actinoids, Z>=84) — kept fraction:")
    print(f"{'dataset':<12} {'n':>7} {'kept':>7} {'kept%':>8} {'dropped%':>9}")
    print("-" * 46)
    for name in SERIES:
        sub = df[df["dataset"] == name]
        if len(sub) == 0:
            continue
        kept = (~sub["denylisted"]).sum()
        print(f"{name:<12} {len(sub):>7} {kept:>7} {100*kept/len(sub):>7.1f}% "
              f"{100*(1-kept/len(sub)):>8.1f}%")

    # --- HHI cap sweep (cost-aware) ---
    print("\nHHI-PRODUCTION CAP (reject if any element's HHI_p > threshold) — kept%:")
    head = f"{'dataset':<12}" + "".join(f"{'<'+str(t):>9}" for t in HHI_THRESHOLDS)
    print(head)
    print("-" * len(head))
    for name in SERIES:
        sub = df[df["dataset"] == name]
        if len(sub) == 0:
            continue
        cells = []
        for t in HHI_THRESHOLDS:
            kept = (sub["max_hhi_p"] <= t).sum()
            cells.append(f"{100*kept/len(sub):>8.1f}%")
        print(f"{name:<12}" + "".join(cells))

    # --- combined: denylist AND HHI cap ---
    print("\nDENYLIST + HHI CAP combined — kept%:")
    head = f"{'dataset':<12}" + "".join(f"{'<'+str(t):>9}" for t in HHI_THRESHOLDS)
    print(head)
    print("-" * len(head))
    for name in SERIES:
        sub = df[df["dataset"] == name]
        if len(sub) == 0:
            continue
        cells = []
        for t in HHI_THRESHOLDS:
            keep_mask = (~sub["denylisted"]) & (sub["max_hhi_p"] <= t)
            kept = keep_mask.sum()
            cells.append(f"{100*kept/len(sub):>8.1f}%")
        print(f"{name:<12}" + "".join(cells))

    # --- reference: HHI_p of a few elements for threshold intuition ---
    print("\nElement HHI_production reference (smact, 0-10000; higher = riskier):")
    refs = ["O", "Si", "Al", "Fe", "Cu", "Ga", "In", "Nd", "Dy", "Eu", "Pm", "Pt"]
    cells = []
    for sym in refs:
        hhi_p, _ = _elem_info(sym)
        cells.append(f"{sym}={hhi_p if hhi_p is not None else 'n/a'}")
    print("  " + "  ".join(cells))


if __name__ == "__main__":
    main()
