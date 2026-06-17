#!/usr/bin/env python3
"""Plot per-element-class frequency shift across the flywheel cascade.

Bins elements into chemistry-relevant classes (alkali, alkaline earth,
transition metal, lanthanide, post-transition metal, metalloid, reactive
non-metal, halogen, chalcogen, noble gas, actinide) using pymatgen's
periodic-table classification. For each class, plots the fraction of
structures containing at least one element of that class, across
{MP20, S0, S1, S2}.

Writes figures/flywheel/element_class_shift.{png,pdf}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Element

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "local_data"
OUT = ROOT / "figures" / "flywheel"
OUT.mkdir(parents=True, exist_ok=True)
PARQUET = LOCAL / "features.parquet"

COL = {
    "MP20": "#525252",   # gray (baseline)
    "S0":   "#1f5a96",   # blue (cascade round 0)
    "S1":   "#2f855a",   # green (cascade round 1)
    "S2":   "#1a4d3a",   # dark green (cascade round 2)
    "S3":   "#0b2e1e",   # darkest green (cascade round 3)
}

SERIES = ("MP20", "S0", "S1", "S2", "S3")

# Element-class definitions. We use a small priority order so each element
# falls into exactly one class — the lanthanide / actinide checks come
# first because pymatgen classifies them also as metals.
def element_class(sym: str) -> str:
    try:
        el = Element(sym)
    except Exception:
        return "other"
    if el.is_lanthanoid:
        return "lanthanide"
    if el.is_actinoid:
        return "actinide"
    if el.is_alkali:
        return "alkali"
    if el.is_alkaline:
        return "alkaline earth"
    if el.is_transition_metal:
        return "transition metal"
    if el.is_post_transition_metal:
        return "post-transition metal"
    if el.is_metalloid:
        return "metalloid"
    if el.is_halogen:
        return "halogen"
    if el.is_chalcogen:
        return "chalcogen"
    if el.is_noble_gas:
        return "noble gas"
    # everything else (H, C, N, P, ...) — the light reactive non-metals
    return "reactive non-metal"


CLASS_ORDER = [
    "lanthanide",
    "actinide",
    "transition metal",
    "post-transition metal",
    "metalloid",
    "alkali",
    "alkaline earth",
    "reactive non-metal",
    "chalcogen",
    "halogen",
    "noble gas",
]


def class_frequencies(df: pd.DataFrame, dataset: str) -> dict[str, float]:
    """Fraction of structures in `dataset` that contain ≥1 element of each class."""
    sub = df[df["dataset"] == dataset]
    n = max(1, len(sub))
    counts = {cls: 0 for cls in CLASS_ORDER}
    for els in sub["elements"]:
        seen = set()
        for e in els.split(";"):
            if not e:
                continue
            cls = element_class(e)
            if cls in counts:
                seen.add(cls)
        for cls in seen:
            counts[cls] += 1
    return {cls: 100 * counts[cls] / n for cls in CLASS_ORDER}


def main() -> None:
    df = pd.read_parquet(PARQUET)
    s1 = pd.concat([df[df.dataset == "S1_v2"], df[df.dataset == "S1_v2_topup"]]).assign(dataset="S1")
    s2 = df[df.dataset == "S2_v2"].assign(dataset="S2")
    s3 = df[df.dataset == "S3_v2_full"].assign(dataset="S3")
    combined = pd.concat(
        [df[df.dataset.isin(["MP20", "S0"])], s1, s2, s3],
        ignore_index=True,
    )

    freqs = {ds: class_frequencies(combined, ds) for ds in SERIES}

    # Order classes by S3 frequency (descending) — S3 is the deepest cascade
    # point so this puts the most-pushed classes first.
    order = sorted(CLASS_ORDER, key=lambda c: -freqs["S3"][c])

    fig, ax = plt.subplots(figsize=(12.5, 4.8), dpi=200)
    x = np.arange(len(order))
    bar_w = 0.17
    n = len(SERIES)

    for i, ds in enumerate(SERIES):
        ys = [freqs[ds][c] for c in order]
        ax.bar(
            x + (i - (n - 1) / 2) * bar_w,
            ys, width=bar_w, label=ds, color=COL[ds], alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("% of structures containing ≥1 element of class")
    ax.set_title("Element-class frequency drift across the flywheel cascade "
                 "(MP20 → S0 → S1 → S2 → S3)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=5)

    # Annotate Δ(MP20→S3) above each cluster — the headline cascade drift.
    for i, cls in enumerate(order):
        delta = freqs["S3"][cls] - freqs["MP20"][cls]
        top = max(freqs[d][cls] for d in SERIES)
        color = "#2f855a" if delta > 0 else "#a8324a" if delta < 0 else "#6b7280"
        ax.text(i, top + 1.4, f"{delta:+.1f}", ha="center", va="bottom",
                fontsize=8, color=color)

    fig.tight_layout()
    png = OUT / "element_class_shift.png"
    pdf = OUT / "element_class_shift.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
