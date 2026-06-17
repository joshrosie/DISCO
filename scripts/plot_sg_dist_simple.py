"""Simple single-panel SG distribution plot: all 230 SGs, model vs dataset.

Bar = MP20 training frequency (gray fill).
Line/markers = Crystalite post-relax frequency.
Vertical bands/ticks mark crystal system boundaries so 230 buckets read cleanly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CIF_CSV = Path("figures/ch2/prerelax_sg_distribution.csv")
TRAIN_CSV = Path("data/mp20/raw/train.csv")
OUT_DIR = Path("figures/ch2")

SYMPREC = 0.1

CRYSTAL_SYSTEMS = [
    # name, lo, hi, short_label
    ("triclinic",     1, 2,    "tric"),
    ("monoclinic",    3, 15,   "monoclinic"),
    ("orthorhombic",  16, 74,  "orthorhombic"),
    ("tetragonal",    75, 142, "tetragonal"),
    ("trigonal",      143, 167,"trigonal"),
    ("hexagonal",     168, 194,"hexagonal"),
    ("cubic",         195, 230,"cubic"),
]


def main() -> None:
    df = pd.read_csv(CIF_CSV)
    df = df[df["symprec"] == SYMPREC].set_index("sg")

    train = pd.read_csv(TRAIN_CSV, usecols=["spacegroup.number"])
    train_sgs = train["spacegroup.number"].dropna().astype(int).values
    train_counts = np.bincount(train_sgs, minlength=231)[1:]
    train_freq = train_counts / train_counts.sum() * 100.0

    gen_freq = np.zeros(230)
    for sg in range(1, 231):
        if sg in df.index:
            gen_freq[sg - 1] = float(df.loc[sg, "freq"]) * 100.0

    fig, ax = plt.subplots(figsize=(14, 5))

    xs = np.arange(1, 231)

    # Training as gray bars (behind)
    ax.bar(
        xs, train_freq, width=1.0, color="lightgray",
        edgecolor="gray", linewidth=0.2, label=f"MP20 train (n={len(train_sgs)})",
    )
    # Model as red line + markers (on top)
    ax.plot(
        xs, gen_freq, color="#d62728", linewidth=1.3,
        marker="o", markersize=3.0, alpha=0.9,
        label="Crystalite (relaxed, n=2500)",
    )

    # Crystal-system bands (no labels inside the plot area — we'll put labels
    # on a dedicated top axis so they don't clash with the legend / bars).
    ymax = max(train_freq.max(), gen_freq.max()) * 1.1
    band_colors = ["#f7f7f7", "#ededed"]
    for i, (name, lo, hi, _) in enumerate(CRYSTAL_SYSTEMS):
        ax.axvspan(lo - 0.5, hi + 0.5, facecolor=band_colors[i % 2],
                   alpha=0.5, zorder=-10)

    ax.set_xlim(0.5, 230.5)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("spacegroup number")
    ax.set_ylabel("frequency (%)")
    ax.set_title(
        f"Spacegroup distribution: Crystalite (relaxed, n=2500) vs. MP20 training  "
        f"(symprec={SYMPREC})"
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    # Top axis dedicated to crystal-system labels + span brackets.
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    # Ticks at crystal-system centers, short labels. Stagger triclinic label
    # slightly right since its band (SG 1-2) is narrow and close to the left edge.
    centers = []
    labels = []
    for name, lo, hi, short in CRYSTAL_SYSTEMS:
        centers.append((lo + hi) / 2)
        labels.append(short)
    ax_top.set_xticks(centers)
    ax_top.set_xticklabels(labels, fontsize=9, color="#555555", style="italic")
    ax_top.tick_params(axis="x", which="both", length=0, pad=2)
    # Hide the top spine but keep the labels visible.
    for spine in ("top", "left", "right", "bottom"):
        ax_top.spines[spine].set_visible(False)

    # Minor ticks on the main x-axis at crystal-system boundaries so you can
    # see where each region starts/ends.
    boundary_ticks = []
    for name, lo, hi, _ in CRYSTAL_SYSTEMS[1:]:  # skip first left edge
        boundary_ticks.append(lo - 0.5)
    ax.set_xticks(boundary_ticks, minor=True)
    ax.tick_params(axis="x", which="minor", length=5, color="#999999")

    fig.tight_layout()
    fig_path = OUT_DIR / "sg_distribution_simple.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written to {fig_path}")


if __name__ == "__main__":
    main()
