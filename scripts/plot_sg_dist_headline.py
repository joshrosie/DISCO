"""A single clean figure: Crystalite (post-relax) SG distribution vs MP20 train.

Two panels:
  (left) Scatter: generated freq vs training freq, log-log. Each SG is a point,
    colored by crystal system, sized by |log ratio|. Diagonal = parity.
    Points above diagonal = over-produced; below = under-produced.
  (right) Filtered bar plot: side-by-side train vs generated for the SGs with
    > 1% mass in either distribution. Sorted by training mass. Crystal-system
    band coloring.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

CIF_CSV = Path("figures/ch2/prerelax_sg_distribution.csv")
TRAIN_CSV = Path("data/mp20/raw/train.csv")
OUT_DIR = Path("figures/ch2")

SYMPREC = 0.1
MIN_FREQ_FOR_BARS = 0.01  # Show SG bars only if train or gen has ≥1% mass

# Crystal system colors (by SG number range)
CRYSTAL_SYSTEMS = [
    ("triclinic",     1, 2,    "#9ecae1"),
    ("monoclinic",    3, 15,   "#6baed6"),
    ("orthorhombic",  16, 74,  "#4292c6"),
    ("tetragonal",    75, 142, "#fdae6b"),
    ("trigonal",      143, 167,"#fd8d3c"),
    ("hexagonal",     168, 194,"#e6550d"),
    ("cubic",         195, 230,"#a63603"),
]


def sg_to_cs(sg: int) -> tuple[str, str]:
    for name, lo, hi, color in CRYSTAL_SYSTEMS:
        if lo <= sg <= hi:
            return name, color
    return "other", "#bbbbbb"


def main() -> None:
    df = pd.read_csv(CIF_CSV)
    df = df[df["symprec"] == SYMPREC].set_index("sg")

    train = pd.read_csv(TRAIN_CSV, usecols=["spacegroup.number"])
    train_sgs = train["spacegroup.number"].dropna().astype(int).values
    train_counts = np.bincount(train_sgs, minlength=231)[1:]
    train_freq = train_counts / train_counts.sum()

    # Build per-SG record.
    rows = []
    for sg in range(1, 231):
        gen = float(df.loc[sg, "freq"]) if sg in df.index else 0.0
        tr = float(train_freq[sg - 1])
        cs_name, cs_color = sg_to_cs(sg)
        rows.append({"sg": sg, "train": tr, "gen": gen,
                     "cs_name": cs_name, "cs_color": cs_color})
    data = pd.DataFrame(rows)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2),
                              gridspec_kw={"width_ratios": [1, 1.6]})

    # --- LEFT: scatter generated vs training, log-log, colored by crystal system ---
    ax = axes[0]
    for name, lo, hi, color in CRYSTAL_SYSTEMS:
        sub = data[(data["sg"] >= lo) & (data["sg"] <= hi)]
        sub = sub[(sub["train"] > 0) | (sub["gen"] > 0)]
        ax.scatter(
            sub["train"].clip(lower=1e-5),
            sub["gen"].clip(lower=1e-5),
            s=26, color=color, edgecolors="black", linewidths=0.4,
            label=name, alpha=0.9,
        )

    # Parity diagonal
    lim = [5e-5, 0.4]
    ax.plot(lim, lim, color="black", linestyle="--", linewidth=1, alpha=0.6,
            label="parity (model = train)")
    # 2× and 0.5× guide lines
    xs = np.linspace(lim[0], lim[1], 100)
    ax.plot(xs, 2 * xs, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.plot(xs, 0.5 * xs, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.text(lim[1] * 0.5, lim[1] * 0.95, "2×", fontsize=8, color="gray")
    ax.text(lim[1] * 0.95, lim[1] * 0.48, "0.5×", fontsize=8, color="gray")

    # Annotate a few landmarks
    annotations = [
        (1, "P1"), (6, "Pm"), (14, "P21/c"), (62, "Pnma"),
        (194, "P63/mmc"), (225, "Fm-3m"),
    ]
    for sg, name in annotations:
        r = data[data["sg"] == sg].iloc[0]
        tx = r["train"] if r["train"] > 0 else 5e-5
        ty = r["gen"] if r["gen"] > 0 else 5e-5
        ax.annotate(
            f"SG={sg}\n{name}", (tx, ty),
            xytext=(8, 4), textcoords="offset points",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("MP20 training frequency")
    ax.set_ylabel("Crystalite generated frequency")
    ax.set_title("Per-spacegroup: generated vs training")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)

    # --- RIGHT: filtered bar plot for SGs with ≥ 1% in either ---
    ax = axes[1]
    keep = data[(data["train"] >= MIN_FREQ_FOR_BARS) | (data["gen"] >= MIN_FREQ_FOR_BARS)]
    keep = keep.sort_values("train", ascending=False).reset_index(drop=True)
    x = np.arange(len(keep))
    width = 0.4
    # Bar colors by crystal system
    colors = keep["cs_color"].values
    ax.bar(
        x - width/2, keep["train"], width=width, color=colors,
        edgecolor="black", linewidth=0.4, label="MP20 train",
    )
    ax.bar(
        x + width/2, keep["gen"], width=width, color=colors, alpha=0.55,
        hatch="///", edgecolor="black", linewidth=0.4, label="Crystalite (relaxed)",
    )
    labels = [f"{int(r.sg)}" for r in keep.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("frequency")
    ax.set_xlabel("spacegroup number (filtered to ≥ 1% in either)")
    ax.set_title(
        f"SG frequencies (symprec={SYMPREC}, n_train={len(train_sgs)}, "
        f"n_gen=2500)"
    )
    ax.grid(True, axis="y", alpha=0.3)

    # Hybrid legend: hatch = generated, solid = train
    legend_elements = [
        Patch(facecolor="lightgray", edgecolor="black", label="MP20 train"),
        Patch(facecolor="lightgray", edgecolor="black", hatch="///",
              alpha=0.55, label="Crystalite (relaxed)"),
    ]
    # Add crystal system colors
    for name, _, _, color in CRYSTAL_SYSTEMS:
        legend_elements.append(Patch(facecolor=color, edgecolor="black", label=name))
    ax.legend(handles=legend_elements, fontsize=7, ncol=2, loc="upper right")

    fig.suptitle(
        "Crystalite's symmetry prior: over-produces simple-operation groups, "
        "under-produces multi-operation groups",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "sg_distribution_headline.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written to {fig_path}")


if __name__ == "__main__":
    main()
