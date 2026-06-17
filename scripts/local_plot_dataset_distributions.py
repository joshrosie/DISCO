#!/usr/bin/env python3
"""Plot distributional shifts across the curation-ladder and the round-on-round
synthetic augmentation.

Reads local_data/features.parquet (produced by local_dataset_features.py).
Generates two PDF/PNG figures into figures/flywheel/:

  curation_ladder_distributions.{png,pdf}
      Three-panel: n-ary distribution, volume-per-atom histogram,
      e_above_hull distribution. Series: MP20, S_raw, S_dedup, S0.
      Visualises what the verifier (msun_like + thermo filter) selects for.

  round_distribution_shift.{png,pdf}
      Four-panel: n-ary distribution, atoms-per-cell, density histogram,
      e_above_hull distribution. Series: MP20 → S0 → S1_v2_full → S2_v2 → S3_v2_full.
      Visualises whether the flywheel concentrates on a survivor distribution
      across iteration rounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOCAL = Path(__file__).resolve().parent.parent / "local_data"
OUT = Path(__file__).resolve().parent.parent / "figures" / "flywheel"
OUT.mkdir(parents=True, exist_ok=True)

PARQUET = LOCAL / "features.parquet"

PALETTE = {
    "MP20":          "#525252",  # neutral dark grey (reference)
    "S_raw":         "#a8324a",
    "S_dedup":       "#dd8452",
    "S0":            "#1f5a96",  # blue (full curation, Round 0)
    "S1_v2":         "#2f855a",  # green (Round 1)
    "S1_v2_full":    "#2f855a",
    "S2_v2":         "#1a4d3a",  # dark green (Round 2)
    "S3_v2_full":    "#0b2e1e",  # darkest green (Round 3)
}


def _subset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    return df[df["dataset"] == dataset]


def _plot_nary_grouped(ax, df, datasets, palette):
    nary_max = int(df["n_ary"].max())
    nary_range = list(range(1, min(nary_max, 8) + 1))
    bar_width = 0.8 / len(datasets)
    for i, name in enumerate(datasets):
        sub = _subset(df, name)
        if len(sub) == 0:
            continue
        counts = sub["n_ary"].value_counts(normalize=True)
        ys = [counts.get(k, 0.0) * 100 for k in nary_range]
        xs = [k + (i - (len(datasets) - 1) / 2) * bar_width for k in nary_range]
        ax.bar(xs, ys, width=bar_width, label=name, color=palette[name], alpha=0.92)
    ax.set_xticks(nary_range)
    ax.set_xlabel("Number of unique elements (n-ary)")
    ax.set_ylabel("% of structures")
    ax.set_title("Composition arity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)


def _plot_hist(ax, df, datasets, palette, column, *, bins, xlim=None,
               xlabel="", title="", clip_pct=(0.5, 99.5)):
    """KDE-ish step histogram, normalised to density."""
    if xlim is None:
        lo, hi = np.percentile(df[column].dropna(), clip_pct)
    else:
        lo, hi = xlim
    edges = np.linspace(lo, hi, bins + 1)
    for name in datasets:
        sub = _subset(df, name)[column].dropna()
        if len(sub) == 0:
            continue
        clipped = sub[(sub >= lo) & (sub <= hi)]
        h, _ = np.histogram(clipped, bins=edges, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centres, h, label=name, color=palette[name], linewidth=2.0)
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel("density")
    ax.set_title(title or column)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    if xlim is not None:
        ax.set_xlim(xlim)


def plot_curation_ladder(df: pd.DataFrame) -> None:
    datasets = ["MP20", "S_raw", "S_dedup", "S0"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), dpi=180)

    _plot_nary_grouped(axes[0, 0], df, datasets, PALETTE)
    _plot_atoms_per_cell(axes[0, 1], df, datasets, PALETTE)
    _plot_hist(
        axes[1, 0], df, datasets, PALETTE,
        column="density", bins=60,
        xlabel="Density (g/cm³)", title="Density",
    )
    # e_above_hull: MP-20 carries Materials Project DFT values; S0 carries
    # NequIP+MP2020 curation-time values. S_raw and S_dedup skip thermo so
    # their column is empty — only MP-20 and S0 plotted. MP-20 is a sharp
    # spike near 0 (DFT-tight), S0 is broader (NequIP estimates + spans the
    # metastable shell). Log y-axis lets both be legible without clipping.
    _plot_hist(
        axes[1, 1], df, ["MP20", "S0"], PALETTE,
        column="e_above_hull", bins=120, xlim=(-0.50, 0.30),
        xlabel="e_above_hull (eV/atom)",
        title="e_above_hull  (MP-20: DFT;  S0: NequIP+MP2020 curation)",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(0.01, 200)
    # mark the metastability filter cap at 0.1 eV/atom
    axes[1, 1].axvline(0.10, color="#9ca3af", linestyle=":", linewidth=0.9, alpha=0.7)
    axes[1, 1].text(0.105, 100, "msun cap", fontsize=7, color="#6b7280", va="top")

    fig.suptitle(
        "Curation-ladder distribution shifts vs MP-20  "
        "(Round 0: 27k augmentation budget)",
        fontsize=11,
    )
    fig.tight_layout()
    png = OUT / "curation_ladder_distributions.png"
    pdf = OUT / "curation_ladder_distributions.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def _plot_atoms_per_cell(ax, df, datasets, palette, n_max: int = 20):
    """Atoms-per-cell as a grouped bar chart over integer values 1..n_max."""
    ns = list(range(1, n_max + 1))
    bar_width = 0.8 / len(datasets)
    for i, name in enumerate(datasets):
        sub = _subset(df, name)["n_atoms"].dropna().astype(int)
        if len(sub) == 0:
            continue
        counts = sub.value_counts(normalize=True)
        ys = [counts.get(n, 0.0) * 100 for n in ns]
        xs = [n + (i - (len(datasets) - 1) / 2) * bar_width for n in ns]
        ax.bar(xs, ys, width=bar_width, label=name, color=palette[name], alpha=0.92)
    ax.set_xlabel("Atoms per cell")
    ax.set_ylabel("% of structures")
    ax.set_title("Atoms per cell")
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)


def plot_round_shift(df: pd.DataFrame) -> None:
    # merge S1_v2 + topup into a single Round-1 series
    s1_full = pd.concat([_subset(df, "S1_v2"), _subset(df, "S1_v2_topup")],
                        ignore_index=True)
    s1_full = s1_full.assign(dataset="S1_v2_full")
    plot_df = pd.concat(
        [_subset(df, "MP20"),
         _subset(df, "S0"),
         s1_full,
         _subset(df, "S2_v2"),
         _subset(df, "S3_v2_full")],
        ignore_index=True,
    )
    datasets = ["MP20", "S0", "S1_v2_full", "S2_v2", "S3_v2_full"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), dpi=180)

    _plot_nary_grouped(axes[0, 0], plot_df, datasets, PALETTE)
    _plot_atoms_per_cell(axes[0, 1], plot_df, datasets, PALETTE)
    _plot_hist(
        axes[1, 0], plot_df, datasets, PALETTE,
        column="density", bins=60,
        xlabel="Density (g/cm³)", title="Density",
    )
    # MP20 e_above_hull is DFT (different distribution from curation-time
    # NequIP+MP2020 values); drop it from this subplot for fair comparison.
    _plot_hist(
        axes[1, 1], plot_df, ["S0", "S1_v2_full", "S2_v2", "S3_v2_full"], PALETTE,
        column="e_above_hull", bins=60, xlim=(-0.15, 0.30),
        xlabel="e_above_hull (eV/atom)",
        title="Curation-time e_above_hull (S0 → S1 → S2 → S3)",
    )

    fig.suptitle(
        "Round-on-round distribution shift  "
        "(MP-20 → S0 → S1 → S2 → S3)",
        fontsize=11,
    )
    fig.tight_layout()
    png = OUT / "round_distribution_shift.png"
    pdf = OUT / "round_distribution_shift.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def main() -> None:
    if not PARQUET.exists():
        sys.exit(f"missing {PARQUET}; run scripts/local_dataset_features.py first")
    df = pd.read_parquet(PARQUET)
    print(df["dataset"].value_counts().sort_index())
    plot_curation_ladder(df)
    plot_round_shift(df)


if __name__ == "__main__":
    main()
