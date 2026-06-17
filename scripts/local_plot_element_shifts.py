#!/usr/bin/env python3
"""Plot per-element distribution shifts across the flywheel cascade.

Two figures into figures/flywheel/:

  element_shifts_topN.{png,pdf}
      Top-N elements by absolute Δ(MP20 → S1) — grouped bar per element
      with MP20 / S0 / S1 frequencies. % of structures containing the
      element. Emphasises which elements are over/underrepresented at
      each cascade step.

  element_shift_scatter.{png,pdf}
      S1 freq (y) vs MP20 freq (x) scatter, one point per element, with
      labels for movers. y=x diagonal marks no-change. Points above the
      line gained, below lost.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "local_data"
OUT = ROOT / "figures" / "flywheel"
OUT.mkdir(parents=True, exist_ok=True)

PARQUET = LOCAL / "features.parquet"

COL = {
    "MP20": "#525252",
    "S0":   "#1f5a96",
    "S1":   "#2f855a",
    "S2":   "#1a4d3a",
    "S3":   "#0b2e1e",
}

SERIES = ("MP20", "S0", "S1", "S2", "S3")


def _frequencies(df: pd.DataFrame, dataset: str) -> dict[str, float]:
    sub = df[df["dataset"] == dataset]
    n = max(1, len(sub))
    c: Counter = Counter()
    for els in sub["elements"]:
        for e in els.split(";"):
            if e:
                c[e] += 1
    return {e: 100 * v / n for e, v in c.items()}


def load_combined() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    df = pd.read_parquet(PARQUET)
    s1 = pd.concat([
        df[df.dataset == "S1_v2"],
        df[df.dataset == "S1_v2_topup"],
    ]).assign(dataset="S1")
    s2 = df[df.dataset == "S2_v2"].assign(dataset="S2")
    s3 = df[df.dataset == "S3_v2_full"].assign(dataset="S3")
    combined = pd.concat(
        [df[df.dataset.isin(["MP20", "S0"])], s1, s2, s3],
        ignore_index=True,
    )

    freqs = {ds: _frequencies(combined, ds) for ds in SERIES}
    return combined, freqs


def plot_top_movers(freqs: dict[str, dict[str, float]], top_n: int = 18) -> None:
    all_els = set().union(*[d.keys() for d in freqs.values()])
    # Rank by max |Δ| across S0/S1/S2/S3 vs MP20.
    def _max_abs_delta(e: str) -> float:
        return max(abs(freqs[ds].get(e, 0) - freqs["MP20"].get(e, 0))
                   for ds in ("S0", "S1", "S2", "S3"))
    movers = sorted(all_els, key=lambda e: -_max_abs_delta(e))[:top_n]
    # order by signed Δ(MP20→S3) for nicer reading (gainers first, then losers)
    movers.sort(key=lambda e: freqs["S3"].get(e, 0) - freqs["MP20"].get(e, 0), reverse=True)

    fig, ax = plt.subplots(figsize=(13, 4.8), dpi=200)
    x = np.arange(len(movers))
    bar_w = 0.17
    n = len(SERIES)

    for i, ds in enumerate(SERIES):
        ys = [freqs[ds].get(e, 0.0) for e in movers]
        ax.bar(
            x + (i - (n - 1) / 2) * bar_w,
            ys, width=bar_w, label=ds, color=COL[ds], alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(movers, fontsize=9)
    ax.set_ylabel("% of structures containing element")
    ax.set_title(f"Top {top_n} elements by max |Δ vs MP20| — flywheel cascade "
                 f"(MP20 → S0 → S1 → S2 → S3)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=5)

    # annotate signed Δ(MP20→S3) on top of each cluster — the deepest-cascade gap
    for i, e in enumerate(movers):
        delta = freqs["S3"].get(e, 0) - freqs["MP20"].get(e, 0)
        top = max(freqs[d].get(e, 0) for d in SERIES)
        color = "#2f855a" if delta > 0 else "#a8324a"
        ax.text(i, top + 0.6, f"{delta:+.1f}", ha="center", va="bottom",
                fontsize=7.5, color=color)

    fig.tight_layout()
    png = OUT / "element_shifts_topN.png"
    pdf = OUT / "element_shifts_topN.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def plot_scatter_s3_vs_mp20(freqs: dict[str, dict[str, float]]) -> None:
    """Scatter MP20 (x) vs S3 (y) frequencies, per element.
    Points above y=x gained; below lost.
    """
    all_els = set().union(*[d.keys() for d in freqs.values()])
    xs = np.array([freqs["MP20"].get(e, 0.0) for e in all_els])
    ys = np.array([freqs["S3"].get(e, 0.0) for e in all_els])
    names = list(all_els)

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    deltas = ys - xs
    colors = ["#2f855a" if d > 0 else "#a8324a" if d < 0 else "#9ca3af"
              for d in deltas]
    ax.scatter(xs, ys, s=24, c=colors, alpha=0.9, edgecolor="white",
               linewidth=0.6)

    hi = max(xs.max(), ys.max()) * 1.08
    ax.plot([0, hi], [0, hi], "--", color="#9ca3af", linewidth=1.0, alpha=0.7)

    label_thresh = 1.5
    for x_, y_, name, d in zip(xs, ys, names, deltas):
        if abs(d) >= label_thresh or x_ > 8:
            ax.annotate(name, (x_, y_),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=8, color="#374151")

    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("MP20: % of structures containing element")
    ax.set_ylabel("S3: % of structures containing element")
    ax.set_title("Element-frequency drift, MP20 → S3 (deepest cascade)\n"
                 "above y=x: gained; below: displaced",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    fig.tight_layout()

    png = OUT / "element_shift_scatter.png"
    pdf = OUT / "element_shift_scatter.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def main() -> None:
    _, freqs = load_combined()
    plot_top_movers(freqs, top_n=18)
    plot_scatter_s3_vs_mp20(freqs)


if __name__ == "__main__":
    main()
