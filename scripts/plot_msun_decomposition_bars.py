"""Stacked-bar decomposition of external LeMat-MSUN into kinds of novelty.

Companion to flywheel_msun_vs_dataset_size.png. Every component is inside the
external MSUN (i.e. novel vs LeMat-Bulk); the split characterises *what kind* of
novelty each model produces — it is a decomposition, NOT a discount.

Partition is nested, matching diagnose_msun_novelty_partition.py:
  1. replay     — matches the model's own synthetic training augmentation
                  (element-sensitive, ltol=0.1)
  2. substitution — among the remaining (train-novel) structures, those that
                    match MP-20 ONLY after lanthanide anonymisation (anon-only
                    match; an as-is MP-20 match would be a matcher-tolerance
                    artifact, not a chemistry claim, so those are bundled into
                    framework-novel)
  3. new-framework novel — the residual: neither replay nor f-block swap

Substitution is a lower bound (MP-20 reference only; LeMat novelty is vs
LeMat-Bulk which is larger). Numbers match thesis Table 4.6 / Appendix H.

Base has no synthetic augmentation (replay = 0) and was not decomposed for
substitution (no MSUN-index JSON available locally), so it is shown as a single
external-MSUN bar for reference.

Writes figures/flywheel/msun_decomposition_bars.{png,pdf}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = Path("figures/flywheel")

# rates as % of valid structures; components sum to external MSUN.
# Numbers from the nested partition (replay -> anon-only substitution -> framework-novel),
# matching thesis Table 4.6 and Appendix H. Substitution counts only anon-only matches;
# as-is MP-20 matcher discrepancies are left in framework-novel.
ROWS = [
    {"label": "Base\n(MP-20)",       "external": 22.60, "new_framework": None, "substitution": None, "replay": 0.0},
    {"label": "M1\n(MP-20+S0)",      "external": 29.34, "new_framework": 23.90, "substitution": 2.20, "replay": 3.24},
    {"label": "M2\n(MP-20+S0+S1)",   "external": 35.06, "new_framework": 28.01, "substitution": 2.74, "replay": 4.32},
]

COL_FRAMEWORK = "#1f5a96"  # blue  – new-framework novel
COL_SUBST     = "#dd8452"  # amber – substitutional (f-block swap)
COL_REPLAY    = "#7c3aed"  # purple – replay
COL_UNDECOMP  = "#9ca3af"  # grey  – external MSUN, not decomposed (Base)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=220)

    x = list(range(len(ROWS)))
    width = 0.62

    for i, row in enumerate(ROWS):
        if row["new_framework"] is None:
            # undecomposed external MSUN (Base)
            ax.bar(i, row["external"], width, color=COL_UNDECOMP, alpha=0.55,
                   edgecolor="white")
            ax.text(i, row["external"] + 0.5, f"{row['external']:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(i, row["external"] / 2, "not\ndecomposed", ha="center",
                    va="center", fontsize=7, color="white")
            continue

        nf, sub, rep = row["new_framework"], row["substitution"], row["replay"]
        ax.bar(i, nf, width, color=COL_FRAMEWORK, edgecolor="white")
        ax.bar(i, sub, width, bottom=nf, color=COL_SUBST, edgecolor="white")
        ax.bar(i, rep, width, bottom=nf + sub, color=COL_REPLAY,
               edgecolor="white", hatch="//")

        # segment labels
        ax.text(i, nf / 2, f"{nf:.1f}", ha="center", va="center",
                color="white", fontsize=8.5)
        ax.text(i, nf + sub / 2, f"{sub:.1f}", ha="center", va="center",
                color="white", fontsize=7.5)
        ax.text(i, nf + sub + rep / 2, f"{rep:.1f}", ha="center", va="center",
                color="white", fontsize=7.5)
        # total on top
        ax.text(i, row["external"] + 0.5, f"{row['external']:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in ROWS], fontsize=9)
    ax.set_ylabel("LeMat-MSUN (% of valid)")
    ax.set_ylim(0, 40)
    ax.set_title("What kind of novelty? — MSUN decomposition", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)

    ax.legend(
        handles=[
            mpatches.Patch(color=COL_FRAMEWORK, label="New-framework novel"),
            mpatches.Patch(color=COL_SUBST, label="Substitutional (f-block swap of MP-20)"),
            mpatches.Patch(facecolor=COL_REPLAY, hatch="//", label="Replay (matches synthetic aug.)"),
            mpatches.Patch(color=COL_UNDECOMP, alpha=0.55, label="External MSUN (not decomposed)"),
        ],
        loc="upper left", frameon=False, fontsize=7.6,
    )
    ax.text(0.02, -0.16,
            "All components are novel vs LeMat-Bulk; the split shows the kind of "
            "novelty.\nSubstitution = anon-only MP-20 match (lower bound, MP-20 reference). "
            "Partition is nested (replay first).",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color="#4b5563")

    fig.tight_layout()
    png = OUT_DIR / "msun_decomposition_bars.png"
    pdf = OUT_DIR / "msun_decomposition_bars.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
