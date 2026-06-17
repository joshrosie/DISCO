#!/usr/bin/env python3
"""Plot the MSUN and SUN replay vs true-train-novel decomposition for each model.

For every model evaluated by LeMat-GenBench we have the canonical breakdown:

    LeMat-MSUN          = M ∧ U ∧ (no match in LeMat-Bulk)
    Replay              = LeMat-MSUN ∧ (matches the model's full
                          curated synthetic augmentation, under
                          pymatgen StructureMatcher(ltol=0.1))
    True train-novel MSUN = LeMat-MSUN − Replay
                          = novel against (LeMat-Bulk ∪ training_synth)

Produces two figures into figures/flywheel/:

    replay_decomposition.{png,pdf}      — MSUN version
    replay_decomposition_sun.{png,pdf}  — SUN version

Both share the same layout: horizontal stacked bar per model with "replay"
hatched and "true train-novel" solid, plus the residual rendered as light
grey for context.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).resolve().parent.parent / "figures" / "flywheel"
OUT.mkdir(parents=True, exist_ok=True)


# --- canonical numbers per model -----------------------------------------
# Sources: outputs/msun_replay/<run_tag>/summary.json
# Plus the public Crystalite and oversample-real baseline rows from
# docs/augmentation/results_log.md (LeMat-GenBench eval, n=2500).
# All rates are percent of the n_valid (LeMat's valid-structure denominator).
#
# Schema: (label, lemat_msun, replay_msun, true_msun,
#                 lemat_sun,  replay_sun,  true_sun)
# Rates in percent.

ROWS = [
    # baselines (no synthetic in training, so replay = 0 by construction)
    ("Public Crystalite (M0)",        22.60, 0.00, 22.60,   1.50, 0.00, 1.50),
    ("Oversample-real (C0)",          24.44, 0.00, 24.44,   1.53, 0.00, 1.53),
    # curation-ladder ablations (Round 0, matched 27k augmentation budget)
    ("M_raw (no filter)",             24.19, 3.28, 20.91,   1.04, 0.17, 0.87),
    ("M_dedup (dedup only)",          24.29, 1.38, 22.91,   0.92, 0.08, 0.84),
    # canonical v2 lineage (chapter primary)
    ("M1 — Round 0 (S0)",             29.34, 3.24, 26.10,   1.74, 0.17, 1.58),
    ("M2 — Round 1 (S0 ∪ S1)",        35.06, 4.32, 30.75,   2.28, 0.75, 1.54),
]

COL_TRUE   = "#1f5a96"   # blue – true train-novel
COL_REPLAY = "#7c3aed"   # purple – replay
COL_REST   = "#e5e7eb"   # light grey – residual ("not MSUN" / "not SUN")


def _plot_decomposition(
    rows: list[tuple],
    lemat_idx: int,
    replay_idx: int,
    true_idx: int,
    *,
    title: str,
    rest_label: str,
    true_label: str,
    metric_short: str,   # "MSUN" or "SUN" — for the outside-bar headline annotation
    out_basename: str,
    xlim_pct: float | None = None,
) -> None:
    labels = [r[0] for r in rows]
    lemat  = [r[lemat_idx]  for r in rows]
    replay = [r[replay_idx] for r in rows]
    true_n = [r[true_idx]   for r in rows]
    rest   = [100.0 - m for m in lemat]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * n_rows + 1.4), dpi=200)
    y = list(range(n_rows))

    ax.barh(y, true_n, color=COL_TRUE, edgecolor="white", label=true_label)
    ax.barh(y, replay, left=true_n, color=COL_REPLAY,
            edgecolor="white", hatch="//", label="Replay (in training augmentation)")
    cumulative = [t + r for t, r in zip(true_n, replay)]
    ax.barh(y, rest, left=cumulative, color=COL_REST,
            edgecolor="white", label=rest_label)

    xlim_top = xlim_pct if xlim_pct is not None else 100
    text_outside_offset = max(xlim_top * 0.01, 0.5)

    for i, (t, r, m) in enumerate(zip(true_n, replay, lemat)):
        # decide whether the segment is big enough for an inline label
        if t > xlim_top * 0.02:
            ax.text(t / 2, i, f"{t:.1f}%" if t >= 1 else f"{t:.2f}%",
                    ha="center", va="center", color="white", fontsize=8.5)
        if r > xlim_top * 0.015:
            label_r = f"{r:.1f}" if r >= 1 else f"{r:.2f}"
            ax.text(t + r / 2, i, label_r, ha="center", va="center",
                    color="white", fontsize=7.5)
        headline = f"{m:.1f}%" if m >= 1 else f"{m:.2f}%"
        ax.text(m + text_outside_offset, i,
                f"LeMat-{metric_short} {headline}", ha="left", va="center",
                fontsize=8, color="#4b5563")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, xlim_top)
    ax.set_xlabel("% of valid LeMat-evaluated structures")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.7)

    ax.legend(
        handles=[
            mpatches.Patch(color=COL_TRUE, label=true_label),
            mpatches.Patch(facecolor=COL_REPLAY, hatch="//",
                           label="Replay (in training augmentation)"),
            mpatches.Patch(color=COL_REST, label=rest_label),
        ],
        loc="lower right", frameon=False, fontsize=8,
        bbox_to_anchor=(1.0, -0.18), ncol=3,
    )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    png = OUT / f"{out_basename}.png"
    pdf = OUT / f"{out_basename}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def main() -> None:
    _plot_decomposition(
        ROWS,
        lemat_idx=1, replay_idx=2, true_idx=3,
        title="MSUN replay decomposition  "
              "(LeMat-MSUN = true train-novel + replay)",
        rest_label="Not MSUN",
        true_label="True train-novel MSUN",
        metric_short="MSUN",
        out_basename="replay_decomposition",
        xlim_pct=100,
    )

    # SUN values are small (0–3% range). Zoom the x-axis so the
    # replay vs true-novel slices are readable.
    _plot_decomposition(
        ROWS,
        lemat_idx=4, replay_idx=5, true_idx=6,
        title="SUN replay decomposition  "
              "(LeMat-SUN = true train-novel + replay)",
        rest_label="Not SUN",
        true_label="True train-novel SUN",
        metric_short="SUN",
        out_basename="replay_decomposition_sun",
        xlim_pct=4.5,
    )


if __name__ == "__main__":
    main()
