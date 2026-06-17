"""Plot the LeMat-MSUN and LeMat-SUN scaling curves across the Flywheel cascade.

For each Flywheel iteration we report:
  - external LeMat-(M)SUN  — the headline figure of merit
  - train-novel (M)SUN     — LeMat-(M)SUN after subtracting structures that
                              match the model's full training augmentation
                              under pymatgen StructureMatcher(ltol=0.1)
  - replay  pp             — external − train-novel
                              (the gap closed by replaying curated data)

M_big is included as an off-trend "one-shot" diamond at the M2 budget,
to show iteration > one-shot at matched effective training-set size.

Writes two plots into figures/flywheel/:
  flywheel_msun_vs_dataset_size.{png,pdf}   — MSUN version
  flywheel_sun_vs_dataset_size.{png,pdf}    — SUN  version
plus the underlying numbers in flywheel_msun_scaling.csv.

All numbers below are the v2 lineage (M1_v2, M2_v2, M3). Sources:
  outputs/msun_replay/m1v2/summary.json
  outputs/msun_replay/m2v2/summary.json
  outputs/msun_replay/m3/summary.json
  outputs/msun_replay/s_big/summary.json
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("figures/flywheel")


# Per-row schema (rates in percent):
#   external_msun, train_novel_msun, replay_msun
#   external_sun,  train_novel_sun,  replay_sun
ROWS = [
    {
        "label": "Base",
        "kind": "flywheel",
        "effective_train_size": 27138,
        "external_msun": 22.60,  "train_novel_msun": None, "replay_msun": None,
        "external_sun":  1.50,   "train_novel_sun":  None, "replay_sun":  None,
    },
    {
        "label": "Oversample",
        "kind": "control",
        "effective_train_size": 54276,
        "external_msun": 24.44,  "train_novel_msun": None, "replay_msun": None,
        "external_sun":  1.53,   "train_novel_sun":  None, "replay_sun":  None,
    },
    {
        "label": "M1",
        "kind": "flywheel",
        "effective_train_size": 54276,
        "external_msun": 29.34,
        "train_novel_msun": 26.10,
        "replay_msun": 3.24,
        "framework_novel_msun": 23.90,   # train-novel − f-block swap of MP20
        "substitution_msun": 2.20,       # anon-only MP20 match among train-novel
        "external_sun":  1.74,
        "train_novel_sun": 1.58,
        "replay_sun":  0.17,
    },
    {
        "label": "M2",
        "kind": "flywheel",
        "effective_train_size": 108552,
        "external_msun": 35.06,
        "train_novel_msun": 30.75,
        "replay_msun": 4.32,
        "framework_novel_msun": 28.01,
        "substitution_msun": 2.74,
        "external_sun":  2.28,
        "train_novel_sun": 1.54,
        "replay_sun":  0.75,
    },
    {
        "label": "M3",
        "kind": "flywheel",
        "effective_train_size": 217104,
        "external_msun": 40.43,
        "train_novel_msun": 32.74,
        "replay_msun": 7.69,
        # framework_novel_msun / substitution_msun: not yet computed for M3/M4
        "external_sun":  2.54,
        "train_novel_sun": 1.97,
        "replay_sun":  0.57,
    },
    {
        "label": "M4",
        "kind": "flywheel",
        "effective_train_size": 434208,
        "external_msun": 48.20,
        "train_novel_msun": 37.34,
        "replay_msun": 10.86,
        "external_sun":  2.90,
        "train_novel_sun": 2.12,
        "replay_sun":  0.78,
    },
    {
        "label": "M_big",
        "kind": "one_shot",
        "effective_train_size": 108552,
        "external_msun": 32.91,
        "train_novel_msun": 27.77,
        "replay_msun": 5.14,
        "external_sun":  1.82,
        "train_novel_sun": 1.61,
        "replay_sun":  0.21,
    },
]


def _plot_one(
    *,
    metric_short: str,            # "MSUN" or "SUN"
    ext_key: str,                 # "external_msun" / "external_sun"
    novel_key: str,               # "train_novel_msun" / "train_novel_sun"
    replay_key: str,              # "replay_msun" / "replay_sun"
    out_basename: str,
    ylim: tuple[float, float],
    framework_key: str | None = None,   # "framework_novel_msun" (None => skip)
    subst_key: str | None = None,       # "substitution_msun"
) -> None:
    flywheel = [row for row in ROWS if row["kind"] == "flywheel"]

    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=220)
    ax.plot(
        [row["effective_train_size"] for row in flywheel],
        [row[ext_key] for row in flywheel],
        marker="o",
        linewidth=2.25,
        markersize=5.5,
        color="#1f5a96",
        label=f"Flywheel: external LeMat-{metric_short}",
    )

    train_novel = [row for row in flywheel if row[novel_key] is not None]
    ax.plot(
        [row["effective_train_size"] for row in train_novel],
        [row[novel_key] for row in train_novel],
        marker="o",
        linestyle="--",
        linewidth=2.0,
        markersize=5.0,
        color="#2f855a",
        label=f"Flywheel: train-novel {metric_short}",
    )

    for row in train_novel:
        replay_dx = 10
        replay_ha = "left"
        if row["label"] == "M2":
            replay_dx = -10
            replay_ha = "right"
        ax.vlines(
            row["effective_train_size"],
            row[novel_key],
            row[ext_key],
            color="#7c3aed",
            linewidth=2.0,
            alpha=0.55,
        )
        ax.annotate(
            f"replay\n+{row[replay_key]:.2f} pp" if row[replay_key] < 1
            else f"replay\n+{row[replay_key]:.1f} pp",
            (
                row["effective_train_size"],
                (row[novel_key] + row[ext_key]) / 2.0,
            ),
            textcoords="offset points",
            xytext=(replay_dx, 0),
            ha=replay_ha,
            va="center",
            fontsize=7,
            color="#5b21b6",
        )

    for row in ROWS:
        if row["kind"] != "flywheel":
            continue
        dx = 0
        dy = 8
        if row["label"] == "M1":
            dx = -11
        if row["label"] == "M2":
            dx = -9
        ext_val = row[ext_key]
        headline = f"{ext_val:.1f}%" if ext_val >= 1 else f"{ext_val:.2f}%"
        ax.annotate(
            f"{row['label']}\n{headline}",
            (row["effective_train_size"], ext_val),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=8,
        )

    for row in train_novel:
        train_label_offset = (-7, -16)
        train_label_ha = "right"
        train_label_va = "top"
        if row["label"] == "M1":
            train_label_offset = (-10, 8)
            train_label_ha = "right"
            train_label_va = "bottom"
        val = row[novel_key]
        ax.annotate(
            f"{val:.1f}%" if val >= 1 else f"{val:.2f}%",
            (row["effective_train_size"], val),
            textcoords="offset points",
            xytext=train_label_offset,
            ha=train_label_ha,
            va=train_label_va,
            fontsize=7,
            color="#276749",
        )

    # Framework-novel (swap) line and one-shot (M_big) diamonds are intentionally
    # not rendered here. This plot is the methodology figure showing the replay
    # decomposition (external LeMat-MSUN vs train-novel MSUN) on the cascade only.
    # M_big lives in flywheel_lemat_*_vs_size; swap decomposition (if needed) is
    # a separate sub-figure.

    ax.set_xscale("log", base=2)
    ax.set_xticks([27138, 54276, 108552, 217104, 434208])
    ax.set_xticklabels(["27k", "54k", "109k", "217k", "434k"])
    ax.set_xlabel("Effective training set size")
    ax.set_ylabel(f"{metric_short} (%)")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.12, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=7.0)
    caption = "Train-novel = unique & novel vs full augmented training set"
    ax.text(
        0.02,
        0.02,
        caption,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.6,
        color="#4b5563",
    )
    fig.tight_layout()

    png_path = OUT_DIR / f"{out_basename}.png"
    pdf_path = OUT_DIR / f"{out_basename}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "flywheel_msun_scaling.csv"
    fieldnames = list(dict.fromkeys(k for row in ROWS for k in row.keys()))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"wrote {csv_path}")

    _plot_one(
        metric_short="MSUN",
        ext_key="external_msun",
        novel_key="train_novel_msun",
        replay_key="replay_msun",
        out_basename="flywheel_msun_vs_dataset_size",
        ylim=(20, 52),
        framework_key="framework_novel_msun",
        subst_key="substitution_msun",
    )

    _plot_one(
        metric_short="SUN",
        ext_key="external_sun",
        novel_key="train_novel_sun",
        replay_key="replay_sun",
        out_basename="flywheel_sun_vs_dataset_size",
        ylim=(1.0, 3.2),
    )


if __name__ == "__main__":
    main()
