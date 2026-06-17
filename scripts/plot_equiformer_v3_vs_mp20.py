#!/usr/bin/env python
"""Plot EquiformerV3 vs MP-20 e_above_hull from the CSV produced by
`equiformer_v3_vs_mp20_ehull.py`.

Generates:
  - histogram overlay: mp20 stored vs eqv3 raw vs eqv3 + MP2020
  - scatter (2 panels): mp20_e_hull vs eqv3 e_hull, raw on left, +MP2020 on right
  - summary numbers printed to stdout (RMSE, bias, Spearman, strict-stable label agreement)

Runs in the main repo env (matplotlib + pandas, no fairchem):

    uv run python scripts/plot_equiformer_v3_vs_mp20.py \\
        --csv results/equiformer_v3_vs_mp20_all.csv \\
        --output-dir figures/augmentation/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results/equiformer_v3_vs_mp20_all.csv"
DEFAULT_OUTDIR = REPO_ROOT / "figures/augmentation"


def summarize(df: pd.DataFrame) -> None:
    print(f"\n=== {len(df)} entries from {df['sample_id'].nunique()} unique samples ===")
    print(f"  num_atoms:  {df['num_atoms'].describe()[['mean', '50%', 'max']].to_dict()}")
    print(f"  mp20_e_hull range: "
          f"[{df['mp20_e_hull'].min():.3f}, {df['mp20_e_hull'].max():.3f}] eV/atom")

    for col, label in [
        ("eqv3_e_hull_raw", "raw eqv3"),
        ("eqv3_e_hull_mp2020", "eqv3 + MP2020"),
    ]:
        sub = df[df[col].notna() & df["mp20_e_hull"].notna()]
        if len(sub) < 2:
            print(f"  {label}: too few valid entries ({len(sub)}) for stats")
            continue
        diff = sub[col].values - sub["mp20_e_hull"].values
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        bias = float(np.mean(diff))
        rho, _ = stats.spearmanr(sub["mp20_e_hull"], sub[col])
        # Strict-stable boundary agreement (DFT says <=0, prediction also says <=0).
        dft_stable = sub["mp20_e_hull"] <= 0
        pred_stable = sub[col] <= 0
        tp = int((dft_stable & pred_stable).sum())
        fp = int((~dft_stable & pred_stable).sum())
        fn = int((dft_stable & ~pred_stable).sum())
        tn = int((~dft_stable & ~pred_stable).sum())
        agree = (tp + tn) / max(1, len(sub))
        # Metastable boundary agreement (DFT <= 0.1).
        dft_meta = sub["mp20_e_hull"] <= 0.1
        pred_meta = sub[col] <= 0.1
        meta_agree = ((dft_meta == pred_meta).sum()) / max(1, len(sub))

        print(f"\n  {label}  (n={len(sub)})")
        print(f"    bias:                  {bias:+.4f} eV/atom")
        print(f"    MAE:                   {mae:.4f} eV/atom")
        print(f"    RMSE:                  {rmse:.4f} eV/atom")
        print(f"    Spearman ρ:            {rho:.4f}")
        print(f"    strict-stable agree:   {agree:.2%}  (tp={tp}, fp={fp}, fn={fn}, tn={tn})")
        print(f"    metastable (<=0.1):    {meta_agree:.2%}")


def plot_histogram(df: pd.DataFrame, outpath: Path, x_clip: tuple[float, float] = (-0.1, 0.5)) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(x_clip[0], x_clip[1], 81)

    mp20 = df["mp20_e_hull"].dropna().clip(*x_clip)
    raw = df["eqv3_e_hull_raw"].dropna().clip(*x_clip)
    mp = df["eqv3_e_hull_mp2020"].dropna().clip(*x_clip)

    ax.hist(mp20, bins=bins, alpha=0.55, label=f"MP-20 stored (DFT, n={len(mp20)})",
            color="#7f7f7f", edgecolor="white", linewidth=0.3)
    ax.hist(raw, bins=bins, alpha=0.55, label=f"EquiformerV3 raw (n={len(raw)})",
            color="#3b6cb7", edgecolor="white", linewidth=0.3)
    ax.hist(mp, bins=bins, alpha=0.55, label=f"EquiformerV3 + MP2020 (n={len(mp)})",
            color="#54a24b", edgecolor="white", linewidth=0.3)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axvline(0.10, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(0.0, ax.get_ylim()[1] * 0.95, " stable", fontsize=8, va="top")
    ax.text(0.10, ax.get_ylim()[1] * 0.95, " metastable cutoff", fontsize=8, va="top")
    ax.set_xlabel("e_above_hull (eV/atom) — clipped to [-0.1, 0.5]")
    ax.set_ylabel("count")
    ax.set_title("EquiformerV3-OAM vs MP-20 stored e_above_hull")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] {outpath}")


def plot_scatter(df: pd.DataFrame, outpath: Path, x_clip: tuple[float, float] = (-0.1, 0.7)) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, col, label, color in [
        (axes[0], "eqv3_e_hull_raw", "raw eqv3", "#3b6cb7"),
        (axes[1], "eqv3_e_hull_mp2020", "eqv3 + MP2020", "#54a24b"),
    ]:
        sub = df[df[col].notna() & df["mp20_e_hull"].notna()]
        x = sub["mp20_e_hull"].values
        y = sub[col].values
        ax.scatter(x, y, s=4, alpha=0.35, color=color, edgecolors="none")
        lo, hi = x_clip
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6, alpha=0.4)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.axvline(0.10, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.axhline(0.10, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("MP-20 stored e_above_hull (eV/atom)")
        ax.set_title(f"{label}  (n={len(sub)})")
        diff = y - x
        bias = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        ax.text(
            0.04, 0.96,
            f"bias = {bias:+.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )
    axes[0].set_ylabel("EquiformerV3 e_above_hull (eV/atom)")
    fig.suptitle("EquiformerV3-OAM e_above_hull vs MP-20 stored", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] {outpath}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default="equiformer_v3_vs_mp20")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"FATAL: csv not found: {args.csv}")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.csv}")
    df = pd.read_csv(args.csv)
    summarize(df)

    plot_histogram(df, args.output_dir / f"{args.prefix}_histogram.png")
    plot_scatter(df, args.output_dir / f"{args.prefix}_scatter.png")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
