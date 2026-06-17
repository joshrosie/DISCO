"""Spacegroup distribution plot for a directory of pre-relax CIF samples.

Classifies each CIF with spglib at a configurable symprec and plots the
resulting distribution against MP20 training distribution.

Run: uv run python scripts/plot_prerelax_sg_distribution.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

CIF_DIR = Path("samples/Crystalite_PCA_16dim/prerelaxed/benchmark_cifs")
TRAIN_CSV = Path("data/mp20/raw/train.csv")
OUT_DIR = Path("figures/ch2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMPRECS = [0.01, 0.1, 0.3]


def classify(struct: Structure, symprec: float) -> int | None:
    try:
        return int(
            SpacegroupAnalyzer(
                struct, symprec=symprec, angle_tolerance=5.0
            ).get_space_group_number()
        )
    except Exception:
        return None


def main() -> None:
    cif_paths = sorted(CIF_DIR.glob("*.cif"))
    print(f"found {len(cif_paths)} CIFs in {CIF_DIR}")

    # Load structures
    structures = []
    for i, p in enumerate(cif_paths):
        try:
            structures.append(Structure.from_file(p))
        except Exception as e:
            print(f"[warn] failed to load {p.name}: {e}")
        if (i + 1) % 250 == 0:
            print(f"  loaded {i+1}/{len(cif_paths)}")
    print(f"loaded {len(structures)} structures")

    # Classify at each symprec
    results: dict[float, list[int | None]] = {}
    for symprec in SYMPRECS:
        print(f"classifying at symprec={symprec}...")
        classified = [classify(s, symprec) for s in structures]
        results[symprec] = classified
        sgs = [c for c in classified if c is not None]
        p1 = sum(1 for c in classified if c == 1)
        print(
            f"  classified: {len(sgs)}/{len(classified)}  "
            f"P1: {p1}/{len(classified)} ({p1/len(classified):.2%})  "
            f"unique_SGs: {len(set(sgs))}"
        )

    # MP20 training distribution
    train = pd.read_csv(TRAIN_CSV, usecols=["spacegroup.number"])
    train_sgs = train["spacegroup.number"].dropna().astype(int).values

    # Plot
    fig, axes = plt.subplots(len(SYMPRECS) + 1, 1, figsize=(12, 3.2 * (len(SYMPRECS) + 1)),
                             sharex=True)
    # Training as reference
    train_counts = np.bincount(train_sgs, minlength=231)[1:]
    train_freq = train_counts / train_counts.sum()
    axes[0].bar(
        np.arange(1, 231), train_freq, color="gray", width=1.0,
        label=f"MP20 train (n={len(train_sgs)})",
    )
    axes[0].set_ylabel("probability")
    axes[0].set_title("Reference: MP20 training SG distribution")
    axes[0].legend(fontsize=8)

    for ax, symprec in zip(axes[1:], SYMPRECS):
        sgs = [c for c in results[symprec] if c is not None]
        if not sgs:
            continue
        counts = np.bincount(sgs, minlength=231)[1:]
        freq = counts / len(structures)
        p1_frac = counts[0] / len(structures)
        ax.bar(np.arange(1, 231), freq, color="C3", width=1.0,
               label=f"Crystalite pre-relax, symprec={symprec}, P1={p1_frac:.1%}")
        # Overlay training curve for comparison
        ax.plot(np.arange(1, 231), train_freq, color="black", linewidth=0.8,
                alpha=0.6, label="MP20 train (overlay)")
        ax.set_ylabel("probability")
        ax.legend(fontsize=8)
        ax.set_title(f"Generated SG distribution (symprec={symprec})")

    axes[-1].set_xlabel("spacegroup number")
    fig.suptitle(
        f"Crystalite pre-relax SG distribution — {len(structures)} samples",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "prerelax_sg_distribution.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"figure written to {fig_path}")

    # CSV summary: one row per (symprec, SG)
    rows = []
    for symprec in SYMPRECS:
        counts = np.bincount(
            [c for c in results[symprec] if c is not None], minlength=231
        )
        for sg in range(1, 231):
            rows.append({
                "symprec": symprec, "sg": sg,
                "count": int(counts[sg]),
                "freq": counts[sg] / len(structures),
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "prerelax_sg_distribution.csv", index=False)
    print(f"table written to {OUT_DIR/'prerelax_sg_distribution.csv'}")


if __name__ == "__main__":
    main()
