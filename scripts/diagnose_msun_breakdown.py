"""Diagnose what's tanking MSUN across ITS methods.

For each method's rows.parquet, computes:

  1. Conditional MSUN gate decomposition: of N delivered samples that are
     target_hit (within tolerance of target=18), how many fail at each gate
     (instability / non-uniqueness / non-novelty)?
  2. Formula concentration in target hits: top compositions by frequency
     within target-hit subset → quantifies "BoK keeps picking the same W/Os
     compounds."
  3. P(msun | target_hit) per method: are methods differentially poisoning
     MSUN, or just finding more target hits some of which fail SUN gating?
  4. Per-method total counts side-by-side table.

Pure post-hoc, no cluster. Drops cleanly into the chapter Q3 writeup.

Usage:
  PYTHONPATH=. uv run python scripts/diagnose_msun_breakdown.py [--target 18.0]
"""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path

import pandas as pd


# Map dir name → friendly method label. Add new methods here as parquets land.
METHOD_DIRS = {
    "its_density_calibration_n1024": "vanilla",
    "its_density_best_of_k64_n1024": "BoK@K=64",
    "its_density_best_of_k128_n1024": "BoK@K=128",
    "its_density_fk_n1024_k64_conservative": "FK cons @K=64",
    "its_density_fk_n1024_k64_aggressive": "FK aggr @K=64",
    "its_density_seq_bok_div_n1024_k128_w0.1_bw1.0_seed1234": "Seq+div@K=128 w=0.1",
}


def _load_parquet(dir_name: str) -> pd.DataFrame | None:
    path = Path(dir_name) / "rows.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _is_hit(df: pd.DataFrame, target: float, tol: float) -> pd.Series:
    return (df["terminal_value"] - target).abs() <= tol


def _gate_decomposition(df: pd.DataFrame, hit_mask: pd.Series) -> dict:
    """Of the target-hit samples, classify by which SUN gate is binding."""
    hits = df[hit_mask]
    n = len(hits)
    if n == 0:
        return {"n_target_hits": 0}
    met = hits["is_metastable"].astype(bool)
    uniq = hits["is_unique"].astype(bool)
    nov = hits["is_novel"].astype(bool)

    # Cascade: if not metastable, "lost to instability"; if metastable but not novel,
    # "lost to novelty"; if metastable+novel but not unique, "lost to uniqueness";
    # all three → kept as MSUN.
    not_met = ~met
    met_not_nov = met & (~nov)
    met_nov_not_uniq = met & nov & (~uniq)
    msun = met & nov & uniq
    return {
        "n_target_hits": int(n),
        "kept_msun": int(msun.sum()),
        "lost_to_instability": int(not_met.sum()),
        "lost_to_non_novelty (met but not novel)": int(met_not_nov.sum()),
        "lost_to_non_uniqueness (met & novel but dup)": int(met_nov_not_uniq.sum()),
        "p_msun_given_hit": float(msun.sum() / n),
    }


def _formula_concentration(df: pd.DataFrame, hit_mask: pd.Series, top_n: int = 8) -> dict:
    """Top compositions among target-hit samples + concentration ratio."""
    formulas = df.loc[hit_mask, "final_reduced_formula"].fillna("").tolist()
    n = len(formulas)
    if n == 0:
        return {"n_target_hits": 0, "n_unique_formulas_in_hits": 0, "top_compositions": []}
    counts = Counter(f for f in formulas if f)
    top = counts.most_common(top_n)
    n_unique = len(counts)
    top_n_share = sum(c for _, c in top) / n if n else 0.0
    return {
        "n_target_hits": n,
        "n_unique_formulas_in_hits": n_unique,
        f"top_{top_n}_share_of_hits": float(top_n_share),
        "top_compositions": top,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=18.0)
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--top_n", type=int, default=8)
    args = ap.parse_args()

    print(f"=== MSUN-breakdown diagnostic (target={args.target} ± {args.tolerance}) ===\n")

    rows_by_method: dict[str, pd.DataFrame] = {}
    for dir_name, label in METHOD_DIRS.items():
        df = _load_parquet(dir_name)
        if df is None:
            print(f"[skip] {dir_name} (no parquet)")
            continue
        rows_by_method[label] = df

    if not rows_by_method:
        print("No parquets found. Pull cluster outputs first.")
        return

    # ----- Section 1: side-by-side gate decomposition -----
    print("\n" + "=" * 90)
    print("SECTION 1: Gate decomposition of target-hit samples")
    print("=" * 90)
    print(f"  (n_hit = samples within ±{args.tolerance} of {args.target})\n")
    print(
        f"{'method':<28}{'n_hit':>8}{'kept_msun':>11}{'instab':>10}{'not_novel':>11}{'not_unique':>12}{'P(msun|hit)':>13}"
    )
    print("-" * 93)
    for label, df in rows_by_method.items():
        hit = _is_hit(df, args.target, args.tolerance)
        d = _gate_decomposition(df, hit)
        if d["n_target_hits"] == 0:
            print(f"{label:<28}{'0 hits':>8}")
            continue
        print(
            f"{label:<28}"
            f"{d['n_target_hits']:>8}"
            f"{d['kept_msun']:>11}"
            f"{d['lost_to_instability']:>10}"
            f"{d['lost_to_non_novelty (met but not novel)']:>11}"
            f"{d['lost_to_non_uniqueness (met & novel but dup)']:>12}"
            f"{d['p_msun_given_hit']:>13.3f}"
        )

    # ----- Section 2: per-method formula concentration -----
    print("\n" + "=" * 90)
    print(f"SECTION 2: Formula concentration in target hits (top {args.top_n})")
    print("=" * 90)
    for label, df in rows_by_method.items():
        hit = _is_hit(df, args.target, args.tolerance)
        c = _formula_concentration(df, hit, top_n=args.top_n)
        print(f"\n  {label}")
        if c["n_target_hits"] == 0:
            print("    (no target hits)")
            continue
        print(
            f"    {c['n_target_hits']} target hits across {c['n_unique_formulas_in_hits']} distinct formulas"
        )
        share_key = f"top_{args.top_n}_share_of_hits"
        print(f"    top-{args.top_n} compositions account for {c[share_key]:.1%} of all target hits")
        for formula, count in c["top_compositions"]:
            share = count / c["n_target_hits"]
            print(f"      {count:>4}× {formula:<20} ({share:.1%} of hits)")

    # ----- Section 3: aggregate sanity check -----
    print("\n" + "=" * 90)
    print("SECTION 3: Per-method top-line summary")
    print("=" * 90)
    print(
        f"{'method':<28}{'n':>6}{'msun_rate':>11}{'tgt_hit':>9}{'msun_tgt':>10}"
        f"{'unique_msun_tgt':>17}{'top1_dist':>11}"
    )
    print("-" * 92)
    for label, df in rows_by_method.items():
        n = len(df)
        msun = df["is_msun"].sum()
        hit = _is_hit(df, args.target, args.tolerance)
        msun_tgt = (df["is_msun"] & hit).sum()
        unique_msun_tgt_formulas = df.loc[df["is_msun"] & hit, "final_reduced_formula"].nunique()
        dist = (df["terminal_value"] - args.target).abs().dropna()
        top1 = float(dist.min()) if len(dist) else float("nan")
        print(
            f"{label:<28}{n:>6}{msun/n:>11.3f}{hit.sum()/n:>9.3f}"
            f"{msun_tgt/n:>10.3f}{unique_msun_tgt_formulas:>17}{top1:>11.4f}"
        )

    print("\n" + "=" * 90)
    print("Reading guide:")
    print("=" * 90)
    print(
        "  - kept_msun / n_hit = P(msun | target_hit). If similar across methods,")
    print("    methods aren't differentially destroying MSUN — they just find more hits.")
    print("  - Top-N compositions share: high % means selection is concentrating on a")
    print("    handful of formulas (uniqueness collapse driver).")
    print("  - unique_msun_tgt: distinct formulas among MSUN-target hits — the actual")
    print("    materials-discovery payload (more is better; chapter framing).")


if __name__ == "__main__":
    main()
