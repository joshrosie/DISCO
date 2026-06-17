"""MP20 chemistry subspace statistics for the Ch 2 failure-mode chapter.

For each chemistry target used in CFG evaluation, count:
  - subset: # MP20 structures whose element set is a subset of the target
  - exact : # MP20 structures whose element set equals the target

These are theoretical ceilings for the uniqueness metric at n=1024: if the
subset-subspace only contains K distinct MP20 compositions, then
unique_msun_at_target is bounded above by K (up to the rate at which CFG
also finds off-manifold structures).

Usage: PYTHONPATH=. uv run python scripts/mp20_subspace_stats.py
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

TARGETS: list[frozenset[str]] = [
    frozenset({"Li", "Mn", "O"}),
    frozenset({"Fe", "Li", "O", "P"}),
    frozenset({"Mn", "Na", "O"}),
    frozenset({"Ca", "Ti", "O"}),
    frozenset({"Zn", "S"}),
    frozenset({"K", "Mg", "F"}),
    frozenset({"Li", "Ni", "O"}),
]

SPLIT_CSVS = [
    "data/mp20/raw/train.csv",
    "data/mp20/raw/val.csv",
    "data/mp20/raw/test.csv",
]


def _parse_elements(raw: str) -> frozenset[str]:
    """Parse the ``elements`` column ("['Co', 'Mn', ...]") into a frozenset."""
    try:
        return frozenset(ast.literal_eval(raw))
    except Exception:
        return frozenset()


def main() -> None:
    # Load + concat all splits; stratify by split for reference.
    dfs = {}
    for path in SPLIT_CSVS:
        p = Path(path)
        if not p.exists():
            print(f"[warn] missing {path}; skipping")
            continue
        df = pd.read_csv(path, usecols=["material_id", "elements", "pretty_formula",
                                         "formation_energy_per_atom", "e_above_hull",
                                         "spacegroup.number"])
        df["elset"] = df["elements"].map(_parse_elements)
        dfs[p.stem] = df

    if "train" not in dfs:
        raise RuntimeError("need train split at least")

    train = dfs["train"]
    print(f"[stats] MP20 train: {len(train)} rows")
    print(f"[stats] distinct train elsets: {train['elset'].nunique()}")
    print(f"[stats] distinct train formulas: {train['pretty_formula'].nunique()}")
    print()
    print(f"{'target':32s}  {'split':8s}  {'subset_hit':>10s}  {'exact_hit':>10s}  "
          f"{'subset_uniq_formula':>20s}")
    print("-" * 92)

    for tgt in TARGETS:
        tag = "-".join(sorted(tgt))
        for split, df in dfs.items():
            subset_mask = df["elset"].map(lambda s: len(s) > 0 and s.issubset(tgt))
            exact_mask = df["elset"].map(lambda s: s == tgt)
            subset_hits = int(subset_mask.sum())
            exact_hits = int(exact_mask.sum())
            subset_uniq_formula = int(df.loc[subset_mask, "pretty_formula"].nunique())
            print(f"{tag:32s}  {split:8s}  {subset_hits:>10d}  {exact_hits:>10d}  "
                  f"{subset_uniq_formula:>20d}")
        print()

    # Summary of train stats only for the chapter table
    print("\n[Ch 2 chapter table — train split only]")
    print(f"{'target':32s}  {'MP20 subset size':>18s}  {'base rate':>10s}  "
          f"{'uniq formulas (ceiling)':>24s}")
    n_train = len(train)
    for tgt in TARGETS:
        tag = "-".join(sorted(tgt))
        subset_mask = train["elset"].map(lambda s: len(s) > 0 and s.issubset(tgt))
        subset_hits = int(subset_mask.sum())
        subset_uniq = int(train.loc[subset_mask, "pretty_formula"].nunique())
        base_rate = subset_hits / n_train
        print(f"{tag:32s}  {subset_hits:>18d}  {base_rate:>9.2%}  "
              f"{subset_uniq:>24d}")


if __name__ == "__main__":
    main()
