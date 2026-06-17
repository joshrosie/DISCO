#!/usr/bin/env python3
"""Print a compact summary from a LeMat-GenBench result JSON.

LeMat currently stores each benchmark result as the string repr of a
BenchmarkResult object.  This helper extracts the flat ``final_scores`` dicts
and reports the metrics we usually need for Crystalite comparisons.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


PUBLIC_CRYSTALITE_MP20 = {
    "valid_rate": 0.972,
    "unique_rate": 0.958,
    "novel_rate": 0.532,
    "stable_rate": 0.127,
    "metastable_rate": 0.516,
    "sun_rate": 0.015,
    "msun_rate": 0.226,
    "mean_e_above_hull": 0.0905,
    "mean_formation_energy": -0.8916,
    "mean_relaxation_rmse": 0.1322,
}


def _parse_final_scores(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if not isinstance(result, str):
        return {}

    match = re.search(r"final_scores=(\{.*?\}), metadata=", result, re.S)
    if match is None:
        return {}

    text = match.group(1)
    text = re.sub(r"np\.(?:float64|float32|int64|int32)\(([^()]*)\)", r"\1", text)
    text = text.replace("nan", "None")
    return ast.literal_eval(text)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def _num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _delta(value: float | None, baseline: float | None, lower_is_better: bool = False) -> str:
    if value is None or baseline is None:
        return ""
    d = value - baseline
    if lower_is_better:
        good = d < 0
    else:
        good = d > 0
    sign = "+" if d >= 0 else ""
    marker = "better" if good else "worse" if d != 0 else "same"
    return f" ({sign}{100.0 * d:.2f} pp, {marker})"


def _delta_num(value: float | None, baseline: float | None, lower_is_better: bool = False) -> str:
    if value is None or baseline is None:
        return ""
    d = value - baseline
    if lower_is_better:
        good = d < 0
    else:
        good = d > 0
    sign = "+" if d >= 0 else ""
    marker = "better" if good else "worse" if d != 0 else "same"
    return f" ({sign}{d:.4f}, {marker})"


def summarize(path: Path, compare_public_crystalite: bool = False) -> None:
    data = json.loads(path.read_text())
    run_info = data.get("run_info", {})
    validity_filtering = data.get("validity_filtering", {})
    scores = {
        name: _parse_final_scores(result)
        for name, result in data.get("results", {}).items()
    }

    total = int(validity_filtering.get("total_input_structures") or run_info.get("n_structures") or 0)
    valid = int(validity_filtering.get("valid_structures") or 0)
    denom = valid or total or 1

    validity = scores.get("validity", {})
    uniqueness = scores.get("uniqueness", {})
    novelty = scores.get("novelty", {})
    sun = scores.get("sun", {})
    stability = scores.get("stability", {})
    distribution = scores.get("distribution", {})
    hhi = scores.get("hhi", {})

    valid_rate = validity.get("overall_validity_ratio") or validity_filtering.get("validity_rate")
    unique_rate = uniqueness.get("uniqueness_ratio")
    novel_rate = novelty.get("novelty_ratio")
    stable_rate = sun.get("stable_rate") or stability.get("stable_ratio")
    metastable_rate = sun.get("metastable_rate") or stability.get("metastable_ratio")
    sun_rate = sun.get("sun_rate")
    msun_rate = sun.get("msun_rate")
    metasun_rate = sun.get("combined_sun_msun_rate") or sun.get("metasun_rate")

    baseline = PUBLIC_CRYSTALITE_MP20 if compare_public_crystalite else {}

    print(f"file: {path}")
    print(f"run: {run_info.get('run_name', '<unknown>')}")
    print(f"config: {run_info.get('config_name', '<unknown>')}")
    print(f"structures: total={total} valid={valid} invalid={total - valid if total else 'n/a'}")
    print()
    print("Metric                         Value")
    print("-------------------------------------------")
    print(f"Valid                          {_pct(valid_rate)}{_delta(valid_rate, baseline.get('valid_rate'))}")
    print(f"Unique                         {_pct(unique_rate)}{_delta(unique_rate, baseline.get('unique_rate'))}")
    print(f"Novel                          {_pct(novel_rate)}{_delta(novel_rate, baseline.get('novel_rate'))}")
    print(f"Stable                         {_pct(stable_rate)}{_delta(stable_rate, baseline.get('stable_rate'))}")
    print(f"Metastable                     {_pct(metastable_rate)}{_delta(metastable_rate, baseline.get('metastable_rate'))}")
    print(f"SUN                            {_pct(sun_rate)}{_delta(sun_rate, baseline.get('sun_rate'))}")
    print(f"MSUN                           {_pct(msun_rate)}{_delta(msun_rate, baseline.get('msun_rate'))}")
    print(f"SUN+MSUN                       {_pct(metasun_rate)}")
    print(f"Mean e above hull              {_num(stability.get('mean_e_above_hull'))}{_delta_num(stability.get('mean_e_above_hull'), baseline.get('mean_e_above_hull'), lower_is_better=True)}")
    print(f"Mean formation energy          {_num(stability.get('mean_formation_energy'))}{_delta_num(stability.get('mean_formation_energy'), baseline.get('mean_formation_energy'), lower_is_better=True)}")
    print(f"Relaxation RMSD                {_num(stability.get('mean_relaxation_RMSE'))}{_delta_num(stability.get('mean_relaxation_RMSE'), baseline.get('mean_relaxation_rmse'), lower_is_better=True)}")
    print(f"FID                            {_num(distribution.get('FrechetDistance'))}")
    print(f"JSD                            {_num(distribution.get('JSDistance'))}")
    print(f"MMD                            {_num(distribution.get('MMD'))}")
    print(f"HHI combined                   {_num(hhi.get('hhi_combined_mean'))}")
    print()
    print("Counts")
    print("-------------------------------------------")
    print(f"unique={uniqueness.get('unique_structures_count', 'n/a')} novel={novelty.get('novel_structures_count', 'n/a')}")
    print(f"stable={sun.get('stable_count', stability.get('stable_count', 'n/a'))} metastable={sun.get('metastable_count', stability.get('metastable_count', 'n/a'))}")
    print(f"sun={sun.get('sun_count', 'n/a')} msun={sun.get('msun_count', 'n/a')} sun+msun={sun.get('metasun_count', 'n/a')}")
    print()
    print("MLIP coverage")
    print("-------------------------------------------")
    for mlip in ("mace", "orb", "uma"):
        n_valid = stability.get(f"stability_n_valid_structures_{mlip}")
        stable = stability.get(f"stability_stable_ratio_{mlip}")
        ehull = stability.get(f"stability_mean_e_above_hull_{mlip}")
        print(f"{mlip}: n={n_valid if n_valid is not None else 'n/a'} stable={_pct(stable)} e_hull={_num(ehull)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument(
        "--compare-public-crystalite",
        action="store_true",
        help="Show deltas against the public LeMat Crystalite/MP-20 leaderboard row.",
    )
    args = parser.parse_args()
    summarize(args.json_path, compare_public_crystalite=args.compare_public_crystalite)


if __name__ == "__main__":
    main()
