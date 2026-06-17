#!/usr/bin/env python3
"""Single canonical extractor for LeMat-GenBench result JSONs.

LeMat serializes its BenchmarkResult dataclasses via repr(), so the per-family
entries under `results.<family>` are Python source strings rather than nested
JSON objects. This module parses them with targeted regexes to recover every
field the chapter cares about: validity, uniqueness, novelty, stable,
metastable, sun, msun, sun+msun, mean e_above_hull, plus counts.

Usage:
    python scripts/extract_lemat_metrics.py results_final/M2_v2.json
    python scripts/extract_lemat_metrics.py --table results_final/*.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Each pattern captures a single named metric somewhere inside the repr string.
# np.float64(...) and plain floats both occur in LeMat reprs.
_NUM = r"(?:np\.float64\()?([\d.eE+-]+)\)?"


def _grab(pattern: str, s: str) -> float | None:
    """Search for `pattern` once; return the captured number or None."""
    m = re.search(pattern, s)
    return float(m.group(1)) if m else None


def _grab_all(pattern: str, s: str) -> list[float]:
    return [float(g) for g in re.findall(pattern, s)]


def extract(path: str | Path) -> dict:
    """Return the canonical metric dict for one LeMat result JSON."""
    with open(path) as f:
        d = json.load(f)
    r = d.get("results", {})
    vf = d.get("validity_filtering", {})

    out: dict[str, float | int | None] = {}
    out["n_input"] = vf.get("total_input_structures")
    out["n_valid"] = vf.get("valid_structures")
    out["validity"] = vf.get("validity_rate")

    s = r.get("uniqueness", "")
    if isinstance(s, str):
        out["uniqueness"] = _grab(r"'Uniqueness':\s*" + _NUM, s)

    s = r.get("novelty", "")
    if isinstance(s, str):
        out["novelty"] = _grab(r"'novelty_score':\s*" + _NUM, s)

    s = r.get("stability", "")
    if isinstance(s, str):
        out["stable"] = _grab(r"'stability_value':\s*" + _NUM, s)
        out["metastable_total"] = _grab(r"'metastability_value':\s*" + _NUM, s)
        out["mean_e_hull"] = _grab(r"'mean_e_above_hull':\s*" + _NUM, s)
    if out.get("stable") is not None and out.get("metastable_total") is not None:
        out["metastable"] = out["metastable_total"] - out["stable"]
    else:
        out["metastable"] = None

    s = r.get("sun", "")
    if isinstance(s, str):
        out["sun"] = _grab(r"'sun_rate':\s*" + _NUM, s)
        out["msun"] = _grab(r"'msun_rate':\s*" + _NUM, s)
        out["sun_msun"] = _grab(r"'combined_sun_msun_rate':\s*" + _NUM, s)
        out["sun_count"] = _grab(r"'stable_count':\s*(\d+)", s)
        out["msun_count"] = _grab(r"'metastable_count':\s*(\d+)", s)
    if out.get("sun_count") is not None:
        out["sun_count"] = int(out["sun_count"])
    if out.get("msun_count") is not None:
        out["msun_count"] = int(out["msun_count"])

    # HHI: element supply-risk / scarcity (LeMat scales to 0-10; higher = riskier).
    s = r.get("hhi", "")
    if isinstance(s, str):
        out["hhi_production"] = _grab(r"'hhi_production_value':\s*" + _NUM, s)
        out["hhi_reserve"] = _grab(r"'hhi_reserve_value':\s*" + _NUM, s)

    return out


def _fmt_pct(v):
    return f"{v*100:.2f}" if isinstance(v, float) else "—"


def _print_table(paths: list[Path]) -> None:
    cols = ["file", "valid%", "uniq%", "novel%", "stable%", "meta%", "sun%", "msun%", "s+m%", "ē_hull", "HHI_prod", "HHI_res", "n_valid"]
    widths = [38, 7, 7, 7, 8, 7, 6, 7, 7, 7, 8, 8, 7]
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for p in paths:
        m = extract(p)
        row = [
            p.stem,
            _fmt_pct(m.get("validity")),
            _fmt_pct(m.get("uniqueness")),
            _fmt_pct(m.get("novelty")),
            _fmt_pct(m.get("stable")),
            _fmt_pct(m.get("metastable")),
            _fmt_pct(m.get("sun")),
            _fmt_pct(m.get("msun")),
            _fmt_pct(m.get("sun_msun")),
            f"{m['mean_e_hull']:.4f}" if m.get("mean_e_hull") is not None else "—",
            f"{m['hhi_production']:.3f}" if m.get("hhi_production") is not None else "—",
            f"{m['hhi_reserve']:.3f}" if m.get("hhi_reserve") is not None else "—",
            str(m.get("n_valid") or "—"),
        ]
        print("  ".join(f"{c:<{w}}" for c, w in zip(row, widths)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--table", action="store_true",
                    help="Print a compact comparison table instead of per-file JSON.")
    args = ap.parse_args()
    if args.table:
        _print_table(args.paths)
    else:
        for p in args.paths:
            print(json.dumps({"path": str(p), **extract(p)}, indent=2, default=str))


if __name__ == "__main__":
    main()
