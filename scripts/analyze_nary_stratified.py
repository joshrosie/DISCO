#!/usr/bin/env python
"""Round 0 stratified analyses.

Slices the SUN/MSUN gain along two axes:

1. **Composition arity (n-ary).** Buckets by number of distinct elements in
   `relaxed_formula`. Tests whether the gain is a composition-shift artifact
   or holds within fixed arity.
2. **Stability distance (e_above_hull).** Buckets by where each sample lands
   on the LeMat-evaluated hull. Tests whether the new MSUN samples are
   hugging the hull (near-stable) or scattered toward the 0.1 cutoff
   (threshold-exploiting).

Plots (saved to `figures/augmentation/`):
- `round0_ehull_distribution.png` — per-model e_hull histogram with SUN/MSUN/neither stacked
- `round0_arity_rates.png` — per-arity SUN+MSUN bar chart, synthetic vs control

Joins:
  outputs/external_eval/<run>/relaxed_cifs/manifest.json    sample_idx -> relaxed_formula, num_sites
  results_final/<run>_..._uma_...json                       valid_structure_ids + per-structure SUN flag
                                                            + per-structure e_above_hull (from `results.stability`)

The LeMat `results.sun` BenchmarkResult dump contains
`individual_values=[...]` of length `n_valid`, in the order of
`validity_filtering.valid_structure_ids`. Encoding: 1.0 = SUN, 0.5 = MSUN,
0.0 = neither (validated against the published aggregates).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from pymatgen.core import Composition

REPO = Path(__file__).resolve().parents[1]


def parse_manifest(path: Path) -> dict[int, str]:
    with path.open() as f:
        records = json.load(f)
    out: dict[int, str] = {}
    for rec in records:
        sid = int(rec["sample_idx"])
        formula = rec.get("relaxed_formula") or rec.get("initial_formula")
        if formula:
            out[sid] = formula
    return out


def parse_lemat_sun(path: Path) -> dict[int, float]:
    with path.open() as f:
        d = json.load(f)
    valid_ids = d["validity_filtering"]["valid_structure_ids"]
    sun_str = d["results"]["sun"]
    m = re.search(r"individual_values=(\[[^\]]+\])", sun_str)
    if m is None:
        raise RuntimeError(f"no individual_values found in {path}")
    values = ast.literal_eval(m.group(1))
    if len(valid_ids) != len(values):
        raise RuntimeError(
            f"length mismatch in {path}: {len(valid_ids)} ids vs {len(values)} values"
        )
    return dict(zip(valid_ids, values))


def _extract_iv_block(text: str, after_marker: str) -> str:
    """Bracket-balanced extraction of the next `individual_values=[...]` after a marker."""
    anchor = text.find(after_marker)
    if anchor < 0:
        raise RuntimeError(f"marker not found: {after_marker!r}")
    iv_key = "individual_values="
    iv_start = text.find(iv_key, anchor)
    if iv_start < 0:
        raise RuntimeError(f"individual_values= not found after {after_marker!r}")
    open_bracket = iv_start + len(iv_key)
    if text[open_bracket] != "[":
        raise RuntimeError("individual_values payload is not a list")
    depth = 0
    i = open_bracket + 1
    while i < len(text):
        c = text[i]
        if c in "[{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "]":
            if depth == 0:
                return text[open_bracket : i + 1]
            depth -= 1
        i += 1
    raise RuntimeError("unterminated individual_values list")


def parse_lemat_ehull(path: Path) -> dict[int, float]:
    """sample_idx -> combined e_above_hull (eV/atom) from the E_HullMetric block."""
    with path.open() as f:
        d = json.load(f)
    valid_ids = d["validity_filtering"]["valid_structure_ids"]
    raw = _extract_iv_block(d["results"]["stability"], "E_HullMetric")
    cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", raw)
    records = ast.literal_eval(cleaned)
    if len(valid_ids) != len(records):
        raise RuntimeError(
            f"length mismatch in {path}: {len(valid_ids)} ids vs {len(records)} ehull"
        )
    return {sid: float(rec["value"]) for sid, rec in zip(valid_ids, records)}


def n_ary_bucket(formula: str) -> str | None:
    try:
        n = len(Composition(formula).elements)
    except Exception:
        return None
    if n == 1:
        return "unary"
    if n == 2:
        return "binary"
    if n == 3:
        return "ternary"
    if n == 4:
        return "quaternary"
    if n >= 5:
        return "5+"
    return None


BUCKETS = ("unary", "binary", "ternary", "quaternary", "5+")

# Parackal et al. (arxiv:2601.21393) enumerate binary + ternary protostructures
# over elements Li (Z=3) through Br (Z=35). Anything outside that compositional
# scope is by-construction unreachable by their 39B-protostructure screen.
PARACKAL_Z_MIN = 3
PARACKAL_Z_MAX = 35
PARACKAL_MAX_NARY = 3


def parackal_scope_classify(formula: str) -> dict:
    """Classify a formula by Parackal-scope inclusion.

    Returns a dict:
      in_scope: bool                — True if both arity AND elements are in scope
      reason:   "in_scope"
              | "high_arity"        — n-ary > 3
              | "out_of_z_range"    — contains element with Z<3 or Z>35
              | "high_arity+oo_z"   — both
              | "unparseable"
      n_ary:    int | None
      max_z:    int | None
      min_z:    int | None
    """
    try:
        comp = Composition(formula)
        elements = list(comp.elements)
    except Exception:
        return {"in_scope": False, "reason": "unparseable", "n_ary": None, "max_z": None, "min_z": None}
    if not elements:
        return {"in_scope": False, "reason": "unparseable", "n_ary": 0, "max_z": None, "min_z": None}
    zs = [int(el.Z) for el in elements]
    n_ary = len(elements)
    max_z = max(zs)
    min_z = min(zs)
    high_arity = n_ary > PARACKAL_MAX_NARY
    out_of_z = (min_z < PARACKAL_Z_MIN) or (max_z > PARACKAL_Z_MAX)
    if high_arity and out_of_z:
        reason = "high_arity+oo_z"
    elif high_arity:
        reason = "high_arity"
    elif out_of_z:
        reason = "out_of_z_range"
    else:
        reason = "in_scope"
    return {
        "in_scope": (not high_arity) and (not out_of_z),
        "reason": reason,
        "n_ary": n_ary,
        "max_z": max_z,
        "min_z": min_z,
    }


def analyze(manifest_path: Path, lemat_path: Path, label: str) -> dict:
    manifest = parse_manifest(manifest_path)
    lemat_sun = parse_lemat_sun(lemat_path)
    bucket_total: Counter = Counter()
    bucket_valid: Counter = Counter()
    bucket_sun: Counter = Counter()
    bucket_msun: Counter = Counter()
    unparseable = 0
    for sid, formula in manifest.items():
        bucket = n_ary_bucket(formula)
        if bucket is None:
            unparseable += 1
            continue
        bucket_total[bucket] += 1
        if sid in lemat_sun:
            bucket_valid[bucket] += 1
            flag = lemat_sun[sid]
            if flag == 1.0:
                bucket_sun[bucket] += 1
            elif flag == 0.5:
                bucket_msun[bucket] += 1
    return {
        "label": label,
        "manifest_path": str(manifest_path),
        "lemat_path": str(lemat_path),
        "buckets": {
            b: {
                "submitted": bucket_total[b],
                "valid": bucket_valid[b],
                "sun": bucket_sun[b],
                "msun": bucket_msun[b],
            }
            for b in BUCKETS
        },
        "unparseable_formulas": unparseable,
        "total_submitted": sum(bucket_total[b] for b in BUCKETS) + unparseable,
        "total_valid": sum(bucket_valid[b] for b in BUCKETS),
        "total_sun": sum(bucket_sun[b] for b in BUCKETS),
        "total_msun": sum(bucket_msun[b] for b in BUCKETS),
    }


def fmt_rate(num: int, den: int) -> str:
    if den == 0:
        return "—"
    return f"{100.0 * num / den:.2f}"


def emit_report(rows: list[dict]) -> None:
    print("# N-ary Stratified Re-scoring of Round 0\n")
    print(f"Inputs:\n")
    for r in rows:
        print(f"- **{r['label']}**")
        print(f"  - manifest: `{r['manifest_path']}`")
        print(f"  - lemat: `{r['lemat_path']}`")
        if r["unparseable_formulas"]:
            print(f"  - unparseable formulas (dropped): {r['unparseable_formulas']}")
    print()

    # n-ary distribution
    print("## n-ary distribution (submitted)\n")
    print("| Model | " + " | ".join(BUCKETS) + " | total |")
    print("|" + "|".join(["---"] + ["---:"] * (len(BUCKETS) + 1)) + "|")
    for r in rows:
        cells = [str(r["buckets"][b]["submitted"]) for b in BUCKETS]
        total = sum(r["buckets"][b]["submitted"] for b in BUCKETS)
        print(f"| {r['label']} | " + " | ".join(cells) + f" | {total} |")
    print()

    print("Same as fractions of submitted:\n")
    print("| Model | " + " | ".join(BUCKETS) + " |")
    print("|" + "|".join(["---"] + ["---:"] * len(BUCKETS)) + "|")
    for r in rows:
        total = sum(r["buckets"][b]["submitted"] for b in BUCKETS) or 1
        cells = [
            f"{100.0 * r['buckets'][b]['submitted'] / total:.1f}%" for b in BUCKETS
        ]
        print(f"| {r['label']} | " + " | ".join(cells) + " |")
    print()

    # per-bucket SUN
    print("## SUN rate per bucket (of valid in bucket)\n")
    print("| Model | " + " | ".join(BUCKETS) + " |")
    print("|" + "|".join(["---"] + ["---:"] * len(BUCKETS)) + "|")
    for r in rows:
        cells = []
        for b in BUCKETS:
            sun = r["buckets"][b]["sun"]
            valid = r["buckets"][b]["valid"]
            cells.append(f"{fmt_rate(sun, valid)}% ({sun}/{valid})")
        print(f"| {r['label']} | " + " | ".join(cells) + " |")
    print()

    # per-bucket MSUN
    print("## MSUN rate per bucket (of valid in bucket)\n")
    print("| Model | " + " | ".join(BUCKETS) + " |")
    print("|" + "|".join(["---"] + ["---:"] * len(BUCKETS)) + "|")
    for r in rows:
        cells = []
        for b in BUCKETS:
            msun = r["buckets"][b]["msun"]
            valid = r["buckets"][b]["valid"]
            cells.append(f"{fmt_rate(msun, valid)}% ({msun}/{valid})")
        print(f"| {r['label']} | " + " | ".join(cells) + " |")
    print()

    # SUN+MSUN
    print("## SUN+MSUN rate per bucket (of valid in bucket)\n")
    print("| Model | " + " | ".join(BUCKETS) + " |")
    print("|" + "|".join(["---"] + ["---:"] * len(BUCKETS)) + "|")
    for r in rows:
        cells = []
        for b in BUCKETS:
            both = r["buckets"][b]["sun"] + r["buckets"][b]["msun"]
            valid = r["buckets"][b]["valid"]
            cells.append(f"{fmt_rate(both, valid)}% ({both}/{valid})")
        print(f"| {r['label']} | " + " | ".join(cells) + " |")
    print()

    # Delta if exactly 2 rows
    if len(rows) == 2:
        a, b = rows[0], rows[1]
        print(f"## Delta per bucket: {b['label']} − {a['label']} (pp)\n")
        print("| Bucket | SUN | MSUN | SUN+MSUN |")
        print("|---|---:|---:|---:|")
        for bk in BUCKETS:
            av, bv = a["buckets"][bk], b["buckets"][bk]
            a_sun = av["sun"] / av["valid"] if av["valid"] else 0
            b_sun = bv["sun"] / bv["valid"] if bv["valid"] else 0
            a_msun = av["msun"] / av["valid"] if av["valid"] else 0
            b_msun = bv["msun"] / bv["valid"] if bv["valid"] else 0
            print(
                f"| {bk} | {(b_sun - a_sun) * 100:+.2f} "
                f"| {(b_msun - a_msun) * 100:+.2f} "
                f"| {((b_sun + b_msun) - (a_sun + a_msun)) * 100:+.2f} |"
            )
        print()

    # Sanity check vs aggregate published numbers
    print("## Sanity check (sum across buckets vs aggregate)\n")
    for r in rows:
        v = r["total_valid"]
        print(
            f"- {r['label']}: valid={v}, "
            f"SUN={r['total_sun']} ({fmt_rate(r['total_sun'], v)}%), "
            f"MSUN={r['total_msun']} ({fmt_rate(r['total_msun'], v)}%), "
            f"SUN+MSUN={r['total_sun'] + r['total_msun']} "
            f"({fmt_rate(r['total_sun'] + r['total_msun'], v)}%)"
        )


EHULL_BANDS = (
    ("below_hull", float("-inf"), 0.0),
    ("near_metastable", 0.0, 0.025),
    ("mid_metastable", 0.025, 0.05),
    ("far_metastable", 0.05, 0.10),
    ("unstable", 0.10, float("inf")),
)


def ehull_band(value: float) -> str:
    for name, lo, hi in EHULL_BANDS:
        if lo < value <= hi or (name == "below_hull" and value <= 0.0):
            return name
    return "unstable"


def analyze_ehull(manifest_path: Path, lemat_path: Path, label: str) -> dict:
    """Per-stability-band SUN/MSUN counts plus the raw e_hull array (for plotting)."""
    manifest = parse_manifest(manifest_path)
    lemat_sun = parse_lemat_sun(lemat_path)
    lemat_ehull = parse_lemat_ehull(lemat_path)
    band_total: Counter = Counter()
    band_sun: Counter = Counter()
    band_msun: Counter = Counter()
    rows: list[tuple[int, float, float, str]] = []
    for sid in lemat_sun:
        if sid not in lemat_ehull or sid not in manifest:
            continue
        e = lemat_ehull[sid]
        flag = lemat_sun[sid]
        band = ehull_band(e)
        band_total[band] += 1
        if flag == 1.0:
            band_sun[band] += 1
        elif flag == 0.5:
            band_msun[band] += 1
        rows.append((sid, e, flag, band))
    band_names = [name for name, _, _ in EHULL_BANDS]
    return {
        "label": label,
        "bands": {
            b: {
                "valid": band_total[b],
                "sun": band_sun[b],
                "msun": band_msun[b],
            }
            for b in band_names
        },
        "raw": rows,
    }


def emit_ehull_report(rows: list[dict]) -> None:
    print("\n# E-Above-Hull Stratification of Round 0\n")
    band_names = [name for name, _, _ in EHULL_BANDS]
    band_ranges = {
        "below_hull": "e_hull ≤ 0 (strict-stable)",
        "near_metastable": "0 < e_hull ≤ 0.025",
        "mid_metastable": "0.025 < e_hull ≤ 0.05",
        "far_metastable": "0.05 < e_hull ≤ 0.10",
        "unstable": "e_hull > 0.10",
    }

    print("Bands:\n")
    for b in band_names:
        print(f"- `{b}`: {band_ranges[b]}")
    print()

    print("## Valid samples per band\n")
    print("| Model | " + " | ".join(band_names) + " | total |")
    print("|" + "|".join(["---"] + ["---:"] * (len(band_names) + 1)) + "|")
    for r in rows:
        cells = [str(r["bands"][b]["valid"]) for b in band_names]
        total = sum(r["bands"][b]["valid"] for b in band_names)
        print(f"| {r['label']} | " + " | ".join(cells) + f" | {total} |")
    print()

    print("Same as fractions of valid:\n")
    print("| Model | " + " | ".join(band_names) + " |")
    print("|" + "|".join(["---"] + ["---:"] * len(band_names)) + "|")
    for r in rows:
        total = sum(r["bands"][b]["valid"] for b in band_names) or 1
        cells = [f"{100.0 * r['bands'][b]['valid'] / total:.1f}%" for b in band_names]
        print(f"| {r['label']} | " + " | ".join(cells) + " |")
    print()

    print("## SUN count per band (note: SUN by construction concentrates in below_hull)\n")
    print("| Model | " + " | ".join(band_names) + " | total |")
    print("|" + "|".join(["---"] + ["---:"] * (len(band_names) + 1)) + "|")
    for r in rows:
        cells = [str(r["bands"][b]["sun"]) for b in band_names]
        total = sum(r["bands"][b]["sun"] for b in band_names)
        print(f"| {r['label']} | " + " | ".join(cells) + f" | {total} |")
    print()

    print("## MSUN count per band (MSUN by construction is 0 < e_hull ≤ 0.10)\n")
    print("| Model | " + " | ".join(band_names) + " | total |")
    print("|" + "|".join(["---"] + ["---:"] * (len(band_names) + 1)) + "|")
    for r in rows:
        cells = [str(r["bands"][b]["msun"]) for b in band_names]
        total = sum(r["bands"][b]["msun"] for b in band_names)
        print(f"| {r['label']} | " + " | ".join(cells) + f" | {total} |")
    print()

    # MSUN distribution within the metastable bands — the load-bearing slice
    msun_bands = ["near_metastable", "mid_metastable", "far_metastable"]
    print("## MSUN distribution within metastable bands (column = % of that model's MSUN)\n")
    print("| Model | " + " | ".join(msun_bands) + " |")
    print("|" + "|".join(["---"] + ["---:"] * len(msun_bands)) + "|")
    for r in rows:
        msun_total = sum(r["bands"][b]["msun"] for b in msun_bands) or 1
        cells = [
            f"{100.0 * r['bands'][b]['msun'] / msun_total:.1f}% ({r['bands'][b]['msun']})"
            for b in msun_bands
        ]
        print(f"| {r['label']} | " + " | ".join(cells) + " |")
    print()

    if len(rows) == 2:
        a, b = rows[0], rows[1]
        print(f"## Delta MSUN counts per band: {b['label']} − {a['label']}\n")
        print("| Band | control MSUN | synthetic MSUN | Δ count |")
        print("|---|---:|---:|---:|")
        for bk in msun_bands:
            ac, bc = a["bands"][bk]["msun"], b["bands"][bk]["msun"]
            print(f"| {bk} | {ac} | {bc} | {bc - ac:+d} |")
        print()


def maybe_make_plots(arity_rows: list[dict], ehull_rows: list[dict], outdir: Path) -> None:
    """Save the two plots that actually add information beyond the tables."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping plots.")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot 1: e_hull distribution per model, stacked by SUN/MSUN/neither
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    bins = [-0.10 + 0.01 * i for i in range(0, 41)]  # -0.10 .. 0.30 step 0.01
    for ax, row in zip(axes, ehull_rows):
        e_sun = [e for _, e, flag, _ in row["raw"] if flag == 1.0]
        e_msun = [e for _, e, flag, _ in row["raw"] if flag == 0.5]
        e_neither = [e for _, e, flag, _ in row["raw"] if flag == 0.0]
        ax.hist(
            [e_neither, e_msun, e_sun],
            bins=bins,
            stacked=True,
            label=["neither", "MSUN", "SUN"],
            color=["#cccccc", "#f0b67f", "#54a24b"],
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0.10, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(row["label"])
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("e_above_hull (eV/atom) — bins clipped to [-0.1, 0.3]")
    axes[0].text(
        0.0, axes[0].get_ylim()[1] * 0.95, " stable",
        fontsize=8, va="top",
    )
    axes[0].text(
        0.10, axes[0].get_ylim()[1] * 0.95, " metastable cutoff",
        fontsize=8, va="top",
    )
    fig.suptitle("Round 0 — e_above_hull distribution (LeMat combined MLIP)", fontsize=11)
    fig.tight_layout()
    p1 = outdir / "round0_ehull_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[plot] {p1}")

    # Plot 2: per-arity SUN+MSUN rate, synthetic vs control
    fig, ax = plt.subplots(figsize=(6.5, 4))
    arity_labels = ["binary", "ternary", "quaternary", "5+"]
    width = 0.35
    xs = list(range(len(arity_labels)))
    rates = []
    for r in arity_rows:
        row_rates = []
        for b in arity_labels:
            valid = r["buckets"][b]["valid"]
            both = r["buckets"][b]["sun"] + r["buckets"][b]["msun"]
            row_rates.append(100.0 * both / valid if valid else 0.0)
        rates.append(row_rates)
    ax.bar(
        [x - width / 2 for x in xs], rates[0], width,
        label=arity_rows[0]["label"], color="#7f8c8d",
    )
    ax.bar(
        [x + width / 2 for x in xs], rates[1], width,
        label=arity_rows[1]["label"], color="#54a24b",
    )
    for i, (a, b) in enumerate(zip(rates[0], rates[1])):
        ax.annotate(
            f"+{b - a:.1f}", xy=(i, max(a, b) + 0.5), ha="center", fontsize=9,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(arity_labels)
    ax.set_ylabel("SUN+MSUN rate (% of valid)")
    ax.set_title("Round 0 — SUN+MSUN per composition arity")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    p2 = outdir / "round0_arity_rates.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[plot] {p2}")


def analyze_parackal_scope(manifest_path: Path, lemat_path: Path, label: str) -> dict:
    """Per-Parackal-scope SUN/MSUN counts.

    Parackal et al. enumerate binary + ternary protostructures over Z=3-35.
    For each Crystalite candidate, classify whether it is:
      - in_scope:          binary or ternary AND all elements in Li-Br
      - high_arity:        ≥4 elements (out of Parackal scope)
      - out_of_z_range:    contains element with Z<3 or Z>35 (out of Parackal scope)
      - high_arity+oo_z:   both (doubly out of scope)
    """
    manifest = parse_manifest(manifest_path)
    lemat_sun = parse_lemat_sun(lemat_path)
    REASONS = ("in_scope", "high_arity", "out_of_z_range", "high_arity+oo_z", "unparseable")
    submitted: Counter = Counter()
    valid: Counter = Counter()
    sun: Counter = Counter()
    msun: Counter = Counter()
    for sid, formula in manifest.items():
        cls = parackal_scope_classify(formula)
        reason = cls["reason"]
        submitted[reason] += 1
        if sid in lemat_sun:
            valid[reason] += 1
            flag = lemat_sun[sid]
            if flag == 1.0:
                sun[reason] += 1
            elif flag == 0.5:
                msun[reason] += 1
    return {
        "label": label,
        "by_reason": {
            r: {
                "submitted": submitted[r],
                "valid": valid[r],
                "sun": sun[r],
                "msun": msun[r],
            }
            for r in REASONS
        },
    }


def emit_parackal_report(rows: list[dict]) -> None:
    print("\n# Parackal et al. (arxiv:2601.21393) Out-of-Scope Analysis\n")
    print(
        "Parackal et al. enumerate **binary + ternary** protostructures over\n"
        "elements **Li (Z=3) through Br (Z=35)**. Candidates outside that scope\n"
        "are by-construction unreachable by their 39B-protostructure screen.\n"
    )
    reason_order = ("in_scope", "high_arity", "out_of_z_range", "high_arity+oo_z", "unparseable")
    pretty = {
        "in_scope": "in Parackal scope (binary/ternary, Z∈[3,35])",
        "high_arity": "≥4-element (out: arity)",
        "out_of_z_range": "element outside Z∈[3,35] (out: chemistry)",
        "high_arity+oo_z": "both ≥4-element AND out-of-Z (doubly out)",
        "unparseable": "unparseable formula",
    }

    print("## Distribution by Parackal-scope class (submitted)\n")
    print("| Model | " + " | ".join(pretty[r] for r in reason_order) + " |")
    print("|" + "|".join(["---"] + ["---:"] * len(reason_order)) + "|")
    for r in rows:
        cells = [str(r["by_reason"][rr]["submitted"]) for rr in reason_order]
        print(f"| {r['label']} | " + " | ".join(cells) + " |")
    print()

    print("Same as fraction of submitted:\n")
    print("| Model | " + " | ".join(pretty[r] for r in reason_order) + " |")
    print("|" + "|".join(["---"] + ["---:"] * len(reason_order)) + "|")
    for r in rows:
        total = sum(r["by_reason"][rr]["submitted"] for rr in reason_order) or 1
        cells = [f"{100.0 * r['by_reason'][rr]['submitted'] / total:.1f}%" for rr in reason_order]
        print(f"| {r['label']} | " + " | ".join(cells) + " |")
    print()

    print("## SUN counts by Parackal scope\n")
    print("| Model | " + " | ".join(pretty[r] for r in reason_order) + " | total |")
    print("|" + "|".join(["---"] + ["---:"] * (len(reason_order) + 1)) + "|")
    for r in rows:
        cells = [str(r["by_reason"][rr]["sun"]) for rr in reason_order]
        total = sum(r["by_reason"][rr]["sun"] for rr in reason_order)
        print(f"| {r['label']} | " + " | ".join(cells) + f" | {total} |")
    print()

    print("## MSUN counts by Parackal scope\n")
    print("| Model | " + " | ".join(pretty[r] for r in reason_order) + " | total |")
    print("|" + "|".join(["---"] + ["---:"] * (len(reason_order) + 1)) + "|")
    for r in rows:
        cells = [str(r["by_reason"][rr]["msun"]) for rr in reason_order]
        total = sum(r["by_reason"][rr]["msun"] for rr in reason_order)
        print(f"| {r['label']} | " + " | ".join(cells) + f" | {total} |")
    print()

    print("## SUN+MSUN counts by Parackal scope\n")
    print("| Model | " + " | ".join(pretty[r] for r in reason_order) + " | total |")
    print("|" + "|".join(["---"] + ["---:"] * (len(reason_order) + 1)) + "|")
    for r in rows:
        cells = [str(r["by_reason"][rr]["sun"] + r["by_reason"][rr]["msun"]) for rr in reason_order]
        total = sum(r["by_reason"][rr]["sun"] + r["by_reason"][rr]["msun"] for rr in reason_order)
        print(f"| {r['label']} | " + " | ".join(cells) + f" | {total} |")
    print()

    # Headline: in-scope vs out-of-scope rollup
    print("## Rollup: in-Parackal-scope vs out-of-scope\n")
    print("| Model | in-scope SUN+MSUN | out-of-scope SUN+MSUN | % out-of-scope |")
    print("|---|---:|---:|---:|")
    for r in rows:
        in_sc = r["by_reason"]["in_scope"]["sun"] + r["by_reason"]["in_scope"]["msun"]
        out_sc = sum(
            r["by_reason"][rr]["sun"] + r["by_reason"][rr]["msun"]
            for rr in ("high_arity", "out_of_z_range", "high_arity+oo_z")
        )
        total = in_sc + out_sc
        pct = (100.0 * out_sc / total) if total else 0.0
        print(f"| {r['label']} | {in_sc} | {out_sc} | {pct:.1f}% |")
    print()


def default_paths() -> tuple[Path, Path, Path, Path]:
    return (
        REPO / "outputs/external_eval/oversample_real_n2500_nequip_relaxed/relaxed_cifs/manifest.json",
        REPO / "results_final/crystalite_oversample_real_n2500_nequip_relaxed_uma_comprehensive_multi_mlip_hull_20260509_171859.json",
        REPO / "outputs/external_eval/synthetic_aug_n2500_nequip_relaxed/relaxed_cifs/manifest.json",
        REPO / "results_final/crystalite_synthetic_aug_n2500_nequip_relaxed_uma_comprehensive_multi_mlip_hull_20260509_072436.json",
    )


def main() -> None:
    ctrl_mf, ctrl_lm, syn_mf, syn_lm = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-manifest", default=str(ctrl_mf))
    parser.add_argument("--control-lemat", default=str(ctrl_lm))
    parser.add_argument("--control-label", default="Oversample-real (control)")
    parser.add_argument("--synthetic-manifest", default=str(syn_mf))
    parser.add_argument("--synthetic-lemat", default=str(syn_lm))
    parser.add_argument("--synthetic-label", default="Synthetic Round 0")
    parser.add_argument(
        "--figures-dir",
        default=str(REPO / "figures/augmentation"),
        help="Output dir for PNGs. Pass --no-plots to skip.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    arity_rows = [
        analyze(Path(args.control_manifest), Path(args.control_lemat), args.control_label),
        analyze(Path(args.synthetic_manifest), Path(args.synthetic_lemat), args.synthetic_label),
    ]
    emit_report(arity_rows)
    ehull_rows = [
        analyze_ehull(Path(args.control_manifest), Path(args.control_lemat), args.control_label),
        analyze_ehull(Path(args.synthetic_manifest), Path(args.synthetic_lemat), args.synthetic_label),
    ]
    emit_ehull_report(ehull_rows)
    parackal_rows = [
        analyze_parackal_scope(Path(args.control_manifest), Path(args.control_lemat), args.control_label),
        analyze_parackal_scope(Path(args.synthetic_manifest), Path(args.synthetic_lemat), args.synthetic_label),
    ]
    emit_parackal_report(parackal_rows)
    if not args.no_plots:
        maybe_make_plots(arity_rows, ehull_rows, Path(args.figures_dir))


if __name__ == "__main__":
    main()
