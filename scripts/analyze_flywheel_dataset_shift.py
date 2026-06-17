#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from pymatgen.core import Composition


NARY_LABELS = ["1", "2", "3", "4", "5+"]
EHULL_BINS = [
    ("stable", -float("inf"), 0.0),
    ("near_meta", 0.0, 0.025),
    ("mid_meta", 0.025, 0.05),
    ("far_meta", 0.05, 0.1),
    ("unstable", 0.1, float("inf")),
]


def _safe_comp(formula: str) -> Composition | None:
    try:
        return Composition(str(formula))
    except Exception:
        return None


def _nary_label(comp: Composition) -> str:
    n = len(comp.elements)
    return str(n) if n <= 4 else "5+"


def _entropy_norm(counter: Counter[str]) -> float:
    n = sum(counter.values())
    k = len(counter)
    if n <= 0 or k <= 1:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = float(c) / float(n)
        h -= p * math.log(p)
    return float(h / math.log(k))


def _tv(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def _js(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    m = {k: 0.5 * (float(a.get(k, 0.0)) + float(b.get(k, 0.0))) for k in keys}

    def kl(p: dict[str, float], q: dict[str, float]) -> float:
        out = 0.0
        for k in keys:
            pk = float(p.get(k, 0.0))
            qk = float(q.get(k, 0.0))
            if pk > 0.0 and qk > 0.0:
                out += pk * math.log(pk / qk)
        return out

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def _normalize(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: float(v) / float(total) for k, v in counter.items()}


def _formula_records_from_mp20(root: Path) -> list[dict]:
    csv_path = root / "raw" / "train.csv"
    if not csv_path.exists():
        csv_path = root / "raw" / "all.csv"
    df = pd.read_csv(csv_path, usecols=["pretty_formula", "e_above_hull"])
    rows: list[dict] = []
    for rec in df.to_dict("records"):
        formula = str(rec["pretty_formula"])
        rows.append(
            {
                "formula": formula,
                "e_above_hull": rec.get("e_above_hull"),
                "source": "MP20",
            }
        )
    return rows


def _formula_records_from_synthetic(root: Path, label: str) -> list[dict]:
    meta_path = root / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    rows: list[dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            formula = rec.get("formula")
            if not formula:
                continue
            rows.append(
                {
                    "formula": str(formula),
                    "e_above_hull": rec.get("e_above_hull"),
                    "source": label,
                    "num_atoms": rec.get("num_atoms"),
                    "stability_label": rec.get("stability_label"),
                }
            )
    return rows


def _summarize(label: str, rows: list[dict]) -> dict:
    formulas: Counter[str] = Counter()
    nary: Counter[str] = Counter()
    elements_by_formula: Counter[str] = Counter()
    elements_by_atoms: Counter[str] = Counter()
    ehull: list[float] = []
    ehull_bins: Counter[str] = Counter()
    parse_failed = 0
    formula_atoms: list[float] = []
    cell_atoms: list[float] = []

    for row in rows:
        formula = str(row.get("formula", "")).strip()
        if not formula:
            continue
        comp = _safe_comp(formula)
        if comp is None:
            parse_failed += 1
            continue
        formulas[comp.reduced_formula] += 1
        nary[_nary_label(comp)] += 1
        formula_atoms.append(float(comp.num_atoms))
        if row.get("num_atoms") is not None:
            try:
                cell_atoms.append(float(row["num_atoms"]))
            except Exception:
                pass
        for el in comp.elements:
            elements_by_formula[el.symbol] += 1
            elements_by_atoms[el.symbol] += float(comp[el])
        e = row.get("e_above_hull")
        try:
            e_f = float(e)
        except Exception:
            continue
        if math.isfinite(e_f):
            ehull.append(e_f)
            for name, lo, hi in EHULL_BINS:
                if lo < e_f <= hi or (name == "stable" and e_f <= hi):
                    ehull_bins[name] += 1
                    break

    n = sum(formulas.values())
    top = formulas.most_common(10)
    out = {
        "label": label,
        "n": int(n),
        "unique_formulas": int(len(formulas)),
        "formula_unique_rate": float(len(formulas) / n) if n else float("nan"),
        "formula_entropy_norm": _entropy_norm(formulas),
        "formula_top1_share": float(top[0][1] / n) if top and n else float("nan"),
        "formula_top10_share": float(sum(c for _, c in top) / n) if n else float("nan"),
        "parse_failed": int(parse_failed),
        "nary_counts": {k: int(nary.get(k, 0)) for k in NARY_LABELS},
        "nary_dist": {k: float(nary.get(k, 0) / n) if n else 0.0 for k in NARY_LABELS},
        "element_formula_dist": _normalize(elements_by_formula),
        "element_atom_dist": _normalize(elements_by_atoms),
        "top_formulas": top,
        "top_elements_by_atom_fraction": sorted(
            _normalize(elements_by_atoms).items(), key=lambda kv: kv[1], reverse=True
        )[:15],
        "formula_atoms_mean": float(sum(formula_atoms) / len(formula_atoms))
        if formula_atoms
        else float("nan"),
        "cell_atoms_mean": float(sum(cell_atoms) / len(cell_atoms))
        if cell_atoms
        else float("nan"),
        "ehull_count": int(len(ehull)),
        "ehull_mean": float(sum(ehull) / len(ehull)) if ehull else float("nan"),
        "ehull_bins": {name: int(ehull_bins.get(name, 0)) for name, _, _ in EHULL_BINS},
    }
    return out


def _combine(*parts: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for part in parts:
        rows.extend(part)
    return rows


def _write_summary_csv(path: Path, summaries: list[dict]) -> None:
    fields = [
        "label",
        "n",
        "unique_formulas",
        "formula_unique_rate",
        "formula_entropy_norm",
        "formula_top1_share",
        "formula_top10_share",
        "formula_atoms_mean",
        "cell_atoms_mean",
        "ehull_count",
        "ehull_mean",
    ] + [f"nary_{k}" for k in NARY_LABELS] + [f"ehull_{name}" for name, _, _ in EHULL_BINS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            row = {k: s.get(k) for k in fields}
            for k in NARY_LABELS:
                row[f"nary_{k}"] = s["nary_counts"][k]
            for name, _, _ in EHULL_BINS:
                row[f"ehull_{name}"] = s["ehull_bins"][name]
            writer.writerow(row)


def _plot_nary(path: Path, summaries: list[dict]) -> None:
    labels = [s["label"] for s in summaries]
    x = range(len(labels))
    bottoms = [0.0] * len(labels)
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=200)
    for bucket, color in zip(NARY_LABELS, colors, strict=True):
        vals = [100.0 * s["nary_dist"].get(bucket, 0.0) for s in summaries]
        ax.bar(x, vals, bottom=bottoms, label=bucket, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Composition arity share (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="n-ary", frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def _plot_element_tv(path: Path, summaries: list[dict]) -> None:
    base = summaries[0]
    labels = [s["label"] for s in summaries[1:]]
    tvs = [_tv(base["element_atom_dist"], s["element_atom_dist"]) for s in summaries[1:]]
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
    ax.bar(labels, tvs, color="#1f5a96")
    ax.set_ylabel("TV distance vs MP20\n(atom-fraction elements)")
    ax.set_ylim(0, max(tvs + [0.05]) * 1.2)
    ax.tick_params(axis="x", rotation=25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mp20-root", type=Path, default=Path("data/mp20"))
    parser.add_argument(
        "--synthetic",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Synthetic dataset root. Can be repeated, e.g. S0=data/synthetic/...",
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures/flywheel_dataset_shift"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    components: list[tuple[str, list[dict]]] = [("MP20", _formula_records_from_mp20(args.mp20_root))]
    for item in args.synthetic:
        if "=" not in item:
            raise ValueError(f"--synthetic must be LABEL=PATH, got {item!r}")
        label, raw_path = item.split("=", 1)
        components.append((label, _formula_records_from_synthetic(Path(raw_path), label)))

    summaries: list[dict] = []
    for label, rows in components:
        summaries.append(_summarize(label, rows))

    cumulative_rows = list(components[0][1])
    for label, rows in components[1:]:
        cumulative_rows = _combine(cumulative_rows, rows)
        summaries.append(_summarize(f"MP20+...+{label}", cumulative_rows))

    with (args.outdir / "dataset_shift_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    _write_summary_csv(args.outdir / "dataset_shift_summary.csv", summaries)
    _plot_nary(args.outdir / "nary_shift.png", summaries)
    _plot_element_tv(args.outdir / "element_tv_vs_mp20.png", summaries)

    base = summaries[0]
    print("# Flywheel Dataset Shift\n")
    print("| Dataset | n | unique formulas | H_formula | nary 1/2/3/4/5+ | e_hull mean | TV elem vs MP20 |")
    print("|---|---:|---:|---:|---|---:|---:|")
    for s in summaries:
        nary_str = "/".join(str(s["nary_counts"][k]) for k in NARY_LABELS)
        tv_elem = _tv(base["element_atom_dist"], s["element_atom_dist"])
        print(
            f"| {s['label']} | {s['n']} | {s['unique_formulas']} | "
            f"{s['formula_entropy_norm']:.3f} | {nary_str} | "
            f"{s['ehull_mean']:.4f} | {tv_elem:.3f} |"
        )
    print(f"\nwrote {args.outdir / 'dataset_shift_summary.csv'}")
    print(f"wrote {args.outdir / 'dataset_shift_summary.json'}")
    print(f"wrote {args.outdir / 'nary_shift.png'}")
    print(f"wrote {args.outdir / 'element_tv_vs_mp20.png'}")


if __name__ == "__main__":
    main()
