#!/usr/bin/env python
"""Score generated CIFs for novelty against the model's actual training set."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.mp20_tokens import tokens_to_structure  # noqa: E402
from src.eval.uniqueness_novelty import (  # noqa: E402
    _filter_by_nary,
    _is_finite_structure,
    compute_uniqueness_novelty,
)


LEMAT_MATCHER_LTOL = 0.1
FLOWMM_MATCHER_KWARGS = {"stol": 0.5, "angle_tol": 10, "ltol": 0.3}


def _make_matcher(
    *,
    preset: str,
    ltol: float,
    stol: float | None,
    angle_tol: float | None,
) -> tuple[StructureMatcher, dict[str, Any]]:
    """Build the matcher used for train-reference novelty.

    LeMat-GenBench's ``structure-matcher`` path uses material-hasher's
    ``PymatgenStructureSimilarity(tolerance=0.1)``, which wraps
    ``pymatgen.StructureMatcher(ltol=0.1)`` and leaves all other pymatgen
    defaults untouched. The old local posthoc script used FlowMM-style
    ``stol=0.5, angle_tol=10, ltol=0.3``; keep that available for provenance,
    but do not use it as the default for LeMat-aligned replay accounting.
    """
    preset = str(preset).strip().lower()
    if preset == "lemat":
        kwargs: dict[str, Any] = {"ltol": float(ltol)}
    elif preset == "flowmm":
        kwargs = dict(FLOWMM_MATCHER_KWARGS)
    elif preset == "custom":
        kwargs = {"ltol": float(ltol)}
        if stol is not None:
            kwargs["stol"] = float(stol)
        if angle_tol is not None:
            kwargs["angle_tol"] = float(angle_tol)
    else:
        raise ValueError(f"Unknown matcher preset: {preset!r}")
    return StructureMatcher(**kwargs), kwargs


def _split_roots(raw: str) -> list[Path]:
    roots: list[Path] = []
    for part in re.split(r"[:;,]", raw):
        part = part.strip()
        if part:
            roots.append(Path(part))
    return roots


def _iter_csv_structures(path: Path) -> Iterable[Structure]:
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cif = row.get("cif") or row.get("cif.conv")
            if not cif:
                continue
            try:
                yield Structure.from_str(cif, fmt="cif")
            except Exception:
                continue


def _load_reference_structures(path: Path, *, limit: int = 0) -> tuple[list[Structure], int]:
    candidates: list[Path] = []
    if path.is_dir():
        for rel in ("raw/train.csv", "train.csv", "samples.pt"):
            candidate = path / rel
            if candidate.exists():
                candidates.append(candidate)
    else:
        candidates.append(path)

    parse_failed = 0
    structs: list[Structure] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".pt":
            records = torch.load(candidate, map_location="cpu", weights_only=False)
            for rec in records:
                if limit and len(structs) >= limit:
                    return structs, parse_failed
                try:
                    structs.append(tokens_to_structure(rec))
                except Exception:
                    parse_failed += 1
            if structs:
                return structs, parse_failed
        elif candidate.suffix == ".csv":
            for struct in _iter_csv_structures(candidate):
                if limit and len(structs) >= limit:
                    return structs, parse_failed
                structs.append(struct)
            if structs:
                return structs, parse_failed

    if path.is_dir():
        cif_paths = sorted((path / "structures").glob("*.cif"))
        if not cif_paths:
            cif_paths = sorted(path.glob("*.cif"))
        for cif_path in cif_paths:
            if limit and len(structs) >= limit:
                break
            try:
                structs.append(Structure.from_file(cif_path))
            except Exception:
                parse_failed += 1
    elif path.suffix == ".cif":
        try:
            structs.append(Structure.from_file(path))
        except Exception:
            parse_failed += 1
    return structs, parse_failed


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    if path.is_dir():
        return [
            {"sample_idx": i, "file": cif.name, "success": True}
            for i, cif in enumerate(sorted(path.glob("*.cif")))
        ]
    raise FileNotFoundError(f"manifest not found: {manifest_path}")


def _load_sample_structures(cifs: Path) -> tuple[list[int], list[Structure], int]:
    manifest = _load_manifest(cifs)
    cif_dir = cifs if cifs.is_dir() else cifs.parent
    sample_ids: list[int] = []
    structs: list[Structure] = []
    parse_failed = 0
    for rec in manifest:
        if rec.get("success") is False:
            continue
        file_name = rec.get("file")
        if not file_name:
            continue
        try:
            struct = Structure.from_file(cif_dir / str(file_name))
        except Exception:
            parse_failed += 1
            continue
        sample_ids.append(int(rec.get("sample_idx", len(sample_ids))))
        structs.append(struct)
    return sample_ids, structs, parse_failed


def _eligible_sample_ids(
    sample_ids: list[int],
    structures: list[Structure],
    *,
    minimum_nary: int,
    maximum_nary: int | None,
) -> list[int]:
    finite_pairs = [
        (sid, struct)
        for sid, struct in zip(sample_ids, structures, strict=True)
        if _is_finite_structure(struct)
    ]
    finite_structs = [struct for _, struct in finite_pairs]
    filtered = _filter_by_nary(
        finite_structs,
        minimum_nary=minimum_nary,
        maximum_nary=maximum_nary,
    )
    return [finite_pairs[finite_idx][0] for finite_idx, _, _ in filtered]


def _extract_iv_block(text: str) -> str:
    iv_key = "individual_values="
    iv_start = text.find(iv_key)
    if iv_start < 0:
        raise RuntimeError("individual_values= not found")
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


def _parse_lemat_sun(path: Path) -> tuple[list[int], dict[int, float]]:
    data = json.loads(path.read_text())
    valid_ids = [int(x) for x in data["validity_filtering"]["valid_structure_ids"]]
    values = ast.literal_eval(_extract_iv_block(data["results"]["sun"]))
    if len(valid_ids) != len(values):
        raise RuntimeError(
            f"LeMat SUN length mismatch: {len(valid_ids)} ids vs {len(values)} values"
        )
    return valid_ids, {sid: float(value) for sid, value in zip(valid_ids, values)}


def _rate(count: int, denom: int) -> float:
    return float(count) / float(denom) if denom else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score generated CIFs for novelty against exact training references."
    )
    parser.add_argument("--cifs", required=True, type=Path, help="CIF dir or manifest.json.")
    parser.add_argument(
        "--train_refs",
        required=True,
        help="Training roots separated by ':'/';'/','; e.g. data/mp20:data/synthetic/S0.",
    )
    parser.add_argument("--lemat_json", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--minimum_nary", type=int, default=1)
    parser.add_argument("--maximum_nary", type=int, default=None)
    parser.add_argument("--reference_limit", type=int, default=0)
    parser.add_argument(
        "--matcher_preset",
        choices=("lemat", "flowmm", "custom"),
        default="lemat",
        help=(
            "Structure matching preset for train-reference novelty. "
            "lemat uses StructureMatcher(ltol=0.1), matching LeMat's "
            "material_hasher PymatgenStructureSimilarity wrapper. "
            "flowmm restores the old local stol=0.5 angle_tol=10 ltol=0.3 settings."
        ),
    )
    parser.add_argument(
        "--matcher_ltol",
        type=float,
        default=LEMAT_MATCHER_LTOL,
        help="ltol for --matcher_preset lemat/custom.",
    )
    parser.add_argument(
        "--matcher_stol",
        type=float,
        default=None,
        help="Optional stol for --matcher_preset custom.",
    )
    parser.add_argument(
        "--matcher_angle_tol",
        type=float,
        default=None,
        help="Optional angle_tol for --matcher_preset custom.",
    )
    args = parser.parse_args()

    train_roots = _split_roots(args.train_refs)
    if not train_roots:
        raise SystemExit("--train_refs resolved to no paths")

    sample_ids, sample_structs, sample_parse_failed = _load_sample_structures(args.cifs)
    if not sample_structs:
        raise SystemExit(f"No CIF structures loaded from {args.cifs}")

    ref_structs: list[Structure] = []
    ref_summaries: list[dict[str, Any]] = []
    for root in train_roots:
        structs, parse_failed = _load_reference_structures(
            root, limit=int(args.reference_limit)
        )
        ref_structs.extend(structs)
        ref_summaries.append(
            {"path": str(root), "num_structures": len(structs), "parse_failed": parse_failed}
        )
    if not ref_structs:
        raise SystemExit(f"No reference structures loaded from {train_roots}")

    matcher, matcher_kwargs = _make_matcher(
        preset=args.matcher_preset,
        ltol=float(args.matcher_ltol),
        stol=args.matcher_stol,
        angle_tol=args.matcher_angle_tol,
    )

    eligible_ids = _eligible_sample_ids(
        sample_ids,
        sample_structs,
        minimum_nary=int(args.minimum_nary),
        maximum_nary=args.maximum_nary,
    )
    novelty = compute_uniqueness_novelty(
        sample_structs,
        ref_structs,
        minimum_nary=int(args.minimum_nary),
        maximum_nary=args.maximum_nary,
        matcher=matcher,
    )
    counts = novelty.get("counts", {})
    is_unique = [bool(v) for v in novelty.get("is_unique", [])]
    is_novel = [bool(v) for v in novelty.get("is_novel", [])]
    is_un = [bool(v) for v in novelty.get("is_un", [])]
    if not (len(eligible_ids) == len(is_unique) == len(is_novel) == len(is_un)):
        raise RuntimeError(
            "novelty flag alignment failed: "
            f"eligible={len(eligible_ids)} unique={len(is_unique)} "
            f"novel={len(is_novel)} un={len(is_un)}"
        )
    per_sample = {
        sid: {"train_unique": unique, "train_novel": novel, "train_un": un}
        for sid, unique, novel, un in zip(eligible_ids, is_unique, is_novel, is_un)
    }

    summary: dict[str, Any] = {
        "cifs": str(args.cifs),
        "train_refs": ref_summaries,
        "minimum_nary": int(args.minimum_nary),
        "maximum_nary": args.maximum_nary,
        "matcher_preset": str(args.matcher_preset),
        "matcher_kwargs": matcher_kwargs,
        "num_samples_loaded": len(sample_structs),
        "num_sample_parse_failed": sample_parse_failed,
        "num_reference_structures": len(ref_structs),
        "train_unique_count": int(counts.get("unique", 0)),
        "train_novel_count": int(counts.get("novel", 0)),
        "train_un_count": int(counts.get("unique_and_novel", 0)),
        "train_novelty_total": int(counts.get("novel_total", counts.get("total", 0))),
        "train_unique_total": int(counts.get("total", 0)),
        "train_unique_rate": float(novelty.get("unique_rate", 0.0)),
        "train_novel_rate": float(novelty.get("novel_rate", 0.0)),
        "train_un_rate": float(novelty.get("un_rate", 0.0)),
    }

    if args.lemat_json is not None:
        valid_ids, sun_values = _parse_lemat_sun(args.lemat_json)
        train_flagged_valid = [sid for sid in valid_ids if sid in per_sample]
        train_novel_valid = [
            sid for sid in train_flagged_valid if per_sample[sid]["train_novel"]
        ]
        train_un_valid = [sid for sid in train_flagged_valid if per_sample[sid]["train_un"]]
        train_novel_sun = [sid for sid in train_novel_valid if sun_values.get(sid) == 1.0]
        train_novel_msun = [sid for sid in train_novel_valid if sun_values.get(sid) == 0.5]
        train_known_sun = [
            sid
            for sid in valid_ids
            if sid in per_sample
            and not per_sample[sid]["train_novel"]
            and sun_values.get(sid) == 1.0
        ]
        train_known_msun = [
            sid
            for sid in valid_ids
            if sid in per_sample
            and not per_sample[sid]["train_novel"]
            and sun_values.get(sid) == 0.5
        ]
        train_un_sun = [sid for sid in train_un_valid if sun_values.get(sid) == 1.0]
        train_un_msun = [sid for sid in train_un_valid if sun_values.get(sid) == 0.5]
        train_dup_msun = [
            sid
            for sid in valid_ids
            if sid in per_sample
            and not per_sample[sid]["train_un"]
            and sun_values.get(sid) == 0.5
        ]
        denom = len(valid_ids)
        summary.update(
            {
                "lemat_json": str(args.lemat_json),
                "lemat_valid_count": denom,
                "lemat_valid_with_train_flags": len(train_flagged_valid),
                "train_novel_valid_count": len(train_novel_valid),
                "train_novel_valid_rate": _rate(len(train_novel_valid), denom),
                "train_novel_sun_count": len(train_novel_sun),
                "train_novel_sun_rate": _rate(len(train_novel_sun), denom),
                "train_novel_msun_count": len(train_novel_msun),
                "train_novel_msun_rate": _rate(len(train_novel_msun), denom),
                "train_novel_sun_msun_count": len(train_novel_sun)
                + len(train_novel_msun),
                "train_novel_sun_msun_rate": _rate(
                    len(train_novel_sun) + len(train_novel_msun), denom
                ),
                "train_known_sun_count": len(train_known_sun),
                "train_known_sun_rate": _rate(len(train_known_sun), denom),
                "train_known_msun_count": len(train_known_msun),
                "train_known_msun_rate": _rate(len(train_known_msun), denom),
                "train_un_valid_count": len(train_un_valid),
                "train_un_valid_rate": _rate(len(train_un_valid), denom),
                "train_un_sun_count": len(train_un_sun),
                "train_un_sun_rate": _rate(len(train_un_sun), denom),
                "train_un_msun_count": len(train_un_msun),
                "train_un_msun_rate": _rate(len(train_un_msun), denom),
                "train_un_sun_msun_count": len(train_un_sun) + len(train_un_msun),
                "train_un_sun_msun_rate": _rate(
                    len(train_un_sun) + len(train_un_msun), denom
                ),
                "train_duplicate_msun_count": len(train_dup_msun),
                "train_duplicate_msun_rate": _rate(len(train_dup_msun), denom),
                "lemat_valid_missing_train_flags": denom - len(train_flagged_valid),
            }
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
