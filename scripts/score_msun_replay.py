#!/usr/bin/env python
"""Score how many LeMat-MSUN samples are duplicates of our training augmentations.

LeMat's MSUN counts structures that are metastable + unique + novel relative
to LeMat-Bulk, using ``pymatgen.StructureMatcher(ltol=0.1)`` (via
material_hasher's ``PymatgenStructureSimilarity``). Since MP20 ⊂ LeMat-Bulk,
the MSUN count already excludes anything matching MP20. The only training
data not covered by LeMat-Bulk is the curated synthetic round(s) S_r.

This script takes the LeMat-MSUN indices from a downloaded LeMat result
JSON, then re-checks each of those samples against the synthetic
augmentation references using the **same matcher LeMat uses**:

    matches reference → replay        (LeMat called it novel, but we curated it)
    no match          → true train-novel-MSUN  (genuinely novel vs LeMat-Bulk ∪ S_r)

This is the exact apples-to-apples replay decomposition.

Usage:
    uv run python scripts/score_msun_replay.py \\
        --lemat_json outputs/external_eval/lemat_results/m1v1_uma.json \\
        --cif_dir outputs/external_eval/.../relaxed_cifs \\
        --reference_roots data/synthetic/crystalite_round0_msun_27k \\
        --output_json outputs/replay_analysis/m1v1.json

Multiple synthetic refs (M2 etc.):
    --reference_roots data/synthetic/crystalite_round0_msun_27k:data/synthetic/crystalite_round1_msun_54k_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Hashable

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from tqdm import tqdm


LEMAT_TOLERANCE = 0.1  # ltol; matches lemat_genbench's PymatgenStructureSimilarity(tolerance=0.1)


def _split_roots(raw: str) -> list[Path]:
    return [Path(p.strip()) for p in re.split(r"[:;,]", raw) if p.strip()]


def _parse_index_list(repr_str: str, key: str) -> list[int] | None:
    """Parse ``'<key>': [1, 2, 3, ...]`` out of a Python repr string."""
    m = re.search(rf"'{key}': \[([^\]]*)\]", repr_str)
    if not m:
        return None
    body = m.group(1).strip()
    if not body:
        return []
    return [int(x.strip()) for x in body.split(",")]


def parse_lemat_indices(lemat_json: Path) -> dict:
    """Extract msun_indices, sun_indices, and validity mapping from a LeMat result JSON."""
    data = json.loads(lemat_json.read_text())

    vf = data.get("validity_filtering", {})
    if isinstance(vf, str):
        valid_ids = _parse_index_list(vf, "valid_structure_ids") or []
    else:
        valid_ids = vf.get("valid_structure_ids", [])

    sun_str = data["results"]["sun"]
    if not isinstance(sun_str, str):
        sys.exit("results.sun is not a repr string — unexpected schema.")

    msun_idx = _parse_index_list(sun_str, "msun_indices")
    sun_idx = _parse_index_list(sun_str, "sun_indices")
    if msun_idx is None or sun_idx is None:
        sys.exit("Could not parse msun_indices/sun_indices from SUN result.")

    n_structures_match = re.search(r"n_structures=(\d+)", sun_str)
    n_valid = int(n_structures_match.group(1)) if n_structures_match else len(valid_ids)

    return {
        "valid_ids": valid_ids,
        "msun_indices": msun_idx,
        "sun_indices": sun_idx,
        "n_valid": n_valid,
    }


def load_structures_from_csv(csv_path: Path, label: str) -> list[tuple[str, Structure]]:
    """Load (id, Structure) pairs from a raw/train.csv with 'material_id' and 'cif' columns."""
    out: list[tuple[str, Structure]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cif = row.get("cif") or row.get("cif.conv") or row.get("cif_string")
            mid = row.get("material_id") or row.get("mp_id") or ""
            if not cif:
                continue
            try:
                out.append((mid, Structure.from_str(cif, fmt="cif")))
            except Exception:
                continue
    print(f"[ref:{label}] loaded {len(out)} structures from {csv_path}", file=sys.stderr)
    return out


def _composition_key(struct: Structure, matcher: StructureMatcher) -> Hashable | None:
    """Use matcher's comparator for fast composition bucketing."""
    comparator = getattr(matcher, "comparator", None) or getattr(matcher, "_comparator", None)
    if comparator is None or not hasattr(comparator, "get_hash"):
        return None
    try:
        return comparator.get_hash(struct.composition)
    except Exception:
        return None


def build_ref_index(
    ref_pairs: list[tuple[str, Structure]],
    matcher: StructureMatcher,
) -> tuple[dict[Hashable, list[tuple[str, Structure]]], list[tuple[str, Structure]]]:
    """Bucket reference structures by composition for fast lookup."""
    by_comp: dict[Hashable, list[tuple[str, Structure]]] = defaultdict(list)
    fallback: list[tuple[str, Structure]] = []
    for rid, s in ref_pairs:
        key = _composition_key(s, matcher)
        if key is None:
            fallback.append((rid, s))
        else:
            by_comp[key].append((rid, s))
    return dict(by_comp), fallback


def find_replay_match(
    sample: Structure,
    ref_by_comp: dict[Hashable, list[tuple[str, Structure]]],
    ref_fallback: list[tuple[str, Structure]],
    matcher: StructureMatcher,
) -> str | None:
    """Return the matching reference id (or None) for a sample structure."""
    key = _composition_key(sample, matcher)
    candidates: list[tuple[str, Structure]]
    if key is None:
        # Fall back to checking all reference structures
        candidates = [pair for bucket in ref_by_comp.values() for pair in bucket] + ref_fallback
    else:
        candidates = ref_by_comp.get(key, []) + ref_fallback

    for rid, ref_struct in candidates:
        try:
            if matcher.fit(sample, ref_struct):
                return rid
        except Exception:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemat_json", required=True, type=Path,
                    help="Downloaded LeMat-GenBench result JSON for the eval.")
    ap.add_argument("--cif_dir", required=True, type=Path,
                    help="Directory of NequIP-relaxed CIFs that was submitted to LeMat.")
    ap.add_argument("--reference_roots", required=True,
                    help="Colon-separated synthetic root paths (each containing raw/train.csv). "
                         "MP20 is covered by LeMat-Bulk so usually only S0 [+ S1 ...] is needed.")
    ap.add_argument("--output_json", required=True, type=Path)
    ap.add_argument("--per_sample_json", type=Path, default=None,
                    help="If set, dump per-sample {is_msun, is_sun, matched_in, matched_against}.")
    ap.add_argument("--ltol", type=float, default=LEMAT_TOLERANCE,
                    help="StructureMatcher ltol (default 0.1, matches LeMat).")
    args = ap.parse_args()

    if not args.lemat_json.exists():
        sys.exit(f"lemat_json not found: {args.lemat_json}")
    if not args.cif_dir.is_dir():
        sys.exit(f"cif_dir is not a directory: {args.cif_dir}")

    # ---- step 1: parse LeMat MSUN/SUN indices (valid-set-indexed) ----
    lemat = parse_lemat_indices(args.lemat_json)
    valid_ids = lemat["valid_ids"]
    msun_valid_idx = set(lemat["msun_indices"])
    sun_valid_idx = set(lemat["sun_indices"])
    print(
        f"[lemat] n_valid={lemat['n_valid']} msun={len(msun_valid_idx)} sun={len(sun_valid_idx)}",
        file=sys.stderr,
    )

    # ---- step 2: map sorted CIF stems → valid indices ----
    sorted_stems = sorted(p.stem for p in args.cif_dir.glob("*.cif"))
    if not sorted_stems:
        sys.exit(f"No CIFs found in {args.cif_dir}")

    # valid_ids[i] = original index of i-th valid structure in sorted CIF list
    # so sorted_stems[valid_ids[i]] is the stem for valid-index i
    msun_stems: list[str] = []
    sun_stems: list[str] = []
    for valid_i, orig_idx in enumerate(valid_ids):
        if orig_idx >= len(sorted_stems):
            continue
        stem = sorted_stems[orig_idx]
        if valid_i in msun_valid_idx:
            msun_stems.append(stem)
        if valid_i in sun_valid_idx:
            sun_stems.append(stem)
    print(
        f"[map] {len(msun_stems)} MSUN stems, {len(sun_stems)} SUN stems",
        file=sys.stderr,
    )

    # ---- step 3: load the actual sample structures we need to re-check ----
    needed_stems = set(msun_stems) | set(sun_stems)
    sample_structs: dict[str, Structure] = {}
    for stem in tqdm(sorted(needed_stems), desc="load samples", file=sys.stderr):
        cif_path = args.cif_dir / f"{stem}.cif"
        try:
            sample_structs[stem] = Structure.from_str(cif_path.read_text(), fmt="cif")
        except Exception as e:
            print(f"[warn] could not load {cif_path}: {e}", file=sys.stderr)

    # ---- step 4: load reference structures ----
    ref_roots = _split_roots(args.reference_roots)
    ref_pairs: list[tuple[str, Structure]] = []
    for root in ref_roots:
        csv_candidates = [root / "raw/train.csv", root / "raw/all.csv"]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if csv_path is None:
            sys.exit(f"No raw/train.csv or raw/all.csv under {root}")
        ref_pairs.extend(load_structures_from_csv(csv_path, label=root.name))
    if not ref_pairs:
        sys.exit("No reference structures loaded.")
    print(f"[ref] total reference structures: {len(ref_pairs)}", file=sys.stderr)

    # ---- step 5: run StructureMatcher(ltol=0.1) on the LeMat-MSUN/SUN samples ----
    matcher = StructureMatcher(ltol=args.ltol)
    ref_by_comp, ref_fallback = build_ref_index(ref_pairs, matcher)
    print(
        f"[ref] composition buckets: {len(ref_by_comp)}, fallback: {len(ref_fallback)}",
        file=sys.stderr,
    )

    per_sample: dict[str, dict] = {}
    for stem in tqdm(msun_stems, desc="MSUN replay check", file=sys.stderr):
        sample = sample_structs.get(stem)
        if sample is None:
            continue
        matched_against = find_replay_match(sample, ref_by_comp, ref_fallback, matcher)
        per_sample[stem] = {
            "is_msun": True,
            "is_sun": False,
            "matched_against": matched_against,
        }
    for stem in tqdm(sun_stems, desc="SUN replay check", file=sys.stderr):
        sample = sample_structs.get(stem)
        if sample is None:
            continue
        matched_against = find_replay_match(sample, ref_by_comp, ref_fallback, matcher)
        per_sample[stem] = {
            "is_msun": False,
            "is_sun": True,
            "matched_against": matched_against,
        }

    # ---- step 6: aggregate ----
    n_valid = lemat["n_valid"]
    n_lemat_msun = len(msun_stems)
    n_lemat_sun = len(sun_stems)
    n_replay_msun = sum(
        1 for s in msun_stems if per_sample.get(s, {}).get("matched_against") is not None
    )
    n_replay_sun = sum(
        1 for s in sun_stems if per_sample.get(s, {}).get("matched_against") is not None
    )
    n_true_msun = n_lemat_msun - n_replay_msun
    n_true_sun = n_lemat_sun - n_replay_sun

    summary = {
        "lemat_json": str(args.lemat_json),
        "cif_dir": str(args.cif_dir),
        "reference_roots": [str(r) for r in ref_roots],
        "n_reference_structures": len(ref_pairs),
        "structure_matcher_ltol": args.ltol,
        "n_valid": n_valid,
        "lemat_msun_count": n_lemat_msun,
        "lemat_sun_count": n_lemat_sun,
        "lemat_msun_rate": n_lemat_msun / max(1, n_valid),
        "lemat_sun_rate": n_lemat_sun / max(1, n_valid),
        "replay_msun_count": n_replay_msun,
        "replay_sun_count": n_replay_sun,
        "replay_msun_rate": n_replay_msun / max(1, n_valid),
        "replay_sun_rate": n_replay_sun / max(1, n_valid),
        "true_train_novel_msun_count": n_true_msun,
        "true_train_novel_sun_count": n_true_sun,
        "true_train_novel_msun_rate": n_true_msun / max(1, n_valid),
        "true_train_novel_sun_rate": n_true_sun / max(1, n_valid),
    }

    print(json.dumps(summary, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {args.output_json}", file=sys.stderr)

    if args.per_sample_json:
        args.per_sample_json.parent.mkdir(parents=True, exist_ok=True)
        args.per_sample_json.write_text(json.dumps(per_sample, indent=2))
        print(f"[done] wrote per-sample to {args.per_sample_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
