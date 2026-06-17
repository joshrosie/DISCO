#!/usr/bin/env python
"""Fetch MP structures via mp-api and write a CSV for the eqv3 hull build.

Uses Materials Project's mp-api to fetch the DFT-relaxed structure for each
material_id in the existing MP DFT hull pickle, so the eqv3 hull scope
exactly matches the NequIP curation hull.

Authentication: reads the MP API key from environment variable MP_API_KEY,
or from pymatgen's default ~/.config/.pmgrc.yaml. Never accept the key as a
CLI arg — keys in command lines end up in shell history, slurm logs, and
process listings.

Usage:

    export MP_API_KEY='<your-key>'
    python scripts/prep_mp_api_for_hull.py \\
        --source-ppd mp_02072023/2023-02-07-ppd-mp.pkl \\
        --output data/mp_full/raw/relaxed.csv

This fetches the exact same material_ids the NequIP curation scores against,
so the eqv3 hull is scope-aligned by construction. Expect ~30 min - 2 hr
wall time depending on rate-limit interactions (~150k entries fetched in
bulk batches of 1000).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPD = REPO_ROOT / "mp_02072023/2023-02-07-ppd-mp.pkl"
DEFAULT_OUTPUT = REPO_ROOT / "data/mp_full/raw/relaxed.csv"


def get_api_key() -> str:
    key = os.environ.get("MP_API_KEY", "").strip()
    if key:
        return key
    # Fall back to pymatgen's default config file.
    pmgrc = Path.home() / ".config" / ".pmgrc.yaml"
    if pmgrc.exists():
        for line in pmgrc.read_text().splitlines():
            if line.startswith("PMG_MAPI_KEY:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    raise RuntimeError(
        "No MP API key found. Set MP_API_KEY env var or add\n"
        "  PMG_MAPI_KEY: <your-key>\n"
        "to ~/.config/.pmgrc.yaml."
    )


def load_material_ids_from_ppd(ppd_path: Path) -> list[str]:
    print(f"[prep-mp] loading material_ids from {ppd_path}", flush=True)
    t0 = time.perf_counter()
    with ppd_path.open("rb") as f:
        ppd = pickle.load(f)
    ids: list[str] = []
    for entry in ppd.all_entries:
        # ComputedEntry entries carry material_id either in .entry_id or in
        # .data["material_id"]. We accept either, take whichever exists.
        mid = getattr(entry, "entry_id", None)
        if isinstance(mid, str) and mid.startswith("mp-"):
            # Some entries use task IDs like "mp-111-GGA" — strip the suffix.
            core = mid.split("-GGA")[0].split("-R2SCAN")[0]
            ids.append(core)
            continue
        data = getattr(entry, "data", {}) or {}
        mid = data.get("material_id")
        if isinstance(mid, str) and mid.startswith("mp-"):
            ids.append(mid)
    ids = sorted(set(ids))
    print(f"[prep-mp] loaded {len(ids)} unique mp-ids in {time.perf_counter() - t0:.1f}s", flush=True)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ppd", type=Path, default=DEFAULT_PPD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of material_ids per bulk API query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="If > 0, fetch only this many material_ids (for testing).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.source_ppd.exists():
        print(f"FATAL: source PPD missing: {args.source_ppd}", file=sys.stderr)
        return 1

    try:
        api_key = get_api_key()
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    try:
        from mp_api.client import MPRester
    except ImportError:
        print(
            "FATAL: mp_api not installed. Run `uv add mp-api` or "
            "`uv pip install mp-api`.",
            file=sys.stderr,
        )
        return 1
    from pymatgen.io.cif import CifWriter
    import pandas as pd

    material_ids = load_material_ids_from_ppd(args.source_ppd)
    if args.limit > 0 and args.limit < len(material_ids):
        material_ids = material_ids[: args.limit]
        print(f"[prep-mp] limit={args.limit}; fetching only first {len(material_ids)}", flush=True)

    rows: list[dict] = []
    n_fetched = 0
    n_failed = 0
    print(
        f"[prep-mp] fetching {len(material_ids)} structures via mp-api, "
        f"chunk={args.chunk_size}",
        flush=True,
    )
    t0 = time.perf_counter()
    with MPRester(api_key, monty_decode=True) as mpr:
        for start in range(0, len(material_ids), args.chunk_size):
            chunk = material_ids[start : start + args.chunk_size]
            try:
                docs = mpr.materials.summary.search(
                    material_ids=chunk,
                    fields=["material_id", "formula_pretty", "structure"],
                )
            except Exception as exc:
                print(
                    f"  [chunk-fail] {start}-{start + len(chunk)}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                n_failed += len(chunk)
                continue
            for doc in docs:
                try:
                    cif_str = str(CifWriter(doc.structure))
                    rows.append(
                        {
                            "material_id": str(doc.material_id),
                            "pretty_formula": str(doc.formula_pretty),
                            "cif": cif_str,
                        }
                    )
                    n_fetched += 1
                except Exception as exc:
                    n_failed += 1
                    if n_failed <= 5:
                        print(
                            f"  [struct-fail] {doc.material_id}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
            elapsed = time.perf_counter() - t0
            eta = (elapsed / max(1, n_fetched)) * (len(material_ids) - n_fetched)
            print(
                f"  {n_fetched}/{len(material_ids)} fetched, {n_failed} failed, "
                f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                flush=True,
            )

    print(
        f"\n[prep-mp] done. {n_fetched} fetched, {n_failed} failed, "
        f"elapsed={(time.perf_counter() - t0)/60:.1f}m",
        flush=True,
    )
    if n_fetched < 10000:
        print("FATAL: too few successful rows; aborting CSV write.", file=sys.stderr)
        return 2

    print(f"[prep-mp] writing CSV: {args.output}", flush=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"[prep-mp] wrote {len(rows)} rows → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
