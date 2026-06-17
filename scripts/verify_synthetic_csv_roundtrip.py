"""Verification gate for the synthetic-CSV → MP20Tokens unification.

Round-trips a handful of MP20 test-set rows through the new path
(write CIF to /tmp/<root>/raw/train.csv → MP20Tokens(split="train")) and
asserts the resulting tokens are bit-for-bit equal to the canonical MP20
test cache at data/mp20/processed/mp20_tokens_test.pt.

If this passes, the unified path is faithful for MP20 inputs. We can then
trust it for synthetic inputs (post-relax CIFs written by CifWriter), which
go through the *same* MP20Tokens preprocess() → build_crystal(niggli=True)
pipeline.

Run:
    uv run python scripts/verify_synthetic_csv_roundtrip.py
"""
from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.mp20_tokens import MP20Tokens  # noqa: E402

MP20_RAW_TEST_CSV = REPO_ROOT / "data" / "mp20" / "raw" / "test.csv"
MP20_TEST_CACHE = REPO_ROOT / "data" / "mp20" / "processed" / "mp20_tokens_test.pt"
N_SAMPLES = 5
SEED = 20260513


def _load_mp20_test_cache() -> dict[str, dict]:
    items = torch.load(str(MP20_TEST_CACHE), map_location="cpu", weights_only=False)
    return {str(it["mp_id"]): it for it in items if "mp_id" in it}


def _load_test_csv_rows() -> pd.DataFrame:
    df = pd.read_csv(MP20_RAW_TEST_CSV)
    return df


def _pick_sample_ids(cache: dict[str, dict], df: pd.DataFrame, n: int) -> list[str]:
    """Pick material_ids that exist in both the cache and the CSV."""
    csv_ids = set(df["material_id"].astype(str).tolist())
    common = [mid for mid in cache.keys() if mid in csv_ids]
    if not common:
        raise RuntimeError("No overlap between MP20 test cache and raw/test.csv.")
    rng = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(common), generator=rng).tolist()
    return [common[i] for i in idx[:n]]


def _make_synthetic_style_root(tmp_root: Path, df: pd.DataFrame, material_ids: list[str]) -> None:
    raw_dir = tmp_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "train.csv"
    rows = df[df["material_id"].astype(str).isin(material_ids)]
    # Mimic the synthetic curation CSV layout: material_id, cif, plus optional props.
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "material_id",
                "cif",
                "formation_energy_per_atom",
                "e_above_hull",
            ],
        )
        writer.writeheader()
        for _, r in rows.iterrows():
            writer.writerow(
                {
                    "material_id": str(r["material_id"]),
                    "cif": str(r["cif"]),
                    "formation_energy_per_atom": r.get("formation_energy_per_atom"),
                    "e_above_hull": r.get("e_above_hull"),
                }
            )


def _compare_records(synth_items: list[dict], mp20_cache: dict[str, dict]) -> tuple[int, list[str]]:
    """Return (#matched, list of mismatch reports)."""
    matched = 0
    failures: list[str] = []
    for item in synth_items:
        mid = str(item["mp_id"])
        if mid not in mp20_cache:
            failures.append(f"{mid}: not present in mp20 test cache")
            continue
        ref = mp20_cache[mid]
        diffs = []
        for key in ("A0", "F1", "Y1", "pad_mask"):
            a = item[key]
            b = ref[key]
            if a.shape != b.shape:
                diffs.append(f"{key}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            if a.dtype != b.dtype:
                diffs.append(f"{key}: dtype {a.dtype} vs {b.dtype}")
                continue
            if key in ("F1", "Y1"):
                # exact equality of float32 tensors when preprocessing is identical
                if not torch.equal(a, b):
                    max_abs = (a - b).abs().max().item()
                    diffs.append(f"{key}: not exactly equal (max abs diff {max_abs:.3e})")
            else:
                if not torch.equal(a, b):
                    diffs.append(f"{key}: not exactly equal")
        if int(item["num_atoms"]) != int(ref["num_atoms"]):
            diffs.append(f"num_atoms: {item['num_atoms']} vs {ref['num_atoms']}")
        if diffs:
            failures.append(f"{mid}: " + "; ".join(diffs))
        else:
            matched += 1
    return matched, failures


def main() -> int:
    if not MP20_RAW_TEST_CSV.exists():
        print(f"ERROR: missing {MP20_RAW_TEST_CSV}", file=sys.stderr)
        return 2
    if not MP20_TEST_CACHE.exists():
        print(f"ERROR: missing {MP20_TEST_CACHE}", file=sys.stderr)
        return 2

    print(f"[verify] loading mp20 test cache: {MP20_TEST_CACHE}")
    cache = _load_mp20_test_cache()
    print(f"[verify] loaded {len(cache)} cached test records")

    print(f"[verify] reading {MP20_RAW_TEST_CSV}")
    df = _load_test_csv_rows()
    print(f"[verify] csv rows: {len(df)}")

    material_ids = _pick_sample_ids(cache, df, N_SAMPLES)
    print(f"[verify] picked {len(material_ids)} material_ids: {material_ids}")

    with tempfile.TemporaryDirectory(prefix="verify_synth_") as tmp:
        tmp_root = Path(tmp)
        _make_synthetic_style_root(tmp_root, df, material_ids)
        print(f"[verify] wrote synthetic-style root at {tmp_root}")
        print(f"[verify]   raw/train.csv:")
        for line in (tmp_root / "raw" / "train.csv").read_text().splitlines()[:2]:
            print(f"[verify]     {line[:120]}{'...' if len(line) > 120 else ''}")

        print("[verify] loading via MP20Tokens(split='train', nmax=20, augment_translate=False) ...")
        ds = MP20Tokens(
            root=str(tmp_root),
            split="train",
            nmax=20,
            augment_translate=False,
        )
        print(f"[verify] MP20Tokens produced {len(ds)} records")

        synth_items = [ds.items[i] for i in range(len(ds))]
        matched, failures = _compare_records(synth_items, cache)

    print()
    print(f"[verify] matched: {matched} / {len(material_ids)}")
    if failures:
        print("[verify] FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[verify] PASS — synthetic-CSV path produces tokens bit-for-bit identical to MP20 cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
