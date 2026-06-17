"""Label cached MP-20 token files with prop__density and prop__hhi.

Density is computed directly from the decoded Structure (mass / volume).
HHI is computed via pymatgen's HHIModel using elemental geological reserves
(matches MatInvent's protocol).

Both labels are added in-place on each item dict, then the .pt files are
saved back to the same path so the dataset cache check sees them on next
load and skips reprocessing.

Run: PYTHONPATH=. uv run python scripts/label_mp20_density_hhi.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.mp20_tokens import tokens_to_structure


def _density(struct) -> float:
    try:
        return float(struct.density)
    except Exception:
        return float("nan")


def _hhi(struct, hhi_model) -> float:
    """HHI score (geological reserves) from composition. Returns NaN on failure."""
    try:
        score = hhi_model.get_hhi(struct.composition)
        if score is None:
            return float("nan")
        # HHIModel.get_hhi returns (hhi_production, hhi_reserve). MatInvent uses reserve.
        if isinstance(score, tuple):
            return float(score[1])
        return float(score)
    except Exception:
        return float("nan")


def label_file(path: Path, hhi_model, write: bool = True) -> dict:
    """Add prop__density and prop__hhi to every item in a cached .pt file.

    Returns a dict of summary stats. Set write=False for a dry run.
    """
    print(f"[label] loading {path}")
    items = torch.load(path, weights_only=False)
    n = len(items)
    n_density_ok = 0
    n_hhi_ok = 0
    n_decode_failed = 0

    t0 = time.time()
    for item in tqdm(items, desc=path.name, dynamic_ncols=True):
        try:
            struct = tokens_to_structure(item)
        except Exception:
            item["prop__density"] = float("nan")
            item["prop__hhi"] = float("nan")
            n_decode_failed += 1
            continue
        d = _density(struct)
        h = _hhi(struct, hhi_model)
        item["prop__density"] = d
        item["prop__hhi"] = h
        if d == d:  # not NaN
            n_density_ok += 1
        if h == h:
            n_hhi_ok += 1
    wall = time.time() - t0

    if write:
        print(f"[label] saving back to {path}")
        torch.save(items, path)

    return {
        "path": str(path),
        "n_total": n,
        "n_density_ok": n_density_ok,
        "n_hhi_ok": n_hhi_ok,
        "n_decode_failed": n_decode_failed,
        "wall_seconds": wall,
    }


def main() -> None:
    from pymatgen.analysis.hhi import HHIModel
    hhi_model = HHIModel()

    candidate_paths = [
        ROOT / "data/mp20/processed/mp20_tokens_train.pt",
        ROOT / "data/mp20/processed/mp20_tokens_val.pt",
        ROOT / "data/mp20/processed/mp20_tokens_test.pt",
        ROOT / "data/mp20/processed/mp20_tokens_all.pt",
    ]
    summaries = []
    for p in candidate_paths:
        if not p.exists():
            print(f"[label] skip (missing): {p}")
            continue
        summaries.append(label_file(p, hhi_model, write=True))
    print()
    print("=== summary ===")
    for s in summaries:
        print(
            f"{s['path']}: n={s['n_total']} density_ok={s['n_density_ok']} "
            f"hhi_ok={s['n_hhi_ok']} decode_failed={s['n_decode_failed']} "
            f"wall={s['wall_seconds']:.1f}s"
        )


if __name__ == "__main__":
    main()
