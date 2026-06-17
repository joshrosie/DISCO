"""Warmup: force MP20Tokens lazy preprocess on synthetic roots.

For each provided root, instantiates `MP20Tokens(root, split='train')` which
triggers `preprocess()` and builds the `processed/` cache. If any CIF fails
to parse / niggli-reduce / tokenize, this surfaces the error in ~5 minutes
on a cheap CPU partition instead of one minute into a 72-hour GPU training.

After this completes, subsequent `MP20Tokens(root, split='train')` calls
load the cache instantly (no preprocess).

Usage:
    uv run python scripts/warmup_mp20tokens_for_synthetic.py \\
        data/synthetic/crystalite_round0_msun_27k \\
        data/synthetic/crystalite_dedup_only_27k \\
        data/synthetic/crystalite_raw_27k
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.mp20_tokens import MP20Tokens  # noqa: E402


def warmup_root(root: Path, nmax: int = 20) -> tuple[int, float]:
    print(f"\n[warmup] {root}: loading MP20Tokens(split='train', nmax={nmax}) ...", flush=True)
    t0 = time.time()
    ds = MP20Tokens(
        root=str(root),
        split="train",
        nmax=nmax,
        augment_translate=False,
    )
    n = len(ds)
    dt = time.time() - t0
    print(f"[warmup] {root}: produced {n} records in {dt:.1f}s", flush=True)

    # Cross-check against summary.json if present
    summary_path = root / "summary.json"
    if summary_path.exists():
        try:
            num_kept = int(json.loads(summary_path.read_text()).get("num_kept", -1))
        except Exception:
            num_kept = None
        if num_kept is not None and num_kept >= 0:
            if n != num_kept:
                print(
                    f"[warmup] WARN: {root}: produced {n} records but summary.json "
                    f"num_kept={num_kept} (some structures may have been filtered "
                    f"by MP20Tokens nmax/VZ guards)",
                    flush=True,
                )
            else:
                print(f"[warmup] {root}: row count matches summary.json num_kept ({num_kept})", flush=True)

    # Spot-check one record
    item = ds[0]
    print(
        f"[warmup] {root}: sample item mp_id={item['mp_id']!r} "
        f"num_atoms={item['num_atoms']} "
        f"Y1=[{item['Y1'][0].item():.4f}, ..., {item['Y1'][-1].item():.4f}]",
        flush=True,
    )
    return n, dt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--nmax", type=int, default=20)
    args = parser.parse_args()

    failures: list[str] = []
    total = 0
    for root in args.roots:
        if not root.is_dir():
            failures.append(f"{root}: not a directory")
            print(f"[warmup] ERROR: {root} is not a directory", file=sys.stderr, flush=True)
            continue
        try:
            n, _ = warmup_root(root, nmax=args.nmax)
            if n == 0:
                failures.append(f"{root}: produced 0 records")
            total += n
        except Exception as exc:
            print(
                f"[warmup] ERROR: {root}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            import traceback
            traceback.print_exc()
            failures.append(f"{root}: {type(exc).__name__}: {exc}")

    print()
    if failures:
        print("[warmup] FAILED — these roots will break M1_v2/training at startup:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"[warmup] PASS — {total} total records tokenized across {len(args.roots)} root(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
