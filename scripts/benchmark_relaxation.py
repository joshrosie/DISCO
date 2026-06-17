#!/usr/bin/env python3
"""Time the Flywheel curation relaxation step against Crystalite sampling.

Replicates the curation pipeline end-to-end on the *same* hardware so the two
costs can sit side by side in the thesis:

  1. SAMPLE   N raw structures from a Crystalite checkpoint (the generation step)
  2. RELAX+SCORE each with NequIP-OAM-L (200 FIRE steps, FrechetCellFilter,
     batch=32) + MP2020 hull, via the same `_build_row_oracle` the curator uses

It reports per-1k seconds for both steps so the claim "Crystalite throughput
balances the MLIP relaxation cost" is backed by one measured number.

Self-contained on the cluster: needs only the checkpoint (dng.pt), the compiled
NequIP model, and the MP PPD pickle. No pre-staged CIFs (the external_eval CIF
dumps are empty placeholders; their samples.pt may also be truncated).

    uv run python scripts/benchmark_relaxation.py \
        --checkpoint dng.pt --n 1000 --device cuda \
        --nequip_compile_path /home/jrosenthal/atom-reps/mlips/aot_batch.nequip.pt2 \
        --ppd_path /home/jrosenthal/atom-reps/mp_02072023/2023-02-07-ppd-mp.pkl

Reuse already-sampled raw tokens instead of re-sampling (skips step 1 timing):
        --samples_pt <report_dir>/samples.pt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _decode(items, n):
    """Token dicts -> pymatgen Structures, skipping undecodable samples."""
    from src.data.mp20_tokens import tokens_to_structure

    structs, skipped = [], 0
    for it in items:
        try:
            structs.append(tokens_to_structure(it))
        except Exception:
            skipped += 1
            continue
        if len(structs) >= n:
            break
    return structs, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="dng.pt",
                    help="Crystalite checkpoint to sample raw structures from.")
    ap.add_argument("--samples_pt", type=Path, default=None,
                    help="Reuse raw token items (list[dict]) instead of sampling.")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)          # THERMO_BATCH
    ap.add_argument("--relax_steps", type=int, default=200)
    ap.add_argument("--fmax", type=float, default=0.005)
    ap.add_argument("--cell_filter", default="frechet")
    ap.add_argument("--optimizer", default="FIRE")
    # sampling knobs (only used when --samples_pt is not given)
    ap.add_argument("--sample_num_steps", type=int, default=None,
                    help="None -> checkpoint's configured value.")
    ap.add_argument("--sample_chunk_size", type=int, default=256)
    ap.add_argument("--sample_mode", default="ema", choices=["ema", "regular"])
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--nequip_compile_path", required=True)
    ap.add_argument("--ppd_path", required=True)
    ap.add_argument("--neighborlist_backend", default="vesin",
                    choices=["vesin", "matscipy", "ase", "alchemiops"],
                    help="Override the batch relaxer's NL backend. The compiled "
                         "model defaults to 'alchemiops', which is broken against "
                         "the installed nvalchemiops (batch_cell_list is now a "
                         "module, not a callable). vesin is GPU/torch like alchemiops.")
    args = ap.parse_args()

    # ---------------------------------------------------------------- sample/load
    sample_elapsed = None
    if args.samples_pt is not None:
        import torch

        if not args.samples_pt.is_file():
            sys.exit(f"missing samples_pt {args.samples_pt}")
        print(f"[bench] loading raw token items from {args.samples_pt}...")
        items = torch.load(args.samples_pt, map_location="cpu", weights_only=False)
        structs, skipped = _decode(items, args.n)
    else:
        from src.data.synthetic_augmentation import (
            generate_synthetic_samples_from_checkpoint,
        )

        print(f"[bench] sampling {args.n} raw structures from {args.checkpoint} "
              f"(steps={args.sample_num_steps or 'ckpt-default'}, mode={args.sample_mode})...")
        t0 = time.perf_counter()
        items = generate_synthetic_samples_from_checkpoint(
            checkpoint=args.checkpoint,
            num_generate=args.n,
            sample_chunk_size=args.sample_chunk_size,
            sample_seed=args.sample_seed,
            sample_num_steps=args.sample_num_steps,
            device=args.device,
            sample_mode=args.sample_mode,
        )
        sample_elapsed = time.perf_counter() - t0
        structs, skipped = _decode(items, args.n)

    if not structs:
        sys.exit("no decodable structures")
    natoms = [len(s) for s in structs]
    print(f"[bench] {len(structs)} structures decoded ({skipped} skipped); "
          f"atoms/cell min={min(natoms)} median={int(statistics.median(natoms))} "
          f"max={max(natoms)}")

    # ---------------------------------------------------------------- oracle
    from src.data.synthetic_augmentation import _build_row_oracle

    print(f"[bench] building NequIP oracle (device={args.device}, batch={args.batch}, "
          f"steps={args.relax_steps}, fmax={args.fmax}, cell_filter={args.cell_filter})...")
    oracle = _build_row_oracle(
        thermo_mlip="nequip",
        thermo_ppd_mp=args.ppd_path,
        thermo_stability_device=args.device,
        thermo_ehull_method="mp2020_like",
        thermo_relax_steps=args.relax_steps,
        thermo_stability_batch=args.batch,
        nequip_compile_path=args.nequip_compile_path,
        nequip_relax_mode="batch",
        nequip_optimizer=args.optimizer,
        nequip_cell_filter=args.cell_filter,
        nequip_fmax=args.fmax,
        nequip_max_force_abort=1e9,
    )

    # The compiled NequIP model's torchsim calc hard-defaults its neighborlist
    # backend to 'alchemiops', which is broken against the installed nvalchemiops
    # (the package renamed/refactored batch_cell_list into a module). Point the
    # transforms at a working backend instead. No env/production change.
    if args.neighborlist_backend != "alchemiops":
        calc = getattr(oracle.relaxer, "calculator", None)
        patched = 0
        for t in getattr(calc, "transforms", []) or []:
            if getattr(t, "backend", None) is not None:
                t.backend = args.neighborlist_backend
                patched += 1
        print(f"[bench] set neighborlist backend -> {args.neighborlist_backend} "
              f"({patched} transform(s) patched)")

    # diagnostic: drive the relaxer directly (no oracle try/except) so a
    # systematic failure surfaces its real traceback instead of being hidden
    # behind the oracle's per-structure err strings.
    print("[bench] diagnostic relax_many on first few structures...")
    import traceback

    probe = structs[: min(4, len(structs))]
    try:
        rel = oracle.relaxer.relax_many(probe, steps=args.relax_steps)
        print(f"[bench]   relax_many OK ({len(rel)} results); "
              f"result[0] keys={list(rel[0].keys()) if isinstance(rel[0], dict) else type(rel[0])}")
    except Exception:
        print("[bench]   relax_many RAISED — real traceback below:\n")
        traceback.print_exc()
        sys.exit("aborting: relaxer itself is failing (see traceback above)")

    # warm-up on a small batch (kernel compilation / model load not counted)
    print("[bench] warm-up (excluded from timing)...")
    _ = oracle.call_many(structs[: min(args.batch, len(structs))])

    print(f"[bench] timing relax+score over {len(structs)} structures...")
    t0 = time.perf_counter()
    results = oracle.call_many(structs)
    relax_elapsed = time.perf_counter() - t0

    n = len(results)
    per_struct = relax_elapsed / n if n else float("nan")
    nsteps = [int(r.get("nsteps", -1)) for r in results if r.get("nsteps", -1) >= 0]
    n_fail = sum(1 for r in results if r.get("err"))

    print("\n" + "=" * 60)
    print(f"[result] structures relaxed:    {n}")
    if sample_elapsed is not None:
        print(f"[result] SAMPLING per 1k:       {sample_elapsed/len(structs)*1000:.1f} s "
              f"(total {sample_elapsed:.1f} s)")
    print(f"[result] RELAX+SCORE total:     {relax_elapsed:.1f} s")
    print(f"[result] RELAX+SCORE per struct:{per_struct:.3f} s")
    print(f"[result] RELAX+SCORE per 1k:    {per_struct*1000:.1f} s")
    print(f"[result] relax throughput:      {n/relax_elapsed:.1f} struct/s")
    if nsteps:
        cap = sum(1 for s in nsteps if s >= args.relax_steps)
        print(f"[result] relax steps used:      median={int(statistics.median(nsteps))} "
              f"mean={statistics.mean(nsteps):.0f}  hit {args.relax_steps}-cap: "
              f"{cap}/{len(nsteps)} ({100*cap/len(nsteps):.0f}%)")
    print(f"[result] relax/score failures:  {n_fail}/{n}")
    if n_fail:
        from collections import Counter

        errs = Counter(r.get("err", "") for r in results if r.get("err"))
        print("[result] failure breakdown:")
        for err, c in errs.most_common(6):
            print(f"           {c:>5}  {err}")
    print(f"\n[compare] Crystalite sampling ~5-22 s/1k (Table 4.2); "
          f"relaxation here is {per_struct*1000:.0f} s/1k.")


if __name__ == "__main__":
    main()
