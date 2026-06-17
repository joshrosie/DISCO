#!/usr/bin/env python
"""Build a self-consistent EquiformerV3-OAM PatchedPhaseDiagram from MP entries.

Protocol 1 (initial hull construction), per
[docs/augmentation/equiformer_v3_setup.md](../docs/augmentation/equiformer_v3_setup.md):

  for each MP entry (already DFT-relaxed):
    1. single-point with EquiformerV3-OAM at the DFT-relaxed geometry
       (default), or full EquiformerV3 relaxation with --relax
    2. construct fresh ComputedStructureEntry with eqv3 e_total
    3. apply MP2020 corrections at construction time
  PatchedPhaseDiagram(entries) → pickle.

Runs in the inner equiformer_v3 venv. CUDA expected on Linux; CPU works for
testing with --limit.

    # Cluster, full run:
    external/equiformer_v3/.venv/bin/python scripts/build_equiformer_v3_hull.py \\
        --output data/hull/equiformer_v3_oam_ppd_v0.pkl

    # Local smoke (n=50 entries to test the pipeline):
    external/equiformer_v3/.venv/bin/python scripts/build_equiformer_v3_hull.py \\
        --limit 50 \\
        --output /tmp/equiformer_v3_oam_ppd_smoke.pkl
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase.filters import FrechetCellFilter, UnitCellFilter
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import FIRE, LBFGS
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = REPO_ROOT / "data/mp20/raw/all.csv"
DEFAULT_CKPT = REPO_ROOT / "external/equiformer_v3/checkpoints/omat24-mptrj-salex_gradient.pt"
DEFAULT_OUTPUT = REPO_ROOT / "data/hull/equiformer_v3_oam_ppd_v0.pkl"

FILTER_CLS = {"frechet": FrechetCellFilter, "unit": UnitCellFilter, "none": None}
OPTIM_CLS = {"FIRE": FIRE, "LBFGS": LBFGS}


def _build_mp2020_parameters(composition: Any, mp2020_compat: Any) -> dict[str, Any]:
    """Build synthetic VASP-style metadata MP2020 needs to apply corrections.

    Mirrors src/utils/sample_stats.py:_build_mp2020_parameters.
    """
    sorted_elements = sorted(
        (el for el in composition.elements if composition[el] > 0),
        key=lambda el: el.X,
    )
    most_electroneg = sorted_elements[-1].symbol if sorted_elements else None
    u_settings = mp2020_compat.u_settings.get(most_electroneg, {})
    hubbards = {
        el.symbol: float(u_settings.get(el.symbol, 0.0))
        for el in composition.elements
        if float(u_settings.get(el.symbol, 0.0)) != 0.0
    }
    run_type = "GGA+U" if hubbards else "GGA"
    return {"run_type": run_type, "hubbards": hubbards, "software": "vasp"}


def load_source_records(csv_path: Path, cif_column: str = "cif.conv") -> list[dict]:
    """Read structure-bearing records from the MP-20 CSV.

    Source-of-truth for structures: data/mp20/raw/all.csv has DFT-relaxed
    structures (the `cif.conv` column) for ~45k MP entries. The MP PPD pickle
    holds ComputedEntry objects (composition + energy only, no structures),
    so it's not usable as a source for re-evaluation with a different MLIP.

    For a "full MP" hull (~150k entries), structures would have to be fetched
    from the MP API or a separate dataset. The MP-20 subset is what
    Crystalite trains on and what novelty is defined against, so it's the
    aligned scope for this chapter's hull.
    """
    import pandas as pd

    print(f"reading MP-20 entries from {csv_path}", flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path, usecols=["material_id", "pretty_formula", cif_column])
    df = df.rename(columns={cif_column: "cif_str"})
    print(f"  loaded {len(df)} rows in {time.perf_counter() - t0:.1f}s", flush=True)
    return df.to_dict("records")


def _score_or_relax_structure(
    *,
    structure,
    calc,
    adaptor: AseAtomsAdaptor,
    relax: bool,
    optimizer: str,
    cell_filter: str,
    fmax: float,
    max_steps: int,
) -> tuple[Any, float, int, float]:
    atoms = adaptor.get_atoms(structure)
    atoms.calc = calc

    t0 = time.perf_counter()
    if not relax:
        e_total = float(atoms.get_potential_energy())
        return structure, e_total, 0, time.perf_counter() - t0

    filter_cls = FILTER_CLS[cell_filter]
    optim_cls = OPTIM_CLS[optimizer]
    wrapped = filter_cls(atoms) if filter_cls is not None else atoms
    opt = optim_cls(wrapped, logfile=None)
    opt.run(fmax=float(fmax), steps=int(max_steps))

    e_total = float(atoms.get_potential_energy())
    relaxed_structure = adaptor.get_structure(atoms)
    # AseAtomsAdaptor can preserve the ASE calculator on the pymatgen
    # Structure as a dynamic `calc` attribute. If left attached, the PPD
    # pickle silently captures the full Equiformer model.
    if hasattr(relaxed_structure, "calc"):
        delattr(relaxed_structure, "calc")
    return relaxed_structure, e_total, int(opt.nsteps), time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument(
        "--cif-column",
        default="cif.conv",
        choices=["cif.conv", "cif"],
        help="Which CIF column to use (cif.conv = conventional cell, what Crystalite saw).",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--entries-jsonl",
        type=Path,
        default=None,
        help="Optional: also save entries as portable JSONL (defaults to <output>.entries.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="If > 0, only process this many entries (for local testing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used if --limit < total entries (for reproducible subsampling).",
    )
    parser.add_argument(
        "--skip-mp2020",
        action="store_true",
        help="Do NOT apply MP2020 corrections (use only if you know what you're doing).",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help=(
            "Relax each MP reference structure with EquiformerV3 before building "
            "the hull. Default is single-point at the input DFT-relaxed geometry."
        ),
    )
    parser.add_argument("--relax-max-steps", type=int, default=500)
    parser.add_argument("--relax-fmax", type=float, default=0.02)
    parser.add_argument(
        "--relax-cell-filter",
        default="frechet",
        choices=list(FILTER_CLS.keys()),
    )
    parser.add_argument(
        "--relax-optimizer",
        default="FIRE",
        choices=list(OPTIM_CLS.keys()),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "gpu"],
        help="Device selection for OCPCalculator.",
    )
    args = parser.parse_args()

    if not args.source_csv.exists():
        print(f"FATAL: source CSV not found: {args.source_csv}", file=sys.stderr)
        return 1
    if not args.checkpoint.exists():
        print(f"FATAL: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entries_jsonl = args.entries_jsonl or args.output.with_suffix(".entries.jsonl")
    summary_path = args.output.with_suffix(".summary.json")

    # 1. Load source records from the MP-20 CSV.
    src_records = load_source_records(args.source_csv, cif_column=args.cif_column)
    if args.limit > 0 and args.limit < len(src_records):
        # PatchedPhaseDiagram requires pure-element terminal entries for every
        # element. When subsampling, always include unaries; sample the rest.
        from pymatgen.core import Composition
        def _arity_safe(formula):
            try:
                return len(Composition(formula).elements)
            except Exception:
                return None
        unaries = [r for r in src_records if _arity_safe(r["pretty_formula"]) == 1]
        rest = [r for r in src_records if _arity_safe(r["pretty_formula"]) != 1]
        rng = np.random.default_rng(args.seed)
        if args.limit <= len(unaries):
            src_records = unaries
            print(
                f"  subsample limit ({args.limit}) ≤ #unaries ({len(unaries)});"
                f"  using all unaries to preserve terminal entries.",
                flush=True,
            )
        else:
            n_rest = args.limit - len(unaries)
            idx = rng.permutation(len(rest))[:n_rest]
            sampled_rest = [rest[i] for i in idx]
            src_records = unaries + sampled_rest
            print(
                f"  subsampled to {len(src_records)} entries "
                f"({len(unaries)} unaries + {len(sampled_rest)} multi-element, seed={args.seed})",
                flush=True,
            )

    # 2. Load EquiformerV3 calculator.
    import torch
    from fairchem.core import OCPCalculator

    if args.device == "auto":
        cpu = not torch.cuda.is_available()
    elif args.device == "cpu":
        cpu = True
    elif args.device in ("cuda", "gpu"):
        cpu = False
    else:
        raise ValueError(f"unknown device: {args.device!r}")
    print(f"loading EquiformerV3-OAM, cpu={cpu}", flush=True)
    t0 = time.perf_counter()
    calc = OCPCalculator(checkpoint_path=str(args.checkpoint), cpu=cpu, seed=0)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    adaptor = AseAtomsAdaptor()
    mp2020 = None if args.skip_mp2020 else MaterialsProject2020Compatibility(check_potcar=False)

    # 3. Score or relax each source entry; construct fresh CSE.
    out_entries: list = []
    n_inference_fail = 0
    n_mp2020_dropped = 0
    n_no_structure = 0
    n_relaxed = 0
    n_relax_steps: list[int] = []
    eval_times: list[float] = []

    protocol = "relaxations" if args.relax else "single-points"
    print(f"running EquiformerV3 {protocol} on {len(src_records)} entries...", flush=True)
    from pymatgen.core import Structure

    bar = tqdm(src_records, total=len(src_records), dynamic_ncols=True)
    for src in bar:
        cif_str = src.get("cif_str")
        if not isinstance(cif_str, str) or not cif_str:
            n_no_structure += 1
            bar.set_postfix(inf_fail=n_inference_fail, no_struct=n_no_structure, dropped=n_mp2020_dropped)
            continue
        try:
            structure = Structure.from_str(cif_str, fmt="cif")
        except Exception:
            n_no_structure += 1
            bar.set_postfix(inf_fail=n_inference_fail, no_struct=n_no_structure, dropped=n_mp2020_dropped)
            continue

        try:
            final_structure, e_total, nsteps, eval_time = _score_or_relax_structure(
                structure=structure,
                calc=calc,
                adaptor=adaptor,
                relax=bool(args.relax),
                optimizer=str(args.relax_optimizer),
                cell_filter=str(args.relax_cell_filter),
                fmax=float(args.relax_fmax),
                max_steps=int(args.relax_max_steps),
            )
            eval_times.append(eval_time)
            n_relax_steps.append(int(nsteps))
            n_relaxed += int(bool(args.relax))
        except Exception as exc:
            n_inference_fail += 1
            bar.set_postfix(inf_fail=n_inference_fail, no_struct=n_no_structure, dropped=n_mp2020_dropped)
            continue

        if not math.isfinite(e_total):
            n_inference_fail += 1
            bar.set_postfix(inf_fail=n_inference_fail, no_struct=n_no_structure, dropped=n_mp2020_dropped)
            continue

        cse_kwargs = {
            "composition": final_structure.composition,
            "energy": e_total,
            "structure": final_structure,
            "entry_id": src.get("material_id"),
        }
        if mp2020 is not None:
            cse_kwargs["parameters"] = _build_mp2020_parameters(final_structure.composition, mp2020)
        cse = ComputedStructureEntry(**cse_kwargs)
        if mp2020 is not None:
            try:
                cse = mp2020.process_entry(cse, on_error="raise")
            except Exception:
                n_mp2020_dropped += 1
                bar.set_postfix(inf_fail=n_inference_fail, no_struct=n_no_structure, dropped=n_mp2020_dropped)
                continue
            if cse is None:
                n_mp2020_dropped += 1
                bar.set_postfix(inf_fail=n_inference_fail, no_struct=n_no_structure, dropped=n_mp2020_dropped)
                continue

        out_entries.append(cse)
    bar.close()

    print(
        f"\ninference done. kept={len(out_entries)}  "
        f"inf_fail={n_inference_fail}  no_struct={n_no_structure}  "
        f"mp2020_dropped={n_mp2020_dropped}",
        flush=True,
    )

    if len(out_entries) < 10:
        print("FATAL: too few successful entries; refusing to build hull.", file=sys.stderr)
        return 2

    # 4. Build PatchedPhaseDiagram.
    print(f"building PatchedPhaseDiagram from {len(out_entries)} entries...", flush=True)
    t0 = time.perf_counter()
    ppd = PatchedPhaseDiagram(out_entries)
    print(f"  built in {time.perf_counter() - t0:.1f}s", flush=True)

    # 5. Save pickle.
    print(f"writing pickle: {args.output}", flush=True)
    with args.output.open("wb") as f:
        pickle.dump(ppd, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 6. Save entries as portable JSONL (for cross-version reconstruction).
    print(f"writing entries jsonl: {entries_jsonl}", flush=True)
    with entries_jsonl.open("w", encoding="utf-8") as f:
        for cse in out_entries:
            f.write(json.dumps(cse.as_dict()) + "\n")

    # 7. Summary.
    summary = {
        "source_csv": str(args.source_csv),
        "cif_column": args.cif_column,
        "checkpoint": str(args.checkpoint),
        "limit": args.limit,
        "seed": args.seed,
        "protocol": "relax" if args.relax else "single_point",
        "relax": bool(args.relax),
        "relax_max_steps": int(args.relax_max_steps),
        "relax_fmax": float(args.relax_fmax),
        "relax_cell_filter": str(args.relax_cell_filter),
        "relax_optimizer": str(args.relax_optimizer),
        "applied_mp2020": mp2020 is not None,
        "n_source_entries": len(src_records),
        "n_kept": len(out_entries),
        "n_inference_fail": n_inference_fail,
        "n_relaxed": int(n_relaxed),
        "n_no_structure": n_no_structure,
        "n_mp2020_dropped": n_mp2020_dropped,
        "eval_time_s": {
            "median": float(np.median(eval_times)) if eval_times else None,
            "p90": float(np.quantile(eval_times, 0.9)) if eval_times else None,
            "max": float(np.max(eval_times)) if eval_times else None,
            "total": float(np.sum(eval_times)) if eval_times else None,
        },
        "relax_steps": {
            "median": float(np.median(n_relax_steps)) if n_relax_steps else None,
            "p90": float(np.quantile(n_relax_steps, 0.9)) if n_relax_steps else None,
            "max": int(np.max(n_relax_steps)) if n_relax_steps else None,
        },
        "n_chemical_systems": len(getattr(ppd, "pds", {})) if hasattr(ppd, "pds") else None,
        "output_pickle": str(args.output),
        "output_entries_jsonl": str(entries_jsonl),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nsummary:\n{json.dumps(summary, indent=2)}", flush=True)
    print(f"\ndone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
