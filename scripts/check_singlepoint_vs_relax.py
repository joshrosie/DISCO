#!/usr/bin/env python
"""Empirical check: does EquiformerV3 single-point at DFT-min give the same
energy as a full EquiformerV3 relaxation?

Tests N random MP-20 entries. For each, runs:
  A. single-point at the DFT-relaxed geometry  → e_singlepoint, max_force
  B. full FIRE+FrechetCellFilter relaxation    → e_relaxed, geometry_rmsd

Reports |Δe| distribution + RMSD distribution + worst-case structures.

If single-points and relax agree on energy to within ~10 meV/atom and
geometries barely move, single-points suffice for the hull build. If they
diverge, full relaxation is required.

Run via the inner equiformer_v3 venv:

    external/equiformer_v3/.venv/bin/python \\
        scripts/check_singlepoint_vs_relax.py --limit 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data/mp20/raw/test.csv"
DEFAULT_CKPT = REPO_ROOT / "external/equiformer_v3/checkpoints/omat24-mptrj-salex_gradient.pt"


def rmsd_atoms(a, b) -> float:
    """Mean-position RMSD between two ASE atoms in Angstroms (matched ordering)."""
    pa = a.get_positions()
    pb = b.get_positions()
    if pa.shape != pb.shape:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1))))


def lattice_strain(a, b) -> float:
    """Frobenius norm of (B^-1 A - I) — rough scalar measure of lattice deformation."""
    ca = a.get_cell().array
    cb = b.get_cell().array
    try:
        m = np.linalg.solve(cb, ca) - np.eye(3)
    except Exception:
        return float("nan")
    return float(np.linalg.norm(m, ord="fro"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/singlepoint_vs_relax.csv")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.csv}", flush=True)
    df = pd.read_csv(args.csv, usecols=["material_id", "pretty_formula", "cif.conv"])
    df = df.rename(columns={"cif.conv": "cif_str"})
    df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"  selected {len(df)} entries (seed={args.seed})", flush=True)

    import torch
    from fairchem.core import OCPCalculator

    cpu = not torch.cuda.is_available()
    print(f"loading EquiformerV3 OAM, cpu={cpu}", flush=True)
    calc = OCPCalculator(checkpoint_path=str(args.checkpoint), cpu=cpu, seed=0)
    adaptor = AseAtomsAdaptor()

    rows: list[dict] = []
    for record in df.itertuples(index=False):
        sample_id = record.material_id
        formula = record.pretty_formula
        try:
            structure = Structure.from_str(record.cif_str, fmt="cif")
        except Exception as exc:
            print(f"  [parse-fail] {sample_id}: {exc}")
            continue

        # Single-point at DFT-min.
        atoms_sp = adaptor.get_atoms(structure)
        atoms_sp.calc = calc
        try:
            t0 = time.perf_counter()
            e_sp = float(atoms_sp.get_potential_energy())
            forces_sp = atoms_sp.get_forces()
            sp_time = time.perf_counter() - t0
        except Exception as exc:
            print(f"  [singlepoint-fail] {sample_id}: {exc}")
            continue
        max_force = float(np.abs(forces_sp).max())
        e_sp_per_atom = e_sp / len(atoms_sp)

        # Full relaxation: FIRE + FrechetCellFilter.
        atoms_rx = adaptor.get_atoms(structure)
        atoms_rx.calc = calc
        try:
            t0 = time.perf_counter()
            opt = FIRE(FrechetCellFilter(atoms_rx), logfile=None)
            opt.run(fmax=args.fmax, steps=args.max_steps)
            e_rx = float(atoms_rx.get_potential_energy())
            relax_time = time.perf_counter() - t0
            nsteps = opt.nsteps
        except Exception as exc:
            print(f"  [relax-fail] {sample_id}: {exc}")
            continue
        e_rx_per_atom = e_rx / len(atoms_rx)
        rmsd = rmsd_atoms(atoms_sp, atoms_rx)
        strain = lattice_strain(atoms_sp, atoms_rx)

        de_total = e_rx - e_sp
        de_per_atom = e_rx_per_atom - e_sp_per_atom
        rows.append(
            {
                "sample_id": sample_id,
                "formula": formula,
                "num_atoms": len(atoms_sp),
                "e_singlepoint_per_atom": e_sp_per_atom,
                "e_relaxed_per_atom": e_rx_per_atom,
                "de_per_atom": de_per_atom,
                "max_force_init": max_force,
                "rmsd_pos_A": rmsd,
                "lattice_strain": strain,
                "relax_nsteps": int(nsteps),
                "sp_time_s": sp_time,
                "relax_time_s": relax_time,
            }
        )
        print(
            f"  {sample_id:>14}  {formula:<14}  n={len(atoms_sp):>2}  "
            f"|de|/atom={abs(de_per_atom):.5f}  fmax_init={max_force:.4f}  "
            f"rmsd={rmsd:.4f}Å  strain={strain:.4f}  "
            f"sp={sp_time:.1f}s  relax={relax_time:.1f}s ({nsteps} steps)",
            flush=True,
        )

    if not rows:
        print("no successful rows", file=sys.stderr)
        return 2

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nwrote {args.output} ({len(out_df)} rows)", flush=True)

    de = out_df["de_per_atom"].abs()
    rmsd = out_df["rmsd_pos_A"]
    fmax_init = out_df["max_force_init"]
    print("\n=== summary ===")
    print(f"  |de|/atom (eV):    median={de.median():.5f}  p90={de.quantile(0.9):.5f}  max={de.max():.5f}")
    print(f"  pos RMSD (Å):      median={rmsd.median():.4f}  p90={rmsd.quantile(0.9):.4f}  max={rmsd.max():.4f}")
    print(f"  init max-force:    median={fmax_init.median():.4f}  p90={fmax_init.quantile(0.9):.4f}  max={fmax_init.max():.4f}")

    threshold_de = 0.010  # 10 meV/atom — well below the 100 meV/atom metastability tolerance
    threshold_rmsd = 0.05  # 50 mÅ — a small lattice perturbation
    de_ok = (de < threshold_de).mean()
    rmsd_ok = (rmsd < threshold_rmsd).mean()
    print(f"\n  |de|/atom < {threshold_de} eV:    {de_ok:.0%} of entries")
    print(f"  pos RMSD  < {threshold_rmsd} Å:     {rmsd_ok:.0%} of entries")
    if de_ok > 0.9 and rmsd_ok > 0.9:
        print("\n  → single-points are a faithful proxy for full relaxation.")
        print("    proceed with single-point hull build.")
    else:
        print("\n  → some entries shift meaningfully under EquiformerV3 relax.")
        print("    consider full relaxation for the hull build,")
        print("    or investigate which chemistries diverge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
