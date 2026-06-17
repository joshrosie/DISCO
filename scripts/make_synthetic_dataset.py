#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.mp20_tokens import NMAX as DEFAULT_NMAX
from src.data.synthetic_augmentation import DEDUP_MODES, FILTER_LEVELS, make_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate/load Crystalite proposals, curate them with optional "
            "relaxation/stability scoring, and write a training-compatible "
            "synthetic dataset."
        )
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Directory containing .cif and/or token .pt samples.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional Crystalite checkpoint used to generate a raw proposal pool internally.",
    )
    parser.add_argument(
        "--train_output_dir",
        default=None,
        help="Optional training output dir used to resolve --checkpoint when needed.",
    )
    parser.add_argument(
        "--checkpoint_preference",
        default="auto",
        choices=["auto", "best", "final", "step_latest", "epoch_latest"],
    )
    parser.add_argument("--num_generate", type=int, default=None)
    parser.add_argument("--sample_chunk_size", type=int, default=256)
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--sample_num_steps", type=int, default=None)
    parser.add_argument("--sample_device", default="cuda")
    parser.add_argument("--sample_mode", default="ema", choices=["ema", "regular"])
    parser.add_argument("--sampler", default=None, choices=["edm", "vfm", "catflow"])
    parser.add_argument(
        "--atom_count_strategy",
        default=None,
        choices=["empirical", "fixed"],
    )
    parser.add_argument("--fixed_num_atoms", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output_dir", required=True, help="Output synthetic dataset directory.")
    parser.add_argument(
        "--real_train_path",
        default=None,
        help=(
            "Legacy alias used for both generation_data_root and "
            "reference_data_root when those are omitted."
        ),
    )
    parser.add_argument(
        "--generation_data_root",
        default=None,
        help=(
            "Dataset root used only for checkpoint generation count priors and "
            "allowed elements. Usually the original training data root."
        ),
    )
    parser.add_argument(
        "--reference_data_root",
        action="append",
        default=None,
        help=(
            "Dataset root/CSV/synthetic dataset used for dedup and novelty filtering. "
            "May be passed multiple times or as an os.pathsep-separated list."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dedup_mode", choices=DEDUP_MODES, default="formula")
    parser.add_argument("--filter_level", choices=FILTER_LEVELS, default="valid")
    parser.add_argument("--nmax", type=int, default=DEFAULT_NMAX)
    parser.add_argument(
        "--ehull_metastable_thresh",
        type=float,
        default=0.1,
        help=(
            "e_above_hull cutoff in eV/atom for metastable/msun_like filtering. "
            "stable_like always uses e_above_hull <= 0."
        ),
    )
    parser.add_argument(
        "--thermo_mlip",
        default=None,
        choices=["chgnet", "nequip", "equiformer_v3"],
        help="Run row-level relaxation/scoring with this MLIP backend.",
    )
    parser.add_argument(
        "--thermo_ppd_mp",
        default="data/mp20/hull/2023-02-07-ppd-mp.pkl",
        help="Patched phase diagram pickle used for e_above_hull scoring.",
    )
    parser.add_argument("--thermo_stability_device", default="cuda")
    parser.add_argument(
        "--thermo_ehull_method",
        default="mp2020_like",
        choices=["uncorrected", "mp2020_like"],
    )
    parser.add_argument("--thermo_relax_steps", type=int, default=200)
    parser.add_argument("--thermo_stability_batch", type=int, default=32)
    parser.add_argument("--nequip_compile_path", default=None)
    parser.add_argument(
        "--nequip_relax_mode",
        default="batch",
        choices=["sequential", "batch"],
    )
    parser.add_argument("--nequip_optimizer", default="FIRE")
    parser.add_argument("--nequip_cell_filter", default="frechet")
    parser.add_argument("--nequip_fmax", type=float, default=0.01)
    parser.add_argument("--nequip_max_force_abort", type=float, default=1e6)
    parser.add_argument(
        "--equiformer_v3_inner_python",
        default=None,
        help="Path to the inner equiformer_v3 venv's python (e.g., external/equiformer_v3/.venv/bin/python).",
    )
    parser.add_argument(
        "--equiformer_v3_wrapper",
        default=None,
        help="Path to scripts/equiformer_v3_inference_wrapper.py.",
    )
    parser.add_argument(
        "--equiformer_v3_checkpoint",
        default=None,
        help="Path to the OAM checkpoint (e.g., external/equiformer_v3/checkpoints/omat24-mptrj-salex_gradient.pt).",
    )
    parser.add_argument("--equiformer_v3_max_steps", type=int, default=500)
    parser.add_argument("--equiformer_v3_fmax", type=float, default=0.02)
    parser.add_argument("--equiformer_v3_cell_filter", default="frechet", choices=["frechet", "unit", "none"])
    parser.add_argument("--equiformer_v3_optimizer", default="FIRE", choices=["FIRE", "LBFGS"])
    parser.add_argument(
        "--equiformer_v3_device",
        default="auto",
        choices=["auto", "cpu", "cuda", "gpu"],
    )
    parser.add_argument(
        "--min_distance",
        type=float,
        default=0.5,
        help="Minimum allowed interatomic distance in Angstrom for valid/filtered tiers.",
    )
    parser.add_argument(
        "--max_abs_energy_per_atom",
        type=float,
        default=100.0,
        help="Conservative relaxed_filtered energy sanity threshold in eV/atom.",
    )
    parser.add_argument(
        "--max_volume_change",
        type=float,
        default=5.0,
        help="Conservative relaxed_filtered absolute fractional volume-change threshold.",
    )
    parser.add_argument(
        "--no_write_cifs",
        action="store_true",
        help=(
            "Skip per-sample CIF files under structures/; the CIF string is still "
            "embedded in raw/train.csv (the source of truth for MP20Tokens). "
            "Use this for large training-only synthetic datasets to reduce inode usage."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.input_dir is None and args.checkpoint is None:
        raise SystemExit("Provide --input_dir and/or --checkpoint.")
    if args.checkpoint is not None and args.num_generate is None:
        raise SystemExit("--checkpoint requires --num_generate.")
    summary = make_synthetic_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        real_train_path=args.real_train_path,
        generation_data_root=args.generation_data_root,
        reference_data_root=args.reference_data_root,
        checkpoint=args.checkpoint,
        train_output_dir=args.train_output_dir,
        checkpoint_preference=args.checkpoint_preference,
        num_generate=args.num_generate,
        sample_chunk_size=args.sample_chunk_size,
        sample_seed=args.sample_seed,
        sample_num_steps=args.sample_num_steps,
        sample_device=args.sample_device,
        sample_mode=args.sample_mode,
        sampler=args.sampler,
        atom_count_strategy=args.atom_count_strategy,
        fixed_num_atoms=args.fixed_num_atoms,
        bf16=bool(args.bf16),
        max_samples=args.max_samples,
        dedup_mode=args.dedup_mode,
        filter_level=args.filter_level,
        nmax=args.nmax,
        min_distance=args.min_distance,
        max_abs_energy_per_atom=args.max_abs_energy_per_atom,
        max_volume_change=args.max_volume_change,
        ehull_metastable_thresh=args.ehull_metastable_thresh,
        thermo_mlip=args.thermo_mlip,
        thermo_ppd_mp=args.thermo_ppd_mp,
        thermo_stability_device=args.thermo_stability_device,
        thermo_ehull_method=args.thermo_ehull_method,
        thermo_relax_steps=args.thermo_relax_steps,
        thermo_stability_batch=args.thermo_stability_batch,
        nequip_compile_path=args.nequip_compile_path,
        nequip_relax_mode=args.nequip_relax_mode,
        nequip_optimizer=args.nequip_optimizer,
        nequip_cell_filter=args.nequip_cell_filter,
        nequip_fmax=args.nequip_fmax,
        nequip_max_force_abort=args.nequip_max_force_abort,
        equiformer_v3_inner_python=args.equiformer_v3_inner_python,
        equiformer_v3_wrapper=args.equiformer_v3_wrapper,
        equiformer_v3_checkpoint=args.equiformer_v3_checkpoint,
        equiformer_v3_max_steps=args.equiformer_v3_max_steps,
        equiformer_v3_fmax=args.equiformer_v3_fmax,
        equiformer_v3_cell_filter=args.equiformer_v3_cell_filter,
        equiformer_v3_optimizer=args.equiformer_v3_optimizer,
        equiformer_v3_device=args.equiformer_v3_device,
        write_cifs=not bool(args.no_write_cifs),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
