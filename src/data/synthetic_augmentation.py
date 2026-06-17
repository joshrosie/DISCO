"""Synthetic Crystalite augmentation datasets and preprocessing helpers."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch.utils.data import Dataset

from src.data.mp20_tokens import (
    NMAX as DEFAULT_NMAX,
    VZ,
    MP20Tokens,
    lattice_to_Y,
    tokens_to_structure,
    translate_frac_coords,
)


FILTER_LEVELS = (
    "raw",
    "valid",
    "relaxed",
    "relaxed_filtered",
    "stable_like",
    "msun_like",
)
DEDUP_MODES = ("none", "formula", "structure")
REQUIRED_KEYS = ("A0", "F1", "Y1", "pad_mask")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _metadata_template(sample_id: str, source_path: str | None = None) -> dict:
    return {
        "sample_id": sample_id,
        "source": "crystalite_synthetic",
        "generator_checkpoint": None,
        "sampling_config": None,
        "pre_relax_path": source_path,
        "post_relax_path": None,
        "formula": None,
        "num_atoms": None,
        "space_group_pre": None,
        "space_group_post": None,
        "valid_parse": True,
        "valid_geometry": True,
        "relax_success": None,
        "energy_per_atom": None,
        "formation_energy_per_atom": None,
        "e_above_hull": None,
        "stability_label": None,
        "is_stable": False,
        "is_metastable": False,
        "is_unique": None,
        "is_novel": None,
        "is_msun": None,
        "dedup_within_synthetic": False,
        "dedup_against_train": False,
        "dedup_key": None,
        "filter_status": "kept",
        "filter_reason": None,
    }


def _canonical_formula_key(formula: str) -> str:
    raw = str(formula).strip()
    if not raw:
        return raw
    try:
        return Composition(raw).reduced_formula
    except Exception:
        return raw


def _validate_structure(
    structure: Structure,
    *,
    nmax: int,
    min_distance: float,
    max_atoms: int | None = None,
) -> tuple[bool, str | None]:
    if len(structure) <= 0:
        return False, "zero_atoms"
    atom_limit = int(max_atoms) if max_atoms is not None else int(nmax)
    if len(structure) > atom_limit or len(structure) > nmax:
        return False, "too_many_atoms"
    try:
        zs = [int(site.specie.Z) for site in structure]
    except Exception:
        return False, "invalid_species"
    if any(z <= 0 or z > VZ for z in zs):
        return False, "invalid_species"
    arrays = [
        np.asarray(structure.frac_coords, dtype=np.float64),
        np.asarray(structure.lattice.matrix, dtype=np.float64),
    ]
    if any(not np.isfinite(a).all() for a in arrays):
        return False, "nan_or_inf"
    if not math.isfinite(float(structure.volume)) or float(structure.volume) <= 0.0:
        return False, "invalid_volume"
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    if ((frac < -1.0e-4) | (frac >= 1.0 + 1.0e-4)).any():
        return False, "fractional_coords_out_of_range"
    if len(structure) > 1:
        dmat = np.asarray(structure.distance_matrix, dtype=np.float64)
        if not np.isfinite(dmat).all():
            return False, "nan_or_inf"
        np.fill_diagonal(dmat, np.inf)
        if float(np.min(dmat)) < float(min_distance):
            return False, "min_distance_too_small"
    return True, None


def _space_group_number(structure: Structure | None, symprec: float = 0.1) -> int | None:
    if structure is None:
        return None
    try:
        return int(SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number())
    except Exception:
        return None


def _filter_requires_relax(filter_level: str) -> bool:
    return FILTER_LEVELS.index(filter_level) >= FILTER_LEVELS.index("relaxed")


def _filter_requires_stability(filter_level: str) -> bool:
    return filter_level in {"stable_like", "msun_like"}


def _passes_filter_level(
    meta: dict,
    structure: Structure,
    *,
    filter_level: str,
    nmax: int,
    max_atoms: int | None,
    min_distance: float,
    max_abs_energy_per_atom: float,
    max_volume_change: float,
) -> tuple[bool, str | None]:
    if filter_level not in FILTER_LEVELS:
        raise ValueError(f"Unknown filter_level={filter_level!r}")

    if filter_level == "raw":
        return True, None

    ok, reason = _validate_structure(
        structure, nmax=nmax, max_atoms=max_atoms, min_distance=min_distance
    )
    if not ok:
        return False, reason
    if filter_level == "valid":
        return True, None

    if meta.get("relax_success") is not True:
        return False, "relax_not_successful"
    if filter_level == "relaxed":
        return True, None

    epa = meta.get("energy_per_atom")
    if epa is not None and abs(float(epa)) > float(max_abs_energy_per_atom):
        return False, "absurd_energy_per_atom"
    vchg = meta.get("volume_change")
    if vchg is not None and abs(float(vchg)) > float(max_volume_change):
        return False, "pathological_volume_change"
    if filter_level == "relaxed_filtered":
        return True, None

    if meta.get("dedup_within_synthetic") or meta.get("dedup_against_train"):
        return False, "not_unique_or_novel"
    stability = meta.get("stability_label")
    if filter_level == "stable_like":
        if stability != "stable":
            return False, "not_stable"
        return True, None
    if stability not in ("stable", "metastable"):
        return False, "not_stable_or_metastable"
    return True, None


def _iter_input_structures(input_dir: Path) -> Iterable[tuple[str, Structure | None, dict]]:
    for cif_path in sorted(input_dir.rglob("*.cif")):
        meta = _metadata_template("", str(cif_path))
        try:
            structure = Structure.from_file(str(cif_path))
        except Exception as exc:
            meta.update(
                valid_parse=False,
                filter_status="rejected",
                filter_reason=f"parse_failed:{type(exc).__name__}",
            )
            yield str(cif_path), None, meta
            continue
        yield str(cif_path), structure, meta

    for pt_path in sorted(input_dir.rglob("*.pt")):
        try:
            records = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            meta = _metadata_template("", str(pt_path))
            meta.update(
                valid_parse=False,
                filter_status="rejected",
                filter_reason=f"parse_failed:{type(exc).__name__}",
            )
            yield str(pt_path), None, meta
            continue
        if isinstance(records, dict) and "samples" in records:
            records = records["samples"]
        if not isinstance(records, list):
            continue
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            source = f"{pt_path}:{idx}"
            meta = _metadata_template("", source)
            for key in ("generator_checkpoint", "sampling_config", "relax_success"):
                if key in rec:
                    meta[key] = rec[key]
            for key in ("energy_per_atom", "e_above_hull", "stability_label"):
                if key in rec:
                    meta[key] = rec[key]
            try:
                if all(k in rec for k in REQUIRED_KEYS):
                    yield source, tokens_to_structure(rec), meta
                elif isinstance(rec.get("structure"), Structure):
                    yield source, rec["structure"], meta
            except Exception as exc:
                meta.update(
                    valid_parse=False,
                    filter_status="rejected",
                    filter_reason=f"parse_failed:{type(exc).__name__}",
                )
                yield source, None, meta


def _path_roots(
    roots: str | os.PathLike | Sequence[str | os.PathLike] | None,
) -> list[Path]:
    if roots is None:
        return []
    if isinstance(roots, (str, os.PathLike)):
        raw_roots: Sequence[str | os.PathLike] = [roots]
    else:
        raw_roots = roots
    out: list[Path] = []
    for root in raw_roots:
        if root is None:
            continue
        if isinstance(root, os.PathLike):
            parts = [root]
        else:
            parts = [p for p in str(root).split(os.pathsep) if p]
        out.extend(Path(p) for p in parts)
    return out


def _synthetic_data_roots(
    roots: str | os.PathLike | Sequence[str | os.PathLike] | None,
) -> list[str | os.PathLike]:
    return [str(path) for path in _path_roots(roots)]


def _read_real_train_formulas(
    real_train_path: str | os.PathLike | Sequence[str | os.PathLike] | None,
) -> set[str]:
    formulas: set[str] = set()
    for path in _path_roots(real_train_path):
        formulas.update(_read_formulas_from_root(path))
    return formulas


def _read_formulas_from_root(path: Path) -> set[str]:
    candidates = []
    if path.is_file():
        candidates.append(path)
    elif path.is_dir():
        candidates.extend(
            [
                path / "raw" / "train.csv",
                path / "train.csv",
                path / "metadata.jsonl",
            ]
        )
    formulas: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".jsonl":
            with candidate.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        formula = json.loads(line).get("formula")
                        if formula:
                            formulas.add(_canonical_formula_key(str(formula)))
        elif candidate.suffix == ".csv":
            with candidate.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    formula = row.get("pretty_formula") or row.get("formula")
                    if formula:
                        formulas.add(_canonical_formula_key(str(formula)))
        if formulas:
            break
    return formulas


def _read_real_train_structures(
    real_train_path: str | os.PathLike | Sequence[str | os.PathLike] | None,
) -> list[Structure]:
    structs: list[Structure] = []
    for path in _path_roots(real_train_path):
        structs.extend(_read_structures_from_root(path))
    return structs


def _read_structures_from_root(path: Path) -> list[Structure]:
    candidates: list[Path] = []
    if path.is_file():
        candidates.append(path)
    elif path.is_dir():
        candidates.extend([path / "raw" / "train.csv", path / "train.csv"])
        structures_dir = path / "structures"
        if structures_dir.exists():
            candidates.extend(sorted(structures_dir.glob("*.cif")))
    structs: list[Structure] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".csv":
            with candidate.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    cif = row.get("cif") or row.get("cif.conv")
                    if not cif:
                        continue
                    try:
                        structs.append(Structure.from_str(cif, fmt="cif"))
                    except Exception:
                        continue
        elif candidate.suffix == ".cif":
            try:
                structs.append(Structure.from_file(str(candidate)))
            except Exception:
                continue
        if structs:
            if candidate.suffix == ".csv":
                break
    return structs


class _ChgnetThermoOracle:
    """Row-level CHGNet relaxation + hull scoring oracle.

    This mirrors the small subset of `NequipFormEnergyOracle` used by the
    synthetic curator: `call_many(structures) -> list[row]`.
    """

    def __init__(
        self,
        *,
        ppd_path: str,
        stability_device: str,
        relax_steps: int = 200,
        apply_mp2020: bool = True,
    ) -> None:
        from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
        from src.eval.stability import load_phase_diagram
        from src.utils.sample_stats import make_chgnet_and_relaxer

        _, self.relaxer, _ = make_chgnet_and_relaxer(stability_device)
        self.ppd = load_phase_diagram(ppd_path)
        self.relax_steps = int(relax_steps)
        self.apply_mp2020 = bool(apply_mp2020)
        self._mp2020_compat = (
            MaterialsProject2020Compatibility(check_potcar=False)
            if self.apply_mp2020
            else None
        )

    @staticmethod
    def _empty_row(*, err: str = "") -> dict:
        return {
            "e_form": float("nan"),
            "e_above_hull": float("nan"),
            "e_total": float("nan"),
            "nsteps": -1,
            "final_structure": None,
            "err": err,
        }

    def _score_relaxed(self, final_struct: Structure, e_total: float, nsteps: int) -> dict:
        from pymatgen.entries.computed_entries import ComputedStructureEntry
        from src.utils.sample_stats import (
            _build_mp2020_parameters,
            compute_e_above_hull_mp2020_like,
            compute_e_above_hull_uncorrected,
        )

        out = self._empty_row()
        out.update(e_total=float(e_total), nsteps=int(nsteps), final_structure=final_struct)
        if self.apply_mp2020:
            e_above_val, fail_reason = compute_e_above_hull_mp2020_like(
                self.ppd,
                final_struct,
                float(e_total),
                mp2020_compat=self._mp2020_compat,
            )
        else:
            e_above_val, fail_reason = compute_e_above_hull_uncorrected(
                self.ppd,
                final_struct,
                float(e_total),
            )
        if fail_reason is not None:
            out["err"] = fail_reason
        elif e_above_val is not None:
            out["e_above_hull"] = float(e_above_val)

        try:
            params = None
            if self.apply_mp2020:
                params = _build_mp2020_parameters(
                    composition=final_struct.composition,
                    mp2020_compat=self._mp2020_compat,
                )
            entry = ComputedStructureEntry(
                composition=final_struct.composition,
                energy=float(e_total),
                structure=final_struct,
                parameters=params,
            )
            if self.apply_mp2020:
                entry = self._mp2020_compat.process_entry(
                    entry.copy(), on_error="raise"
                )
                if entry is None:
                    out["err"] = out["err"] or "mp2020_removed"
                    return out
            out["e_form"] = float(self.ppd.get_form_energy_per_atom(entry))
        except Exception as exc:
            out["err"] = out["err"] or f"entry_exc:{type(exc).__name__}"
        return out

    def call_many(self, structs: list[Structure]) -> list[dict]:
        rows: list[dict] = []
        for struct in structs:
            try:
                relaxation = self.relaxer.relax(
                    struct,
                    steps=self.relax_steps,
                    verbose=False,
                )
                final_struct = relaxation["final_structure"]
                energies = getattr(relaxation["trajectory"], "energies", [])
                if not energies:
                    rows.append(self._empty_row(err="no_energy"))
                else:
                    rows.append(
                        self._score_relaxed(
                            final_struct,
                            float(energies[-1]),
                            int(len(energies)),
                        )
                    )
            except Exception as exc:
                rows.append(self._empty_row(err=f"relax_exc_{type(exc).__name__}"))
        return rows


def _build_row_oracle(
    *,
    thermo_mlip: str | None,
    thermo_ppd_mp: str | os.PathLike | None,
    thermo_stability_device: str,
    thermo_ehull_method: str,
    thermo_relax_steps: int,
    thermo_stability_batch: int,
    nequip_compile_path: str | None,
    nequip_relax_mode: str,
    nequip_optimizer: str,
    nequip_cell_filter: str,
    nequip_fmax: float,
    nequip_max_force_abort: float,
    equiformer_v3_inner_python: str | None = None,
    equiformer_v3_wrapper: str | None = None,
    equiformer_v3_checkpoint: str | None = None,
    equiformer_v3_max_steps: int = 500,
    equiformer_v3_fmax: float = 0.02,
    equiformer_v3_cell_filter: str = "frechet",
    equiformer_v3_optimizer: str = "FIRE",
    equiformer_v3_device: str = "auto",
):
    if thermo_mlip is None:
        return None
    if thermo_ppd_mp is None:
        raise ValueError("Relaxed/msun curation requires --thermo_ppd_mp.")
    mlip = str(thermo_mlip).strip().lower()
    apply_mp2020 = str(thermo_ehull_method).strip().lower() == "mp2020_like"
    if mlip == "nequip":
        if not nequip_compile_path:
            raise ValueError("--thermo_mlip nequip requires --nequip_compile_path.")
        from src.eval.oracles import NequipFormEnergyOracle

        return NequipFormEnergyOracle(
            nequip_compile_path=str(nequip_compile_path),
            ppd_path=str(thermo_ppd_mp),
            stability_device=str(thermo_stability_device),
            optimizer=str(nequip_optimizer),
            cell_filter=str(nequip_cell_filter),
            fmax=float(nequip_fmax),
            max_force_abort=float(nequip_max_force_abort),
            relax_steps=int(thermo_relax_steps),
            apply_mp2020=bool(apply_mp2020),
            relax_mode=str(nequip_relax_mode),
            batch_size=int(thermo_stability_batch),
        )
    if mlip == "chgnet":
        return _ChgnetThermoOracle(
            ppd_path=str(thermo_ppd_mp),
            stability_device=str(thermo_stability_device),
            relax_steps=int(thermo_relax_steps),
            apply_mp2020=bool(apply_mp2020),
        )
    if mlip in ("equiformer_v3", "equiformerv3", "eqv3"):
        if not equiformer_v3_inner_python:
            raise ValueError("--thermo_mlip equiformer_v3 requires --equiformer_v3_inner_python.")
        if not equiformer_v3_wrapper:
            raise ValueError("--thermo_mlip equiformer_v3 requires --equiformer_v3_wrapper.")
        if not equiformer_v3_checkpoint:
            raise ValueError("--thermo_mlip equiformer_v3 requires --equiformer_v3_checkpoint.")
        from src.eval.oracles import EquiformerV3FormEnergyOracle

        return EquiformerV3FormEnergyOracle(
            inner_python=str(equiformer_v3_inner_python),
            wrapper_path=str(equiformer_v3_wrapper),
            checkpoint_path=str(equiformer_v3_checkpoint),
            ppd_path=str(thermo_ppd_mp),
            fmax=float(equiformer_v3_fmax),
            max_steps=int(equiformer_v3_max_steps),
            cell_filter=str(equiformer_v3_cell_filter),
            optimizer=str(equiformer_v3_optimizer),
            apply_mp2020=bool(apply_mp2020),
            device=str(equiformer_v3_device),
        )
    raise ValueError(f"Unsupported thermo_mlip={thermo_mlip!r}.")


def generate_synthetic_samples_from_checkpoint(
    *,
    checkpoint: str | os.PathLike,
    train_output_dir: str | os.PathLike | None = None,
    checkpoint_preference: str = "auto",
    data_root: str | os.PathLike | None = None,
    dataset_name: str | None = None,
    nmax: int | None = None,
    num_generate: int,
    sample_chunk_size: int = 256,
    sample_seed: int | None = None,
    sample_num_steps: int | None = None,
    device: str = "cuda",
    sample_mode: str = "ema",
    sampler: str | None = None,
    atom_count_strategy: str | None = None,
    fixed_num_atoms: int | None = None,
    bf16: bool = False,
) -> list[dict]:
    """Generate raw MP20-token samples from a Crystalite checkpoint.

    This intentionally mirrors the DNG checkpoint evaluator's sampler path but
    returns row-level sample items for the curator instead of metrics.
    """

    from tqdm import tqdm
    from src.crystalite import mod1
    from src.crystalite.sampler import (
        clamp_lattice_latent as _clamp_lattice_latent,
        edm_sampler,
    )
    from src.data.mp20_tokens import MP20Tokens
    from src.data.type_encoding import build_type_encoding
    from src.eval_crystalite_ckpt import (
        _apply_ema_state_dict,
        _build_count_distribution,
        _build_model_from_ckpt,
        _catflow_predict_fn_for_checkpoint,
        _cfg_triplet,
        _cfg_value,
        _load_checkpoint,
        _resolve_checkpoint_path,
        _resolve_sampler_name,
        _sample_num_atoms,
    )
    from src.models.lattice_repr import lattice_latent_to_y1
    from src.utils.dataset import compute_allowed_elements

    ckpt_path = _resolve_checkpoint_path(
        checkpoint=str(checkpoint),
        train_output_dir=str(train_output_dir or ""),
        preference=str(checkpoint_preference),
    )
    ckpt = _load_checkpoint(ckpt_path)
    model, model_args = _build_model_from_ckpt(ckpt=ckpt, device=torch.device("cpu"))
    resolved_data_root = str(
        data_root
        if data_root is not None
        else model_args.get("metrics_data_root", model_args.get("data_root", "data/mp20"))
    )
    resolved_nmax = int(_cfg_value(nmax, model_args, "nmax", DEFAULT_NMAX))
    resolved_seed = int(_cfg_value(sample_seed, model_args, "sample_seed", 123))
    resolved_steps = int(_cfg_value(sample_num_steps, model_args, "sample_num_steps", 100))
    sampler_name = _resolve_sampler_name(sampler, ckpt, model_args)
    lattice_repr = str(_cfg_value(None, model_args, "lattice_repr", "y1"))
    resolved_atom_count_strategy = str(
        _cfg_value(atom_count_strategy, model_args, "atom_count_strategy", "empirical")
    ).lower()

    torch.manual_seed(resolved_seed)
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    torch_device = torch.device(device if use_cuda else "cpu")
    model = model.to(torch_device)
    model.eval()
    if sample_mode == "ema":
        ema_state = ckpt.get("ema_state_dict")
        if ema_state is not None:
            _apply_ema_state_dict(model, ema_state)

    type_encoding_name = str(
        ckpt.get("type_encoding", model_args.get("type_encoding", "atomic_number"))
    )
    type_encoding = build_type_encoding(type_encoding_name, vz=VZ)
    train_split = "train" if (Path(resolved_data_root) / "raw" / "train.csv").exists() else "all"
    ds_train = MP20Tokens(
        root=resolved_data_root,
        augment_translate=True,
        split=train_split,
        nmax=resolved_nmax,
    )
    train_allowed_mask = compute_allowed_elements(ds_train)
    count_probs = None
    if resolved_atom_count_strategy == "empirical":
        count_probs = _build_count_distribution(ds_train, nmax=resolved_nmax)

    sigma_min = float(_cfg_value(None, model_args, "sigma_min", 0.002))
    sigma_max = float(_cfg_value(None, model_args, "sigma_max", 80.0))
    rho = float(_cfg_value(None, model_args, "rho", 7.0))
    S_churn = float(_cfg_value(None, model_args, "S_churn", 20.0))
    S_min = float(_cfg_value(None, model_args, "S_min", 0.0))
    S_max = float(_cfg_value(None, model_args, "S_max", 999.0))
    S_noise = float(_cfg_value(None, model_args, "S_noise", 1.0))
    sigma_data_type = float(_cfg_value(None, model_args, "sigma_data_type", 1.0))
    sigma_data_coord = float(_cfg_value(None, model_args, "sigma_data_coord", 0.25))
    sigma_data_lattice = float(_cfg_value(None, model_args, "sigma_data_lattice", 1.0))
    aa_frac_max_scale = float(_cfg_value(None, model_args, "aa_frac_max_scale", 0.0))
    aa_rho_types = float(_cfg_value(None, model_args, "aa_rho_types", 0.0))
    aa_rho_coords = float(_cfg_value(None, model_args, "aa_rho_coords", 0.0))
    aa_rho_lattice = float(_cfg_value(None, model_args, "aa_rho_lattice", 0.0))
    vfm_eps = float(_cfg_value(None, model_args, "vfm_eps", 1e-4))
    vfm_base_std_type = float(_cfg_value(None, model_args, "vfm_base_std_type", 1.0))
    vfm_base_std_coord = float(_cfg_value(None, model_args, "vfm_base_std_coord", 1.0))
    vfm_base_std_lattice = float(_cfg_value(None, model_args, "vfm_base_std_lattice", 1.0))
    vfm_time_grid = str(_cfg_value(None, model_args, "vfm_time_grid", "uniform"))
    vfm_time_power = float(_cfg_value(None, model_args, "vfm_time_power", 2.0))
    vfm_lattice_prior_mode = str(
        _cfg_value(None, model_args, "vfm_lattice_prior_mode", "gaussian")
    )
    vfm_lattice_prior_log_length_loc = _cfg_triplet(
        None,
        model_args,
        "vfm_lattice_prior_log_length_loc",
        (1.5893, 1.7169, 1.9784),
    )
    vfm_lattice_prior_log_length_scale = _cfg_triplet(
        None,
        model_args,
        "vfm_lattice_prior_log_length_scale",
        (0.2510, 0.2696, 0.3562),
    )
    vfm_lattice_prior_angle_min_deg = float(
        _cfg_value(None, model_args, "vfm_lattice_prior_angle_min_deg", 60.0)
    )
    vfm_lattice_prior_angle_max_deg = float(
        _cfg_value(None, model_args, "vfm_lattice_prior_angle_max_deg", 120.0)
    )
    catflow_predict_fn = _catflow_predict_fn_for_checkpoint(model_args)
    autocast_dtype = torch.bfloat16 if bf16 else None
    generator = torch.Generator(device=torch_device).manual_seed(resolved_seed)

    sample_items: list[dict] = []
    with torch.no_grad():
        for start in tqdm(
            range(0, int(num_generate), max(1, int(sample_chunk_size))),
            desc="synthetic/generate",
            dynamic_ncols=True,
        ):
            bsz = min(max(1, int(sample_chunk_size)), int(num_generate) - start)
            num_atoms = _sample_num_atoms(
                bsz=bsz,
                nmax=resolved_nmax,
                strategy=resolved_atom_count_strategy,
                fixed_num_atoms=fixed_num_atoms,
                count_probs=count_probs,
                device=torch_device,
                generator=generator,
            )
            arange = torch.arange(resolved_nmax, device=torch_device)[None, :]
            pad_mask = arange >= num_atoms[:, None]
            real_mask = ~pad_mask
            if sampler_name == "edm":
                samples = edm_sampler(
                    model=model,
                    pad_mask=pad_mask,
                    type_dim=type_encoding.type_dim,
                    num_steps=resolved_steps,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    rho=rho,
                    S_churn=S_churn,
                    S_min=S_min,
                    S_max=S_max,
                    S_noise=S_noise,
                    sigma_data_type=sigma_data_type,
                    sigma_data_coord=sigma_data_coord,
                    sigma_data_lat=sigma_data_lattice,
                    generator=generator,
                    autocast_dtype=autocast_dtype,
                    fixed_atom_types=None,
                    skip_type_scaling=False,
                    aa_frac_max_scale=aa_frac_max_scale,
                    aa_rho_types=aa_rho_types,
                    aa_rho_coords=aa_rho_coords,
                    aa_rho_lattice=aa_rho_lattice,
                    lattice_repr=lattice_repr,
                )
            else:
                raise RuntimeError(
                    "Only 'edm' sampling is supported on this branch; the vfm and "
                    f"catflow samplers were removed. Got sampler_name={sampler_name!r}."
                )
            pad_mask_cpu = pad_mask.to("cpu")
            real_mask_cpu = real_mask.to("cpu")
            atom_idx = type_encoding.decode_logits_to_A0(
                type_logits=samples["type"].detach().cpu(),
                pad_mask=pad_mask_cpu,
                allowed_mask=train_allowed_mask,
            )
            atom_idx = torch.where(real_mask_cpu, atom_idx, torch.zeros_like(atom_idx))
            frac_coords = mod1(samples["frac"].detach().cpu() + 0.5).clamp(0.0, 1.0)
            frac_coords = torch.where(
                real_mask_cpu[..., None], frac_coords, torch.zeros_like(frac_coords)
            )
            lattice_latent = _clamp_lattice_latent(
                samples["lat"].detach().cpu(), lattice_repr=lattice_repr
            )
            lattice = lattice_latent_to_y1(lattice_latent, lattice_repr=lattice_repr)
            lattice = _clamp_lattice_latent(lattice, lattice_repr="y1")
            for i in range(bsz):
                sample_items.append(
                    {
                        "A0": atom_idx[i],
                        "F1": frac_coords[i],
                        "Y1": lattice[i],
                        "pad_mask": pad_mask_cpu[i],
                    }
                )
    return sample_items


def make_synthetic_dataset(
    *,
    input_dir: str | os.PathLike | None = None,
    output_dir: str | os.PathLike,
    real_train_path: str | os.PathLike | None = None,
    generation_data_root: str | os.PathLike | None = None,
    reference_data_root: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    checkpoint: str | os.PathLike | None = None,
    train_output_dir: str | os.PathLike | None = None,
    checkpoint_preference: str = "auto",
    num_generate: int | None = None,
    sample_chunk_size: int = 256,
    sample_seed: int | None = None,
    sample_num_steps: int | None = None,
    sample_device: str = "cuda",
    sample_mode: str = "ema",
    sampler: str | None = None,
    atom_count_strategy: str | None = None,
    fixed_num_atoms: int | None = None,
    bf16: bool = False,
    max_samples: int | None = None,
    dedup_mode: str = "formula",
    filter_level: str = "valid",
    nmax: int = DEFAULT_NMAX,
    min_distance: float = 0.5,
    max_abs_energy_per_atom: float = 100.0,
    max_volume_change: float = 5.0,
    ehull_metastable_thresh: float = 0.1,
    thermo_mlip: str | None = None,
    thermo_ppd_mp: str | os.PathLike | None = None,
    thermo_stability_device: str = "cuda",
    thermo_ehull_method: str = "mp2020_like",
    thermo_relax_steps: int = 200,
    thermo_stability_batch: int = 32,
    nequip_compile_path: str | None = None,
    nequip_relax_mode: str = "batch",
    nequip_optimizer: str = "FIRE",
    nequip_cell_filter: str = "frechet",
    nequip_fmax: float = 0.01,
    nequip_max_force_abort: float = 1e6,
    equiformer_v3_inner_python: str | None = None,
    equiformer_v3_wrapper: str | None = None,
    equiformer_v3_checkpoint: str | None = None,
    equiformer_v3_max_steps: int = 500,
    equiformer_v3_fmax: float = 0.02,
    equiformer_v3_cell_filter: str = "frechet",
    equiformer_v3_optimizer: str = "FIRE",
    equiformer_v3_device: str = "auto",
    write_cifs: bool = True,
    row_oracle=None,
) -> dict:
    if dedup_mode not in DEDUP_MODES:
        raise ValueError(f"Unknown dedup_mode={dedup_mode!r}")
    if filter_level not in FILTER_LEVELS:
        raise ValueError(f"Unknown filter_level={filter_level!r}")

    if input_dir is None and checkpoint is None:
        raise ValueError("Provide either input_dir or checkpoint.")
    output_path = Path(output_dir)
    structures_dir = output_path / "structures"
    output_path.mkdir(parents=True, exist_ok=True)
    if write_cifs:
        structures_dir.mkdir(parents=True, exist_ok=True)

    generation_root = generation_data_root or real_train_path
    novelty_reference_root = reference_data_root or real_train_path
    real_formulas = _read_real_train_formulas(novelty_reference_root)
    real_structures_by_formula: dict[str, list[Structure]] = {}
    if dedup_mode == "structure":
        for struct in _read_real_train_structures(novelty_reference_root):
            real_structures_by_formula.setdefault(
                _canonical_formula_key(struct.composition.reduced_formula), []
            ).append(struct)
    seen_formulas: set[str] = set()
    seen_structures_by_formula: dict[str, list[Structure]] = {}
    matcher = StructureMatcher() if dedup_mode == "structure" else None
    oracle = row_oracle or _build_row_oracle(
        thermo_mlip=thermo_mlip,
        thermo_ppd_mp=thermo_ppd_mp,
        thermo_stability_device=thermo_stability_device,
        thermo_ehull_method=thermo_ehull_method,
        thermo_relax_steps=thermo_relax_steps,
        thermo_stability_batch=thermo_stability_batch,
        nequip_compile_path=nequip_compile_path,
        nequip_relax_mode=nequip_relax_mode,
        nequip_optimizer=nequip_optimizer,
        nequip_cell_filter=nequip_cell_filter,
        nequip_fmax=nequip_fmax,
        nequip_max_force_abort=nequip_max_force_abort,
        equiformer_v3_inner_python=equiformer_v3_inner_python,
        equiformer_v3_wrapper=equiformer_v3_wrapper,
        equiformer_v3_checkpoint=equiformer_v3_checkpoint,
        equiformer_v3_max_steps=equiformer_v3_max_steps,
        equiformer_v3_fmax=equiformer_v3_fmax,
        equiformer_v3_cell_filter=equiformer_v3_cell_filter,
        equiformer_v3_optimizer=equiformer_v3_optimizer,
        equiformer_v3_device=equiformer_v3_device,
    )
    if _filter_requires_relax(filter_level) and oracle is None:
        raise ValueError(
            f"filter_level={filter_level!r} requires row_oracle or --thermo_mlip."
        )

    summary = {
        "num_input": 0,
        "num_parse_failed": 0,
        "num_invalid_geometry": 0,
        "num_prescreen_kept": 0,
        "num_relax_attempted": 0,
        "num_relax_failed": 0,
        "num_scoring_failed": 0,
        "num_unstable": 0,
        "num_dedup_within_synthetic": 0,
        "num_dedup_against_train": 0,
        "num_kept": 0,
        "filter_level": filter_level,
        "dedup_mode": dedup_mode,
        "nmax": int(nmax),
        "generation_data_root": str(generation_root) if generation_root else None,
        "reference_data_root": [
            str(path) for path in _path_roots(novelty_reference_root)
        ],
        "thermo_ppd_mp": str(thermo_ppd_mp) if thermo_ppd_mp is not None else None,
    }

    metadata_path = output_path / "metadata.jsonl"
    rejected_path = output_path / "rejected.jsonl"
    raw_dir = output_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_csv_path = raw_dir / "train.csv"
    _CSV_FIELDS = ["material_id", "cif", "formation_energy_per_atom", "e_above_hull"]
    with metadata_path.open("w", encoding="utf-8") as meta_f, rejected_path.open(
        "w", encoding="utf-8"
    ) as rej_f, train_csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.DictWriter(csv_f, fieldnames=_CSV_FIELDS)
        csv_writer.writeheader()
        pending_relax: list[tuple[str, Structure, dict, str]] = []

        def _write_reject(meta: dict) -> None:
            rej_f.write(json.dumps(meta, default=_json_default) + "\n")

        def _finalize_candidate(
            *,
            source: str,
            structure: Structure,
            meta: dict,
            sample_id: str,
            oracle_result: dict | None = None,
        ) -> None:
            final_structure = structure
            if oracle_result is not None:
                result = oracle_result
                final_structure = result.get("final_structure")
                err = result.get("err", "") or ""
                meta["relax_success"] = (
                    final_structure is not None and not err.startswith("relax_exc")
                )
                meta["energy_per_atom"] = (
                    float(result["e_total"]) / float(len(final_structure))
                    if final_structure is not None
                    and result.get("e_total") is not None
                    and math.isfinite(float(result.get("e_total")))
                    and len(final_structure) > 0
                    else None
                )
                meta["formation_energy_per_atom"] = result.get("e_form")
                meta["e_above_hull"] = result.get("e_above_hull")
                meta["relax_nsteps"] = result.get("nsteps")
                meta["relax_error"] = err or None
                meta["space_group_post"] = _space_group_number(final_structure)
                if final_structure is None:
                    meta.update(
                        filter_status="rejected",
                        filter_reason=err or "relax_failed",
                    )
                    summary["num_relax_failed"] += 1
                    _write_reject(meta)
                    return
                e_above = meta.get("e_above_hull")
                has_e_above = e_above is not None and math.isfinite(float(e_above))
                if not has_e_above:
                    summary["num_scoring_failed"] += 1
                    if _filter_requires_stability(filter_level):
                        meta.update(
                            filter_status="rejected",
                            filter_reason=err or "scoring_failed",
                        )
                        _write_reject(meta)
                        return
                else:
                    e_above_f = float(e_above)
                    meta["is_stable"] = bool(e_above_f <= 0.0)
                    meta["is_metastable"] = bool(
                        e_above_f <= float(ehull_metastable_thresh)
                    )
                    if meta["is_stable"]:
                        meta["stability_label"] = "stable"
                    elif meta["is_metastable"]:
                        meta["stability_label"] = "metastable"
                    else:
                        meta["stability_label"] = "unstable"
                if (
                    _filter_requires_stability(filter_level)
                    and filter_level == "stable_like"
                    and not meta["is_stable"]
                ):
                    meta.update(
                        filter_status="rejected",
                        filter_reason="not_stable",
                    )
                    summary["num_unstable"] += 1
                    _write_reject(meta)
                    return
                if (
                    _filter_requires_stability(filter_level)
                    and filter_level != "stable_like"
                    and not meta["is_metastable"]
                ):
                    meta.update(
                        filter_status="rejected",
                        filter_reason="not_stable_or_metastable",
                    )
                    summary["num_unstable"] += 1
                    _write_reject(meta)
                    return
                if meta["relax_success"] is True:
                    meta["volume_change"] = (
                        float(final_structure.volume) - float(structure.volume)
                    ) / max(float(structure.volume), 1.0e-12)

            final_formula = final_structure.composition.reduced_formula
            final_formula_key = _canonical_formula_key(final_formula)
            meta["formula"] = final_formula
            meta["num_atoms"] = int(len(final_structure))
            meta["dedup_key"] = (
                final_formula_key if dedup_mode in ("formula", "structure") else None
            )
            if dedup_mode == "formula":
                duplicate_within = final_formula_key in seen_formulas
                duplicate_against_train = final_formula_key in real_formulas
            elif dedup_mode == "structure":
                duplicate_within = any(
                    matcher.fit(final_structure, s)
                    for s in seen_structures_by_formula.get(final_formula_key, [])
                )
                real_candidates = real_structures_by_formula.get(final_formula_key, [])
                duplicate_against_train = (
                    any(matcher.fit(final_structure, s) for s in real_candidates)
                    if real_candidates
                    else final_formula_key in real_formulas
                )
            else:
                duplicate_within = False
                duplicate_against_train = False
            meta["dedup_within_synthetic"] = bool(duplicate_within)
            meta["dedup_against_train"] = bool(duplicate_against_train)
            meta["is_unique"] = not bool(duplicate_within)
            meta["is_novel"] = not bool(duplicate_against_train)
            meta["is_msun"] = (
                bool(meta.get("is_metastable"))
                and bool(meta["is_unique"])
                and bool(meta["is_novel"])
            )

            if duplicate_within:
                summary["num_dedup_within_synthetic"] += 1
                meta["filter_status"] = "rejected"
                meta["filter_reason"] = "dedup_within_synthetic"
            elif duplicate_against_train:
                summary["num_dedup_against_train"] += 1
                meta["filter_status"] = "rejected"
                meta["filter_reason"] = "dedup_against_train"
            else:
                keep, reason = _passes_filter_level(
                    meta,
                    final_structure,
                    filter_level=filter_level,
                    nmax=nmax,
                    max_atoms=nmax,
                    min_distance=min_distance,
                    max_abs_energy_per_atom=max_abs_energy_per_atom,
                    max_volume_change=max_volume_change,
                )
                if not keep:
                    meta["filter_status"] = "rejected"
                    meta["filter_reason"] = reason
                    if reason == "relax_not_successful":
                        summary["num_relax_failed"] += 1
                    elif reason in {"not_stable", "not_stable_or_metastable"}:
                        summary["num_unstable"] += 1
                    else:
                        summary["num_invalid_geometry"] += 1

            if meta["filter_status"] == "kept":
                cif_str = str(CifWriter(final_structure))
                # CIF round-trip validation. Some NequIP-relaxed structures land
                # with near-singular cells that pymatgen's CIF parser can't process
                # (it calls lattice.d_hkl which inverts the cell matrix). Catch
                # those at curation time so downstream MP20Tokens preprocess never
                # sees an unparseable row.
                try:
                    rt = Structure.from_str(cif_str, fmt="cif")
                    rt.lattice.d_hkl((1, 0, 0))
                except Exception as exc:
                    meta["filter_status"] = "cif_roundtrip_failed"
                    meta["cif_roundtrip_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    summary.setdefault("num_cif_roundtrip_failed", 0)
                    summary["num_cif_roundtrip_failed"] += 1
                    _write_reject(meta)
                    return
                seen_formulas.add(final_formula_key)
                if dedup_mode == "structure":
                    seen_structures_by_formula.setdefault(final_formula_key, []).append(
                        final_structure
                    )
                if write_cifs:
                    cif_name = f"{sample_id}.cif"
                    cif_path = structures_dir / cif_name
                    cif_path.write_text(cif_str, encoding="utf-8")
                    meta["post_relax_path"] = str(cif_path)
                # Emit MP20Tokens-compatible raw/train.csv row. Tokenization (Niggli
                # reduction + canonical lattice orientation) is done at load time by
                # MP20Tokens → preprocess() → build_crystal(niggli=True), matching MP20.
                csv_writer.writerow(
                    {
                        "material_id": sample_id,
                        "cif": cif_str,
                        "formation_energy_per_atom": meta.get(
                            "formation_energy_per_atom"
                        ),
                        "e_above_hull": meta.get("e_above_hull"),
                    }
                )
                summary["num_kept"] += 1
                meta_f.write(json.dumps(meta, default=_json_default) + "\n")
            else:
                _write_reject(meta)

        def _flush_pending_relax() -> None:
            if not pending_relax:
                return
            batch = list(pending_relax)
            pending_relax.clear()
            structs = [entry[1] for entry in batch]
            summary["num_relax_attempted"] += len(structs)
            results = oracle.call_many(structs)
            if len(results) != len(batch):
                raise RuntimeError(
                    "row oracle returned mismatched result count: "
                    f"{len(results)} for {len(batch)} structures"
                )
            for (source, structure, meta, sample_id), result in zip(
                batch, results, strict=True
            ):
                _finalize_candidate(
                    source=source,
                    structure=structure,
                    meta=meta,
                    sample_id=sample_id,
                    oracle_result=result,
                )
                if max_samples is not None and summary["num_kept"] >= int(max_samples):
                    break

        raw_iterables: list[Iterable[tuple[str, Structure | None, dict]]] = []
        if input_dir is not None:
            raw_iterables.append(_iter_input_structures(Path(input_dir)))
        if checkpoint is not None:
            generated_items = generate_synthetic_samples_from_checkpoint(
                checkpoint=checkpoint,
                train_output_dir=train_output_dir,
                checkpoint_preference=checkpoint_preference,
                data_root=generation_root,
                nmax=nmax,
                num_generate=int(num_generate or max_samples or 0),
                sample_chunk_size=sample_chunk_size,
                sample_seed=sample_seed,
                sample_num_steps=sample_num_steps,
                device=sample_device,
                sample_mode=sample_mode,
                sampler=sampler,
                atom_count_strategy=atom_count_strategy,
                fixed_num_atoms=fixed_num_atoms,
                bf16=bf16,
            )

            def _generated_iter():
                for idx, rec in enumerate(generated_items):
                    source = f"{checkpoint}:generated:{idx}"
                    meta = _metadata_template("", source)
                    meta["generator_checkpoint"] = str(checkpoint)
                    meta["sampling_config"] = {
                        "num_generate": int(num_generate or len(generated_items)),
                        "sample_seed": sample_seed,
                        "sample_num_steps": sample_num_steps,
                        "sampler": sampler,
                        "sample_mode": sample_mode,
                    }
                    try:
                        yield source, tokens_to_structure(rec), meta
                    except Exception as exc:
                        meta.update(
                            valid_parse=False,
                            filter_status="rejected",
                            filter_reason=f"parse_failed:{type(exc).__name__}",
                        )
                        yield source, None, meta

            raw_iterables.append(_generated_iter())

        for raw_iterable in raw_iterables:
            for source, structure, meta in raw_iterable:
                summary["num_input"] += 1
                sample_id = f"synthetic_{summary['num_input']:06d}"
                meta["sample_id"] = sample_id
                meta["pre_relax_path"] = meta.get("pre_relax_path") or source
                if structure is None:
                    summary["num_parse_failed"] += 1
                    rej_f.write(json.dumps(meta, default=_json_default) + "\n")
                    continue
                try:
                    formula = structure.composition.reduced_formula
                    formula_key = _canonical_formula_key(formula)
                    meta["formula"] = formula
                    meta["num_atoms"] = int(len(structure))
                    meta["dedup_key"] = (
                        formula_key if dedup_mode in ("formula", "structure") else None
                    )
                except Exception as exc:
                    summary["num_parse_failed"] += 1
                    meta.update(
                        valid_parse=False,
                        filter_status="rejected",
                        filter_reason=f"parse_failed:{type(exc).__name__}",
                    )
                    rej_f.write(json.dumps(meta, default=_json_default) + "\n")
                    continue

                meta["space_group_pre"] = _space_group_number(structure)
                ok, reason = _validate_structure(
                    structure,
                    nmax=nmax,
                    max_atoms=nmax,
                    min_distance=min_distance,
                )
                if filter_level != "raw" and not ok:
                    meta.update(
                        valid_geometry=False,
                        filter_status="rejected",
                        filter_reason=reason,
                    )
                    summary["num_invalid_geometry"] += 1
                    rej_f.write(json.dumps(meta, default=_json_default) + "\n")
                    continue
                summary["num_prescreen_kept"] += 1

                if _filter_requires_relax(filter_level) and oracle is not None:
                    pending_relax.append((source, structure, meta, sample_id))
                    if len(pending_relax) >= max(1, int(thermo_stability_batch)):
                        _flush_pending_relax()
                else:
                    _finalize_candidate(
                        source=source,
                        structure=structure,
                        meta=meta,
                        sample_id=sample_id,
                    )
                if max_samples is not None and summary["num_kept"] >= int(max_samples):
                    break
            if pending_relax and (
                max_samples is None or summary["num_kept"] < int(max_samples)
            ):
                _flush_pending_relax()
            if max_samples is not None and summary["num_kept"] >= int(max_samples):
                break

    summary["synthetic_metadata_path"] = str(metadata_path)
    summary["synthetic_train_csv_path"] = str(train_csv_path)
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if max_samples is not None and summary["num_kept"] < int(max_samples):
        raise RuntimeError(
            f"make_synthetic_dataset undershot max_samples={int(max_samples)}: "
            f"only {summary['num_kept']} kept after generating "
            f"{summary.get('num_relax_attempted', '?')} candidates "
            f"(yield "
            f"{summary['num_kept']/max(1, summary.get('num_relax_attempted', 1))*100:.1f}%). "
            "Increase num_generate (current candidate pool was too small) or lower "
            "max_samples. Partial dataset has been written but is short of target."
        )
    return summary


class AugmentedCrystalDataset(Dataset):
    """Dataset wrapper for real-only, synthetic-concat, and oversampled-real runs."""

    def __init__(
        self,
        real_dataset: Dataset,
        synthetic_dataset: Dataset | None = None,
        *,
        augmentation_mode: str = "none",
        num_extra_samples: int | None = None,
        synthetic_ratio: float | None = None,
        seed: int = 0,
    ):
        if augmentation_mode not in ("none", "synthetic_concat", "oversample_real"):
            raise ValueError(f"Unknown augmentation_mode={augmentation_mode!r}")
        if synthetic_ratio is not None:
            raise NotImplementedError("synthetic_ratio sampling is not implemented yet.")
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.augmentation_mode = augmentation_mode
        self.seed = int(seed)
        self.synthetic_ratio = synthetic_ratio
        if augmentation_mode == "synthetic_concat":
            if synthetic_dataset is None:
                raise ValueError("synthetic_concat requires synthetic_dataset.")
            self.num_extra_samples = len(synthetic_dataset)
        elif augmentation_mode == "oversample_real":
            if num_extra_samples is None or int(num_extra_samples) < 0:
                raise ValueError("oversample_real requires nonnegative num_extra_samples.")
            self.num_extra_samples = int(num_extra_samples)
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            self._oversample_indices = torch.randint(
                low=0,
                high=len(real_dataset),
                size=(self.num_extra_samples,),
                generator=gen,
            ).tolist()
        else:
            self.num_extra_samples = 0
            self._oversample_indices = []
        self.items = self._build_items_view()

    def _build_items_view(self) -> list[dict]:
        real_items = list(getattr(self.real_dataset, "items", []))
        if len(real_items) != len(self.real_dataset):
            real_items = [self.real_dataset[i] for i in range(len(self.real_dataset))]
        if self.augmentation_mode == "synthetic_concat":
            assert self.synthetic_dataset is not None
            synthetic_items = list(getattr(self.synthetic_dataset, "items", []))
            if len(synthetic_items) != len(self.synthetic_dataset):
                synthetic_items = [
                    self.synthetic_dataset[i] for i in range(len(self.synthetic_dataset))
                ]
            return real_items + synthetic_items
        if self.augmentation_mode == "oversample_real":
            return real_items + [real_items[i] for i in self._oversample_indices]
        return real_items

    def __len__(self) -> int:
        return len(self.real_dataset) + int(self.num_extra_samples)

    def __getitem__(self, idx: int) -> dict:
        real_len = len(self.real_dataset)
        if idx < real_len:
            return self.real_dataset[idx]
        extra_idx = idx - real_len
        if self.augmentation_mode == "synthetic_concat":
            assert self.synthetic_dataset is not None
            return self.synthetic_dataset[extra_idx]
        if self.augmentation_mode == "oversample_real":
            return self.real_dataset[self._oversample_indices[extra_idx]]
        raise IndexError(idx)


def build_augmented_train_dataset(
    real_dataset: Dataset,
    *,
    augmentation_mode: str = "none",
    synthetic_data: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    num_extra_samples: int | None = None,
    seed: int = 0,
    nmax: int = DEFAULT_NMAX,
) -> tuple[Dataset, dict]:
    synthetic_dataset = None
    synthetic_roots: list[str | os.PathLike] = []
    if augmentation_mode == "synthetic_concat":
        if synthetic_data is None:
            raise ValueError("--synthetic_data is required for synthetic_concat.")
        synthetic_roots = _synthetic_data_roots(synthetic_data)
        if not synthetic_roots:
            raise ValueError("--synthetic_data did not resolve to any dataset roots.")
        synthetic_datasets = [
            MP20Tokens(
                root=str(root),
                split="train",
                nmax=nmax,
                augment_translate=True,
            )
            for root in synthetic_roots
        ]
        # Track metadata.jsonl paths per root for wandb logging (MP20Tokens itself
        # doesn't carry this; it lives in the curation directory alongside raw/).
        synthetic_metadata_paths_list = [
            str(Path(root) / "metadata.jsonl")
            for root in synthetic_roots
            if (Path(root) / "metadata.jsonl").exists()
        ]
        if len(synthetic_datasets) == 1:
            synthetic_dataset = synthetic_datasets[0]
            synthetic_dataset.metadata_paths = synthetic_metadata_paths_list
        else:
            synthetic_dataset = torch.utils.data.ConcatDataset(synthetic_datasets)
            synthetic_dataset.items = [
                item for ds in synthetic_datasets for item in getattr(ds, "items", [])
            ]
            synthetic_dataset.metadata_paths = synthetic_metadata_paths_list
        if num_extra_samples is not None and int(num_extra_samples) != len(synthetic_dataset):
            raise ValueError(
                "--num_extra_samples should be omitted or match synthetic dataset length "
                f"for synthetic_concat ({len(synthetic_dataset)})."
            )

    ds = AugmentedCrystalDataset(
        real_dataset,
        synthetic_dataset,
        augmentation_mode=augmentation_mode,
        num_extra_samples=num_extra_samples,
        seed=seed,
    )
    synthetic_metadata_paths = None
    if synthetic_dataset is not None:
        synthetic_metadata_paths = getattr(synthetic_dataset, "metadata_paths", None)
        if synthetic_metadata_paths is None and getattr(
            synthetic_dataset, "metadata_path", Path()
        ).exists():
            synthetic_metadata_paths = [str(synthetic_dataset.metadata_path)]

    composition = {
        "real_train_count": len(real_dataset),
        "synthetic_train_count": len(synthetic_dataset) if synthetic_dataset is not None else 0,
        "synthetic_dataset_count": len(synthetic_roots),
        "oversampled_real_count": int(num_extra_samples or 0)
        if augmentation_mode == "oversample_real"
        else 0,
        "augmentation_mode": augmentation_mode,
        "effective_train_count": len(ds),
        "synthetic_metadata_path": (
            synthetic_metadata_paths[0]
            if synthetic_metadata_paths and len(synthetic_metadata_paths) == 1
            else (os.pathsep.join(synthetic_metadata_paths) if synthetic_metadata_paths else None)
        ),
        "synthetic_metadata_paths": synthetic_metadata_paths,
    }
    return ds, composition
