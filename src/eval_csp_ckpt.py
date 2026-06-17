from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

# Ensure repository root is on PYTHONPATH when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.mp20_tokens import MP20Tokens, VZ, tokens_to_structure
from src.data.type_encoding import build_type_encoding
from src.eval.csp_eval import RecEval, RecEvalBatch
from src.eval.sample_runtime import (
    SamplingContext,
    SamplingRequest,
    _generate_csp_items_for_indices,
    _safe_crystal_from_tokens,
    run_sampling_request,
)
from src.eval.stability import _compute_thermo_metrics
from src.eval_crystalite_ckpt import (
    _apply_ema_state_dict,
    _build_model_from_ckpt,
    _cfg_value,
    _load_checkpoint,
    _print_final_metrics,
    _resolve_checkpoint_path,
    _seed_everything,
)
from src.training.config import _compute_topk_target_count, _normalize_topk_list
from src.utils.dataset import ensure_dataset_splits
from src.utils.stability_logger import StabilityLogger, _ThermoConfig


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    out = float(value)
    return out


def _resolve_optional_str(*values: Any, default: str) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "none":
            continue
        return text
    return default


def _require_csp_checkpoint(model_args: Mapping[str, Any]) -> None:
    if not bool(model_args.get("csp", False)):
        raise ValueError("eval_csp_ckpt.py requires a checkpoint trained with --csp.")


def _resolve_source_split(
    *,
    data_root: str,
    dataset_name: str,
    requested_split: str,
) -> str:
    split = str(requested_split).strip().lower()
    if split == "all":
        return "all"

    has_split = ensure_dataset_splits(data_root, dataset_name)
    if split == "auto":
        return "val" if has_split else "all"

    if not has_split:
        raise FileNotFoundError(
            f"Dataset at {data_root} does not provide train/val splits; "
            f"cannot use --source_split={split}."
        )

    if split == "test":
        test_csv = Path(data_root) / "raw" / "test.csv"
        if not test_csv.exists():
            raise FileNotFoundError(
                f"Requested --source_split=test but missing {test_csv}."
            )
    return split


def _resolve_report_dir(report_dir: str, ckpt_path: Path) -> Path:
    if report_dir:
        return Path(report_dir)
    if ckpt_path.parent.name == "checkpoints" and ckpt_path.parent.parent.exists():
        return ckpt_path.parent.parent / "csp_eval_reports"
    return ckpt_path.parent / "csp_eval_reports"


def _build_thermo_logger(
    args: argparse.Namespace,
    model_args: Mapping[str, Any],
) -> StabilityLogger:
    thermo_cfg = _ThermoConfig(
        batch_size=int(
            _cfg_value(
                args.thermo_stability_batch,
                model_args,
                "thermo_stability_batch",
                32,
            )
        ),
        relax_steps=int(
            _cfg_value(args.thermo_relax_steps, model_args, "thermo_relax_steps", 200)
        ),
        ppd_path=str(
            _cfg_value(
                args.thermo_ppd_mp,
                model_args,
                "thermo_ppd_mp",
                "data/mp20/hull/2023-02-07-ppd-mp.pkl",
            )
        ),
        device=str(
            _cfg_value(
                args.thermo_stability_device,
                model_args,
                "thermo_stability_device",
                "cuda",
            )
        ),
        ehull_method=str(
            _cfg_value(
                args.thermo_ehull_method,
                model_args,
                "thermo_ehull_method",
                "uncorrected",
            )
        ),
        mlip=str(_cfg_value(args.thermo_mlip, model_args, "thermo_mlip", "chgnet")),
        nequip_compile_path=str(
            _cfg_value(
                args.nequip_compile_path,
                model_args,
                "nequip_compile_path",
                "",
            )
        ),
        nequip_optimizer=str(
            _cfg_value(args.nequip_optimizer, model_args, "nequip_optimizer", "FIRE")
        ),
        nequip_cell_filter=str(
            _cfg_value(args.nequip_cell_filter, model_args, "nequip_cell_filter", "none")
        ),
        nequip_fmax=float(
            _cfg_value(args.nequip_fmax, model_args, "nequip_fmax", 0.01)
        ),
        nequip_max_force_abort=float(
            _cfg_value(
                args.nequip_max_force_abort,
                model_args,
                "nequip_max_force_abort",
                1e6,
            )
        ),
        nequip_relax_mode=str(
            _cfg_value(
                args.nequip_relax_mode,
                model_args,
                "nequip_relax_mode",
                "sequential",
            )
        ),
    )
    return StabilityLogger(gamma_cfg=None, thermo_cfg=thermo_cfg)


def _configure_sampling_args(
    args: argparse.Namespace,
    model_args: Mapping[str, Any],
) -> None:
    args.sample_chunk_size = int(
        _cfg_value(args.sample_chunk_size, model_args, "sample_chunk_size", 256)
    )
    args.sample_num_steps = int(
        _cfg_value(args.sample_num_steps, model_args, "sample_num_steps", 100)
    )
    args.sample_temperature = float(
        _cfg_value(args.sample_temperature, model_args, "sample_temperature", 1.0)
    )
    args.sigma_min = float(_cfg_value(args.sigma_min, model_args, "sigma_min", 0.002))
    args.sigma_max = float(_cfg_value(args.sigma_max, model_args, "sigma_max", 80.0))
    args.rho = float(_cfg_value(args.rho, model_args, "rho", 7.0))
    args.S_churn = float(_cfg_value(args.S_churn, model_args, "S_churn", 20.0))
    args.S_min = float(_cfg_value(args.S_min, model_args, "S_min", 0.0))
    args.S_max = float(_cfg_value(args.S_max, model_args, "S_max", 999.0))
    args.S_noise = float(_cfg_value(args.S_noise, model_args, "S_noise", 1.0))
    args.sigma_data_type = float(
        _cfg_value(args.sigma_data_type, model_args, "sigma_data_type", 1.0)
    )
    args.sigma_data_coord = float(
        _cfg_value(args.sigma_data_coord, model_args, "sigma_data_coord", 0.25)
    )
    args.sigma_data_lattice = float(
        _cfg_value(args.sigma_data_lattice, model_args, "sigma_data_lattice", 1.0)
    )
    args.aa_frac_max_scale = float(
        _cfg_value(args.aa_frac_max_scale, model_args, "aa_frac_max_scale", 0.0)
    )
    args.aa_rho_types = float(
        _cfg_value(args.aa_rho_types, model_args, "aa_rho_types", 0.0)
    )
    args.aa_rho_coords = float(
        _cfg_value(args.aa_rho_coords, model_args, "aa_rho_coords", 0.0)
    )
    args.aa_rho_lattice = float(
        _cfg_value(args.aa_rho_lattice, model_args, "aa_rho_lattice", 0.0)
    )
    args.sampler = str(_cfg_value(getattr(args, "sampler", None), model_args, "sampler", "edm"))
    args.lattice_repr = str(_cfg_value(None, model_args, "lattice_repr", "y1"))
    args.atom_count_strategy = "fixed"
    args.discrete_types = False
    args.sample_vis_count = 0


def _compute_reconstruction_metrics(
    *,
    sample_items: list[dict[str, Any]],
    csp_indices: torch.Tensor | None,
    source_ds: Any,
    source_label: str,
    ctx: SamplingContext,
    base_seed: int,
    topk_list: list[int],
    topk_samples: int,
) -> dict[str, float]:
    metrics_out: dict[str, float] = {}
    if not sample_items or csp_indices is None:
        return metrics_out

    pred_crys = []
    gt_crys = []
    for i, item in enumerate(sample_items):
        gt_item = source_ds[csp_indices[i].item()]
        pred_crys.append(_safe_crystal_from_tokens(item, i))
        gt_crys.append(_safe_crystal_from_tokens(gt_item, i))

    rec_eval = RecEval(pred_crys, gt_crys)
    rec_metrics = rec_eval.get_metrics()
    metrics_out[f"csp_{source_label}_match_rate"] = _float_or_nan(
        rec_metrics["match_rate"]
    )
    metrics_out[f"csp_{source_label}_mean_rms"] = _float_or_nan(
        rec_metrics["rms_dist"]
    )

    if not topk_list:
        return metrics_out

    topk_target_count = _compute_topk_target_count(len(sample_items), topk_samples)
    if topk_target_count <= 0:
        print(
            f"[csp-topk] Skipping {source_label}: "
            f"--topk_samples={topk_samples} left zero available targets."
        )
        return metrics_out

    if topk_target_count < topk_samples:
        print(
            f"[csp-topk] Truncating {source_label} targets "
            f"from requested {topk_samples} to {topk_target_count} "
            "due to available samples."
        )

    target_indices = csp_indices[:topk_target_count]
    gt_topk = [
        _safe_crystal_from_tokens(source_ds[target_indices[i].item()], i)
        for i in range(topk_target_count)
    ]
    pred_topk_batches = [
        [_safe_crystal_from_tokens(sample_items[i], i) for i in range(topk_target_count)]
    ]
    chunk_size = max(1, int(ctx.args.sample_chunk_size or topk_target_count))
    tag_seed = sum(ord(c) for c in "eval")
    label_seed = sum(ord(c) for c in source_label)

    for candidate_idx in range(1, topk_list[-1]):
        candidate_seed = (
            int(base_seed)
            + tag_seed * 1009
            + label_seed * 9176
            + candidate_idx * 1000003
        )
        candidate_items = _generate_csp_items_for_indices(
            target_indices=target_indices,
            csp_source_ds=source_ds,
            ctx=ctx,
            seed=candidate_seed,
            chunk_size=chunk_size,
        )
        pred_topk_batches.append(
            [_safe_crystal_from_tokens(item, i) for i, item in enumerate(candidate_items)]
        )

    rec_topk = RecEvalBatch(pred_topk_batches, gt_topk)
    for k in topk_list:
        topk_metrics = rec_topk.get_match_rate_and_rms_for_k(k)
        metrics_out[f"csp_{source_label}_top{k}_match_rate"] = _float_or_nan(
            topk_metrics["match_rate"]
        )
        metrics_out[f"csp_{source_label}_top{k}_mean_rms"] = _float_or_nan(
            topk_metrics["rms_dist"]
        )

    return metrics_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline CSP checkpoint evaluator."
    )
    parser.add_argument("--train_output_dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--checkpoint_preference",
        type=str,
        default="auto",
        choices=["auto", "best", "final", "step_latest", "epoch_latest"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument(
        "--source_split",
        type=str,
        default="auto",
        choices=["auto", "train", "val", "test", "all"],
    )
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--sample_chunk_size", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--sample_num_steps", type=int, default=None)
    parser.add_argument("--sample_mode", type=str, default="ema", choices=["ema", "regular"])
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=["edm"],
        help="Sampling backend for CSP generation.",
    )
    parser.add_argument("--sample_temperature", type=float, default=None)
    parser.add_argument("--sigma_min", type=float, default=None)
    parser.add_argument("--sigma_max", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--S_churn", type=float, default=None)
    parser.add_argument("--S_min", type=float, default=None)
    parser.add_argument("--S_max", type=float, default=None)
    parser.add_argument("--S_noise", type=float, default=None)
    parser.add_argument("--sigma_data_type", type=float, default=None)
    parser.add_argument("--sigma_data_coord", type=float, default=None)
    parser.add_argument("--sigma_data_lattice", type=float, default=None)
    parser.add_argument("--aa_frac_max_scale", type=float, default=None)
    parser.add_argument("--aa_rho_types", type=float, default=None)
    parser.add_argument("--aa_rho_coords", type=float, default=None)
    parser.add_argument("--aa_rho_lattice", type=float, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--topk_list", type=int, nargs="*", default=[])
    parser.add_argument("--topk_samples", type=int, default=0)

    parser.add_argument("--report_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--save_samples_pt", action="store_true")

    parser.add_argument("--thermo_count", type=int, default=0)
    parser.add_argument("--thermo_stability_batch", type=int, default=None)
    parser.add_argument("--thermo_relax_steps", type=int, default=None)
    parser.add_argument("--thermo_stability_device", type=str, default=None)
    parser.add_argument("--thermo_mlip", type=str, default=None, choices=["chgnet", "nequip"])
    parser.add_argument(
        "--thermo_ehull_method",
        type=str,
        default=None,
        choices=["uncorrected", "mp2020_like"],
    )
    parser.add_argument("--thermo_ppd_mp", type=str, default=None)
    parser.add_argument("--nequip_compile_path", type=str, default=None)
    parser.add_argument(
        "--nequip_relax_mode",
        type=str,
        default=None,
        choices=["sequential", "batch"],
    )
    parser.add_argument("--nequip_optimizer", type=str, default=None)
    parser.add_argument("--nequip_cell_filter", type=str, default=None)
    parser.add_argument("--nequip_fmax", type=float, default=None)
    parser.add_argument("--nequip_max_force_abort", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)

    ckpt_path = _resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        train_output_dir=args.train_output_dir,
        preference=args.checkpoint_preference,
    )
    ckpt = _load_checkpoint(ckpt_path)
    model, model_args = _build_model_from_ckpt(ckpt=ckpt, device=torch.device("cpu"))
    _require_csp_checkpoint(model_args)

    dataset_name = _resolve_optional_str(
        args.dataset_name,
        model_args.get("metrics_dataset_name"),
        model_args.get("dataset_name"),
        default="mp20",
    )
    data_root = _resolve_optional_str(
        args.data_root,
        model_args.get("metrics_data_root"),
        model_args.get("data_root"),
        default="data/mp20",
    )
    nmax = int(_cfg_value(args.nmax, model_args, "nmax", 20))
    sample_seed = int(_cfg_value(args.sample_seed, model_args, "sample_seed", 123))
    _configure_sampling_args(args, model_args)

    if args.topk_samples < 0:
        raise ValueError("--topk_samples must be >= 0.")
    topk_list = _normalize_topk_list(list(args.topk_list))

    resolved_split = _resolve_source_split(
        data_root=data_root,
        dataset_name=dataset_name,
        requested_split=args.source_split,
    )
    source_ds = MP20Tokens(
        root=data_root,
        augment_translate=False,
        split=resolved_split,
        nmax=nmax,
    )
    if len(source_ds) == 0:
        raise ValueError(
            f"Source dataset split '{resolved_split}' is empty at {data_root}."
        )

    if args.num_samples is None:
        num_samples = len(source_ds)
    else:
        num_samples = int(args.num_samples)
    if num_samples <= 0:
        raise ValueError("--num_samples must be >= 1.")

    if topk_list and args.topk_samples == 0:
        args.topk_samples = min(num_samples, len(source_ds))

    _seed_everything(sample_seed)
    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")
    model = model.to(device)

    if args.sample_mode == "ema":
        ema_state = ckpt.get("ema_state_dict", None)
        if ema_state is not None:
            _apply_ema_state_dict(model, ema_state)
            print("[ckpt] Using EMA weights for sampling.")
        else:
            print("[ckpt] EMA state not found; falling back to regular model weights.")
    else:
        print("[ckpt] Using regular model weights for sampling.")

    type_encoding_name = str(
        ckpt.get("type_encoding", model_args.get("type_encoding", "atomic_number"))
    )
    type_encoding = build_type_encoding(type_encoding_name, vz=VZ)

    metrics_count = max(num_samples, int(args.topk_samples or 0))
    ctx = SamplingContext(
        args=args,
        model=model,
        ema=None,
        device=device,
        nmax=nmax,
        type_encoding=type_encoding,
        count_probs=None,
        train_allowed_mask=None,
        train_element_dist=None,
        ref_stats=None,
        ref_structs=[],
        enable_evaluator_metrics=False,
        novelty_ref_structs=None,
        full_train_novelty_ref_structs=None,
        thermo_logger=None,
        thermo_reference_cache={},
        sample_dir=Path("."),
        ase_view=None,
        wandb_enabled=False,
    )
    request = SamplingRequest(
        tag="eval",
        step=0,
        base_seed=sample_seed,
        use_ema=False,
        metrics_count=metrics_count,
        csp_source_ds=source_ds,
        csp_source_label=resolved_split,
    )

    print(
        f"[data] dataset={dataset_name} split={resolved_split} "
        f"size={len(source_ds)} nmax={nmax}"
    )
    print(
        f"[sample] Starting CSP eval sampling: num_samples={num_samples}, "
        f"chunk_size={max(1, int(args.sample_chunk_size or metrics_count))}, "
        f"num_steps={args.sample_num_steps}, seed={sample_seed}, sampler={args.sampler}"
    )

    batch = run_sampling_request(request, ctx)
    sample_items = batch.sample_items
    print(f"[sample] Finished sampling. generated={len(sample_items)}")

    metrics_out: dict[str, float] = {
        "summary/num_samples_requested": float(num_samples),
        "summary/num_samples_generated": float(len(sample_items)),
        "summary/source_size": float(len(source_ds)),
    }
    metrics_out.update(
        _compute_reconstruction_metrics(
            sample_items=sample_items,
            csp_indices=batch.csp_indices,
            source_ds=source_ds,
            source_label=resolved_split,
            ctx=ctx,
            base_seed=sample_seed,
            topk_list=topk_list,
            topk_samples=int(args.topk_samples),
        )
    )

    if int(args.thermo_count) > 0:
        thermo_count = min(int(args.thermo_count), len(sample_items))
        print(f"[eval] Running thermo metrics on up to {thermo_count} structures.")
        thermo_structs = []
        for item in sample_items[:thermo_count]:
            try:
                thermo_structs.append(tokens_to_structure(item))
            except Exception:
                continue

        if thermo_structs:
            logger = _build_thermo_logger(args, model_args)
            thermo_metrics = _compute_thermo_metrics(
                logger,
                thermo_structs,
                tag="eval",
                step=0,
                enabled=True,
                show_progress=True,
            )
            for key, val in thermo_metrics.items():
                metrics_out[key] = _float_or_nan(val)
        else:
            print("[eval] No constructed structures available for thermo metrics.")

    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = f"csp_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_report_dir = _resolve_report_dir(args.report_dir, ckpt_path)
    report_dir = base_report_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "meta": {
            "run_name": run_name,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_step": int(ckpt.get("step", -1)),
            "checkpoint_is_csp": True,
            "dataset_name": dataset_name,
            "data_root": data_root,
            "source_split_requested": str(args.source_split),
            "source_split": resolved_split,
            "source_size": len(source_ds),
            "nmax": nmax,
            "sample_mode": args.sample_mode,
            "num_samples": num_samples,
            "sample_chunk_size": int(args.sample_chunk_size),
            "sample_num_steps": int(args.sample_num_steps),
            "sample_seed": sample_seed,
            "sampler": str(args.sampler),
            "topk_list": topk_list,
            "topk_samples": int(args.topk_samples),
            "thermo_count": int(args.thermo_count),
        },
        "metrics": {k: float(v) for k, v in metrics_out.items()},
    }
    metrics_json = report_dir / "metrics.json"
    metrics_json.write_text(json.dumps(report, indent=2))

    if args.save_samples_pt:
        samples_pt = report_dir / "samples.pt"
        torch.save(sample_items, samples_pt)
        print(f"[save] Sample tensors: {samples_pt}")
        if batch.csp_indices is not None:
            source_indices_pt = report_dir / "source_indices.pt"
            torch.save(batch.csp_indices, source_indices_pt)
            print(f"[save] Source indices: {source_indices_pt}")

    print(f"[save] Metrics JSON: {metrics_json}")
    _print_final_metrics(report["metrics"])
    return report


if __name__ == "__main__":
    main()
