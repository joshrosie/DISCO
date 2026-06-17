"""Shared inference utilities for loading Crystalite checkpoints.

Used by sampling / ablation / sweep scripts that need to load a trained
model for generation and evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.crystalite import CrystaliteModel
from src.data.mp20_tokens import VZ
from src.data.type_encoding import build_type_encoding


def cfg(model_args: dict, key: str, default: Any) -> Any:
    """Read a model_args key, falling back to *default* when the value is None."""
    v = model_args.get(key)
    return v if v is not None else default


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def apply_ema(model: torch.nn.Module, ema_state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_state:
                param.copy_(ema_state[name].to(device=param.device, dtype=param.dtype))


def build_model(ckpt: dict[str, Any], device: torch.device) -> tuple[CrystaliteModel, dict]:
    """Build a CrystaliteModel from a checkpoint dict and move it to *device*."""
    ma = dict(ckpt.get("model_args", {}))
    type_dim = int(ckpt.get("type_dim", ma.get("type_dim", VZ + 1)))
    model = CrystaliteModel(
        d_model=int(ma.get("d_model", 512)),
        n_heads=int(ma.get("n_heads", 8)),
        n_layers=int(ma.get("n_layers", 18)),
        vz=VZ,
        type_dim=type_dim,
        n_freqs=int(ma.get("coord_n_freqs", ma.get("n_freqs", 32))),
        coord_embed_mode=str(ma.get("coord_embed_mode", "fourier")),
        coord_head_mode=str(ma.get("coord_head_mode", "direct")),
        coord_rff_dim=ma.get("coord_rff_dim", None),
        coord_rff_sigma=float(ma.get("coord_rff_sigma", 1.0)),
        lattice_embed_mode=str(ma.get("lattice_embed_mode", "mlp")),
        lattice_rff_dim=int(ma.get("lattice_rff_dim", 256)),
        lattice_rff_sigma=float(ma.get("lattice_rff_sigma", 5.0)),
        lattice_repr=str(ma.get("lattice_repr", "y1")),
        dropout=float(ma.get("dropout", 0.0)),
        attn_dropout=float(ma.get("attn_dropout", 0.0)),
        use_distance_bias=bool(ma.get("use_distance_bias", False)),
        use_edge_bias=bool(ma.get("use_edge_bias", False)),
        edge_bias_n_freqs=int(ma.get("edge_bias_n_freqs", 8)),
        edge_bias_hidden_dim=int(ma.get("edge_bias_hidden_dim", 128)),
        edge_bias_n_rbf=int(ma.get("edge_bias_n_rbf", 16)),
        edge_bias_rbf_max=float(ma.get("edge_bias_rbf_max", 2.0)),
        pbc_radius=int(ma.get("pbc_radius", 1)),
        dist_slope_init=float(ma.get("dist_slope_init", -1.0)),
        use_noise_gate=bool(ma.get("use_noise_gate", True)),
        gem_per_layer=bool(ma.get("gem_per_layer", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ma


def build_count_distribution(dataset, nmax: int) -> torch.Tensor:
    """Empirical atom-count distribution from a token dataset."""
    counts = torch.zeros(nmax + 1, dtype=torch.float64)
    for i in range(len(dataset)):
        n = int(dataset[i]["num_atoms"])
        if 1 <= n <= nmax:
            counts[n] += 1
    return counts[1:] / counts[1:].sum()


def build_sampler_kwargs(model_args: dict, *, num_steps: int) -> dict[str, Any]:
    """Extract EDM sampler hyperparameters from checkpoint model_args."""
    return dict(
        num_steps=num_steps,
        sigma_min=float(cfg(model_args, "sigma_min", 0.002)),
        sigma_max=float(cfg(model_args, "sigma_max", 80.0)),
        rho=float(cfg(model_args, "rho", 7.0)),
        S_churn=float(cfg(model_args, "S_churn", 20.0)),
        S_min=float(cfg(model_args, "S_min", 0.0)),
        S_max=float(cfg(model_args, "S_max", 999.0)),
        S_noise=float(cfg(model_args, "S_noise", 1.0)),
        sigma_data_type=float(cfg(model_args, "sigma_data_type", 1.0)),
        sigma_data_coord=float(cfg(model_args, "sigma_data_coord", 0.25)),
        sigma_data_lat=float(cfg(model_args, "sigma_data_lattice", 1.0)),
        aa_frac_max_scale=float(cfg(model_args, "aa_frac_max_scale", 0.0)),
        aa_rho_types=float(cfg(model_args, "aa_rho_types", 0.0)),
        aa_rho_coords=float(cfg(model_args, "aa_rho_coords", 0.0)),
        aa_rho_lattice=float(cfg(model_args, "aa_rho_lattice", 0.0)),
        lattice_repr=str(cfg(model_args, "lattice_repr", "y1")),
    )


def load_model_for_sampling(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[CrystaliteModel, dict, Any]:
    """One-shot: load checkpoint → build model → apply EMA → return (model, model_args, type_encoding).

    Loads to CPU first, applies EMA, then moves to the target device.
    """
    ckpt = load_checkpoint(checkpoint_path)
    model, model_args = build_model(ckpt, device=torch.device("cpu"))
    ema_state = ckpt.get("ema_state_dict")
    if ema_state:
        apply_ema(model, ema_state)
    model = model.to(device)

    type_encoding_name = str(
        ckpt.get("type_encoding", model_args.get("type_encoding", "atomic_number"))
    )
    type_encoding = build_type_encoding(type_encoding_name, vz=VZ)
    return model, model_args, type_encoding
