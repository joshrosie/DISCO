from __future__ import annotations

import math

import torch


def compute_type_distribution_stats(
    type_scores: torch.Tensor,
    pad_mask: torch.Tensor,
) -> dict[str, float]:
    real_mask = ~pad_mask.bool()
    if not real_mask.any():
        return {
            "type_entropy": 0.0,
            "type_entropy_norm": 0.0,
            "type_max_prob": 0.0,
        }

    probs = torch.softmax(type_scores.to(dtype=torch.float32), dim=-1)
    probs_real = probs[real_mask]
    entropy = -(probs_real * probs_real.clamp_min(1e-12).log()).sum(dim=-1)
    max_prob = probs_real.amax(dim=-1)

    num_classes = int(type_scores.shape[-1])
    if num_classes > 1:
        entropy_norm = entropy / math.log(float(num_classes))
    else:
        entropy_norm = torch.zeros_like(entropy)

    return {
        "type_entropy": float(entropy.mean().item()),
        "type_entropy_norm": float(entropy_norm.mean().item()),
        "type_max_prob": float(max_prob.mean().item()),
    }


__all__ = ["compute_type_distribution_stats"]
