"""End-to-end DNG eval pipeline for a generated sample set.

Composes the primitives in ``src/eval/`` (evaluator, structure stats,
novelty, thermo / SUN / MSUN) into one call that returns a flat
``metrics`` dict. Used by the FK ablation driver, the W&B sweep, and
local smoke scripts.

Extracted from ``scripts/ablation_fk._run_eval`` so it can be imported
without depending on a CLI script.
"""

from __future__ import annotations

from typing import Any

from src.eval.dng_eval import (
    collect_constructed_structures,
    compute_evaluator_metrics,
    compute_novelty_metrics,
    compute_structure_stats_metrics,
    compute_sun_msun_from_thermo_rates,
    float_or_nan,
)
from src.eval.stability import _compute_thermo_metrics
from src.utils.stability_logger import StabilityLogger, _ThermoConfig


def run_eval(
    sample_items: list[dict],
    *,
    ref_structs: list,
    novelty_ref_structs: list,
    sample_seed: int,
    thermo_count: int,
    thermo_cfg: _ThermoConfig | None,
    compute_novelty: bool = True,
    compute_wasserstein: bool = True,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    n = len(sample_items)
    metrics["num_samples"] = float(n)

    # Validity + diagnostics + Wasserstein
    ev = compute_evaluator_metrics(
        sample_items,
        limit=n,
        ref_structs=ref_structs,
        sample_seed=sample_seed,
        include_diagnostics=True,
        include_wasserstein=compute_wasserstein,
        wasserstein_max_samples=10000,
    )
    if ev.valid_rate is not None:
        metrics["valid_rate"] = float_or_nan(ev.valid_rate)
    if ev.comp_valid_rate is not None:
        metrics["comp_valid_rate"] = float_or_nan(ev.comp_valid_rate)
    if ev.struct_valid_rate is not None:
        metrics["struct_valid_rate"] = float_or_nan(ev.struct_valid_rate)
    for k, v in ev.diag_metrics.items():
        metrics[f"diag/{k}"] = float_or_nan(v)
    if compute_wasserstein and ev.dist_metrics:
        for k, v in ev.dist_metrics.items():
            metrics[k] = float_or_nan(v)

    # Structure stats
    stats = compute_structure_stats_metrics(sample_items, total_count=n, include_summary_stats=True)
    for k, v in stats.metrics.items():
        metrics[f"stats/{k}"] = float_or_nan(v)

    # Uniqueness / novelty
    un_rate = None
    novelty_metrics: dict[str, Any] = {}
    if compute_novelty and novelty_ref_structs:
        nov = compute_novelty_metrics(sample_items, novelty_ref_structs, limit=n, minimum_nary=1)
        novelty_metrics = nov.novelty_metrics
        if nov.unique_rate is not None:
            metrics["unique_rate"] = float_or_nan(nov.unique_rate)
        if nov.novel_rate is not None:
            metrics["novel_rate"] = float_or_nan(nov.novel_rate)
        if nov.un_rate is not None:
            un_rate = nov.un_rate
            metrics["un_rate"] = float_or_nan(nov.un_rate)

    # Thermo / SUN / MSUN
    if thermo_count > 0 and thermo_cfg is not None:
        thermo_n = min(thermo_count, n)
        thermo_structs = collect_constructed_structures(
            sample_items, pred_crys_list=ev.pred_crys_list, count=thermo_n,
        )
        if thermo_structs:
            logger = StabilityLogger(gamma_cfg=None, thermo_cfg=thermo_cfg)
            thermo_metrics = _compute_thermo_metrics(
                logger, thermo_structs, tag="eval", step=0, enabled=True, show_progress=True,
            )
            for k, v in thermo_metrics.items():
                metrics[k] = float_or_nan(v)
            sun_summary = compute_sun_msun_from_thermo_rates(
                un_rate=un_rate, thermo_metrics=thermo_metrics, thermo_tag="eval",
            )
            for k, v in sun_summary.items():
                metrics[k] = float_or_nan(v)

    return metrics
