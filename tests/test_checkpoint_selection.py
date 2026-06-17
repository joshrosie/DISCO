from __future__ import annotations

import pytest

from src.utils.checkpoint import select_primary_candidate_from_sampling


def test_dng_auto_prefers_train_msun_when_available():
    candidate = select_primary_candidate_from_sampling(
        is_csp=False,
        step=100,
        epoch=2,
        best_ckpt_selector="auto",
        dng_payloads={
            "precise": {
                "MSUN": 0.40,
                "Train_MSUN": 0.25,
            }
        },
    )

    assert candidate is not None
    assert candidate.selector_name == "precise/Train_MSUN"
    assert candidate.selector_value == pytest.approx(0.25)
    assert candidate.raw_metrics == {
        "precise/Train_MSUN": pytest.approx(0.25),
        "precise/MSUN": pytest.approx(0.40),
    }


def test_dng_auto_falls_back_to_raw_msun_without_train_msun():
    candidate = select_primary_candidate_from_sampling(
        is_csp=False,
        step=100,
        epoch=2,
        best_ckpt_selector="auto",
        dng_payloads={"precise": {"MSUN": 0.40}},
    )

    assert candidate is not None
    assert candidate.selector_name == "precise/MSUN"
    assert candidate.selector_value == pytest.approx(0.40)


def test_dng_legacy_keeps_raw_msun_order():
    candidate = select_primary_candidate_from_sampling(
        is_csp=False,
        step=100,
        epoch=2,
        best_ckpt_selector="legacy",
        dng_payloads={
            "sample": {"MSUN": 0.30, "Train_MSUN": 0.20},
            "precise": {"MSUN": 0.40, "Train_MSUN": 0.25},
        },
    )

    assert candidate is not None
    assert candidate.selector_name == "sample/MSUN"
    assert candidate.selector_value == pytest.approx(0.30)


def test_dng_explicit_train_msun_selector_requires_train_msun():
    candidate = select_primary_candidate_from_sampling(
        is_csp=False,
        step=100,
        epoch=2,
        best_ckpt_selector="dng_precise_train_msun",
        dng_payloads={"precise": {"MSUN": 0.40}},
    )

    assert candidate is None

