from __future__ import annotations

import copy

import pytest

from neural_state_machine.phase3c_diagnostics import runner
from neural_state_machine.phase3c_diagnostics.contracts import (
    REGISTERED_ARMS,
    REGISTERED_CONDITIONS,
    REGISTERED_SEEDS,
    ModelId,
)

TRAINING_DECISIONS = 2_000


def _validator():
    validator = getattr(runner, "validate_registered_evidence", None)
    assert validator is not None, "registered evidence validator must be implemented"
    return validator


def _metric(value: float | None = 0.25, reason: str | None = None) -> dict[str, object]:
    return {"value": value, "reason": reason, "sample_count": 1}


def _d1() -> dict[str, object]:
    return {
        "provenance_rows": [{}] * TRAINING_DECISIONS,
        "donor_summary": {
            "fixed_points": 1,
            "current_decision_matches": 2,
            "same_block_donors": 3,
            "relative_to_delivery": {"past": 1, "current": 1, "future": 1},
            "displacements": {"0": 1},
        },
        "call_classes": {
            "real_decision_calls": TRAINING_DECISIONS,
            "drain_calls": 5,
            "no_arrival_zero": 1,
            "cancellation_to_zero": 1,
            "one_source": 1,
            "collisions": 1,
        },
        "lags": [
            {
                "lag": lag,
                "feedback_reward_product": _metric(),
                "feedback_reward_correlation": _metric(),
                "action_cells": {},
                "label_cells": {},
            }
            for lag in range(11)
        ],
        "final_100": {"sample_count": 100, "feedback_sum": 0.0, "feedback_mean": 0.0},
    }


def _model_rows() -> list[dict[str, object]]:
    return [
        {
            "model_id": ModelId(
                "original", 7, arm="td0", condition="normal"
            ).to_dict(),
            "d1": _d1(),
        },
        {
            "model_id": ModelId(
                "permutation", 7, arm="eligibility", mode="block10", replicate=0
            ).to_dict(),
            "d1": _d1(),
        },
        {
            "model_id": ModelId(
                "reference", 7, reference_kind="supervised_ridge"
            ).to_dict(),
            "information_access": "training labels",
        },
    ]


def _execution() -> dict[str, object]:
    d0 = [
        {
            "seed": seed,
            "arm": arm,
            "condition": condition,
            "parameter_digest": "0" * 64,
            "scalar_call_count": TRAINING_DECISIONS + 5,
        }
        for seed in REGISTERED_SEEDS
        for condition in REGISTERED_CONDITIONS
        for arm in REGISTERED_ARMS
    ]
    component = {
        "eligibility_reconstruction_max_residual": 1e-13,
        "update_reconstruction_max_residual": 2e-13,
        "source_vs_reference_cosine": _metric(None, "zero_norm"),
        "other_vs_reference_cosine": _metric(),
        "actual_vs_reference_cosine": _metric(),
    }
    accounting: list[dict[str, object]] = []
    for seed in REGISTERED_SEEDS:
        for condition in REGISTERED_CONDITIONS:
            accounting.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "arm": "td0",
                    "real_steps": TRAINING_DECISIONS,
                    "drain_steps": 5,
                    "drain_weight_changes": 0,
                }
            )
            accounting.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "arm": "eligibility",
                    "maximum_drain_residual": 3e-13,
                    "history_components": [component] * TRAINING_DECISIONS,
                }
            )
    return {
        "diagnostic_valid": True,
        "d0": d0,
        "accounting": accounting,
    }


def _manifest() -> dict[str, object]:
    return {
        "profile": "registered-v1",
        "registered_valid_profile": True,
        "configuration": {"training_decisions": TRAINING_DECISIONS},
    }


def test_registered_evidence_validator_accepts_complete_d0_d1_d3_schema() -> None:
    _validator()(_manifest(), _execution(), _model_rows())


def test_registered_evidence_validator_rejects_anonymous_row_without_d1() -> None:
    rows = _model_rows()
    rows[0].pop("d1")
    with pytest.raises(RuntimeError, match="D1|d1"):
        _validator()(_manifest(), _execution(), rows)


def test_registered_evidence_validator_rejects_missing_d0_trajectory() -> None:
    execution = _execution()
    execution["d0"] = execution["d0"][:-1]
    with pytest.raises(RuntimeError, match="D0|d0"):
        _validator()(_manifest(), execution, _model_rows())


def test_registered_evidence_validator_rejects_missing_accounting_trajectory() -> None:
    execution = _execution()
    execution["accounting"] = execution["accounting"][:-1]
    with pytest.raises(RuntimeError, match="accounting"):
        _validator()(_manifest(), execution, _model_rows())


def test_registered_evidence_validator_rejects_large_reconstruction_residual() -> None:
    execution = copy.deepcopy(_execution())
    eligibility = next(row for row in execution["accounting"] if row["arm"] == "eligibility")
    eligibility["history_components"][0]["update_reconstruction_max_residual"] = 1e-4
    with pytest.raises(RuntimeError, match="residual"):
        _validator()(_manifest(), execution, _model_rows())
