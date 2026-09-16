from __future__ import annotations

import pytest

from neural_state_machine.phase3c_diagnostics import report
from neural_state_machine.phase3c_diagnostics.contracts import ModelId


def _scores(correct: int) -> list[dict[str, object]]:
    return [
        {
            "evaluation_id": evaluation_id,
            "overall": {"correct": correct, "total": 200},
        }
        for evaluation_id in range(-1, 8)
    ]


def _row(model: ModelId, correct: int, **extra: object) -> dict[str, object]:
    return {
        "model_id": model.to_dict(),
        "parameter_digest": "0" * 64,
        "scores": _scores(correct),
        **extra,
    }


def _registered_slice() -> list[dict[str, object]]:
    rows = [
        _row(
            ModelId("original", 7, arm="td0", condition="normal"),
            151,
        ),
        _row(
            ModelId("original", 7, arm="td0", condition="original_shuffle"),
            197,
            d1={"donor_summary": {"current_decision_matches": 3}},
        ),
    ]
    rows.extend(
        _row(
            ModelId("permutation", 7, arm="td0", mode="block10", replicate=replicate),
            130 + replicate,
            d1={"donor_summary": {"current_decision_matches": replicate}},
        )
        for replicate in range(32)
    )
    rows.extend(
        _row(
            ModelId("reference", 7, reference_kind=kind),
            160 + index,
            information_access=f"privileged-{kind}",
        )
        for index, kind in enumerate(
            ("supervised_ridge", "immediate_identified", "source_visible_delayed")
        )
    )
    return rows


def _execution() -> dict[str, object]:
    return {
        "diagnostic_valid": True,
        "accounting": [
            {
                "seed": 7,
                "condition": "normal",
                "arm": "eligibility",
                "maximum_drain_residual": 2e-13,
                "history_components": [
                    {
                        "eligibility_reconstruction_max_residual": 1e-13,
                        "update_reconstruction_max_residual": 3e-13,
                        "source_vs_reference_cosine": {
                            "value": None,
                            "reason": "zero_norm",
                            "sample_count": 1,
                        },
                        "other_vs_reference_cosine": {
                            "value": 0.25,
                            "reason": None,
                            "sample_count": 1,
                        },
                        "actual_vs_reference_cosine": {
                            "value": None,
                            "reason": "zero_norm",
                            "sample_count": 1,
                        },
                    }
                ],
            }
        ],
    }


def _auxiliary() -> dict[str, object]:
    return {
        "reset_scores": [{"key": "reset/a"}, {"key": "reset/b"}],
        "reverse_checks": [{"key": "reverse/a"}],
        "drain_scores": [{"key": "drain/a"}, {"key": "drain/b"}, {"key": "drain/c"}],
    }


def test_evidence_summary_indexes_scores_distributions_references_and_accounting() -> None:
    builder = getattr(report, "build_evidence_summary", None)
    assert builder is not None, "report evidence summary must be implemented"
    rows = _registered_slice()

    evidence = builder(rows, _execution(), _auxiliary())

    assert evidence["counts"] == {
        "models": len(rows),
        "main_scores": len(rows) * 9,
        "permutation_models": 32,
        "privileged_references": 3,
    }
    assert len(evidence["model_scores"]) == len(rows)
    assert sum(len(row["scores"]) for row in evidence["model_scores"]) == len(rows) * 9

    distributions = evidence["permutation_distributions"]
    assert len(distributions) == 1
    distribution = distributions[0]
    assert (distribution["seed"], distribution["arm"], distribution["mode"]) == (
        7,
        "td0",
        "block10",
    )
    assert distribution["summary"]["count"] == 32
    assert distribution["summary"]["minimum"] == 130
    assert distribution["summary"]["maximum"] == 161
    assert distribution["summary"]["count_at_least_150"] == 12
    assert distribution["original_shuffle_score"] == 197
    assert distribution["original_shuffle_rank_descending"] == 1

    assert len(evidence["privileged_references"]) == 3
    assert all("information_access" in row for row in evidence["privileged_references"])
    assert evidence["auxiliary_counts"] == {
        "reset_scores": 2,
        "reverse_checks": 1,
        "drain_scores": 3,
    }
    assert evidence["accounting"]["maximum_drain_residual"] == 2e-13
    assert evidence["accounting"]["maximum_eligibility_reconstruction_residual"] == 1e-13
    assert evidence["accounting"]["maximum_update_reconstruction_residual"] == 3e-13
    assert evidence["accounting"]["undefined_metric_reasons"] == {"zero_norm": 2}


def test_evidence_summary_rejects_incomplete_permutation_distribution() -> None:
    builder = getattr(report, "build_evidence_summary", None)
    assert builder is not None, "report evidence summary must be implemented"
    rows = _registered_slice()
    rows = [
        row
        for row in rows
        if not (
            row["model_id"]["family"] == "permutation"
            and row["model_id"]["replicate"] == 31
        )
    ]

    with pytest.raises(RuntimeError, match="32"):
        builder(rows, _execution(), _auxiliary())
