from __future__ import annotations

from pathlib import Path

import pytest

from neural_state_machine.phase3c_diagnostics.contracts import (
    REGISTERED_ARMS,
    REGISTERED_CONDITIONS,
    REGISTERED_SEEDS,
    ModelId,
)
from neural_state_machine.phase3c_diagnostics.manifest import canonical_json_bytes, sha256_file
from neural_state_machine.phase3c_diagnostics.runner import verify_attempt

TRAINING_DECISIONS = 2_000


def _execution_accounting() -> list[dict[str, object]]:
    metric = {"value": None, "reason": "zero_norm", "sample_count": 1}
    component = {
        "eligibility_reconstruction_max_residual": 1e-13,
        "update_reconstruction_max_residual": 2e-13,
        "source_vs_reference_cosine": metric,
        "other_vs_reference_cosine": metric,
        "actual_vs_reference_cosine": metric,
    }
    rows: list[dict[str, object]] = []
    for seed in REGISTERED_SEEDS:
        for condition in REGISTERED_CONDITIONS:
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "arm": "td0",
                    "real_steps": TRAINING_DECISIONS,
                    "drain_steps": 5,
                    "drain_weight_changes": 0,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "arm": "eligibility",
                    "maximum_drain_residual": 3e-13,
                    "history_components": [component] * TRAINING_DECISIONS,
                }
            )
    return rows


def _attempt_missing_d1(tmp_path: Path) -> tuple[Path, Path]:
    model = ModelId("original", 7, arm="td0", condition="normal")
    model_key = model.stable_key()
    manifest_path = tmp_path / "manifest.json"
    main_score_keys = [
        f"{model_key}/evaluation={evaluation_id}" for evaluation_id in range(-1, 8)
    ]
    manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "profile": "registered-v1",
                "registered_valid_profile": True,
                "configuration": {"training_decisions": TRAINING_DECISIONS},
                "model_ids": [model.to_dict()],
                "main_score_keys": main_score_keys,
                "auxiliary_keys": {
                    "reset_scores": ["reset/a"],
                    "reverse_checks": ["reverse/a"],
                    "drain_scores": ["drain/a"],
                },
            }
        )
    )

    attempt_dir = tmp_path / "attempt"
    rows_dir = attempt_dir / "rows"
    rows_dir.mkdir(parents=True)
    row_path = rows_dir / "0000.json"
    row_path.write_bytes(
        canonical_json_bytes(
            {
                "model_id": model.to_dict(),
                "parameter_digest": "0" * 64,
                "scores": [
                    {
                        "evaluation_id": evaluation_id,
                        "overall": {"correct": 100, "total": 200},
                    }
                    for evaluation_id in range(-1, 8)
                ],
            }
        )
    )
    auxiliary_path = attempt_dir / "auxiliary.json"
    auxiliary_path.write_bytes(
        canonical_json_bytes(
            {
                "reset_scores": [{"key": "reset/a"}],
                "reverse_checks": [{"key": "reverse/a"}],
                "drain_scores": [{"key": "drain/a"}],
            }
        )
    )
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
    execution = {
        "schema_version": 1,
        "attempt_id": "missing-d1",
        "manifest_sha256": sha256_file(manifest_path),
        "complete": True,
        "diagnostic_valid": True,
        "models": [{"model_key": model_key, "row_sha256": sha256_file(row_path)}],
        "d0": d0,
        "accounting": _execution_accounting(),
        "auxiliary": {"path": "auxiliary.json", "sha256": sha256_file(auxiliary_path)},
    }
    (attempt_dir / "execution-manifest.json").write_bytes(canonical_json_bytes(execution))
    return manifest_path, attempt_dir


def test_verify_attempt_applies_registered_d1_d3_validator(tmp_path: Path) -> None:
    manifest, attempt = _attempt_missing_d1(tmp_path)
    with pytest.raises(RuntimeError, match="D1|d1"):
        verify_attempt(manifest, attempt)
