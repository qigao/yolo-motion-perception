from __future__ import annotations

from pathlib import Path

import pytest

from neural_state_machine.phase3c_diagnostics.contracts import ModelId
from neural_state_machine.phase3c_diagnostics.manifest import canonical_json_bytes, sha256_file
from neural_state_machine.phase3c_diagnostics.runner import verify_attempt


def _attempt(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = ModelId("original", 7, arm="td0", condition="normal")
    model_key = model.stable_key()
    main_keys = tuple(
        f"{model_key}/evaluation={evaluation_id}" for evaluation_id in range(-1, 8)
    )
    auxiliary_keys = {
        "reset_scores": ("reset/a",),
        "reverse_checks": ("reverse/a",),
        "drain_scores": ("drain/a",),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "model_ids": [model.to_dict()],
                "main_score_keys": list(main_keys),
                "auxiliary_keys": {
                    name: list(values) for name, values in auxiliary_keys.items()
                },
            }
        )
    )

    attempt_dir = tmp_path / "attempt"
    rows = attempt_dir / "rows"
    rows.mkdir(parents=True)
    row_path = rows / "0000.json"
    row_path.write_bytes(
        canonical_json_bytes(
            {
                "model_id": model.to_dict(),
                "scores": [
                    {"evaluation_id": evaluation_id} for evaluation_id in range(-1, 8)
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
    execution = {
        "manifest_sha256": sha256_file(manifest_path),
        "complete": True,
        "diagnostic_valid": True,
        "models": [
            {"model_key": model_key, "row_sha256": sha256_file(row_path)}
        ],
        "auxiliary": {
            "path": "auxiliary.json",
            "sha256": sha256_file(auxiliary_path),
        },
    }
    (attempt_dir / "execution-manifest.json").write_bytes(canonical_json_bytes(execution))
    return manifest_path, attempt_dir, auxiliary_path


def test_offline_verify_accepts_exact_auxiliary_key_grid(tmp_path: Path) -> None:
    manifest, attempt, _ = _attempt(tmp_path)
    assert verify_attempt(manifest, attempt)["diagnostic_valid"] is True


def test_offline_verify_rejects_missing_auxiliary_key(tmp_path: Path) -> None:
    manifest, attempt, auxiliary = _attempt(tmp_path)
    auxiliary.write_bytes(
        canonical_json_bytes(
            {
                "reset_scores": [],
                "reverse_checks": [{"key": "reverse/a"}],
                "drain_scores": [{"key": "drain/a"}],
            }
        )
    )
    execution_path = attempt / "execution-manifest.json"
    execution = __import__("json").loads(execution_path.read_text(encoding="utf-8"))
    execution["auxiliary"]["sha256"] = sha256_file(auxiliary)
    execution_path.write_bytes(canonical_json_bytes(execution))

    with pytest.raises(RuntimeError, match="auxiliary"):
        verify_attempt(manifest, attempt)


def test_offline_verify_rejects_auxiliary_hash_mutation(tmp_path: Path) -> None:
    manifest, attempt, auxiliary = _attempt(tmp_path)
    auxiliary.write_bytes(auxiliary.read_bytes() + b"\n")

    with pytest.raises(RuntimeError, match="auxiliary.*hash|hash.*auxiliary"):
        verify_attempt(manifest, attempt)
