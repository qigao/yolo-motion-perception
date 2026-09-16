from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from neural_state_machine.phase3c_diagnostics.contracts import ModelId
from neural_state_machine.phase3c_diagnostics.manifest import canonical_json_bytes, sha256_file


def _scores(correct: int) -> list[dict[str, object]]:
    return [
        {
            "evaluation_id": evaluation_id,
            "overall": {"correct": correct, "total": 200},
        }
        for evaluation_id in range(-1, 8)
    ]


def _models() -> list[tuple[ModelId, int, str | None]]:
    rows: list[tuple[ModelId, int, str | None]] = [
        (ModelId("original", 7, arm="td0", condition="normal"), 151, None),
        (
            ModelId("original", 7, arm="td0", condition="original_shuffle"),
            197,
            None,
        ),
    ]
    rows.extend(
        (
            ModelId("permutation", 7, arm="td0", mode="block10", replicate=replicate),
            130 + replicate,
            None,
        )
        for replicate in range(32)
    )
    rows.extend(
        (
            ModelId("reference", 7, reference_kind=kind),
            160 + index,
            f"privileged-{kind}",
        )
        for index, kind in enumerate(
            ("supervised_ridge", "immediate_identified", "source_visible_delayed")
        )
    )
    return rows


def _verified_attempt(tmp_path: Path) -> tuple[Path, Path]:
    models = _models()
    manifest_path = tmp_path / "manifest.json"
    main_score_keys = [
        f"{model.stable_key()}/evaluation={evaluation_id}"
        for model, _, _ in models
        for evaluation_id in range(-1, 8)
    ]
    manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "profile": "registered-v1",
                "model_ids": [model.to_dict() for model, _, _ in models],
                "main_score_keys": main_score_keys,
                "auxiliary_keys": {
                    "reset_scores": ["reset/a"],
                    "reverse_checks": ["reverse/a"],
                    "drain_scores": ["drain/a"],
                },
            }
        )
    )

    attempt_dir = tmp_path / "attempt-cli"
    rows_dir = attempt_dir / "rows"
    rows_dir.mkdir(parents=True)
    execution_models: list[dict[str, str]] = []
    for index, (model, correct, information_access) in enumerate(models):
        payload: dict[str, object] = {
            "model_id": model.to_dict(),
            "parameter_digest": "0" * 64,
            "scores": _scores(correct),
        }
        if information_access is not None:
            payload["information_access"] = information_access
        path = rows_dir / f"{index:04d}.json"
        path.write_bytes(canonical_json_bytes(payload))
        execution_models.append(
            {"model_key": model.stable_key(), "row_sha256": sha256_file(path)}
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
        "schema_version": 1,
        "attempt_id": "cli",
        "manifest_sha256": sha256_file(manifest_path),
        "complete": True,
        "diagnostic_valid": True,
        "models": execution_models,
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
                        "actual_vs_reference_cosine": {
                            "value": None,
                            "reason": "zero_norm",
                            "sample_count": 1,
                        },
                    }
                ],
            }
        ],
        "auxiliary": {
            "path": "auxiliary.json",
            "sha256": sha256_file(auxiliary_path),
        },
    }
    (attempt_dir / "execution-manifest.json").write_bytes(canonical_json_bytes(execution))
    return manifest_path, attempt_dir


def test_report_cli_embeds_verified_complete_evidence_index(tmp_path: Path) -> None:
    manifest, attempt = _verified_attempt(tmp_path)
    output = tmp_path / "report"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_phase3c_failure.py",
            "report",
            "--manifest",
            str(manifest),
            "--attempt-dir",
            str(attempt),
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    evidence = summary["evidence"]
    assert evidence["counts"] == {
        "models": 37,
        "main_scores": 333,
        "permutation_models": 32,
        "privileged_references": 3,
    }
    assert len(evidence["model_scores"]) == 37
    assert len(evidence["permutation_distributions"]) == 1
    assert evidence["auxiliary_counts"] == {
        "reset_scores": 1,
        "reverse_checks": 1,
        "drain_scores": 1,
    }
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "Models: `37`" in markdown
    assert "Main scores: `333`" in markdown
