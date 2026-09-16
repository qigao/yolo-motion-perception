from __future__ import annotations

from pathlib import Path

import pytest

from neural_state_machine.phase3c_diagnostics.contracts import ModelId
from neural_state_machine.phase3c_diagnostics.manifest import (
    canonical_json_bytes,
    sha256_file,
)
from neural_state_machine.phase3c_diagnostics.report import (
    AttributionCandidate,
    build_report_summary,
)
from neural_state_machine.phase3c_diagnostics.runner import (
    StageResult,
    run_fail_closed_stages,
    validate_complete_keys,
    verify_attempt,
)


def test_d0_failure_stops_later_stages() -> None:
    calls: list[str] = []

    def stage(name: str, passed: bool):
        def run() -> StageResult:
            calls.append(name)
            return StageResult(name, passed)
        return run

    results = run_fail_closed_stages(
        stage("d0", False), stage("d1", True), stage("d2", True), stage("d3", True)
    )
    assert [row.name for row in results] == ["d0"]
    assert calls == ["d0"]


def test_complete_key_validation_rejects_missing_duplicate_or_extra_rows() -> None:
    expected = ("a", "b", "c")
    for observed in (("a", "b"), ("a", "b", "c", "c"), ("a", "b", "c", "d")):
        with pytest.raises(ValueError, match="grid"):
            validate_complete_keys(expected, observed)


def test_valid_unresolved_report_keeps_original_behavior_false() -> None:
    summary = build_report_summary(
        diagnostic_valid=True,
        original_behavior_passed=False,
        candidates=(AttributionCandidate("temporal_realign", "unresolved", ("d1.rank",)),),
    )
    assert summary["diagnostic_valid"] is True
    assert summary["original_phase3c"]["behavior_passed"] is False
    assert summary["candidates"][0]["status"] == "unresolved"


def _minimal_attempt(
    tmp_path: Path,
    *,
    evaluation_ids: tuple[int, ...],
) -> tuple[Path, Path, Path]:
    model = ModelId("original", 7, arm="td0", condition="normal")
    model_key = model.stable_key()
    expected_score_keys = tuple(
        f"{model_key}/evaluation={evaluation_id}" for evaluation_id in range(-1, 8)
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "model_ids": [model.to_dict()],
                "main_score_keys": list(expected_score_keys),
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
                "scores": [
                    {"evaluation_id": evaluation_id}
                    for evaluation_id in evaluation_ids
                ],
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
    }
    (attempt_dir / "execution-manifest.json").write_bytes(
        canonical_json_bytes(execution)
    )
    return manifest_path, attempt_dir, row_path


def test_offline_verify_rejects_missing_main_score(tmp_path: Path) -> None:
    manifest, attempt, _ = _minimal_attempt(
        tmp_path,
        evaluation_ids=tuple(range(-1, 7)),
    )

    with pytest.raises(RuntimeError, match="score"):
        verify_attempt(manifest, attempt)


def test_offline_verify_rejects_row_hash_mutation(tmp_path: Path) -> None:
    manifest, attempt, row_path = _minimal_attempt(
        tmp_path,
        evaluation_ids=tuple(range(-1, 8)),
    )
    row_path.write_bytes(row_path.read_bytes() + b"\n")

    with pytest.raises(RuntimeError, match="hash"):
        verify_attempt(manifest, attempt)
