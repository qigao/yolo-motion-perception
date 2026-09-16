from __future__ import annotations

import pytest

from neural_state_machine.phase3c_diagnostics.report import (
    AttributionCandidate,
    build_report_summary,
)
from neural_state_machine.phase3c_diagnostics.runner import (
    StageResult,
    run_fail_closed_stages,
    validate_complete_keys,
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
