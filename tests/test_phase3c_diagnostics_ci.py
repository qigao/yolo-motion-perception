from __future__ import annotations

from pathlib import Path


def test_ci_remains_python_312_only_and_does_not_run_registered_measurement() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.12"]' in text
    assert '"3.10"' not in text and '"3.11"' not in text and '"3.13"' not in text
    assert "--approve-measurement" not in text


def test_diagnostic_lock_anchors_actual_run_414_environment() -> None:
    text = Path("requirements/phase3c-diagnostics-python312.lock").read_text(encoding="utf-8")
    assert "numpy==2.5.3" in text
    assert "pytest==9.1.1" in text
    assert "ruff==0.15.22" in text


def test_manual_measurement_workflow_is_bounded_and_fail_closed() -> None:
    text = Path(".github/workflows/phase3c-diagnostics.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "timeout-minutes: 120" in text
    assert "contents: read" in text
    assert "--expected-manifest-sha256" in text
    assert "--approve-measurement" in text
    assert "actions/upload-artifact" in text
    assert "if: always()" in text
    assert "python-version: '3.12'" in text or 'python-version: "3.12"' in text
