from __future__ import annotations

import subprocess
import sys


def test_diagnostic_cli_exposes_prepare_run_verify_report_without_executing() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/diagnose_phase3c_failure.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "prepare" in result.stdout
    assert "run" in result.stdout
    assert "verify" in result.stdout
    assert "report" in result.stdout
