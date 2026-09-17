from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import benchmark_r1_e1_reservoir as cli


_HEAD = "132df8a9a100959c63117f8673789120290cbf5d"


def _forbidden_measurement():
    pytest.fail("registered measurement was invoked outside the measure command")


def _fake_measurement() -> dict[str, object]:
    return {
        "registered_measurement": True,
        "compatibility": [],
        "arms": [],
    }


def test_protocol_command_does_not_run_registered_measurement(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    monkeypatch.setattr(
        cli,
        "protocol_smoke",
        lambda: {"registered_measurement": False, "valid": True},
    )

    assert cli.main(["protocol"]) == 0
    assert json.loads(capsys.readouterr().out)["registered_measurement"] is False


def test_prepare_command_does_not_run_registered_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    root = tmp_path / "prospective"

    assert cli.main(["prepare", "--output", str(root), "--scientific-head", _HEAD]) == 0
    assert (root / "manifest.json").exists()
    assert not (root / "result.json").exists()


def test_verify_no_result_command_does_not_run_registered_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    root = tmp_path / "prospective"
    assert cli.main(["prepare", "--output", str(root), "--scientific-head", _HEAD]) == 0

    assert cli.main(["verify", "--root", str(root), "--no-result-ok"]) == 0


def test_measure_command_is_the_only_command_that_invokes_registered_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "prospective"
    assert cli.main(["prepare", "--output", str(root), "--scientific-head", _HEAD]) == 0
    manifest_sha = (root / "manifest.sha256").read_text().strip()
    calls = []

    def fake_run():
        calls.append("measure")
        return _fake_measurement()

    monkeypatch.setattr(cli, "run_registered_measurement", fake_run)
    monkeypatch.setattr(cli, "_current_head", lambda: _HEAD)

    assert cli.main([
        "measure",
        "--root",
        str(root),
        "--manifest-sha256",
        manifest_sha,
    ]) == 0
    assert calls == ["measure"]
    assert (root / "result.json").exists()
    assert not (root / "report.md").exists()


def test_measure_rejects_head_mismatch_before_running_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "prospective"
    assert cli.main(["prepare", "--output", str(root), "--scientific-head", _HEAD]) == 0
    manifest_sha = (root / "manifest.sha256").read_text().strip()
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    monkeypatch.setattr(cli, "_current_head", lambda: "0" * 40)

    with pytest.raises(SystemExit):
        cli.main([
            "measure",
            "--root",
            str(root),
            "--manifest-sha256",
            manifest_sha,
        ])
