from __future__ import annotations

import json
from pathlib import Path

import pytest


_HEAD = "182f15f6a8ba6e4aca21827cefa0416ec5008dfe"


def _cli():
    from scripts import benchmark_r1_e2_reservoir as cli

    return cli


def _forbidden_measurement():
    pytest.fail("registered measurement was invoked outside the measure command")


def test_e2_protocol_command_does_not_measure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _cli()
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    monkeypatch.setattr(
        cli,
        "protocol_smoke",
        lambda: {"registered_measurement": False, "valid": True},
    )

    assert cli.main(["protocol"]) == 0
    assert json.loads(capsys.readouterr().out)["registered_measurement"] is False


def test_e2_prepare_and_verify_no_result_do_not_measure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    root = tmp_path / "prospective"

    assert cli.main(["prepare", "--output", str(root), "--scientific-head", _HEAD]) == 0
    assert (root / "manifest.json").exists()
    assert not (root / "result.json").exists()
    assert cli.main(["verify", "--root", str(root), "--no-result-ok"]) == 0


def test_e2_measure_rejects_head_mismatch_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    root = tmp_path / "prospective"
    assert cli.main(["prepare", "--output", str(root), "--scientific-head", _HEAD]) == 0
    manifest_sha = (root / "manifest.sha256").read_text().strip()
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    monkeypatch.setattr(cli, "_current_head", lambda: "0" * 40)

    with pytest.raises(SystemExit, match="scientific head mismatch"):
        cli.main(
            [
                "measure",
                "--root",
                str(root),
                "--manifest-sha256",
                manifest_sha,
            ]
        )


def test_permanent_ci_runs_e2_preflight_but_never_measure() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "R1 E2 focused tests" in workflow
    assert "R1 E2 protocol smoke" in workflow
    assert "Prepare prospective R1 E2 manifest without measurement" in workflow
    assert "Verify prospective R1 E2 manifest" in workflow
    assert "benchmark_r1_e2_reservoir.py measure" not in workflow


def test_e2_measure_rejects_manifest_hash_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    root = tmp_path / "prospective"
    assert cli.main(["prepare", "--output", str(root), "--scientific-head", _HEAD]) == 0
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    monkeypatch.setattr(cli, "_current_head", lambda: _HEAD)

    with pytest.raises(SystemExit, match="manifest sha256"):
        cli.main(
            [
                "measure",
                "--root",
                str(root),
                "--manifest-sha256",
                "0" * 64,
            ]
        )


def test_e2_measure_preflights_prospective_evidence_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    root = tmp_path / "prospective"
    assert cli.main(["prepare", "--output", str(root), "--scientific-head", _HEAD]) == 0
    manifest_sha = (root / "manifest.sha256").read_text().strip()
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    monkeypatch.setattr(cli, "_current_head", lambda: _HEAD)

    def invalid_preflight(*args, **kwargs):
        raise cli.EvidenceInvalid("measurement runtime does not match sealed manifest")

    monkeypatch.setattr(cli, "verify_evidence", invalid_preflight)

    with pytest.raises(SystemExit, match="runtime"):
        cli.main(
            [
                "measure",
                "--root",
                str(root),
                "--manifest-sha256",
                manifest_sha,
            ]
        )


def test_e2_measure_rejects_existing_result_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    root = tmp_path / "prospective"
    assert cli.main(["prepare", "--output", str(root), "--scientific-head", _HEAD]) == 0
    manifest_sha = (root / "manifest.sha256").read_text().strip()
    monkeypatch.setattr(cli, "run_registered_measurement", _forbidden_measurement)
    monkeypatch.setattr(cli, "_current_head", lambda: _HEAD)
    monkeypatch.setattr(
        cli,
        "verify_evidence",
        lambda *args, **kwargs: {
            "valid": True,
            "prospective_only": False,
            "scientific_head": _HEAD,
            "manifest_sha256": manifest_sha,
            "registered_arm_count": 20,
        },
    )

    with pytest.raises(SystemExit, match="already exists"):
        cli.main(
            [
                "measure",
                "--root",
                str(root),
                "--manifest-sha256",
                manifest_sha,
            ]
        )
