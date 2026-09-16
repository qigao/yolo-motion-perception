from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_state_machine.phase_c4a_diagnostics.evidence import prepare_attribution_manifest
from scripts import diagnose_phase_c4a_failure as diagnose_cli
from scripts import verify_phase_c4a_failure_diagnostics as verify_cli


ROOT = Path(__file__).resolve().parents[1]
_LOCK = ROOT / "requirements/phase-c4a-diagnostics-python312.lock"
_INPUT = ROOT / "requirements/phase-c4a-diagnostics.in"


def test_protocol_cli_runs_registered_d0_only(monkeypatch, capsys):
    called = {"d0": 0}

    def fake_d0(root: Path):
        assert root == ROOT
        called["d0"] += 1
        return tuple(range(6))

    monkeypatch.setattr(diagnose_cli, "run_d0_integrity", fake_d0)
    assert diagnose_cli.main(["protocol"]) == 0
    assert called == {"d0": 1}
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "condition_count": 6,
        "integrity_valid": True,
        "stage": "protocol",
    }


def test_prepare_cli_writes_only_registered_manifest(tmp_path: Path):
    assert diagnose_cli.main(["prepare", "--output", str(tmp_path)]) == 0
    assert [path.name for path in tmp_path.iterdir()] == ["manifest.json"]
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["stage"] == "c4a-failure-attribution"
    assert payload["diagnostic_config"]["permutation_replicates"] == 32
    assert payload["permutation_lineage"] == 0x43344144
    assert {
        "scripts/diagnose_phase_c4a_failure.py",
        "scripts/verify_phase_c4a_failure_diagnostics.py",
        "requirements/phase-c4a-diagnostics.in",
        "requirements/phase-c4a-diagnostics-python312.lock",
    } <= set(payload["scientific_hashes"])


def test_verify_cli_accepts_prospective_manifest(tmp_path: Path):
    assert diagnose_cli.main(["prepare", "--output", str(tmp_path)]) == 0
    assert verify_cli.main(["--root", str(tmp_path), "--no-result-ok"]) == 0


def test_measure_rejects_manifest_mismatch_before_any_permutation_fit(
    tmp_path: Path,
    monkeypatch,
):
    manifest_dir = tmp_path / "prospective"
    manifest = prepare_attribution_manifest(ROOT, manifest_dir)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["scientific_head"] = "0" * 40
    manifest.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    calls = {"grid": 0}

    def forbidden_grid(*args, **kwargs):
        calls["grid"] += 1
        raise AssertionError("permutation fitting must not run before manifest preflight")

    monkeypatch.setattr(diagnose_cli, "run_permutation_grid_for_seed", forbidden_grid)
    with pytest.raises(ValueError, match="manifest scientific_head mismatch"):
        diagnose_cli.main(
            [
                "measure",
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "result"),
            ]
        )
    assert calls == {"grid": 0}
    assert not (tmp_path / "result").exists()


def test_measure_cli_exposes_no_registered_constant_weakening_flags(tmp_path: Path):
    manifest_dir = tmp_path / "prospective"
    manifest = prepare_attribution_manifest(ROOT, manifest_dir)
    with pytest.raises(SystemExit):
        diagnose_cli.main(
            [
                "measure",
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "result"),
                "--replicates",
                "2",
            ]
        )


def test_diagnostic_input_and_python312_lock_are_exact():
    assert _INPUT.read_text(encoding="utf-8").splitlines() == [
        "numpy==2.5.3",
        "pytest==9.1.1",
        "ruff==0.15.22",
    ]

    lock_lines = [
        line
        for line in _LOCK.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert lock_lines == [
        "iniconfig==2.3.0",
        "numpy==2.5.3",
        "packaging==26.3",
        "pip==26.2.1",
        "pluggy==1.6.0",
        "Pygments==2.21.0",
        "pytest==9.1.1",
        "ruff==0.15.22",
        "setuptools==84.0.0",
        "wheel==0.48.0",
    ]
    header = _LOCK.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("# Phase C4-A failure-attribution diagnostics")
    assert "preflight" not in header.lower()
