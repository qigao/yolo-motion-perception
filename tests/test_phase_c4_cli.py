from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import benchmark_phase_c4_delay_marginalized_credit as benchmark_cli
from scripts import verify_phase_c4_delay_marginalized_credit as verify_cli


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_c4a_cli_writes_only_manifest(tmp_path):
    assert benchmark_cli.main(["prepare-c4a", "--output", str(tmp_path)]) == 0
    manifest = tmp_path / "c4a-manifest.json"
    assert manifest.is_file()
    assert not (tmp_path / "c4a-result.json").exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["stage"] == "c4a"
    assert payload["expected_scalar_design_rows"] == 2005


def test_verify_cli_accepts_prospective_c4a_manifest(tmp_path):
    assert benchmark_cli.main(["prepare-c4a", "--output", str(tmp_path)]) == 0
    assert verify_cli.main(["--stage", "c4a", "--root", str(tmp_path), "--no-result-ok"]) == 0


def test_measure_cli_requires_manifest_argument(tmp_path):
    with pytest.raises(SystemExit):
        benchmark_cli.main(["measure-c4a", "--output", str(tmp_path)])


def test_protocol_cli_is_a_distinct_nonmeasurement_command(monkeypatch, capsys):
    called = {"protocol": 0}

    def fake_protocol():
        called["protocol"] += 1
        return ()

    monkeypatch.setattr(benchmark_cli, "run_phase_c4_protocol_gate", fake_protocol)
    assert benchmark_cli.main(["protocol"]) == 0
    assert called == {"protocol": 1}
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"protocol_valid": True, "result_count": 0, "stage": "protocol"}
