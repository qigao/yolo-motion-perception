import json
from pathlib import Path

import pytest

from neural_state_machine.phase3b_delayed_benchmark import DelayedCreditConfig
from scripts.benchmark_phase3b_delayed_credit import build_payload, write_evidence
from scripts.verify_phase3b_delayed_credit import _validate


def test_phase3b_payload_is_explicitly_diagnostic_only(tmp_path: Path) -> None:
    frozen = tmp_path / "docs/experiments/phase-3a-action-value.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"frozen")
    payload = build_payload(
        tmp_path,
        seeds=(7,),
        config=DelayedCreditConfig(
            training_episodes=100,
            evaluation_blocks=2,
            checkpoint_interval=50,
        ),
    )
    assert payload["experiment"] == "phase-3b-delayed-credit"
    assert payload["diagnostic_only"] is True
    assert payload["all_passed"] is False
    _validate(payload)


def test_writer_accepts_only_approved_path(tmp_path: Path) -> None:
    payload = {
        "experiment": "phase-3b-delayed-credit",
        "schema_version": 1,
        "diagnostic_only": True,
        "decision_status": "diagnostic-only-before-full-gate",
        "frozen_phase3a_sha256": "0" * 64,
        "results": [],
    }
    with pytest.raises(ValueError, match="approved"):
        write_evidence(payload, tmp_path / "other.json", tmp_path)


def test_writer_is_deterministic_for_identical_payload(tmp_path: Path) -> None:
    root = tmp_path
    target = root / "docs/experiments/phase-3b-delayed-credit.json"
    payload = {
        "experiment": "phase-3b-delayed-credit",
        "schema_version": 1,
        "diagnostic_only": True,
        "decision_status": "diagnostic-only-before-full-gate",
        "frozen_phase3a_sha256": "0" * 64,
        "results": [],
    }
    write_evidence(payload, target, root)
    first = target.read_bytes()
    write_evidence(payload, target, root)
    assert target.read_bytes() == first
    assert json.loads(first)["schema_version"] == 1
