import copy
import json
from pathlib import Path

import pytest

from neural_state_machine.phase3b_delayed_benchmark import DelayedCreditConfig
from scripts.benchmark_phase3b_delayed_credit import build_payload, write_evidence
from scripts.verify_phase3b_delayed_credit import _portable_projection, _validate


def _small_payload(tmp_path: Path) -> dict[str, object]:
    frozen = tmp_path / "docs/experiments/phase-3a-action-value.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_bytes(b"frozen")
    return build_payload(
        tmp_path,
        seeds=(7,),
        config=DelayedCreditConfig(
            training_episodes=100,
            evaluation_blocks=2,
            checkpoint_interval=50,
        ),
    )


def test_phase3b_payload_uses_schema_v2_and_arm_a_only(tmp_path: Path) -> None:
    payload = _small_payload(tmp_path)

    assert payload["experiment"] == "phase-3b-delayed-credit"
    assert payload["schema_version"] == 2
    assert payload["protocol_valid"] is True
    assert type(payload["behavior_passed"]) is bool
    assert payload["all_passed"] == (
        payload["protocol_valid"] and payload["behavior_passed"]
    )
    assert payload["seeds"] == [7]
    assert payload["reward_delays"] == [0, 1, 3, 5]
    assert {row["arm"] for row in payload["results"]} == {"td0"}
    _validate(payload)


def test_writer_refuses_protocol_invalid_payload(tmp_path: Path) -> None:
    target = tmp_path / "docs/experiments/phase-3b-delayed-credit.json"
    payload = {
        "experiment": "phase-3b-delayed-credit",
        "schema_version": 2,
        "protocol_valid": False,
        "behavior_passed": False,
        "all_passed": False,
    }
    with pytest.raises(ValueError, match="protocol_valid"):
        write_evidence(payload, target, tmp_path)
    assert not target.exists()


def test_writer_is_deterministic_for_protocol_valid_payload(tmp_path: Path) -> None:
    payload = _small_payload(tmp_path)
    target = tmp_path / "docs/experiments/phase-3b-delayed-credit.json"

    write_evidence(payload, target, tmp_path)
    first = target.read_bytes()
    write_evidence(payload, target, tmp_path)

    assert target.read_bytes() == first
    assert json.loads(first)["schema_version"] == 2


def test_portable_projection_ignores_environment_local_parameter_digests(
    tmp_path: Path,
) -> None:
    payload = _small_payload(tmp_path)
    other = copy.deepcopy(payload)

    for index, result in enumerate(other["results"]):
        local_digest = ("a" if index % 2 == 0 else "b") * 64
        result["parameter_digest"] = local_digest
        result["shuffled_parameter_digest"] = ("c" if index % 2 == 0 else "d") * 64
        for checkpoint in result["normal_checkpoints"]:
            checkpoint["parameter_digest"] = "e" * 64
        for checkpoint in result["shuffled_checkpoints"]:
            checkpoint["parameter_digest"] = "f" * 64

    assert _portable_projection(payload) == _portable_projection(other)


def test_portable_projection_keeps_protocol_and_behavior_fields(tmp_path: Path) -> None:
    payload = _small_payload(tmp_path)
    other = copy.deepcopy(payload)
    other["results"][0]["normal_timeline"]["terminal_drain_count"] += 1

    assert _portable_projection(payload) != _portable_projection(other)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "protocol",
        "arm",
        "timeline_digest",
    ],
)
def test_verifier_schema_rejects_protocol_corruption(
    tmp_path: Path, mutation: str
) -> None:
    payload = _small_payload(tmp_path)
    broken = copy.deepcopy(payload)
    if mutation == "schema":
        broken["schema_version"] = 1
    elif mutation == "protocol":
        broken["protocol_valid"] = False
    elif mutation == "arm":
        broken["results"][0]["arm"] = "td_lambda"
    else:
        broken["results"][0]["normal_timeline"]["delivery_timeline_digest"] = "bad"

    with pytest.raises(RuntimeError):
        _validate(broken)
