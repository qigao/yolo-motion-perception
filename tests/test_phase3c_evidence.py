from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import neural_state_machine.phase3c_benchmark as phase3c_benchmark
import pytest

from neural_state_machine.memory_benchmark import AccuracyCount
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_formal_contract import (
    APPROVED_FORMAL_COMMIT,
    REQUIRED_THEOREMS,
)
import scripts.benchmark_phase3c_anonymous_credit as benchmark_script


SMALL = AnonymousCreditConfig(
    hidden_size=8,
    recurrent_radius=0.9,
    step_size=0.1,
    training_decisions=10,
    evaluation_blocks=1,
    checkpoint_interval=10,
    delay_support=(1, 3, 5),
    discount=0.9,
    trace_decay=0.8,
)
PHASE3B_BASE_SHA = "a5ab079d56fe569aded44348f9591d226ce83009"
APPROVED_EVIDENCE = Path("docs/experiments/phase-3c-anonymous-temporal-credit.json")


def _registered_per_delay(correct: int) -> tuple[tuple[int, AccuracyCount], ...]:
    return tuple((delay, AccuracyCount(correct, 40)) for delay in range(1, 6))


def _evidence_api() -> tuple[object, object, object]:
    payload_builder = getattr(benchmark_script, "build_measurement_payload", None)
    writer = getattr(benchmark_script, "write_evidence", None)
    approved = getattr(benchmark_script, "_APPROVED", None)
    assert callable(payload_builder)
    assert callable(writer)
    assert approved == APPROVED_EVIDENCE
    return payload_builder, writer, approved


def _verifier_module() -> object:
    try:
        return importlib.import_module("scripts.verify_phase3c_anonymous_credit")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Phase 3C verifier is missing: {exc}")


def _synthetic_registered_payload() -> dict[str, object]:
    payload_builder, _, _ = _evidence_api()
    reduced = payload_builder(seeds=(7,), config=SMALL)
    template_rows = reduced["results"]
    assert isinstance(template_rows, list) and len(template_rows) == 2
    rows: list[dict[str, object]] = []
    for seed in (7, 17, 29):
        for template in template_rows:
            row = copy.deepcopy(template)
            assert isinstance(row, dict)
            row["seed"] = seed
            protocol = row["protocol"]
            assert isinstance(protocol, dict)
            protocol["seed"] = seed
            rows.append(row)
    payload = copy.deepcopy(reduced)
    payload["seeds"] = [7, 17, 29]
    payload["config"] = {
        "hidden_size": 64,
        "recurrent_radius": 0.9,
        "step_size": 0.1,
        "training_decisions": 2000,
        "evaluation_blocks": 20,
        "checkpoint_interval": 100,
        "discount": 0.9,
        "trace_decay": 0.8,
    }
    payload["results"] = rows
    payload["behavior_passed"] = all(bool(row["behavior_passed"]) for row in rows)
    payload["all_passed"] = bool(
        payload["formal_valid"]
        and payload["protocol_valid"]
        and payload["behavior_passed"]
    )
    return payload


def test_measurement_boundary_stops_before_evaluation_when_protocol_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement = getattr(phase3c_benchmark, "run_phase3c_measurement", None)
    assert callable(measurement)

    def fail_protocol(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced protocol invalid")

    def forbidden_evaluate(*args: object, **kwargs: object) -> object:
        raise AssertionError("behavioral evaluation must not run")

    monkeypatch.setattr(phase3c_benchmark, "run_phase3c_protocol_gate", fail_protocol)
    monkeypatch.setattr(phase3c_benchmark, "_evaluate", forbidden_evaluate)

    with pytest.raises(RuntimeError, match="forced protocol invalid"):
        measurement(seed=7, arm="td0", config=SMALL)


def test_reduced_measurement_produces_behavioral_count_shapes() -> None:
    measurement = getattr(phase3c_benchmark, "run_phase3c_measurement", None)
    assert callable(measurement)

    result = measurement(seed=7, arm="td0", config=SMALL)

    assert result.post_training.total == 10
    assert result.state_reset.total == 10
    assert result.shuffled_control.total == 10
    assert tuple(total.total for _, total in result.per_delay) == (2, 2, 2, 2, 2)
    assert tuple(total.total for _, total in result.reset_per_delay) == (2, 2, 2, 2, 2)
    assert tuple(total.total for _, total in result.shuffled_per_delay) == (2, 2, 2, 2, 2)


def test_reduced_shuffled_control_reuses_action_and_schedule_lineage() -> None:
    measurement = getattr(phase3c_benchmark, "run_phase3c_measurement", None)
    assert callable(measurement)

    result = measurement(seed=7, arm="eligibility", config=SMALL)

    assert result.protocol.action_digest == result.shuffled_action_digest
    assert result.protocol.audit.delay_digest == result.shuffled_delay_digest
    assert result.protocol.audit.due_step_digest == result.shuffled_due_step_digest
    assert result.protocol.audit.multiplicity_digest == result.shuffled_multiplicity_digest


def test_reduced_shuffled_control_preserves_reward_multiset_but_changes_assignment() -> None:
    measurement = getattr(phase3c_benchmark, "run_phase3c_measurement", None)
    assert callable(measurement)

    result = measurement(seed=7, arm="td0", config=SMALL)

    assert result.reward_block_multisets_equal is True
    assert result.protocol.latent_reward_digest != result.shuffled_latent_reward_digest


def test_registered_behavior_thresholds_accept_exact_boundaries() -> None:
    predicate = getattr(phase3c_benchmark, "registered_behavior_passed", None)
    assert callable(predicate)

    assert predicate(
        post_training=AccuracyCount(180, 200),
        per_delay=_registered_per_delay(34),
        state_reset=AccuracyCount(100, 200),
        reset_per_delay=_registered_per_delay(20),
        shuffled_control=AccuracyCount(149, 200),
    ) is True


@pytest.mark.parametrize(
    ("overrides"),
    [
        {"post_training": AccuracyCount(179, 200)},
        {"per_delay": ((1, AccuracyCount(33, 40)),) + _registered_per_delay(34)[1:]},
        {"state_reset": AccuracyCount(101, 200)},
        {"reset_per_delay": ((1, AccuracyCount(21, 40)),) + _registered_per_delay(20)[1:]},
        {"shuffled_control": AccuracyCount(150, 200)},
    ],
)
def test_registered_behavior_thresholds_reject_each_boundary_violation(
    overrides: dict[str, object],
) -> None:
    predicate = getattr(phase3c_benchmark, "registered_behavior_passed", None)
    assert callable(predicate)
    values: dict[str, object] = {
        "post_training": AccuracyCount(180, 200),
        "per_delay": _registered_per_delay(34),
        "state_reset": AccuracyCount(100, 200),
        "reset_per_delay": _registered_per_delay(20),
        "shuffled_control": AccuracyCount(149, 200),
    }
    values.update(overrides)

    assert predicate(**values) is False


def test_reduced_measurement_payload_exposes_separate_fail_closed_gates() -> None:
    payload_builder, _, _ = _evidence_api()
    payload = payload_builder(seeds=(7,), config=SMALL)

    assert payload["experiment"] == "phase-3c-anonymous-temporal-credit"
    assert payload["schema_version"] == 1
    assert payload["phase3b_base_sha"] == PHASE3B_BASE_SHA
    assert payload["formal_contract"]["commit"] == APPROVED_FORMAL_COMMIT
    assert payload["formal_contract"]["theorems"] == list(REQUIRED_THEOREMS)
    assert payload["formal_valid"] is True
    assert payload["protocol_valid"] is True
    assert type(payload["behavior_passed"]) is bool
    assert payload["all_passed"] == (
        payload["formal_valid"]
        and payload["protocol_valid"]
        and payload["behavior_passed"]
    )
    assert [(row["seed"], row["arm"]) for row in payload["results"]] == [
        (7, "td0"),
        (7, "eligibility"),
    ]


def test_writer_refuses_invalid_formal_or_protocol_gate(tmp_path: Path) -> None:
    payload_builder, writer, _ = _evidence_api()
    target = tmp_path / APPROVED_EVIDENCE
    base = payload_builder(seeds=(7,), config=SMALL)

    for field in ("formal_valid", "protocol_valid"):
        broken = copy.deepcopy(base)
        broken[field] = False
        with pytest.raises(ValueError, match=field):
            writer(broken, target, tmp_path)
    assert not target.exists()


def test_writer_allows_valid_negative_result_and_is_atomic_deterministic(
    tmp_path: Path,
) -> None:
    payload_builder, writer, _ = _evidence_api()
    payload = payload_builder(seeds=(7,), config=SMALL)
    payload["behavior_passed"] = False
    payload["all_passed"] = False
    target = tmp_path / APPROVED_EVIDENCE

    writer(payload, target, tmp_path)
    first = target.read_bytes()
    writer(payload, target, tmp_path)

    assert target.read_bytes() == first
    decoded = json.loads(first)
    assert decoded["formal_valid"] is True
    assert decoded["protocol_valid"] is True
    assert decoded["behavior_passed"] is False


def test_writer_refuses_any_nonapproved_path(tmp_path: Path) -> None:
    payload_builder, writer, _ = _evidence_api()
    payload = payload_builder(seeds=(7,), config=SMALL)

    with pytest.raises(ValueError, match="approved"):
        writer(payload, tmp_path / "wrong.json", tmp_path)


def test_verifier_accepts_registered_schema_shape() -> None:
    verifier = _verifier_module()
    payload = _synthetic_registered_payload()

    verifier._validate(payload)


def test_portable_projection_excludes_only_environment_local_parameter_digests() -> None:
    verifier = _verifier_module()
    payload = _synthetic_registered_payload()
    other = copy.deepcopy(payload)
    for index, row in enumerate(other["results"]):
        protocol = row["protocol"]
        protocol["parameter_digest"] = ("a" if index % 2 == 0 else "b") * 64
        row["shuffled_parameter_digest"] = ("c" if index % 2 == 0 else "d") * 64

    assert verifier._portable_projection(payload) == verifier._portable_projection(other)


def test_portable_projection_keeps_behavior_and_protocol_evidence() -> None:
    verifier = _verifier_module()
    payload = _synthetic_registered_payload()
    other = copy.deepcopy(payload)
    other["results"][0]["protocol"]["audit"]["delay_digest"] = "f" * 64

    assert verifier._portable_projection(payload) != verifier._portable_projection(other)


@pytest.mark.parametrize(
    "mutation",
    [
        "base_sha",
        "formal_sha",
        "theorems",
        "schema",
        "seeds",
        "config",
        "duplicate_row",
        "digest",
        "protocol_false",
        "all_passed",
    ],
)
def test_verifier_rejects_registered_evidence_corruption(mutation: str) -> None:
    verifier = _verifier_module()
    broken = _synthetic_registered_payload()

    if mutation == "base_sha":
        broken["phase3b_base_sha"] = "0" * 40
    elif mutation == "formal_sha":
        broken["formal_contract"]["commit"] = "0" * 40
    elif mutation == "theorems":
        broken["formal_contract"]["theorems"] = ["wrong"]
    elif mutation == "schema":
        broken["schema_version"] = 2
    elif mutation == "seeds":
        broken["seeds"] = [7, 17]
    elif mutation == "config":
        broken["config"]["training_decisions"] = 1990
    elif mutation == "duplicate_row":
        broken["results"][1] = copy.deepcopy(broken["results"][0])
    elif mutation == "digest":
        broken["results"][0]["protocol"]["audit"]["delay_digest"] = "bad"
    elif mutation == "protocol_false":
        broken["protocol_valid"] = False
        broken["behavior_passed"] = False
        broken["all_passed"] = False
    else:
        broken["all_passed"] = not broken["all_passed"]

    with pytest.raises(RuntimeError):
        verifier._validate(broken)
