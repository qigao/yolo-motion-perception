from __future__ import annotations

import hashlib

import numpy as np
import pytest

from neural_state_machine.phase3c_benchmark import (
    AnonymousCreditConfig,
    Phase3CProtocolResult,
    run_phase3c_protocol_gate,
)
from neural_state_machine.phase3c_formal_contract import APPROVED_FORMAL_COMMIT
from scripts.benchmark_phase3c_anonymous_credit import build_payload


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


def _by_arm(results: tuple[Phase3CProtocolResult, ...]) -> dict[str, Phase3CProtocolResult]:
    return {result.arm: result for result in results}


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_protocol_result_has_no_behavioral_surface() -> None:
    results = run_phase3c_protocol_gate(seeds=(7,), config=SMALL)

    assert len(results) == 2
    for result in results:
        assert not hasattr(result, "post_training")
        assert not hasattr(result, "state_reset")
        assert not hasattr(result, "shuffled_control")
        assert not hasattr(result, "behavior_passed")
        assert not hasattr(result, "accuracy")
        assert result.immediate_continuity is True
        assert result.repeatable is True


def test_arms_share_pre_generated_action_and_protocol_lineage() -> None:
    results = _by_arm(run_phase3c_protocol_gate(seeds=(7,), config=SMALL))
    td0 = results["td0"]
    eligibility = results["eligibility"]

    expected_rng = np.random.default_rng(np.random.SeedSequence([7, 0x33414354]))
    expected_actions = tuple(int(expected_rng.integers(2)) for _ in range(SMALL.training_decisions))
    expected_action_digest = hashlib.sha256(bytes(expected_actions)).hexdigest()

    assert td0.action_digest == expected_action_digest
    assert eligibility.action_digest == expected_action_digest
    assert td0.training_fixture_digest == eligibility.training_fixture_digest
    assert td0.evaluation_fixture_digest == eligibility.evaluation_fixture_digest
    assert td0.latent_reward_digest == eligibility.latent_reward_digest
    assert td0.audit.delay_digest == eligibility.audit.delay_digest
    assert td0.audit.due_step_digest == eligibility.audit.due_step_digest
    assert td0.audit.multiplicity_digest == eligibility.audit.multiplicity_digest
    assert td0.audit.aggregate_feedback_digest == eligibility.audit.aggregate_feedback_digest
    assert td0.audit.learner_call_digest == eligibility.audit.learner_call_digest


@pytest.mark.parametrize("seed", [7, 17, 29])
def test_protocol_timeline_is_exactly_once_and_fully_drained(seed: int) -> None:
    results = run_phase3c_protocol_gate(seeds=(seed,), config=SMALL)

    for result in results:
        audit = result.audit
        assert audit.action_count == SMALL.training_decisions
        assert audit.latent_record_count == SMALL.training_decisions
        assert audit.delivered_record_count == SMALL.training_decisions
        assert audit.real_feedback_count == SMALL.training_decisions
        assert audit.drain_feedback_count > 0
        assert audit.queue_pending_final == 0
        assert audit.latent_reward_sum == audit.aggregate_feedback_sum
        assert audit.inversion_count > 0
        assert audit.collision_step_count > 0
        assert audit.source_relabel_invariant is True
        assert audit.hidden_multiplicity_invariant is True
        expected_reset_count = 0 if result.arm == "td0" else 1
        assert audit.trace_reset_count == expected_reset_count


@pytest.mark.parametrize("seed", [7, 17, 29])
def test_registered_seeds_pass_exact_phase3a_immediate_continuity(seed: int) -> None:
    results = run_phase3c_protocol_gate(seeds=(seed,), config=SMALL)

    assert {result.arm for result in results} == {"td0", "eligibility"}
    assert all(result.immediate_continuity is True for result in results)


def test_protocol_gate_is_repeatable_across_public_calls() -> None:
    first = run_phase3c_protocol_gate(seeds=(7,), config=SMALL)
    second = run_phase3c_protocol_gate(seeds=(7,), config=SMALL)

    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"training_decisions": 11},
        {"checkpoint_interval": 3},
        {"delay_support": (0, 1, 3, 5)},
        {"discount": -0.1},
        {"trace_decay": 1.1},
    ],
)
def test_config_rejects_values_that_break_registered_protocol(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "hidden_size": 8,
        "recurrent_radius": 0.9,
        "step_size": 0.1,
        "training_decisions": 10,
        "evaluation_blocks": 1,
        "checkpoint_interval": 10,
        "delay_support": (1, 3, 5),
        "discount": 0.9,
        "trace_decay": 0.8,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        AnonymousCreditConfig(**values)


def test_protocol_cli_payload_binds_formal_gate_and_contains_only_structure() -> None:
    payload = build_payload(seeds=(7,), config=SMALL)

    assert payload["experiment"] == "phase-3c-anonymous-temporal-credit"
    assert payload["schema_version"] == 1
    assert payload["formal_valid"] is True
    assert payload["formal_contract"]["commit"] == APPROVED_FORMAL_COMMIT
    assert payload["protocol_valid"] is True
    assert payload["seeds"] == [7]
    assert payload["delay_support"] == [1, 3, 5]
    assert [(row["seed"], row["arm"]) for row in payload["results"]] == [
        (7, "td0"),
        (7, "eligibility"),
    ]

    forbidden = {
        "behavior_passed",
        "all_passed",
        "post_training",
        "state_reset",
        "shuffled_control",
        "per_delay",
        "reset_per_delay",
        "shuffled_per_delay",
        "accuracy",
    }
    assert _all_keys(payload).isdisjoint(forbidden)


def test_protocol_cli_payload_exposes_required_gate_p_structure() -> None:
    payload = build_payload(seeds=(7,), config=SMALL)

    for row in payload["results"]:
        audit = row["audit"]
        assert row["immediate_continuity"] is True
        assert row["repeatable"] is True
        assert audit["inversion_count"] > 0
        assert audit["collision_step_count"] > 0
        assert audit["latent_record_count"] == SMALL.training_decisions
        assert audit["delivered_record_count"] == SMALL.training_decisions
        assert audit["queue_pending_final"] == 0
        assert audit["source_relabel_invariant"] is True
        assert audit["hidden_multiplicity_invariant"] is True
