from __future__ import annotations

import neural_state_machine.phase3c_benchmark as phase3c_benchmark
import pytest

from neural_state_machine.memory_benchmark import AccuracyCount
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig


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


def _registered_per_delay(correct: int) -> tuple[tuple[int, AccuracyCount], ...]:
    return tuple((delay, AccuracyCount(correct, 40)) for delay in range(1, 6))


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
