from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neural_state_machine.action_value import NormalizedActionValue
from neural_state_machine.phase3b_learners import DelayedTD0Adapter
from neural_state_machine.phase3c_learners import (
    AnonymousCreditUpdate,
    AnonymousCurrentStepTD0,
    NormalizedAnonymousEligibilityCredit,
)


def _seeded_pair(cls, *args, seed: int = 101, **kwargs):
    left = cls(*args, **kwargs)
    right = NormalizedActionValue(*args[:2], step_size=kwargs.get("step_size", 0.1))
    return left, right, np.random.default_rng(seed), np.random.default_rng(seed)


def test_arm_a_one_step_is_byte_exact_phase3a() -> None:
    arm, baseline, arm_rng, base_rng = _seeded_pair(
        AnonymousCurrentStepTD0, 2, 2, step_size=0.1
    )
    hidden = np.array([0.25, -0.5], dtype=np.float64)

    arm_decision = arm.select_for_training(hidden, (0, 1), arm_rng)
    base_decision = baseline.select_for_training(hidden, (0, 1), base_rng)
    arm_update = arm.learn(1.0)
    base_update = baseline.learn(1.0)

    assert arm_decision.action_index == base_decision.action_index
    assert arm_decision.action_values.tobytes() == base_decision.action_values.tobytes()
    assert arm.parameter_snapshot().tobytes() == baseline.parameter_snapshot().tobytes()
    assert arm.parameter_digest() == baseline.parameter_digest()
    assert arm_update.reward == base_update.reward
    assert arm_update.prediction == base_update.prediction_before
    assert arm_update.td_error == base_update.td_error
    assert arm_update.applied is True


def test_arm_a_has_no_fifo_history_and_drain_is_noop() -> None:
    arm = AnonymousCurrentStepTD0(2, 2)
    assert not isinstance(arm, DelayedTD0Adapter)
    assert not hasattr(arm, "_credits")

    before = arm.parameter_snapshot()
    update = arm.learn_drain(1.0)

    assert isinstance(update, AnonymousCreditUpdate)
    assert update.applied is False
    assert arm.parameter_snapshot().tobytes() == before.tobytes()


def test_arm_a_zero_aggregate_is_real_feedback() -> None:
    arm = AnonymousCurrentStepTD0(2, 2)
    arm._weights[:] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    before = arm.parameter_snapshot()
    arm.select_for_training(np.array([1.0, 0.0]), (0,), np.random.default_rng(7))
    update = arm.learn(0.0)

    assert update.applied is True
    assert update.reward == 0.0
    assert arm.parameter_snapshot().tobytes() != before.tobytes()


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf")])
def test_arm_a_invalid_feedback_is_atomic(bad: object) -> None:
    arm = AnonymousCurrentStepTD0(2, 2)
    arm.select_for_training(np.array([0.2, -0.1]), (1,), np.random.default_rng(11))
    before = arm.parameter_snapshot()

    with pytest.raises(ValueError, match="reward"):
        arm.learn(bad)

    assert arm.parameter_snapshot().tobytes() == before.tobytes()
    assert arm.has_pending_feedback is True


def test_arm_b_accumulates_decision_based_trace_and_prediction() -> None:
    arm = NormalizedAnonymousEligibilityCredit(
        1, 2, step_size=0.1, discount=0.9, trace_decay=0.8
    )
    arm._weights[:] = np.array([[0.5, 0.25], [-0.4, 0.1]], dtype=np.float64)

    hidden0 = np.array([1.0])
    feature0 = np.array([1.0, 1.0])
    q0 = float(np.dot(arm._weights[0], feature0))
    arm.select_for_training(hidden0, (0,), np.random.default_rng(13))
    expected_e0 = np.zeros_like(arm.parameter_snapshot())
    expected_e0[0] = feature0 / float(np.dot(feature0, feature0))
    np.testing.assert_allclose(arm.eligibility_snapshot(), expected_e0, rtol=0, atol=0)
    assert arm.prediction_trace == q0

    arm.learn(0.25)
    retained_e = arm.eligibility_snapshot()
    retained_p = arm.prediction_trace

    hidden1 = np.array([0.0])
    feature1 = np.array([0.0, 1.0])
    q1 = float(np.dot(arm._weights[1], feature1))
    arm.select_for_training(hidden1, (1,), np.random.default_rng(17))
    expected_e1 = retained_e * 0.72
    expected_e1[1] += feature1 / float(np.dot(feature1, feature1))

    np.testing.assert_allclose(arm.eligibility_snapshot(), expected_e1, rtol=0, atol=1e-15)
    assert arm.prediction_trace == pytest.approx(0.72 * retained_p + q1, abs=1e-15)


def test_arm_b_feedback_does_not_clear_trace() -> None:
    arm = NormalizedAnonymousEligibilityCredit(1, 2)
    arm.select_for_training(np.array([0.5]), (0,), np.random.default_rng(19))
    before_e = arm.eligibility_snapshot()
    before_p = arm.prediction_trace

    arm.learn(1.0)

    assert arm.eligibility_snapshot().tobytes() == before_e.tobytes()
    assert arm.prediction_trace == before_p


def test_arm_b_same_scalar_is_indistinguishable_from_collision_origin() -> None:
    left = NormalizedAnonymousEligibilityCredit(1, 2)
    right = NormalizedAnonymousEligibilityCredit(1, 2)
    left_rng = np.random.default_rng(23)
    right_rng = np.random.default_rng(23)
    hidden = np.array([0.25])

    left.select_for_training(hidden, (0, 1), left_rng)
    right.select_for_training(hidden, (0, 1), right_rng)
    left.learn(0.0)
    right.learn(+1.0 + -1.0)

    assert left.parameter_digest() == right.parameter_digest()
    assert left.eligibility_snapshot().tobytes() == right.eligibility_snapshot().tobytes()
    assert left.prediction_trace == right.prediction_trace


def test_arm_b_drain_uses_persistent_trace_without_decay() -> None:
    arm = NormalizedAnonymousEligibilityCredit(1, 2, step_size=0.1)
    arm.select_for_training(np.array([1.0]), (0,), np.random.default_rng(29))
    arm.learn(0.0)
    trace_before = arm.eligibility_snapshot()
    prediction_before = arm.prediction_trace
    weights_before = arm.parameter_snapshot()

    update = arm.learn_drain(1.0)

    assert update.applied is True
    assert arm.eligibility_snapshot().tobytes() == trace_before.tobytes()
    assert arm.prediction_trace == prediction_before
    assert arm.parameter_snapshot().tobytes() != weights_before.tobytes()


def test_arm_b_has_no_episode_reset_surface_and_end_run_clears_once() -> None:
    arm = NormalizedAnonymousEligibilityCredit(1, 2)
    assert not hasattr(arm, "reset_episode")
    assert arm.trace_reset_count == 0
    arm.select_for_training(np.array([0.5]), (0,), np.random.default_rng(31))
    arm.learn(0.0)

    arm.end_run()

    assert arm.trace_reset_count == 1
    assert np.array_equal(arm.eligibility_snapshot(), np.zeros((2, 2)))
    assert arm.prediction_trace == 0.0
    with pytest.raises(RuntimeError):
        arm.end_run()
    with pytest.raises(RuntimeError):
        arm.select_for_training(np.array([0.5]), (0,), np.random.default_rng(31))


def test_arm_b_cannot_end_run_with_real_step_feedback_pending() -> None:
    arm = NormalizedAnonymousEligibilityCredit(1, 2)
    arm.select_for_training(np.array([0.5]), (0,), np.random.default_rng(37))

    with pytest.raises(RuntimeError, match="pending"):
        arm.end_run()


def test_both_arms_rho_zero_immediate_boundary_is_byte_exact_phase3a() -> None:
    hidden = np.array([0.125, -0.375], dtype=np.float64)
    reward = -1.0
    for learner in (
        AnonymousCurrentStepTD0(2, 2, step_size=0.1),
        NormalizedAnonymousEligibilityCredit(
            2, 2, step_size=0.1, discount=0.0, trace_decay=0.8
        ),
    ):
        baseline = NormalizedActionValue(2, 2, step_size=0.1)
        learner_rng = np.random.default_rng(41)
        baseline_rng = np.random.default_rng(41)

        learner_decision = learner.select_for_training(hidden, (0, 1), learner_rng)
        baseline_decision = baseline.select_for_training(hidden, (0, 1), baseline_rng)
        learner_update = learner.learn(reward)
        baseline_update = baseline.learn(reward)

        assert learner_decision.action_index == baseline_decision.action_index
        assert learner.parameter_snapshot().tobytes() == baseline.parameter_snapshot().tobytes()
        assert learner.parameter_digest() == baseline.parameter_digest()
        assert learner_update.reward == baseline_update.reward
        assert learner_update.prediction == baseline_update.prediction_before
        assert learner_update.td_error == baseline_update.td_error


def test_update_dataclass_is_frozen() -> None:
    update = AnonymousCreditUpdate(1.0, 0.0, 1.0, True)
    with pytest.raises(FrozenInstanceError):
        update.applied = False
