from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from neural_state_machine.action_value import NormalizedActionValue
from neural_state_machine.phase_c4_delay_model import DelayLaw
from neural_state_machine.phase_c4_learner import (
    DelayMarginalizedAnonymousCredit,
    MarginalizedCreditUpdate,
)


def _drive_zero_step(
    learner: DelayMarginalizedAnonymousCredit,
    hidden: float,
    action: int,
    seed: int,
) -> MarginalizedCreditUpdate:
    learner.select_for_training(
        np.array([hidden], dtype=np.float64),
        (action,),
        np.random.default_rng(seed),
    )
    return learner.learn(0.0)


def _prepared_registered_learner() -> DelayMarginalizedAnonymousCredit:
    learner = DelayMarginalizedAnonymousCredit(1, 2, step_size=0.1)
    for index, (hidden, action) in enumerate(
        ((2.0, 1), (9.0, 0), (3.0, 0), (8.0, 1), (4.0, 0))
    ):
        _drive_zero_step(learner, hidden, action, 100 + index)
    learner.select_for_training(
        np.array([7.0], dtype=np.float64),
        (1,),
        np.random.default_rng(200),
    )
    return learner


def _assert_history_equal(left, right) -> None:
    assert len(left) == len(right)
    for left_row, right_row in zip(left, right, strict=True):
        assert left_row.decision_index == right_row.decision_index
        assert left_row.action_index == right_row.action_index
        assert left_row.denominator == right_row.denominator
        np.testing.assert_array_equal(left_row.feature, right_row.feature)


def test_registered_learner_starts_zero_with_bounded_anonymous_state():
    learner = DelayMarginalizedAnonymousCredit(2, 2)
    assert learner.law == DelayLaw.registered()
    assert learner.has_pending_feedback is False
    assert learner.history_snapshot() == ()
    np.testing.assert_array_equal(learner.parameter_snapshot(), np.zeros((2, 3)))

    forbidden = {
        "_eligibility",
        "_prediction_trace",
        "_credits",
        "_source_ids",
        "_delays",
        "_due_steps",
        "_multiplicities",
        "_latent_rewards",
    }
    assert forbidden.isdisjoint(vars(learner))


def test_registered_prediction_uses_current_weights_at_feedback_time():
    left = _prepared_registered_learner()
    right = _prepared_registered_learner()

    left_weights = np.array([[0.5, 1.0], [-0.2, 0.3]], dtype=np.float64)
    right_weights = np.array([[1.5, 1.0], [-0.2, 0.3]], dtype=np.float64)
    left._weights[:] = left_weights
    right._weights[:] = right_weights

    update_left = left.learn(0.0)
    update_right = right.learn(0.0)

    z = np.zeros((2, 2), dtype=np.float64)
    z[0] = (np.array([4.0, 1.0]) + np.array([3.0, 1.0])) / 3.0
    z[1] = np.array([2.0, 1.0]) / 3.0
    expected_left = float(np.sum(left_weights * z))
    expected_right = float(np.sum(right_weights * z))

    assert update_left.feedback_step == 5
    assert update_right.feedback_step == 5
    assert update_left.prediction == pytest.approx(expected_left, rel=0.0, abs=1e-15)
    assert update_right.prediction == pytest.approx(expected_right, rel=0.0, abs=1e-15)
    assert update_right.prediction - update_left.prediction == pytest.approx(
        expected_right - expected_left, rel=0.0, abs=1e-15
    )


def test_registered_history_retains_only_last_five_completed_decisions():
    learner = DelayMarginalizedAnonymousCredit(1, 2)
    for index in range(12):
        _drive_zero_step(learner, float(index), index % 2, 300 + index)

    history = learner.history_snapshot()
    assert tuple(row.decision_index for row in history) == (7, 8, 9, 10, 11)
    assert all(not row.feature.flags.writeable for row in history)
    assert {field.name for field in fields(history[0])} == {
        "decision_index",
        "action_index",
        "feature",
        "denominator",
    }


def test_registered_drain_ages_history_without_synthetic_decisions():
    learner = DelayMarginalizedAnonymousCredit(1, 2)
    for index in range(4):
        _drive_zero_step(learner, float(index + 1), index % 2, 400 + index)

    assert tuple(row.decision_index for row in learner.history_snapshot()) == (0, 1, 2, 3)
    expected_after = (
        (0, 1, 2, 3),
        (1, 2, 3),
        (2, 3),
        (3,),
        (),
    )
    for offset, expected_indices in enumerate(expected_after):
        update = learner.learn_drain(0.0)
        assert update.feedback_step == 4 + offset
        assert tuple(row.decision_index for row in learner.history_snapshot()) == expected_indices
        assert learner.has_pending_feedback is False


def test_drain_rejects_pending_real_decision_atomically():
    learner = DelayMarginalizedAnonymousCredit(1, 2)
    learner.select_for_training(
        np.array([0.5]),
        (0,),
        np.random.default_rng(17),
    )
    before = learner.parameter_snapshot()
    history = learner.history_snapshot()

    with pytest.raises(RuntimeError, match="current decision"):
        learner.learn_drain(0.0)

    assert learner.parameter_snapshot().tobytes() == before.tobytes()
    _assert_history_equal(learner.history_snapshot(), history)
    assert learner.has_pending_feedback is True


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), object()])
def test_invalid_feedback_preserves_online_state(bad: object):
    learner = DelayMarginalizedAnonymousCredit(1, 2)
    learner.select_for_training(
        np.array([0.25]),
        (1,),
        np.random.default_rng(19),
    )
    before_weights = learner.parameter_snapshot()
    before_history = learner.history_snapshot()

    with pytest.raises(ValueError, match="reward"):
        learner.learn(bad)

    assert learner.parameter_snapshot().tobytes() == before_weights.tobytes()
    _assert_history_equal(learner.history_snapshot(), before_history)
    assert learner.has_pending_feedback is True


def test_same_scalar_feedback_is_indistinguishable_from_collision_origin():
    left = DelayMarginalizedAnonymousCredit(1, 2)
    right = DelayMarginalizedAnonymousCredit(1, 2)
    for index in range(4):
        for learner in (left, right):
            learner.select_for_training(
                np.array([float(index + 1)]),
                (index % 2,),
                np.random.default_rng(500 + index),
            )
        left.learn(0.0)
        right.learn(+1.0 + -1.0)

    assert left.parameter_digest() == right.parameter_digest()
    _assert_history_equal(left.history_snapshot(), right.history_snapshot())


def test_immediate_boundary_is_byte_exact_phase3a_across_multiple_steps():
    learner = DelayMarginalizedAnonymousCredit(
        2,
        2,
        step_size=0.1,
        law=DelayLaw.immediate(),
    )
    baseline = NormalizedActionValue(2, 2, step_size=0.1)
    learner_rng = np.random.default_rng(41)
    baseline_rng = np.random.default_rng(41)
    cases = (
        (np.array([0.125, -0.375]), -1.0),
        (np.array([0.5, 0.25]), 1.0),
        (np.array([-0.75, 0.125]), -1.0),
        (np.array([0.0, 0.5]), 1.0),
    )

    for step, (hidden, reward) in enumerate(cases):
        learner_decision = learner.select_for_training(hidden, (0, 1), learner_rng)
        baseline_decision = baseline.select_for_training(hidden, (0, 1), baseline_rng)
        learner_update = learner.learn(reward)
        baseline_update = baseline.learn(reward)

        assert learner_decision.action_index == baseline_decision.action_index
        assert learner_decision.action_values.tobytes() == baseline_decision.action_values.tobytes()
        assert learner_update.feedback_step == step
        assert learner_update.reward == baseline_update.reward
        assert learner_update.prediction == baseline_update.prediction_before
        assert learner_update.td_error == baseline_update.td_error
        assert learner.parameter_snapshot().tobytes() == baseline.parameter_snapshot().tobytes()
        assert learner.parameter_digest() == baseline.parameter_digest()
        assert learner.history_snapshot() == ()


def test_update_dataclass_is_frozen():
    update = MarginalizedCreditUpdate(0, 1.0, 0.0, 1.0, True)
    with pytest.raises(FrozenInstanceError):
        update.applied = False
