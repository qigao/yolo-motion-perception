import numpy as np
import pytest

from neural_state_machine.action_value import NormalizedActionValue


def test_extreme_finite_hidden_state_is_rejected_before_pending_credit() -> None:
    learner = NormalizedActionValue(hidden_size=2, action_count=2)
    before = learner.parameter_snapshot()

    with pytest.raises(ValueError, match="magnitude"):
        learner.select_for_training(
            np.array([1e308, 1.0], dtype=np.float64),
            (0,),
            np.random.default_rng(101),
        )

    assert learner.has_pending_feedback is False
    np.testing.assert_array_equal(learner.parameter_snapshot(), before)


def test_overflowing_finite_reward_update_is_rejected_without_state_corruption() -> None:
    learner = NormalizedActionValue(hidden_size=2, action_count=2)
    hidden = np.array([0.25, -0.5], dtype=np.float64)
    learner._weights[1, 0] = np.finfo(np.float64).max * 0.995
    learner.select_for_training(hidden, (1,), np.random.default_rng(103))
    before = learner.parameter_snapshot()
    before_digest = learner.parameter_digest()

    with pytest.raises(ValueError, match="overflow"):
        learner.learn(1e308)

    assert learner.has_pending_feedback is True
    np.testing.assert_array_equal(learner.parameter_snapshot(), before)
    assert learner.parameter_digest() == before_digest
