import numpy as np
import pytest

from neural_state_machine.action_value import NormalizedActionValue
from neural_state_machine.phase3b_learners import (
    DelayedTD0Adapter,
    EpisodeResetEligibilityTrace,
)


def test_td0_adapter_preserves_normalized_action_value_updates() -> None:
    adapter = DelayedTD0Adapter(hidden_size=2, action_count=2, step_size=0.1)
    baseline = NormalizedActionValue(hidden_size=2, action_count=2, step_size=0.1)
    hidden = np.array([0.25, -0.5], dtype=np.float64)
    adapter.select_for_training(hidden, (1,), np.random.default_rng(113))
    baseline.select_for_training(hidden, (1,), np.random.default_rng(113))
    adapter.learn(1.0)
    baseline.learn(1.0)
    np.testing.assert_array_equal(adapter.parameter_snapshot(), baseline.parameter_snapshot())
    assert adapter.parameter_digest() == baseline.parameter_digest()
    assert adapter.has_pending_feedback is False


def test_episode_reset_trace_clears_only_eligibility() -> None:
    learner = EpisodeResetEligibilityTrace(
        hidden_size=2,
        action_count=2,
        discount=0.9,
        trace_decay=0.8,
    )
    learner.select_for_training(
        np.array([0.25, -0.5], dtype=np.float64),
        (0,),
        np.random.default_rng(127),
    )
    learner.learn(1.0)
    before_weights = learner.parameter_snapshot()
    assert np.any(learner.eligibility_snapshot() != 0.0)

    learner.reset_episode()

    np.testing.assert_array_equal(learner.eligibility_snapshot(), np.zeros((2, 3)))
    np.testing.assert_array_equal(learner.parameter_snapshot(), before_weights)
    assert learner.has_pending_feedback is False


def test_episode_reset_rejects_pending_feedback() -> None:
    learner = EpisodeResetEligibilityTrace(hidden_size=2, action_count=2)
    learner.select_for_training(
        np.array([0.25, -0.5], dtype=np.float64),
        (0,),
        np.random.default_rng(131),
    )
    with pytest.raises(RuntimeError, match="pending"):
        learner.reset_episode()
    assert learner.has_pending_feedback is True
