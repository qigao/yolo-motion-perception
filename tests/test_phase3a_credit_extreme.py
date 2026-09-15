import numpy as np
import pytest

from neural_state_machine.phase3a_credit_compare import EligibilityTraceActionValue


def test_trace_selection_rejects_overflow_without_retaining_pending_credit() -> None:
    learner = EligibilityTraceActionValue(
        hidden_size=2,
        action_count=2,
        discount=1.0,
        trace_decay=1.0,
    )
    learner._eligibility[0, 0] = np.finfo(np.float64).max

    with pytest.raises(ValueError, match="eligibility trace"):
        learner.select_for_training(
            np.array([0.25, -0.5], dtype=np.float64),
            (0,),
            np.random.default_rng(107),
        )

    assert learner.has_pending_feedback is False
    assert np.isfinite(learner.eligibility_snapshot()).all()


def test_trace_update_rejects_overflow_without_mutating_weights_or_pending_credit() -> None:
    learner = EligibilityTraceActionValue(hidden_size=2, action_count=2)
    learner._eligibility[1, 0] = 1.0
    learner._weights[1, 0] = np.finfo(np.float64).max * 0.995
    learner.select_for_training(
        np.array([0.25, -0.5], dtype=np.float64),
        (1,),
        np.random.default_rng(109),
    )
    before = learner.parameter_snapshot()
    before_digest = learner.parameter_digest()

    with pytest.raises(ValueError, match="eligibility-trace"):
        learner.learn(1e308)

    assert learner.has_pending_feedback is True
    np.testing.assert_array_equal(learner.parameter_snapshot(), before)
    assert learner.parameter_digest() == before_digest
