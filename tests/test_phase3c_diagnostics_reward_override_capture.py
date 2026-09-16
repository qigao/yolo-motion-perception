from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig, _execute_training
from neural_state_machine.phase3c_diagnostics import replay


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


@pytest.mark.parametrize("arm", ["td0", "eligibility"])
def test_reward_override_capture_matches_untouched_endpoint_and_exposes_actual_calls(
    arm: str,
) -> None:
    baseline = _execute_training(7, arm, SMALL, immediate_control=False)
    reward_override = tuple(reversed(baseline.protocol.rewards))
    expected = _execute_training(
        7,
        arm,
        SMALL,
        immediate_control=False,
        reward_override=reward_override,
    )

    capture = getattr(replay, "capture_reward_override_training", None)
    assert capture is not None, "D1 reward-override capture must be implemented"
    observed = capture(
        7,
        arm,
        SMALL,
        reward_override=reward_override,
        attempt_id="d1-test",
    )

    assert observed.protocol == expected.protocol
    assert observed.final_parameter_digest == expected.protocol.parameter_digest
    assert observed.queue_pending_final == 0
    assert tuple(step.reward for step in observed.steps) == reward_override
    assert observed.scalar_calls == tuple(
        step.feedback for step in (*observed.steps, *observed.drain_steps)
    )
    assert len(observed.scalar_calls) == (
        SMALL.training_decisions + expected.protocol.audit.drain_feedback_count
    )


@pytest.mark.parametrize("arm", ["td0", "eligibility"])
def test_single_pass_capture_returns_the_same_endpoint_learner_for_scoring(arm: str) -> None:
    baseline = _execute_training(7, arm, SMALL, immediate_control=False)
    reward_override = tuple(reversed(baseline.protocol.rewards))
    expected = _execute_training(
        7,
        arm,
        SMALL,
        immediate_control=False,
        reward_override=reward_override,
    )

    capture = getattr(replay, "capture_reward_override_execution", None)
    assert capture is not None, "single-pass D1 capture endpoint must be implemented"
    observed = capture(
        7,
        arm,
        SMALL,
        reward_override=reward_override,
        attempt_id="d1-single-pass",
    )

    assert observed.replay.protocol == expected.protocol
    assert observed.learner.parameter_digest() == expected.protocol.parameter_digest
    np.testing.assert_array_equal(
        observed.learner.parameter_snapshot(),
        expected.learner.parameter_snapshot(),
    )
