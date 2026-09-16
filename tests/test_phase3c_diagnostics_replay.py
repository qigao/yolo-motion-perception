from __future__ import annotations

import pytest

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig, _execute_training
from neural_state_machine.phase3c_diagnostics.replay import (
    D0Failure,
    run_diagnostic_replay,
    validate_call_stream,
)

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
def test_diagnostic_replay_matches_untouched_endpoint(arm: str) -> None:
    observed = run_diagnostic_replay(7, arm, SMALL)
    untouched = _execute_training(7, arm, SMALL, immediate_control=False)
    assert observed.protocol == untouched.protocol
    assert observed.final_parameter_digest == untouched.protocol.parameter_digest
    assert observed.queue_pending_final == 0
    assert len(observed.hidden_bytes) == SMALL.training_decisions
    assert len(observed.action_values) == SMALL.training_decisions
    assert len(observed.scalar_calls) >= SMALL.training_decisions


def test_call_stream_reports_first_corrupted_scalar() -> None:
    with pytest.raises(D0Failure) as raised:
        validate_call_stream((0.0, 1.0, -1.0), (0.0, 2.0, -1.0), attempt_id="a1")
    assert raised.value.path == "scalar_calls[1]"
    assert raised.value.step == 1
