from __future__ import annotations

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_diagnostics import runner_core
from neural_state_machine.phase3c_diagnostics.provenance import (
    apply_permutation,
    diagnostic_permutation,
    identity_permutation,
)
from neural_state_machine.phase3c_diagnostics.replay import (
    capture_reward_override_execution,
    run_diagnostic_replay,
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


def _helper():
    helper = getattr(runner_core, "_d1_audit_for_replay", None)
    assert helper is not None, "runner must bind D1 audit to the captured replay"
    return helper


def test_normal_d1_row_uses_actual_replay_calls_and_training_labels() -> None:
    replay = run_diagnostic_replay(7, "td0", SMALL)
    audit = _helper()(
        7,
        replay,
        identity_permutation(SMALL.training_decisions),
        replay.protocol.rewards,
        SMALL,
    )

    assert len(audit["provenance_rows"]) == SMALL.training_decisions
    assert audit["donor_summary"]["fixed_points"] == SMALL.training_decisions
    assert audit["call_classes"]["real_decision_calls"] == SMALL.training_decisions
    assert audit["lags"][0]["feedback_reward_product"]["sample_count"] == 10
    assert audit["provenance_rows"][0]["slot_label"] in (0, 1)


def test_permutation_d1_row_uses_the_same_capture_that_produces_endpoint_weights() -> None:
    normal = run_diagnostic_replay(7, "td0", SMALL)
    permutation = diagnostic_permutation(7, SMALL.training_decisions, "block10", 0)
    reward_override = apply_permutation(normal.protocol.rewards, permutation)
    captured = capture_reward_override_execution(
        7,
        "td0",
        SMALL,
        reward_override=reward_override,
        attempt_id="d1-row",
    )

    audit = _helper()(7, captured.replay, permutation, normal.protocol.rewards, SMALL)

    assert captured.learner.parameter_digest() == captured.replay.final_parameter_digest
    assert len(audit["provenance_rows"]) == SMALL.training_decisions
    assert tuple(row["delivered_reward"] for row in audit["provenance_rows"]) == reward_override
    assert audit["donor_summary"]["same_block_donors"] == SMALL.training_decisions
