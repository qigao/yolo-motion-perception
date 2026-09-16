from __future__ import annotations

import numpy as np

from neural_state_machine.action_value import NormalizedActionValue
from neural_state_machine.action_value_benchmark import ActionValueBenchmarkConfig
from neural_state_machine.phase3a_credit_compare import (
    EligibilityTraceActionValue,
    credit_comparison_benchmark_payload,
    run_credit_comparison,
    run_credit_comparison_benchmark,
)


def test_trace_learner_matches_action_value_protocol_and_accumulates_trace() -> None:
    learner = EligibilityTraceActionValue(
        hidden_size=2,
        action_count=2,
        step_size=0.1,
        discount=0.9,
        trace_decay=0.8,
    )
    baseline = NormalizedActionValue(hidden_size=2, action_count=2, step_size=0.1)
    rng = np.random.default_rng(7)
    hidden = np.array([0.25, -0.5])
    learner.select_for_training(hidden, (0, 1), rng)
    baseline.select_for_training(hidden, (0, 1), np.random.default_rng(7))
    learner.learn(1.0)
    baseline.learn(1.0)
    first_trace = learner.eligibility_snapshot()
    assert np.any(first_trace != 0.0)

    learner.select_for_training(np.array([-0.5, 0.25]), (0, 1), rng)
    learner.learn(-1.0)
    assert learner.has_pending_feedback is False
    assert learner.parameter_digest() != baseline.parameter_digest()


def test_credit_comparison_keeps_frozen_fixture_and_action_lineages() -> None:
    config = ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=100,
        evaluation_blocks=2,
        checkpoint_interval=50,
    )
    result = run_credit_comparison(seed=7, config=config)

    assert result.action_sequences_equal is True
    assert result.training_fixture_digest == result.lambda_training_fixture_digest
    assert result.evaluation_fixture_digest == result.lambda_evaluation_fixture_digest
    assert result.td0_post_training.total == result.lambda_post_training.total == 20
    assert result.td0_per_delay[0][1].total == result.lambda_per_delay[0][1].total == 4


def test_credit_comparison_benchmark_rejects_unordered_seed_protocol() -> None:
    try:
        run_credit_comparison_benchmark((7, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate seeds must be rejected")


def test_credit_comparison_benchmark_payload_is_diagnostic_only() -> None:
    config = ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=100,
        evaluation_blocks=2,
        checkpoint_interval=50,
    )
    payload = credit_comparison_benchmark_payload(
        run_credit_comparison_benchmark((7,), config)
    )
    assert payload["diagnostic_only"] is True
    assert payload["seeds"] == [7]
    assert payload["results"][0]["td_lambda"]["per_delay"]["5"]["total"] == 4
