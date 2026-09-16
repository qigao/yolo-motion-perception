from __future__ import annotations

import hashlib
import inspect
from dataclasses import fields

import numpy as np

import neural_state_machine.phase_c4_benchmark as phase_c4_benchmark
from neural_state_machine.memory_benchmark import AccuracyCount
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_diagnostics.evaluation import evaluation_manifest_rows
from neural_state_machine.phase_c4_benchmark import (
    PhaseC4Config,
    PhaseC4ProtocolResult,
    registered_c4_gate,
    run_phase_c4_protocol_gate,
    secondary_evaluation_manifest,
)


def _small_config() -> PhaseC4Config:
    return PhaseC4Config(
        hidden_size=8,
        recurrent_radius=0.9,
        step_size=0.1,
        training_decisions=20,
        evaluation_blocks=2,
        checkpoint_interval=10,
    )


def _per_delay(correct: int) -> tuple[tuple[int, AccuracyCount], ...]:
    return tuple((delay, AccuracyCount(correct, 40)) for delay in range(1, 6))


def test_config_freezes_registered_defaults():
    config = PhaseC4Config()
    assert config.hidden_size == 64
    assert config.recurrent_radius == 0.9
    assert config.step_size == 0.1
    assert config.training_decisions == 2_000
    assert config.evaluation_blocks == 20
    assert config.checkpoint_interval == 100
    assert config.ridge_penalty == 1e-6
    assert config.delay_support == (1, 3, 5)


def test_protocol_result_surface_contains_no_behavior_measurement():
    names = {field.name for field in fields(PhaseC4ProtocolResult)}
    forbidden = {
        "post_training",
        "state_reset",
        "shuffled_control",
        "operator_passed",
        "behavior_passed",
        "secondary_scores",
        "accuracy",
    }
    assert forbidden.isdisjoint(names)


def test_protocol_gate_uses_fixed_public_horizon_and_is_repeatable():
    first = run_phase_c4_protocol_gate(seeds=(7,), config=_small_config())
    second = run_phase_c4_protocol_gate(seeds=(7,), config=_small_config())

    assert first == second
    assert len(first) == 1
    result = first[0]
    assert result.seed == 7
    assert result.audit.decision_count == 20
    assert result.audit.latent_reward_count == 20
    assert result.audit.delivered_reward_count == 20
    assert result.audit.real_feedback_count == 20
    assert result.audit.drain_feedback_count == 5
    assert result.audit.pending_final == 0
    assert result.audit.current_weight_probe_passed is True
    assert result.audit.bounded_history_passed is True
    assert result.audit.immediate_continuity_passed is True
    assert result.audit.repeatable is True


def test_protocol_action_lineage_is_exact_and_behavior_free():
    result = run_phase_c4_protocol_gate(seeds=(17,), config=_small_config())[0]
    rng = np.random.default_rng(np.random.SeedSequence([17, 0x33414354]))
    expected_actions = tuple(int(rng.integers(2)) for _ in range(20))
    expected_digest = hashlib.sha256(bytes(expected_actions)).hexdigest()

    assert result.action_digest == expected_digest
    source = inspect.getsource(phase_c4_benchmark)
    assert "fit_anonymous_batch_probe" not in source
    assert "phase3c_diagnostics" not in source


def test_secondary_evaluation_manifest_matches_frozen_task12_lineage():
    c4_rows = secondary_evaluation_manifest(PhaseC4Config())
    c3_rows = evaluation_manifest_rows(AnonymousCreditConfig())

    assert len(c4_rows) == 27
    assert c4_rows == c3_rows
    assert len({(row["seed"], row["evaluation_id"]) for row in c4_rows}) == 27


def test_registered_gate_accepts_only_exact_frozen_thresholds():
    assert registered_c4_gate(
        AccuracyCount(180, 200),
        _per_delay(34),
        AccuracyCount(100, 200),
        _per_delay(20),
        AccuracyCount(149, 200),
    )
    assert not registered_c4_gate(
        AccuracyCount(179, 200),
        _per_delay(34),
        AccuracyCount(100, 200),
        _per_delay(20),
        AccuracyCount(149, 200),
    )
    assert not registered_c4_gate(
        AccuracyCount(180, 200),
        ((1, AccuracyCount(33, 40)), *_per_delay(34)[1:]),
        AccuracyCount(100, 200),
        _per_delay(20),
        AccuracyCount(149, 200),
    )
    for reset_correct in (19, 21):
        assert not registered_c4_gate(
            AccuracyCount(180, 200),
            _per_delay(34),
            AccuracyCount(100, 200),
            ((1, AccuracyCount(reset_correct, 40)), *_per_delay(20)[1:]),
            AccuracyCount(149, 200),
        )
    assert not registered_c4_gate(
        AccuracyCount(180, 200),
        _per_delay(34),
        AccuracyCount(100, 200),
        _per_delay(20),
        AccuracyCount(150, 200),
    )


def test_registered_gate_rejects_wrong_totals():
    assert not registered_c4_gate(
        AccuracyCount(180, 199),
        _per_delay(34),
        AccuracyCount(100, 200),
        _per_delay(20),
        AccuracyCount(149, 200),
    )
    malformed = ((1, AccuracyCount(34, 39)), *_per_delay(34)[1:])
    assert not registered_c4_gate(
        AccuracyCount(180, 200),
        malformed,
        AccuracyCount(100, 200),
        _per_delay(20),
        AccuracyCount(149, 200),
    )
