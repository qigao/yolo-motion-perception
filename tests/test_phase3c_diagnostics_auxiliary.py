from __future__ import annotations

from dataclasses import replace

import numpy as np

from neural_state_machine.action_value_benchmark import _build_fixture_bundle
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig, _execute_training
from neural_state_machine.phase3c_diagnostics.auxiliary import (
    drain_weight_snapshots,
    learner_view_digest,
    registered_auxiliary_keys,
    replay_with_metadata,
    score_weight_snapshot,
)
from neural_state_machine.phase3c_diagnostics.replay import run_diagnostic_replay


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


def test_registered_auxiliary_key_grid_is_complete_and_unique() -> None:
    keys = registered_auxiliary_keys(AnonymousCreditConfig())
    assert len(keys.reset_score_keys) == 189
    assert len(set(keys.reset_score_keys)) == 189
    assert len(keys.reverse_check_keys) == 108
    assert len(set(keys.reverse_check_keys)) == 108
    assert keys.drain_score_keys
    assert len(keys.drain_score_keys) == len(set(keys.drain_score_keys))


def test_metadata_only_variants_leave_learner_view_digest_exact() -> None:
    replay = run_diagnostic_replay(7, "td0", SMALL)
    base = learner_view_digest(replay)
    relabeled = tuple(
        {
            **row,
            "multiplicity": int(row["multiplicity"]) + 1000,
            "records": tuple(
                {**record, "source_step": int(record["source_step"]) + 10000}
                for record in row["records"]
            ),
        }
        for row in replay.metadata
    )
    assert learner_view_digest(replay_with_metadata(replay, relabeled)) == base
    deleted = tuple({"delivery_step": row["delivery_step"]} for row in replay.metadata)
    assert learner_view_digest(replay_with_metadata(replay, deleted)) == base
    reversed_records = tuple(
        {**row, "records": tuple(reversed(row["records"]))}
        for row in replay.metadata
    )
    assert learner_view_digest(replay_with_metadata(replay, reversed_records)) == base

    corrupted = replace(
        replay,
        scalar_calls=(replay.scalar_calls[0] + 1.0, *replay.scalar_calls[1:]),
    )
    assert learner_view_digest(corrupted) != base


def test_drain_snapshot_lifecycle_is_exact() -> None:
    for arm in ("td0", "eligibility"):
        replay = run_diagnostic_replay(7, arm, SMALL)
        snapshots = drain_weight_snapshots(replay, arm)
        labels = tuple(label for label, _ in snapshots)
        assert labels[0] == "pre_drain"
        assert labels[1 : 1 + len(replay.drain_steps)] == tuple(
            f"after_drain_{step.step}" for step in replay.drain_steps
        )
        assert len(snapshots) == 1 + len(replay.drain_steps) + int(arm == "eligibility")
        if arm == "eligibility":
            assert labels[-1] == "post_end_run"
        np.testing.assert_array_equal(snapshots[0][1], replay.drain_steps[0].weights_before)
        for (_, weights), step in zip(
            snapshots[1 : 1 + len(replay.drain_steps)], replay.drain_steps, strict=True
        ):
            np.testing.assert_array_equal(weights, step.weights_after)


def test_weight_snapshot_scoring_is_order_invariant_and_nonmutating() -> None:
    execution = _execute_training(7, "eligibility", SMALL, immediate_control=False)
    weights = execution.learner.parameter_snapshot()
    before = np.array(weights, copy=True)
    fixtures = _build_fixture_bundle(7, SMALL.action_value_config).evaluation
    score = score_weight_snapshot(7, weights, fixtures, SMALL)

    assert score.total == len(fixtures)
    assert score.canonical_actions == score.reversed_actions_mapped
    assert 0 <= score.reset_correct <= score.total
    np.testing.assert_array_equal(weights, before)
