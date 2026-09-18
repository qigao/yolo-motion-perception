from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.r1_e1_yolo_like import (
    BEHAVIOR_CLASSES,
    CORRUPTION_ARMS,
    CleanFixtureSet,
    CorruptionPlan,
    build_clean_fixture_set,
    build_corruption_plan,
)
from neural_state_machine.r1_e2_reservoir import E2Architecture, E2ReservoirSpec


def _api():
    from neural_state_machine.r1_e2_yolo_episode import (
        E2_EPISODE_SEEDS,
        EPISODE_CLASSES,
        EPISODE_CORRUPTIONS,
        EPISODE_TEMPORAL_WINDOW,
        build_e2_episode_corruption_plan,
        build_e2_episode_fixture_set,
        evaluate_yolo_episode_set,
        run_yolo_episode_arm,
    )

    return (
        E2_EPISODE_SEEDS,
        EPISODE_CLASSES,
        EPISODE_CORRUPTIONS,
        EPISODE_TEMPORAL_WINDOW,
        build_e2_episode_fixture_set,
        build_e2_episode_corruption_plan,
        evaluate_yolo_episode_set,
        run_yolo_episode_arm,
    )


def test_e2c_registry_is_frozen() -> None:
    seeds, classes, corruptions, window, *_ = _api()

    assert seeds == (7, 17, 29, 43, 61)
    assert classes == BEHAVIOR_CLASSES == ("approach", "touch", "pick_up", "pass_by")
    assert corruptions == CORRUPTION_ARMS == (
        "clean",
        "drop10",
        "wrong10",
        "occlusion4",
        "jitter",
        "mixed",
    )
    assert window == 4


def test_e2c_fixture_adapter_is_exactly_frozen_e1c_fixture() -> None:
    *_, build_fixture, _, _, _ = _api()

    for training in (True, False):
        e2 = build_fixture(17, training=training)
        reference = build_clean_fixture_set(17, training=training)

        assert e2.fixture_digest == reference.fixture_digest
        assert len(e2.sequences) == len(reference.sequences)
        for left, right in zip(e2.sequences[:4], reference.sequences[:4], strict=True):
            assert left.label == right.label
            assert left.behavior == right.behavior
            assert left.pair_index == right.pair_index
            assert left.fixture_index == right.fixture_index
            np.testing.assert_array_equal(left.frames, right.frames)


def test_e2c_corruption_adapter_is_exactly_frozen_e1c_plan() -> None:
    *_, build_fixture, build_plan, _, _ = _api()
    evaluation = build_fixture(29, training=False)

    e2 = build_plan(29, evaluation)
    reference = build_corruption_plan(29, evaluation)

    assert e2.digest == reference.digest
    assert len(e2.entries) == len(reference.entries) == 200
    assert e2.entries[:4] == reference.entries[:4]


def test_e2c_small_paired_group_exercises_all_three_baselines_without_full_measurement() -> None:
    *_, build_fixture, build_plan, evaluate, _ = _api()
    training_full = build_fixture(7, training=True)
    evaluation_full = build_fixture(7, training=False)
    plan_full = build_plan(7, evaluation_full)

    training = CleanFixtureSet(
        sequences=training_full.sequences[:4],
        fixture_digest=training_full.fixture_digest,
    )
    evaluation = CleanFixtureSet(
        sequences=evaluation_full.sequences[:4],
        fixture_digest=evaluation_full.fixture_digest,
    )
    plan = CorruptionPlan(
        entries=plan_full.entries[:4],
        digest=plan_full.digest,
    )

    result = evaluate(
        E2ReservoirSpec(
            architecture=E2Architecture.FLAT,
            input_size=9,
            seed=7,
        ),
        training,
        evaluation,
        plan,
    )

    assert result.reset_groups_equal is True
    assert result.frame_only.coefficient_digest
    assert result.reservoir_instantaneous.coefficient_digest
    assert result.reservoir_temporal_mean.coefficient_digest
    assert result.reservoir_parameter_digest

    for baseline in (
        result.frame_only,
        result.reservoir_instantaneous,
        result.reservoir_temporal_mean,
    ):
        assert tuple(name for name, _ in baseline.arm_metrics) == CORRUPTION_ARMS
        assert all(metrics.total == 4 for _, metrics in baseline.arm_metrics)
        assert len(baseline.prediction_digests) == len(CORRUPTION_ARMS)

    # B0 sees only frame 19, which is group-identical and corruption-invariant.
    assert all(metrics.correct == 1 for _, metrics in result.frame_only.arm_metrics)

    # Reset before frame 16 removes active-episode memory for B1/B2.
    assert result.reservoir_instantaneous.reset_correct == 1
    assert result.reservoir_instantaneous.reset_total == 4
    assert result.reservoir_temporal_mean.reset_correct == 1
    assert result.reservoir_temporal_mean.reset_total == 4


def test_e2c_frame_only_predictions_are_corruption_invariant_on_paired_group() -> None:
    *_, build_fixture, build_plan, evaluate, _ = _api()
    training_full = build_fixture(17, training=True)
    evaluation_full = build_fixture(17, training=False)
    plan_full = build_plan(17, evaluation_full)
    result = evaluate(
        E2ReservoirSpec(
            architecture=E2Architecture.GROUPED4,
            input_size=9,
            seed=17,
        ),
        CleanFixtureSet(training_full.sequences[:4], training_full.fixture_digest),
        CleanFixtureSet(evaluation_full.sequences[:4], evaluation_full.fixture_digest),
        CorruptionPlan(plan_full.entries[:4], plan_full.digest),
    )

    digests = tuple(digest for _, digest in result.frame_only.prediction_digests)
    assert len(set(digests)) == 1


def test_registered_e2c_runner_rejects_nonregistered_counts_and_seed_lineage() -> None:
    *_, build_fixture, build_plan, _, run = _api()
    training = build_fixture(7, training=True)
    evaluation = build_fixture(7, training=False)
    plan = build_plan(7, evaluation)

    with pytest.raises(ValueError, match="registered training count"):
        run(
            E2ReservoirSpec(
                architecture=E2Architecture.FLAT,
                input_size=9,
                seed=7,
            ),
            CleanFixtureSet(training.sequences[:4], training.fixture_digest),
            evaluation,
            plan,
        )

    with pytest.raises(ValueError, match="seed"):
        run(
            E2ReservoirSpec(
                architecture=E2Architecture.FLAT,
                input_size=9,
                seed=17,
            ),
            training,
            evaluation,
            plan,
        )


def test_e2c_runner_rejects_wrong_input_width() -> None:
    *_, build_fixture, build_plan, _, run = _api()
    training = build_fixture(7, training=True)
    evaluation = build_fixture(7, training=False)
    plan = build_plan(7, evaluation)

    with pytest.raises(ValueError, match="input_size"):
        run(
            E2ReservoirSpec(
                architecture=E2Architecture.FLAT,
                input_size=8,
                seed=7,
            ),
            training,
            evaluation,
            plan,
        )
