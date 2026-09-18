from __future__ import annotations

import numpy as np
import pytest


def _api():
    from neural_state_machine.r1_e2_composition import (
        COMPOSITION_CLASSES,
        COMPOSITION_HISTORIES,
        E2_COMPOSITION_SEEDS,
        EVAL_GROUPS_PER_HISTORY,
        TRAIN_GROUPS_PER_HISTORY,
        build_composition_fixture_sets,
        classification_metrics,
        representation_geometry,
    )

    return (
        COMPOSITION_CLASSES,
        COMPOSITION_HISTORIES,
        E2_COMPOSITION_SEEDS,
        TRAIN_GROUPS_PER_HISTORY,
        EVAL_GROUPS_PER_HISTORY,
        build_composition_fixture_sets,
        classification_metrics,
        representation_geometry,
    )


def test_composition_registry_is_frozen() -> None:
    classes, histories, seeds, train_groups, eval_groups, *_ = _api()

    assert classes == ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")
    assert histories == (5, 10, 20, 40)
    assert seeds == (7, 17, 29, 43, 61)
    assert train_groups == 100
    assert eval_groups == 25


def test_training_fixtures_lock_event_timing_counts_and_shared_nuisance() -> None:
    classes, histories, _, _, _, build, _, _ = _api()
    fixture_sets = build(7, training=True)

    assert tuple(item.history for item in fixture_sets) == histories
    event_index = {"A": 0, "B": 1, "C": 2}

    for fixture_set in fixture_sets:
        assert len(fixture_set.sequences) == 600
        assert len(fixture_set.fixture_digest) == 64
        first_group = fixture_set.sequences[:6]
        assert [sequence.label for sequence in first_group] == list(range(6))
        assert all(sequence.frames.shape == (5 + fixture_set.history, 7) for sequence in first_group)

        nuisance = first_group[0].frames[:, 3:]
        assert set(np.unique(nuisance).tolist()) == {-0.25, 0.25}
        for sequence in first_group[1:]:
            np.testing.assert_array_equal(sequence.frames[:, 3:], nuisance)

        for sequence, order in zip(first_group, classes, strict=True):
            expected = np.zeros((5 + fixture_set.history, 3), dtype=np.float64)
            for frame_index, symbol in zip((0, 2, 4), order, strict=True):
                expected[frame_index, event_index[symbol]] = 1.0
            np.testing.assert_array_equal(sequence.frames[:, :3], expected)


def test_evaluation_fixture_counts_and_rng_lineage_are_deterministic() -> None:
    _, histories, _, _, _, build, _, _ = _api()

    first = build(17, training=False)
    same = build(17, training=False)
    changed = build(29, training=False)

    assert tuple(item.history for item in first) == histories
    assert all(len(item.sequences) == 150 for item in first)
    assert [item.fixture_digest for item in first] == [item.fixture_digest for item in same]
    assert [item.fixture_digest for item in first] != [item.fixture_digest for item in changed]


def test_composition_fixture_arrays_are_immutable() -> None:
    *_, build, _, _ = _api()
    sequence = build(7, training=False)[0].sequences[0]

    assert sequence.frames.dtype == np.float64
    assert not sequence.frames.flags.writeable
    with pytest.raises(ValueError):
        sequence.frames[0, 0] = 9.0


def test_classification_metrics_report_six_class_confusion_and_macro_f1() -> None:
    *_, metrics, _ = _api()
    labels = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    predictions = labels.copy()

    result = metrics(labels, predictions)

    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.confusion_counts == tuple(
        tuple(2 if row == col else 0 for col in range(6))
        for row in range(6)
    )


def test_representation_geometry_uses_all_six_classes_and_is_finite() -> None:
    *_, geometry = _api()
    labels = np.repeat(np.arange(6, dtype=np.int64), 2)
    states = np.vstack(
        [
            np.array([float(label), -float(label)], dtype=np.float64) + offset
            for label in range(6)
            for offset in (
                np.array([-0.1, 0.1], dtype=np.float64),
                np.array([0.1, -0.1], dtype=np.float64),
            )
        ]
    )

    result = geometry(states, labels)

    assert len(result.centroid_distances) == 15
    assert len(result.within_class_dispersion) == 6
    assert all(value >= 0.0 and np.isfinite(value) for value in result.centroid_distances)
    assert all(value >= 0.0 and np.isfinite(value) for value in result.within_class_dispersion)
    assert result.separation_ratio > 0.0
    assert np.isfinite(result.separation_ratio)


@pytest.mark.parametrize("seed", [-1, 5, True, 7.0, "7", np.int64(7)])
def test_composition_fixture_builder_rejects_unregistered_seed(seed: object) -> None:
    *_, build, _, _ = _api()

    with pytest.raises(ValueError, match="registered"):
        build(seed, training=True)


def test_composition_history_runner_uses_exact_same_trajectories_for_both_readouts() -> None:
    from neural_state_machine.r1_e2_composition import (
        CompositionFixtureSet,
        build_composition_fixture_sets,
        evaluate_composition_history,
    )
    from neural_state_machine.r1_e2_reservoir import E2Architecture, E2ReservoirSpec

    training = build_composition_fixture_sets(7, training=True)[0]
    evaluation = build_composition_fixture_sets(7, training=False)[0]
    # Keep one nuisance group per class for a fast structural smoke; the registered
    # top-level runner is responsible for enforcing the full 100/25 group counts.
    training_small = CompositionFixtureSet(
        history=training.history,
        sequences=training.sequences[:6],
        fixture_digest=training.fixture_digest,
    )
    evaluation_small = CompositionFixtureSet(
        history=evaluation.history,
        sequences=evaluation.sequences[:6],
        fixture_digest=evaluation.fixture_digest,
    )

    result = evaluate_composition_history(
        E2ReservoirSpec(
            architecture=E2Architecture.FLAT,
            input_size=7,
            seed=7,
        ),
        training_small,
        evaluation_small,
    )

    assert result.history == 5
    assert result.instantaneous.metrics.total == 6
    assert result.temporal_mean.metrics.total == 6
    assert result.instantaneous.geometry is not None
    assert result.temporal_mean.geometry is not None
    assert result.instantaneous.coefficient_digest
    assert result.temporal_mean.coefficient_digest
    assert result.instantaneous.prediction_digest
    assert result.temporal_mean.prediction_digest
    assert result.instantaneous.reset_prediction_digest
    assert result.temporal_mean.reset_prediction_digest


def test_composition_history_runner_reset_control_is_exact_chance_for_paired_group() -> None:
    from neural_state_machine.r1_e2_composition import (
        CompositionFixtureSet,
        evaluate_composition_history,
    )
    from neural_state_machine.r1_e2_reservoir import E2Architecture, E2ReservoirSpec

    training = build_composition_fixture_sets(17, training=True)[0]
    evaluation = build_composition_fixture_sets(17, training=False)[0]
    training_small = CompositionFixtureSet(
        history=training.history,
        sequences=training.sequences[:6],
        fixture_digest=training.fixture_digest,
    )
    evaluation_small = CompositionFixtureSet(
        history=evaluation.history,
        sequences=evaluation.sequences[:6],
        fixture_digest=evaluation.fixture_digest,
    )

    result = evaluate_composition_history(
        E2ReservoirSpec(
            architecture=E2Architecture.GROUPED4,
            input_size=7,
            seed=17,
        ),
        training_small,
        evaluation_small,
    )

    assert result.reset_groups_equal is True
    assert result.instantaneous.reset_correct == 1
    assert result.instantaneous.reset_total == 6
    assert result.temporal_mean.reset_correct == 1
    assert result.temporal_mean.reset_total == 6


def test_registered_composition_runner_rejects_nonregistered_fixture_counts() -> None:
    from neural_state_machine.r1_e2_composition import (
        CompositionFixtureSet,
        build_composition_fixture_sets,
        run_composition_arm,
    )
    from neural_state_machine.r1_e2_reservoir import E2Architecture, E2ReservoirSpec

    training = build_composition_fixture_sets(7, training=True)
    evaluation = build_composition_fixture_sets(7, training=False)
    first = training[0]
    bad_training = (
        CompositionFixtureSet(
            history=first.history,
            sequences=first.sequences[:6],
            fixture_digest=first.fixture_digest,
        ),
        *training[1:],
    )

    with pytest.raises(ValueError, match="registered training count"):
        run_composition_arm(
            E2ReservoirSpec(
                architecture=E2Architecture.FLAT,
                input_size=7,
                seed=7,
            ),
            bad_training,
            evaluation,
        )


def test_composition_runner_rejects_wrong_input_width_and_seed_mismatch() -> None:
    from neural_state_machine.r1_e2_composition import (
        build_composition_fixture_sets,
        run_composition_arm,
    )
    from neural_state_machine.r1_e2_reservoir import E2Architecture, E2ReservoirSpec

    training = build_composition_fixture_sets(7, training=True)
    evaluation = build_composition_fixture_sets(7, training=False)

    with pytest.raises(ValueError, match="input_size"):
        run_composition_arm(
            E2ReservoirSpec(
                architecture=E2Architecture.FLAT,
                input_size=6,
                seed=7,
            ),
            training,
            evaluation,
        )

    with pytest.raises(ValueError, match="seed"):
        run_composition_arm(
            E2ReservoirSpec(
                architecture=E2Architecture.FLAT,
                input_size=7,
                seed=17,
            ),
            training,
            evaluation,
        )
