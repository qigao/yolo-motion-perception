from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.r1_e1_reservoir import ReservoirArchitecture, ReservoirSpec
from neural_state_machine.r1_e1_yolo_like import (
    BEHAVIOR_CLASSES,
    CORRUPTION_ARMS,
    build_clean_fixture_set,
    build_corruption_plan,
    corrupt_sequence,
    run_yolo_like_robustness,
)


def _group_by_pair(fixtures):
    groups = {}
    for fixture in fixtures:
        groups.setdefault(fixture.pair_index, []).append(fixture)
    return groups


def test_clean_evaluation_fixtures_have_registered_shape_and_pairing() -> None:
    fixtures = build_clean_fixture_set(7, training=False)

    assert len(fixtures.sequences) == 200
    assert fixtures.fixture_digest
    groups = _group_by_pair(fixtures.sequences)
    assert len(groups) == 50
    for group in groups.values():
        assert [sequence.label for sequence in group] == [0, 1, 2, 3]
        for sequence in group:
            assert sequence.frames.shape == (20, 9)
            assert sequence.frames.dtype == np.float64
            assert not sequence.frames.flags.writeable
        nuisance = [sequence.frames[:, 7:9] for sequence in group]
        for values in nuisance[1:]:
            np.testing.assert_array_equal(values, nuisance[0])
        assert set(np.unique(nuisance[0])).issubset({-0.05, 0.05})


def test_clean_templates_match_registered_behavior_formulas() -> None:
    fixtures = build_clean_fixture_set(11, training=False)
    first_group = fixtures.sequences[:4]
    assert tuple(sequence.behavior for sequence in first_group) == BEHAVIOR_CLASSES

    approach, touch, pick_up, pass_by = [sequence.frames for sequence in first_group]
    t = np.arange(16, dtype=np.float64)

    np.testing.assert_allclose(approach[:16, 0], 1.0 - 0.8 * t / 15.0)
    np.testing.assert_array_equal(approach[:16, 1], 0.0)
    np.testing.assert_array_equal(approach[:16, 2], 0.8)
    np.testing.assert_array_equal(approach[:16, 3:6], 0.0)
    np.testing.assert_array_equal(approach[:16, 6], 1.0)

    np.testing.assert_allclose(touch[:11, 0], 1.0 - np.arange(11) / 10.0)
    np.testing.assert_array_equal(touch[:11, 1], 0.0)
    np.testing.assert_array_equal(touch[:11, 2], 1.0)
    np.testing.assert_array_equal(touch[11:16, 0], 0.0)
    np.testing.assert_array_equal(touch[11:16, 1], 1.0)
    np.testing.assert_array_equal(touch[11:16, 2], 0.0)

    np.testing.assert_allclose(pick_up[:9, 0], 1.0 - np.arange(9) / 8.0)
    np.testing.assert_array_equal(pick_up[:9, 1], 0.0)
    np.testing.assert_array_equal(pick_up[:9, 2], 1.0)
    np.testing.assert_array_equal(pick_up[9:16, 0], 0.0)
    np.testing.assert_array_equal(pick_up[9:16, 1], 1.0)
    np.testing.assert_array_equal(pick_up[9:16, 2], 1.0)
    np.testing.assert_array_equal(pick_up[9, 3:5], 0.0)
    np.testing.assert_array_equal(pick_up[10:16, 3:5], 1.0)
    np.testing.assert_allclose(pick_up[9:16, 5], np.arange(7) / 6.0)

    np.testing.assert_allclose(pass_by[:8, 0], 1.0 - 0.9 * np.arange(8) / 7.0)
    np.testing.assert_allclose(pass_by[8:16, 0], 0.1 + 0.9 * np.arange(8) / 7.0)
    np.testing.assert_array_equal(pass_by[:16, 1], 0.0)
    np.testing.assert_array_equal(pass_by[:16, 2], 1.0)
    np.testing.assert_array_equal(pass_by[:16, 3:6], 0.0)

    for frames in (approach, touch, pick_up, pass_by):
        np.testing.assert_array_equal(frames[16:20, 0], 0.5)
        np.testing.assert_array_equal(frames[16:20, 1:6], 0.0)
        np.testing.assert_array_equal(frames[16:20, 6], 1.0)


def test_registered_corruption_plan_is_deterministic_and_has_exact_masks() -> None:
    fixtures = build_clean_fixture_set(17, training=False)
    first = build_corruption_plan(17, fixtures)
    second = build_corruption_plan(17, fixtures)

    assert first.digest == second.digest
    assert CORRUPTION_ARMS == ("clean", "drop10", "wrong10", "occlusion4", "jitter", "mixed")
    assert len(first.entries) == 200
    for entry in first.entries:
        assert len(entry.drop_indices) == 2
        assert len(set(entry.drop_indices)) == 2
        assert all(0 <= index <= 15 for index in entry.drop_indices)
        assert len(entry.wrong_indices) == 2
        assert len(set(entry.wrong_indices)) == 2
        assert all(0 <= index <= 15 for index in entry.wrong_indices)
        assert entry.occlusion_indices == (8, 9, 10, 11)
        assert len(entry.mixed_drop_indices) == 2
        assert len(set(entry.mixed_drop_indices)) == 2
        assert entry.mixed_wrong_index not in entry.mixed_drop_indices
        assert all(0 <= index <= 15 for index in (*entry.mixed_drop_indices, entry.mixed_wrong_index))


def test_corruption_arms_obey_missing_donor_jitter_and_boundary_rules() -> None:
    fixtures = build_clean_fixture_set(29, training=False)
    plan = build_corruption_plan(29, fixtures)
    group = fixtures.sequences[:4]
    by_label = {sequence.label: sequence for sequence in group}

    for sequence in group:
        entry = plan.entries[sequence.fixture_index]
        clean = sequence.frames

        dropped = corrupt_sequence(sequence, by_label, entry, "drop10")
        for index in entry.drop_indices:
            np.testing.assert_array_equal(dropped[index], np.zeros(9))
        np.testing.assert_array_equal(dropped[16:20], clean[16:20])

        wrong = corrupt_sequence(sequence, by_label, entry, "wrong10")
        donor = by_label[(sequence.label + 1) % 4].frames
        for index in entry.wrong_indices:
            np.testing.assert_array_equal(wrong[index, :7], donor[index, :7])
            np.testing.assert_array_equal(wrong[index, 7:9], clean[index, 7:9])
        np.testing.assert_array_equal(wrong[16:20], clean[16:20])

        occluded = corrupt_sequence(sequence, by_label, entry, "occlusion4")
        np.testing.assert_array_equal(occluded[8:12], np.zeros((4, 9)))
        np.testing.assert_array_equal(occluded[16:20], clean[16:20])

        jittered = corrupt_sequence(sequence, by_label, entry, "jitter")
        np.testing.assert_array_equal(jittered[:16, 1], clean[:16, 1])
        np.testing.assert_array_equal(jittered[:16, 6], clean[:16, 6])
        assert np.all((jittered[:16, [0, 2, 3, 4, 5]] >= 0.0) & (jittered[:16, [0, 2, 3, 4, 5]] <= 1.0))
        assert np.all((jittered[:16, 7:9] >= -0.1) & (jittered[:16, 7:9] <= 0.1))
        np.testing.assert_array_equal(jittered[16:20], clean[16:20])

        mixed = corrupt_sequence(sequence, by_label, entry, "mixed")
        for index in entry.mixed_drop_indices:
            np.testing.assert_array_equal(mixed[index], np.zeros(9))
        assert entry.mixed_wrong_index not in entry.mixed_drop_indices
        np.testing.assert_array_equal(mixed[16:20], clean[16:20])


def test_yolo_like_run_reports_all_registered_arms_and_frozen_probe_metrics() -> None:
    result = run_yolo_like_robustness(
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 9, 7)
    )

    assert result.seed == 7
    assert result.architecture == int(ReservoirArchitecture.SHALLOW)
    assert result.budget == 64
    assert result.clean.total == 200
    assert tuple(name for name, _ in result.corrupted) == CORRUPTION_ARMS[1:]
    assert all(score.total == 200 for _, score in result.corrupted)
    assert len(result.confusion_matrices) == 6
    for _, matrix in result.confusion_matrices:
        assert matrix.shape == (4, 4)
        assert matrix.dtype == np.int64
        assert int(np.sum(matrix)) == 200
    assert result.train_fixture_digest
    assert result.evaluation_fixture_digest
    assert result.corruption_digest
    assert result.coefficient_digest
    assert result.reservoir_parameter_digest
    assert np.isfinite(result.macro_corrupted_accuracy)
    assert np.isfinite(result.worst_corrupted_accuracy)


@pytest.mark.parametrize("input_size", [1, 8, 10])
def test_yolo_like_rejects_wrong_reservoir_input_size(input_size: int) -> None:
    with pytest.raises(ValueError, match="input_size 9"):
        run_yolo_like_robustness(
            ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, input_size, 7)
        )
