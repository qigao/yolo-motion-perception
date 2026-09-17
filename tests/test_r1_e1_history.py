from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from neural_state_machine.r1_e1_history import (
    HISTORY_CLASSES,
    HISTORY_HORIZONS,
    HistoryAccuracy,
    build_history_fixtures,
    contiguous_separability_horizon,
    normalized_correct_class_margins,
    run_history_arm,
)
from neural_state_machine.r1_e1_reservoir import ReservoirArchitecture, ReservoirSpec


def test_registered_history_fixture_counts_and_class_order() -> None:
    fixtures = build_history_fixtures(7)

    assert HISTORY_CLASSES == ("AB", "BA", "AA", "BB")
    assert HISTORY_HORIZONS == (1, 5, 20, 40)
    for horizon in HISTORY_HORIZONS:
        training = fixtures.training[horizon]
        evaluation = fixtures.evaluation[horizon]
        assert len(training) == 800
        assert len(evaluation) == 200
        assert Counter(sequence.label for sequence in training) == {0: 200, 1: 200, 2: 200, 3: 200}
        assert Counter(sequence.label for sequence in evaluation) == {0: 50, 1: 50, 2: 50, 3: 50}


def test_history_frames_encode_events_at_exact_steps_with_shared_nuisance() -> None:
    fixtures = build_history_fixtures(17)
    group = [sequence for sequence in fixtures.evaluation[5] if sequence.group_id == 0]
    assert [sequence.label for sequence in group] == [0, 1, 2, 3]
    assert all(len(sequence.frames) == 8 for sequence in group)

    expected_events = {
        0: ((1.0, 0.0), (0.0, 1.0)),
        1: ((0.0, 1.0), (1.0, 0.0)),
        2: ((1.0, 0.0), (1.0, 0.0)),
        3: ((0.0, 1.0), (0.0, 1.0)),
    }
    for sequence in group:
        first, second = expected_events[sequence.label]
        assert tuple(sequence.frames[0][:2]) == first
        assert tuple(sequence.frames[1][:2]) == (0.0, 0.0)
        assert tuple(sequence.frames[2][:2]) == second
        assert all(tuple(frame[:2]) == (0.0, 0.0) for frame in sequence.frames[3:])
        assert all(not frame.flags.writeable for frame in sequence.frames)

    for frame_index in range(8):
        nuisance = group[0].frames[frame_index][2:]
        assert set(nuisance.tolist()) <= {-0.25, 0.25}
        for sequence in group[1:]:
            np.testing.assert_array_equal(sequence.frames[frame_index][2:], nuisance)


def test_history_fixture_digests_repeat_and_change_with_seed() -> None:
    first = build_history_fixtures(29)
    same = build_history_fixtures(29)
    changed = build_history_fixtures(43)

    assert first.combined_digest == same.combined_digest
    assert first.training_digests == same.training_digests
    assert first.evaluation_digests == same.evaluation_digests
    assert first.combined_digest != changed.combined_digest


def test_normalized_margin_matches_registered_formula_and_is_finite() -> None:
    scores = np.array([[3.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, 1.0]])
    labels = np.array([0, 2])

    margins = normalized_correct_class_margins(scores, labels)

    expected = (
        (3.0 - 1.0) / np.linalg.norm(scores[0]),
        (1.0 - 2.0) / np.linalg.norm(scores[1]),
    )
    np.testing.assert_allclose(margins, expected)
    assert np.all(np.isfinite(margins))


@pytest.mark.parametrize(
    ("accuracies", "expected"),
    [
        ((0.90, 0.85, 0.79, 1.0), 5),
        ((0.80, 0.80, 0.80, 0.80), 40),
        ((0.79, 1.0, 1.0, 1.0), None),
    ],
)
def test_contiguous_separability_horizon(
    accuracies: tuple[float, ...], expected: int | None
) -> None:
    per_horizon = tuple(
        (horizon, HistoryAccuracy(int(round(value * 100)), 100))
        for horizon, value in zip(HISTORY_HORIZONS, accuracies, strict=True)
    )

    assert contiguous_separability_horizon(per_horizon, threshold=0.80) == expected


def test_registered_history_arm_enforces_reset_and_metric_controls() -> None:
    fixtures = build_history_fixtures(7)
    result = run_history_arm(
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 6, 7),
        fixtures,
    )

    assert result.valid is True
    assert result.parameter_digest_before == result.parameter_digest_after
    assert result.fixture_digest == fixtures.combined_digest
    assert tuple(item.horizon for item in result.horizons) == HISTORY_HORIZONS
    assert result.macro_accuracy == pytest.approx(
        np.mean([item.evaluation.accuracy for item in result.horizons])
    )
    for item in result.horizons:
        assert item.training.total == 800
        assert item.evaluation.total == 200
        assert item.reset == HistoryAccuracy(50, 200)
        assert item.all_reset_groups_equal is True
        assert len(item.margins) == 200
        assert all(np.isfinite(item.margins))
        for summary in (item.within_class_cosine, item.between_class_cosine):
            assert summary.count > 0
            assert 0.0 <= summary.minimum <= summary.maximum <= 2.0
            assert np.isfinite(summary.median)
            assert np.isfinite(summary.mean)
        for digest in (item.coefficient_digest, item.prediction_digest, item.reset_prediction_digest):
            assert len(digest) == 64


def test_history_arm_reuses_same_fixtures_across_architectures() -> None:
    fixtures = build_history_fixtures(17)
    shallow = run_history_arm(
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 6, 17), fixtures
    )
    grouped = run_history_arm(
        ReservoirSpec(ReservoirArchitecture.GROUPED2, 64, 6, 17), fixtures
    )

    assert shallow.fixture_digest == grouped.fixture_digest == fixtures.combined_digest
    assert shallow.parameter_digest_before != grouped.parameter_digest_before


@pytest.mark.parametrize("seed", [-1, True, 7.0, "7", np.int64(7)])
def test_history_fixtures_reject_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        build_history_fixtures(seed)
