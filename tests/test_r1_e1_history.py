from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.r1_e1_history import (
    HISTORY_CLASSES,
    HISTORY_HORIZONS,
    _cosine_distance,
    _normalized_margins,
    build_history_fixture_sets,
    run_history_separability,
)
from neural_state_machine.r1_e1_reservoir import ReservoirArchitecture, ReservoirSpec


def test_history_class_and_horizon_registration_is_locked() -> None:
    assert HISTORY_CLASSES == ((0, 1), (1, 0), (0, 0), (1, 1))
    assert HISTORY_HORIZONS == (1, 5, 20, 40)


def test_training_fixture_counts_encoding_and_shared_nuisance() -> None:
    fixture_sets = build_history_fixture_sets(7, training=True)

    assert tuple(item.horizon for item in fixture_sets) == HISTORY_HORIZONS
    for fixture_set in fixture_sets:
        assert len(fixture_set.sequences) == 800
        assert fixture_set.fixture_digest
        first_group = fixture_set.sequences[:4]
        assert [sequence.label for sequence in first_group] == [0, 1, 2, 3]
        assert all(sequence.frames.shape == (3 + fixture_set.horizon, 6) for sequence in first_group)
        nuisance = first_group[0].frames[:, 2:]
        assert set(np.unique(nuisance).tolist()) == {-0.25, 0.25}
        for sequence in first_group[1:]:
            np.testing.assert_array_equal(sequence.frames[:, 2:], nuisance)

        # 0=AB, 1=BA, 2=AA, 3=BB; step 1 and all tail event channels are neutral.
        expected = (
            ((1.0, 0.0), (0.0, 1.0)),
            ((0.0, 1.0), (1.0, 0.0)),
            ((1.0, 0.0), (1.0, 0.0)),
            ((0.0, 1.0), (0.0, 1.0)),
        )
        for sequence, (first, second) in zip(first_group, expected, strict=True):
            assert tuple(sequence.frames[0, :2]) == first
            assert tuple(sequence.frames[1, :2]) == (0.0, 0.0)
            assert tuple(sequence.frames[2, :2]) == second
            np.testing.assert_array_equal(sequence.frames[3:, :2], 0.0)


def test_evaluation_fixture_counts_and_lineage_are_deterministic() -> None:
    first = build_history_fixture_sets(17, training=False)
    second = build_history_fixture_sets(17, training=False)
    different = build_history_fixture_sets(29, training=False)

    assert all(len(item.sequences) == 200 for item in first)
    assert [item.fixture_digest for item in first] == [item.fixture_digest for item in second]
    assert [item.fixture_digest for item in first] != [item.fixture_digest for item in different]


def test_fixture_arrays_are_readonly_and_independent() -> None:
    fixture = build_history_fixture_sets(7, training=False)[0].sequences[0]
    assert fixture.frames.dtype == np.float64
    assert not fixture.frames.flags.writeable
    with pytest.raises(ValueError):
        fixture.frames[0, 0] = 9.0


def test_normalized_margin_definition_and_tie_behavior() -> None:
    scores = np.array([[2.0, 1.0, 0.0, -1.0], [1.0, 1.0, 0.0, 0.0]])
    labels = np.array([0, 0], dtype=np.int64)
    margins = _normalized_margins(scores, labels)

    assert margins[0] == pytest.approx(1.0 / np.linalg.norm(scores[0]))
    assert margins[1] == 0.0


def test_cosine_distance_has_zero_norm_protection() -> None:
    zero = np.zeros(3)
    unit = np.array([1.0, 0.0, 0.0])
    opposite = np.array([-1.0, 0.0, 0.0])

    assert _cosine_distance(zero, zero) == 0.0
    assert _cosine_distance(zero, unit) == 1.0
    assert _cosine_distance(unit, unit) == pytest.approx(0.0)
    assert _cosine_distance(unit, opposite) == pytest.approx(2.0)


def test_real_shallow64_history_run_satisfies_registered_controls() -> None:
    result = run_history_separability(
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 6, 7)
    )

    assert result.seed == 7
    assert result.architecture == 0
    assert result.budget == 64
    assert tuple(item.horizon for item in result.horizons) == HISTORY_HORIZONS
    assert np.isfinite(result.macro_accuracy)
    assert result.contiguous_80_horizon in (None, 1, 5, 20, 40)
    for item in result.horizons:
        assert item.total == 200
        assert 0 <= item.correct <= 200
        assert item.reset_correct == 50
        assert item.reset_total == 200
        assert item.reset_groups_equal
        assert item.coefficient_digest
        assert item.prediction_digest
        assert item.reset_prediction_digest
        assert item.train_fixture_digest
        assert item.evaluation_fixture_digest
        assert item.reservoir_parameter_digest
        for summary in (
            item.normalized_margin,
            item.within_class_cosine_distance,
            item.between_class_cosine_distance,
        ):
            assert summary.count > 0
            assert np.isfinite(
                [summary.minimum, summary.median, summary.mean, summary.maximum]
            ).all()


def test_history_run_is_exactly_repeatable() -> None:
    spec = ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 6, 17)
    assert run_history_separability(spec) == run_history_separability(spec)


def test_history_run_requires_six_input_channels() -> None:
    with pytest.raises(ValueError, match="input_size 6"):
        run_history_separability(
            ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 5, 7)
        )


@pytest.mark.parametrize("seed", [-1, True, 7.0, "7", None, np.int64(7)])
def test_fixture_builder_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        build_history_fixture_sets(seed, training=True)
