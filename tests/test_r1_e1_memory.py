from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from neural_state_machine.r1_e1_memory import (
    MEMORY_DELAYS,
    MemoryAccuracy,
    build_memory_fixtures,
    contiguous_memory_horizon,
    run_memory_arm,
    run_phase2a_compatibility,
)
from neural_state_machine.r1_e1_reservoir import ReservoirArchitecture, ReservoirSpec


def test_registered_memory_fixture_counts_balance_every_delay_and_class() -> None:
    fixtures = build_memory_fixtures(7)

    assert MEMORY_DELAYS == (1, 2, 5, 10, 20, 40, 80)
    assert len(fixtures.training) == 2_800
    assert len(fixtures.evaluation) == 280
    assert Counter((episode.delay, episode.label) for episode in fixtures.training) == {
        (delay, label): 200 for delay in MEMORY_DELAYS for label in (0, 1)
    }
    assert Counter((episode.delay, episode.label) for episode in fixtures.evaluation) == {
        (delay, label): 20 for delay in MEMORY_DELAYS for label in (0, 1)
    }


def test_memory_fixture_frames_use_phase2a_semantics_and_are_immutable() -> None:
    fixtures = build_memory_fixtures(17)
    episode = next(item for item in fixtures.evaluation if item.delay == 80 and item.label == 1)

    np.testing.assert_array_equal(episode.cue_stimulus, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(episode.decision_stimulus, [0.0, 0.0, 0.0, 1.0])
    assert len(episode.delay_stimuli) == 80
    assert all(frame.shape == (4,) for frame in episode.delay_stimuli)
    assert all(np.all(frame[[0, 1, 3]] == 0.0) for frame in episode.delay_stimuli)
    assert all(-0.25 <= frame[2] <= 0.25 for frame in episode.delay_stimuli)
    assert not episode.cue_stimulus.flags.writeable
    assert not episode.decision_stimulus.flags.writeable
    assert all(not frame.flags.writeable for frame in episode.delay_stimuli)


def test_memory_fixture_rng_is_repeatable_and_seed_sensitive() -> None:
    first = build_memory_fixtures(29)
    same = build_memory_fixtures(29)
    changed = build_memory_fixtures(43)

    assert first.training_digest == same.training_digest
    assert first.evaluation_digest == same.evaluation_digest
    assert first.training_digest != changed.training_digest
    assert first.evaluation_digest != changed.evaluation_digest


def test_paired_left_right_fixtures_share_delay_distractors() -> None:
    fixtures = build_memory_fixtures(61)

    for episodes in (fixtures.training, fixtures.evaluation):
        grouped: dict[tuple[int, int], list] = {}
        for episode in episodes:
            grouped.setdefault((episode.delay, episode.pair_id), []).append(episode)
        assert all(len(pair) == 2 for pair in grouped.values())
        for pair in grouped.values():
            assert {episode.label for episode in pair} == {0, 1}
            for left, right in zip(pair[0].delay_stimuli, pair[1].delay_stimuli, strict=True):
                np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((0.90, 0.86, 0.85, 0.84, 1.0, 1.0, 1.0), 5),
        ((0.90, 0.86, 0.85, 0.90, 0.85, 0.85, 0.85), 80),
        ((0.84, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), None),
    ],
)
def test_contiguous_memory_horizon_stops_at_first_subthreshold_delay(
    scores: tuple[float, ...], expected: int | None
) -> None:
    per_delay = tuple(
        (delay, MemoryAccuracy(int(round(score * 100)), 100))
        for delay, score in zip(MEMORY_DELAYS, scores, strict=True)
    )

    assert contiguous_memory_horizon(per_delay, threshold=0.85) == expected


def test_registered_shallow64_arm_enforces_exact_reset_control_and_digests() -> None:
    result = run_memory_arm(
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 4, 7),
        build_memory_fixtures(7),
    )

    assert result.training.total == 2_800
    assert result.evaluation.total == 280
    assert tuple(delay for delay, _ in result.per_delay) == MEMORY_DELAYS
    assert all(score.total == 40 for _, score in result.per_delay)
    assert result.reset == MemoryAccuracy(140, 280)
    assert all(score == MemoryAccuracy(20, 40) for _, score in result.reset_per_delay)
    assert result.all_reset_pairs_equal is True
    assert result.valid is True
    assert result.parameter_digest_before == result.parameter_digest_after
    for digest in (
        result.parameter_digest_before,
        result.fixture_digest,
        result.coefficient_digest,
        result.prediction_digest,
        result.reset_prediction_digest,
    ):
        assert len(digest) == 64


def test_memory_arm_reuses_fixtures_independently_of_architecture_and_budget() -> None:
    fixtures = build_memory_fixtures(17)
    shallow = run_memory_arm(
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 4, 17),
        fixtures,
    )
    grouped = run_memory_arm(
        ReservoirSpec(ReservoirArchitecture.GROUPED2, 64, 4, 17),
        fixtures,
    )

    assert shallow.fixture_digest == fixtures.combined_digest
    assert grouped.fixture_digest == fixtures.combined_digest
    assert shallow.fixture_digest == grouped.fixture_digest
    assert shallow.parameter_digest_before != grouped.parameter_digest_before


def test_phase2a_compatibility_replays_frozen_seed7_exactly() -> None:
    result = run_phase2a_compatibility(7)

    assert result.seed == 7
    assert result.hidden_recurrence_matches is True
    assert result.training_matches is True
    assert result.evaluation_matches is True
    assert result.per_delay_matches is True
    assert result.reset_matches is True
    assert result.probe_digest_matches is True
    assert result.prediction_digest_matches is True
    assert result.reset_prediction_digest_matches is True
    assert result.legacy_acceptance_required is True
    assert result.legacy_acceptance_passed is True
    assert result.passed is True


def test_phase2a_compatibility_does_not_retroactively_require_acceptance_for_new_seed() -> None:
    result = run_phase2a_compatibility(43)

    assert result.legacy_acceptance_required is False
    assert result.passed is (
        result.hidden_recurrence_matches
        and result.training_matches
        and result.evaluation_matches
        and result.per_delay_matches
        and result.reset_matches
        and result.probe_digest_matches
        and result.prediction_digest_matches
        and result.reset_prediction_digest_matches
    )


@pytest.mark.parametrize("seed", [-1, True, 7.0, "7", np.int64(7)])
def test_memory_fixture_and_compatibility_reject_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        build_memory_fixtures(seed)
    with pytest.raises(ValueError, match="seed"):
        run_phase2a_compatibility(seed)
