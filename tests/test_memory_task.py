import numpy as np
import pytest

from neural_state_machine import Cue, DelayedCueTask


def test_left_and_right_share_an_identical_decision_stimulus() -> None:
    task = DelayedCueTask()
    left = task.build_episode(Cue.LEFT, 3, np.random.default_rng(11))
    right = task.build_episode(Cue.RIGHT, 3, np.random.default_rng(11))

    np.testing.assert_array_equal(left.cue_stimulus, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(right.cue_stimulus, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(left.decision_stimulus, right.decision_stimulus)
    np.testing.assert_array_equal(left.decision_stimulus, [0.0, 0.0, 0.0, 1.0])


def test_episode_contains_only_seeded_single_channel_delay_vectors() -> None:
    episode = DelayedCueTask().build_episode(
        Cue.LEFT, 5, np.random.default_rng(11)
    )

    assert len(episode.delay_stimuli) == 5
    for delay_stimulus in episode.delay_stimuli:
        assert delay_stimulus.dtype == np.float64
        assert not delay_stimulus.flags.writeable
        np.testing.assert_array_equal(delay_stimulus[[0, 1, 3]], [0.0, 0.0, 0.0])
        assert np.isfinite(delay_stimulus[2])
        assert -0.25 <= delay_stimulus[2] <= 0.25


def test_episode_arrays_are_independent_readonly_float64_copies() -> None:
    task = DelayedCueTask()
    first = task.build_episode(Cue.LEFT, 2, np.random.default_rng(3))
    second = task.build_episode(Cue.LEFT, 2, np.random.default_rng(3))

    arrays = (first.cue_stimulus, first.decision_stimulus, *first.delay_stimuli)
    assert all(array.dtype == np.float64 and not array.flags.writeable for array in arrays)
    assert len({id(array) for array in arrays}) == len(arrays)
    assert not np.shares_memory(first.cue_stimulus, second.cue_stimulus)
    assert not np.shares_memory(first.decision_stimulus, second.decision_stimulus)
    assert all(
        not np.shares_memory(first_delay, second_delay)
        for first_delay, second_delay in zip(first.delay_stimuli, second.delay_stimuli)
    )


def test_same_seed_reproduces_every_episode_array() -> None:
    task = DelayedCueTask()
    first = task.build_episode(Cue.RIGHT, 4, np.random.default_rng(19))
    second = task.build_episode(Cue.RIGHT, 4, np.random.default_rng(19))

    np.testing.assert_array_equal(first.cue_stimulus, second.cue_stimulus)
    np.testing.assert_array_equal(first.decision_stimulus, second.decision_stimulus)
    assert all(
        np.array_equal(first_delay, second_delay)
        for first_delay, second_delay in zip(first.delay_stimuli, second.delay_stimuli)
    )


def test_different_seed_changes_a_delay_distractor() -> None:
    task = DelayedCueTask()
    first = task.build_episode(Cue.RIGHT, 4, np.random.default_rng(19))
    second = task.build_episode(Cue.RIGHT, 4, np.random.default_rng(20))

    assert any(
        not np.array_equal(first_delay, second_delay)
        for first_delay, second_delay in zip(first.delay_stimuli, second.delay_stimuli)
    )


@pytest.mark.parametrize(
    ("cue", "action_index", "expected"),
    [(Cue.LEFT, 0, 1.0), (Cue.LEFT, 1, -1.0), (Cue.RIGHT, 0, -1.0), (Cue.RIGHT, 1, 1.0)],
)
def test_reward_scores_the_fixture_cue_against_a_legal_action(
    cue: Cue, action_index: int, expected: float
) -> None:
    episode = DelayedCueTask().build_episode(cue, 1, np.random.default_rng(4))

    assert DelayedCueTask().reward(episode, action_index) == expected


@pytest.mark.parametrize("action_index", [-1, 2, True, 1.5, None, "0"])
def test_reward_rejects_non_legal_action_indices(action_index: object) -> None:
    episode = DelayedCueTask().build_episode(Cue.LEFT, 1, np.random.default_rng(4))

    with pytest.raises(ValueError):
        DelayedCueTask().reward(episode, action_index)


@pytest.mark.parametrize("cue", [0, 1, "left", None, True])
def test_build_episode_rejects_values_that_are_not_cue_members(cue: object) -> None:
    with pytest.raises(ValueError):
        DelayedCueTask().build_episode(cue, 1, np.random.default_rng(4))


@pytest.mark.parametrize("delay_steps", [0, 6, True, 1.5])
def test_build_episode_rejects_delay_lengths_outside_the_protocol(delay_steps: object) -> None:
    with pytest.raises(ValueError):
        DelayedCueTask().build_episode(Cue.LEFT, delay_steps, np.random.default_rng(4))


@pytest.mark.parametrize("rng", [11, np.random.RandomState(11), None])
def test_build_episode_rejects_non_generator_random_sources(rng: object) -> None:
    with pytest.raises(ValueError):
        DelayedCueTask().build_episode(Cue.LEFT, 1, rng)


def test_episode_fixture_is_frozen() -> None:
    episode = DelayedCueTask().build_episode(Cue.LEFT, 1, np.random.default_rng(4))

    with pytest.raises(AttributeError):
        episode.correct_action_index = 1


def test_reward_rejects_non_episode_fixtures() -> None:
    with pytest.raises(ValueError):
        DelayedCueTask().reward(object(), 0)
