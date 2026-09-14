from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import neural_state_machine.learning_diagnostics as diagnostics
from neural_state_machine import DelayedCueTask


def test_learning_diagnostics_config_defaults_are_the_frozen_protocol() -> None:
    assert diagnostics.LearningDiagnosticsConfig() == diagnostics.LearningDiagnosticsConfig(
        hidden_size=64,
        recurrent_radius=0.9,
        learning_rate=0.05,
        temperature=1.0,
        training_episodes=2_000,
        evaluation_blocks=20,
        checkpoint_interval=100,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("hidden_size", 0),
        ("hidden_size", -1),
        ("hidden_size", True),
        ("hidden_size", 8.0),
        ("training_episodes", 0),
        ("training_episodes", -10),
        ("training_episodes", False),
        ("training_episodes", 20.0),
        ("evaluation_blocks", 0),
        ("evaluation_blocks", -1),
        ("evaluation_blocks", True),
        ("evaluation_blocks", 2.0),
        ("checkpoint_interval", 0),
        ("checkpoint_interval", -10),
        ("checkpoint_interval", True),
        ("checkpoint_interval", 10.0),
    ],
)
def test_learning_diagnostics_config_rejects_invalid_integer_fields(
    name: str, value: object
) -> None:
    with pytest.raises(ValueError):
        diagnostics.LearningDiagnosticsConfig(**{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("recurrent_radius", -0.1),
        ("recurrent_radius", 1.0),
        ("recurrent_radius", True),
        ("recurrent_radius", np.nan),
        ("recurrent_radius", np.inf),
        ("learning_rate", 0.0),
        ("learning_rate", -0.1),
        ("learning_rate", True),
        ("learning_rate", np.nan),
        ("learning_rate", np.inf),
        ("temperature", 0.0),
        ("temperature", -0.1),
        ("temperature", True),
        ("temperature", np.nan),
        ("temperature", np.inf),
    ],
)
def test_learning_diagnostics_config_rejects_invalid_float_fields(
    name: str, value: object
) -> None:
    with pytest.raises(ValueError):
        diagnostics.LearningDiagnosticsConfig(**{name: value})


@pytest.mark.parametrize(
    ("training_episodes", "checkpoint_interval"),
    [(21, 10), (20, 15), (20, 30)],
)
def test_learning_diagnostics_config_requires_ten_case_divisible_intervals(
    training_episodes: int, checkpoint_interval: int
) -> None:
    with pytest.raises(ValueError):
        diagnostics.LearningDiagnosticsConfig(
            training_episodes=training_episodes,
            checkpoint_interval=checkpoint_interval,
        )


def test_hidden_dataset_defensively_owns_immutable_validated_arrays() -> None:
    states = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    labels = np.array([0, 1, 0, 1], dtype=np.int32)
    delays = np.array([1, 2, 3, 5], dtype=np.int32)
    expected_states = np.ascontiguousarray(states, dtype=np.float64)
    state_hasher = hashlib.sha256()
    state_hasher.update(str(expected_states.shape).encode("ascii"))
    state_hasher.update(expected_states.tobytes(order="C"))
    dataset = diagnostics._HiddenDataset(
        states=states,
        labels=labels,
        delays=delays,
        fixture_digest="a" * 64,
        state_digest=state_hasher.hexdigest(),
    )
    states[:] = -1
    labels[:] = 9
    delays[:] = 9

    assert dataset.states.dtype == np.dtype(np.float64)
    assert dataset.labels.dtype == np.dtype(np.int64)
    assert dataset.delays.dtype == np.dtype(np.int64)
    assert dataset.states.flags.c_contiguous
    assert dataset.labels.flags.c_contiguous
    assert dataset.delays.flags.c_contiguous
    assert not dataset.states.flags.writeable
    assert not dataset.labels.flags.writeable
    assert not dataset.delays.flags.writeable
    assert not np.shares_memory(dataset.states, states)
    assert not np.shares_memory(dataset.labels, labels)
    assert not np.shares_memory(dataset.delays, delays)
    np.testing.assert_array_equal(dataset.labels, [0, 1, 0, 1])
    np.testing.assert_array_equal(dataset.delays, [1, 2, 3, 5])
    with pytest.raises(FrozenInstanceError):
        dataset.fixture_digest = "changed"


@pytest.mark.parametrize(
    ("states", "labels", "delays", "fixture_digest", "state_digest"),
    [
        (np.empty((0, 2)), [0], [1], "a" * 64, "b" * 64),
        (np.array([[np.nan, 0.0], [0.0, 0.0]]), [0, 1], [1, 2], "a" * 64, "b" * 64),
        (np.ones((2, 2)), [0, 2], [1, 2], "a" * 64, "b" * 64),
        (np.ones((2, 2)), [0, 0], [1, 2], "a" * 64, "b" * 64),
        (np.ones((2, 2)), [0, 1], [0, 2], "a" * 64, "b" * 64),
        (np.ones((2, 2)), [0, 1], [1], "a" * 64, "b" * 64),
        (np.ones((2, 2)), [0, 1], [1, 2], 7, "b" * 64),
        (np.ones((2, 2)), [0, 1], [1, 2], "a" * 64, "not-a-digest"),
    ],
)
def test_hidden_dataset_rejects_invalid_data(
    states: object,
    labels: object,
    delays: object,
    fixture_digest: object,
    state_digest: object,
) -> None:
    with pytest.raises(ValueError):
        diagnostics._HiddenDataset(
            states=states,
            labels=labels,
            delays=delays,
            fixture_digest=fixture_digest,
            state_digest=state_digest,
        )


class _MetadataGuardEpisode:
    def __init__(self, episode) -> None:
        self.cue_stimulus = episode.cue_stimulus
        self.delay_stimuli = episode.delay_stimuli
        self.decision_stimulus = episode.decision_stimulus
        self._label = episode.correct_action_index
        self._delay = episode.delay_steps
        self.metadata_open = False

    @property
    def correct_action_index(self) -> int:
        if not self.metadata_open:
            raise AssertionError("label read before hidden collection finished")
        return self._label

    @property
    def delay_steps(self) -> int:
        if not self.metadata_open:
            raise AssertionError("delay read before hidden collection finished")
        return self._delay


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def reset_state(self) -> None:
        return None

    def advance(self, stimulus: np.ndarray) -> np.ndarray:
        assert isinstance(stimulus, np.ndarray)
        assert stimulus.dtype == np.dtype(np.float64)
        assert stimulus.ndim == 1
        self.calls.append(stimulus.copy())
        return np.array([float(len(self.calls)), 0.0], dtype=np.float64)


def test_hidden_collection_finishes_numeric_policy_calls_before_reading_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_episodes = diagnostics._build_fixtures(
        DelayedCueTask(), np.random.default_rng(4), 1
    )
    episodes = tuple(_MetadataGuardEpisode(episode) for episode in original_episodes)
    policy = _RecordingPolicy()
    collected = 0

    def numeric_hidden(
        received_policy: _RecordingPolicy,
        episode: _MetadataGuardEpisode,
        *,
        reset_before_decision: bool,
    ) -> np.ndarray:
        assert reset_before_decision is False
        nonlocal collected
        received_policy.reset_state()
        received_policy.advance(episode.cue_stimulus)
        for stimulus in episode.delay_stimuli:
            received_policy.advance(stimulus)
        hidden = received_policy.advance(episode.decision_stimulus)
        collected += 1
        if collected == len(episodes):
            for guarded in episodes:
                guarded.metadata_open = True
        return hidden

    monkeypatch.setattr(diagnostics, "_decision_hidden", numeric_hidden)

    dataset = diagnostics._collect_hidden_dataset(policy, episodes)

    assert dataset.states.shape == (10, 2)
    assert len(policy.calls) == sum(episode._delay + 2 for episode in episodes)
    assert all(call.dtype == np.dtype(np.float64) and call.ndim == 1 for call in policy.calls)


@pytest.mark.parametrize("seed", [7, 17, 29])
def test_diagnostic_fixtures_are_balanced_immutable_and_repeatable(seed: int) -> None:
    config = diagnostics.LearningDiagnosticsConfig()
    first = diagnostics._build_diagnostic_fixtures(seed, config)
    second = diagnostics._build_diagnostic_fixtures(seed, config)

    assert len(first.training_episodes) == 2_000
    assert len(first.evaluation_episodes) == 200
    for fixtures, blocks in ((first.training_episodes, 200), (first.evaluation_episodes, 20)):
        for block_index in range(blocks):
            block = fixtures[block_index * 10 : (block_index + 1) * 10]
            assert Counter(episode.correct_action_index for episode in block) == {0: 5, 1: 5}
            assert Counter(episode.delay_steps for episode in block) == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
            assert len({(episode.correct_action_index, episode.delay_steps) for episode in block}) == 10
    assert Counter(first.evaluation.labels.tolist()) == {0: 100, 1: 100}
    assert Counter(first.evaluation.delays.tolist()) == {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}
    assert not first.training.states.flags.writeable
    assert not first.evaluation.states.flags.writeable
    assert len({id(episode.delay_stimuli[0]) for episode in first.training_episodes}) == 2_000
    assert first.training.fixture_digest == second.training.fixture_digest
    assert first.evaluation.fixture_digest == second.evaluation.fixture_digest
    assert first.training.state_digest == second.training.state_digest
    assert first.evaluation.state_digest == second.evaluation.state_digest
    assert not np.shares_memory(first.training.states, second.training.states)
    assert not np.shares_memory(first.evaluation.states, second.evaluation.states)
