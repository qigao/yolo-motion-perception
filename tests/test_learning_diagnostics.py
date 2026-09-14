from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import neural_state_machine.learning_diagnostics as diagnostics
from neural_state_machine import DelayedCueTask
from neural_state_machine.memory_probe import FittedLinearProbe, fit_linear_probe
from neural_state_machine.reward_learning import _build_fixtures


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
    task = DelayedCueTask()
    expected_training = _build_fixtures(
        task,
        np.random.default_rng(np.random.SeedSequence([seed, 0x54524149])),
        config.training_episodes // 10,
    )
    expected_evaluation = _build_fixtures(
        task,
        np.random.default_rng(np.random.SeedSequence([seed, 0x4556414C])),
        config.evaluation_blocks,
    )

    assert len(first.training_episodes) == 2_000
    assert len(first.evaluation_episodes) == 200
    _assert_ordered_fixture_stimuli_equal(first.training_episodes, expected_training)
    _assert_ordered_fixture_stimuli_equal(first.evaluation_episodes, expected_evaluation)
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
    distractors = [
        stimulus
        for fixtures in (first.training_episodes, first.evaluation_episodes)
        for episode in fixtures
        for stimulus in episode.delay_stimuli
    ]
    assert all(not stimulus.flags.writeable for stimulus in distractors)
    assert len({id(stimulus) for stimulus in distractors}) == len(distractors)
    assert first.training.fixture_digest == second.training.fixture_digest
    assert first.evaluation.fixture_digest == second.evaluation.fixture_digest
    assert first.training.state_digest == second.training.state_digest
    assert first.evaluation.state_digest == second.evaluation.state_digest
    assert not np.shares_memory(first.training.states, second.training.states)
    assert not np.shares_memory(first.evaluation.states, second.evaluation.states)


def _assert_ordered_fixture_stimuli_equal(actual, expected) -> None:
    assert len(actual) == len(expected)
    for actual_episode, expected_episode in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            actual_episode.cue_stimulus,
            expected_episode.cue_stimulus,
        )
        assert len(actual_episode.delay_stimuli) == len(expected_episode.delay_stimuli)
        for actual_stimulus, expected_stimulus in zip(
            actual_episode.delay_stimuli,
            expected_episode.delay_stimuli,
            strict=True,
        ):
            np.testing.assert_array_equal(actual_stimulus, expected_stimulus)
        np.testing.assert_array_equal(
            actual_episode.decision_stimulus,
            expected_episode.decision_stimulus,
        )


def _hidden_dataset(
    states: np.ndarray,
    labels: np.ndarray,
    delays: np.ndarray,
) -> diagnostics._HiddenDataset:
    canonical_states = np.ascontiguousarray(states, dtype=np.float64)
    state_hasher = hashlib.sha256()
    state_hasher.update(str(canonical_states.shape).encode("ascii"))
    state_hasher.update(canonical_states.tobytes(order="C"))
    return diagnostics._HiddenDataset(
        states=states,
        labels=labels,
        delays=delays,
        fixture_digest="f" * 64,
        state_digest=state_hasher.hexdigest(),
    )


def test_normalized_signed_margins_match_the_literal_ridge_equation() -> None:
    states = np.array([[3.0, 4.0], [2.0, -1.0], [-4.0, -3.0]], dtype=np.float64)
    labels = np.array([1, 0, 0], dtype=np.int64)
    dataset = _hidden_dataset(states, labels, np.array([1, 2, 3], dtype=np.int64))
    probe = FittedLinearProbe(np.array([3.0, 4.0], dtype=np.float64), bias=-2.0)

    signed = np.where(labels == 0, -1.0, 1.0)
    expected = signed * (states @ probe.weights + probe.bias) / np.linalg.norm(
        probe.weights
    )

    margins = diagnostics._normalized_signed_margins(probe, dataset)

    assert margins.shape == labels.shape
    assert margins.dtype == np.dtype(np.float64)
    assert not margins.flags.writeable
    np.testing.assert_allclose(margins, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("states", "labels", "delays"),
    [
        (np.empty((0, 2)), np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
        (np.ones((2, 2)), np.array([0, 0]), np.array([1, 1])),
        (np.ones((2, 2)), np.array([0]), np.array([1])),
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), np.array([0, 1]), np.array([1, 2])),
    ],
)
def test_normalized_signed_margins_rejects_invalid_geometry_without_mutation(
    states: np.ndarray,
    labels: np.ndarray,
    delays: np.ndarray,
) -> None:
    dataset = object.__new__(diagnostics._HiddenDataset)
    object.__setattr__(dataset, "states", states)
    object.__setattr__(dataset, "labels", labels)
    object.__setattr__(dataset, "delays", delays)
    object.__setattr__(dataset, "fixture_digest", "f" * 64)
    object.__setattr__(dataset, "state_digest", "s" * 64)
    probe = FittedLinearProbe(np.array([1.0, -2.0]), bias=0.25)
    states_before = states.copy()
    labels_before = labels.copy()
    delays_before = delays.copy()
    weights_before = probe.weights.copy()
    bias_before = probe.bias

    with pytest.raises(ValueError):
        diagnostics._normalized_signed_margins(probe, dataset)

    np.testing.assert_array_equal(states, states_before)
    np.testing.assert_array_equal(labels, labels_before)
    np.testing.assert_array_equal(delays, delays_before)
    np.testing.assert_array_equal(probe.weights, weights_before)
    assert probe.bias == bias_before


def test_normalized_signed_margins_rejects_a_zero_norm_probe_without_mutation() -> None:
    dataset = _hidden_dataset(
        np.array([[1.0, 2.0], [-1.0, 3.0]]),
        np.array([0, 1]),
        np.array([1, 2]),
    )
    probe = FittedLinearProbe(np.zeros(2, dtype=np.float64), bias=0.25)
    states_before = dataset.states.copy()
    labels_before = dataset.labels.copy()
    weights_before = probe.weights.copy()

    with pytest.raises(ValueError, match="norm"):
        diagnostics._normalized_signed_margins(probe, dataset)

    np.testing.assert_array_equal(dataset.states, states_before)
    np.testing.assert_array_equal(dataset.labels, labels_before)
    np.testing.assert_array_equal(probe.weights, weights_before)


def test_run_geometry_uses_linear_percentiles_and_orders_every_delay() -> None:
    training = _hidden_dataset(
        np.array(
            [[-4.0, 0.0], [-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0],
             [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
            dtype=np.float64,
        ),
        np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64),
        np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.int64),
    )
    evaluation_states = np.array(
        [[-5.0, 0.0], [-4.0, 0.0], [-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0],
         [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        dtype=np.float64,
    )
    evaluation_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    evaluation_delays = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.int64)
    evaluation = _hidden_dataset(evaluation_states, evaluation_labels, evaluation_delays)

    geometry = diagnostics._run_geometry(training, evaluation)

    fitted = fit_linear_probe(
        training.states, training.labels, regularization=1e-3
    )
    signed = np.where(evaluation_labels == 0, -1.0, 1.0)
    margins = signed * (
        evaluation_states @ fitted.weights + fitted.bias
    ) / np.linalg.norm(fitted.weights)
    expected_summary = diagnostics.MarginSummary(
        minimum=float(np.min(margins)),
        percentile_10=float(np.percentile(margins, 10, method="linear")),
        median=float(np.median(margins)),
    )

    assert geometry.evaluation_margins == expected_summary
    assert tuple(delay for delay, _ in geometry.per_delay) == (1, 2, 3, 4, 5)
    assert tuple(delay for delay, _ in geometry.per_delay_margins) == (1, 2, 3, 4, 5)
    by_delay = dict(geometry.per_delay_margins)
    first_delay_margins = margins[evaluation_delays == 1]
    assert by_delay[1] == diagnostics.MarginSummary(
        minimum=float(np.min(first_delay_margins)),
        percentile_10=float(
            np.percentile(first_delay_margins, 10, method="linear")
        ),
        median=float(np.median(first_delay_margins)),
    )
    assert geometry.delay_five_to_one_median_ratio == (
        by_delay[5].median / by_delay[1].median
    )


def test_run_geometry_rejects_evaluation_with_a_missing_delay_without_mutation() -> None:
    training = _hidden_dataset(
        np.array([[-1.0, 0.0], [1.0, 0.0]]),
        np.array([0, 1]),
        np.array([1, 2]),
    )
    evaluation = _hidden_dataset(
        np.array([[-2.0, 0.0], [2.0, 0.0]]),
        np.array([0, 1]),
        np.array([1, 1]),
    )
    training_before = training.states.copy()
    evaluation_before = evaluation.states.copy()

    with pytest.raises(ValueError, match="delay"):
        diagnostics._run_geometry(training, evaluation)

    np.testing.assert_array_equal(training.states, training_before)
    np.testing.assert_array_equal(evaluation.states, evaluation_before)


@pytest.mark.parametrize("seed", [7, 17, 29])
def test_frozen_ridge_geometry_passes_each_real_seed_without_mutating_fixtures(
    seed: int,
) -> None:
    fixtures = diagnostics._build_diagnostic_fixtures(
        seed, diagnostics.LearningDiagnosticsConfig()
    )
    training_digest = fixtures.training.state_digest
    evaluation_digest = fixtures.evaluation.state_digest
    training_fixture_digest = fixtures.training.fixture_digest
    evaluation_fixture_digest = fixtures.evaluation.fixture_digest

    geometry = diagnostics._run_geometry(fixtures.training, fixtures.evaluation)

    assert geometry.training == diagnostics.AccuracyCount(2_000, 2_000)
    assert geometry.evaluation == diagnostics.AccuracyCount(200, 200)
    assert geometry.per_delay == tuple(
        (delay, diagnostics.AccuracyCount(40, 40)) for delay in range(1, 6)
    )
    assert geometry.training_margins.minimum > 0.0
    assert geometry.evaluation_margins.minimum > 0.0
    assert all(summary.minimum > 0.0 for _, summary in geometry.per_delay_margins)
    assert np.isfinite(geometry.delay_five_to_one_median_ratio)
    assert geometry.geometry_passed
    assert fixtures.training.state_digest == training_digest
    assert fixtures.evaluation.state_digest == evaluation_digest
    assert fixtures.training.fixture_digest == training_fixture_digest
    assert fixtures.evaluation.fixture_digest == evaluation_fixture_digest
