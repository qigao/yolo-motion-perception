from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Self

import numpy as np
import pytest

import neural_state_machine.learning_diagnostics as diagnostics
from neural_state_machine import DelayedCueTask
from neural_state_machine.memory_probe import FittedLinearProbe, fit_linear_probe
from neural_state_machine.reward_learning import _build_fixtures, run_reward_learning_benchmark
from neural_state_machine.reward_readout import RewardReadoutDecision


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


def _policy_matrix_digests(policy: object) -> tuple[str, str, str]:
    digests = []
    for matrix in (
        policy._input_weights,
        policy._recurrent_weights,
        policy._output_weights,
    ):
        values = np.ascontiguousarray(matrix, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        digests.append(digest.hexdigest())
    return tuple(digests)


def _perfect_geometry_datasets() -> tuple[
    diagnostics._HiddenDataset,
    diagnostics._HiddenDataset,
]:
    training_labels = np.tile(np.array([0, 1], dtype=np.int64), 10)
    training = _hidden_dataset(
        np.where(training_labels == 0, -1.0, 1.0).reshape(-1, 1),
        training_labels,
        np.tile(np.arange(1, 6, dtype=np.int64), 4),
    )
    evaluation_labels = np.tile(np.array([0, 1], dtype=np.int64), 100)
    evaluation = _hidden_dataset(
        np.where(evaluation_labels == 0, -1.0, 1.0).reshape(-1, 1),
        evaluation_labels,
        np.repeat(np.arange(1, 6, dtype=np.int64), 40),
    )
    return training, evaluation


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_policies = []
    original_factory = diagnostics._new_frozen_policy

    def capture_policy(
        received_seed: int,
        config: diagnostics.LearningDiagnosticsConfig,
    ):
        policy = original_factory(received_seed, config)
        created_policies.append(policy)
        return policy

    monkeypatch.setattr(diagnostics, "_new_frozen_policy", capture_policy)
    fixtures = diagnostics._build_diagnostic_fixtures(
        seed, diagnostics.LearningDiagnosticsConfig()
    )
    assert len(created_policies) == 2
    policy_digests_before = tuple(
        _policy_matrix_digests(policy) for policy in created_policies
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
    assert tuple(
        _policy_matrix_digests(policy) for policy in created_policies
    ) == policy_digests_before


def test_geometry_gate_rejects_a_single_overall_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, evaluation = _perfect_geometry_datasets()
    probe = FittedLinearProbe(np.array([1.0]), bias=0.0)
    original_predict = FittedLinearProbe.predict

    monkeypatch.setattr(diagnostics, "fit_linear_probe", lambda *_args, **_kwargs: probe)

    def predict_with_one_wrong_choice(
        self: FittedLinearProbe, states: np.ndarray
    ) -> np.ndarray:
        choices = original_predict(self, states)
        if states.shape == evaluation.states.shape:
            altered = choices.copy()
            altered[0] = 1 - altered[0]
            altered.flags.writeable = False
            return altered
        return choices

    monkeypatch.setattr(FittedLinearProbe, "predict", predict_with_one_wrong_choice)

    geometry = diagnostics._run_geometry(training, evaluation)

    assert geometry.evaluation == diagnostics.AccuracyCount(199, 200)
    assert geometry.evaluation_margins.minimum > 0.0
    assert geometry.geometry_passed is False


def test_geometry_gate_rejects_a_single_delay_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, evaluation = _perfect_geometry_datasets()
    probe = FittedLinearProbe(np.array([1.0]), bias=0.0)
    original_predict = FittedLinearProbe.predict

    monkeypatch.setattr(diagnostics, "fit_linear_probe", lambda *_args, **_kwargs: probe)

    def predict_with_delay_three_miss(
        self: FittedLinearProbe, states: np.ndarray
    ) -> np.ndarray:
        choices = original_predict(self, states)
        if states.shape == evaluation.states.shape:
            altered = choices.copy()
            altered[80] = 1 - altered[80]
            altered.flags.writeable = False
            return altered
        return choices

    monkeypatch.setattr(FittedLinearProbe, "predict", predict_with_delay_three_miss)

    geometry = diagnostics._run_geometry(training, evaluation)

    assert geometry.per_delay[2] == (3, diagnostics.AccuracyCount(39, 40))
    assert geometry.evaluation_margins.minimum > 0.0
    assert geometry.geometry_passed is False


def test_geometry_gate_rejects_an_imbalanced_delay_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, evaluation = _perfect_geometry_datasets()
    imbalanced_delays = np.concatenate(
        (
            np.full(41, 1, dtype=np.int64),
            np.full(39, 2, dtype=np.int64),
            np.repeat(np.arange(3, 6, dtype=np.int64), 40),
        )
    )
    imbalanced_evaluation = _hidden_dataset(
        evaluation.states, evaluation.labels, imbalanced_delays
    )
    probe = FittedLinearProbe(np.array([1.0]), bias=0.0)
    monkeypatch.setattr(diagnostics, "fit_linear_probe", lambda *_args, **_kwargs: probe)

    geometry = diagnostics._run_geometry(training, imbalanced_evaluation)

    assert geometry.evaluation == diagnostics.AccuracyCount(200, 200)
    assert geometry.evaluation_margins.minimum > 0.0
    assert geometry.per_delay == (
        (1, diagnostics.AccuracyCount(41, 41)),
        (2, diagnostics.AccuracyCount(39, 39)),
        (3, diagnostics.AccuracyCount(40, 40)),
        (4, diagnostics.AccuracyCount(40, 40)),
        (5, diagnostics.AccuracyCount(40, 40)),
    )
    assert geometry.geometry_passed is False


def test_geometry_gate_rejects_a_zero_normalized_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, evaluation = _perfect_geometry_datasets()
    zero_margin_states = evaluation.states.copy()
    zero_margin_states[0, 0] = 0.0
    zero_margin_evaluation = _hidden_dataset(
        zero_margin_states, evaluation.labels, evaluation.delays
    )
    probe = FittedLinearProbe(np.array([1.0]), bias=0.0)
    monkeypatch.setattr(diagnostics, "fit_linear_probe", lambda *_args, **_kwargs: probe)

    geometry = diagnostics._run_geometry(training, zero_margin_evaluation)

    assert geometry.evaluation == diagnostics.AccuracyCount(200, 200)
    assert geometry.per_delay == tuple(
        (delay, diagnostics.AccuracyCount(40, 40)) for delay in range(1, 6)
    )
    assert geometry.evaluation_margins.minimum == 0.0
    assert geometry.geometry_passed is False


def test_supervised_readout_uses_stable_softmax_and_immutable_greedy_snapshots() -> None:
    learning_rate = 0.25
    temperature = 2.0
    readout = diagnostics._DiagnosticSupervisedReadout(
        hidden_size=2,
        learning_rate=learning_rate,
        temperature=temperature,
    )

    assert readout._weights.dtype == np.dtype(np.float64)
    assert readout._biases.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(readout._weights, np.zeros((2, 2)))
    np.testing.assert_array_equal(readout._biases, np.zeros(2))
    assert readout.learning_rate == learning_rate
    assert readout.temperature == temperature

    tie = readout.select_greedy(np.array([1.0, -1.0], dtype=np.float64))
    assert tie.action_index == 0
    assert not tie.probabilities.flags.writeable
    with pytest.raises(ValueError):
        tie.probabilities[0] = 1.0

    readout._weights[:] = [[1_000.0, -1_000.0], [-1_000.0, 1_000.0]]
    readout._biases[:] = [10.0, -10.0]
    hidden = np.array([3.0, -2.0], dtype=np.float64)
    expected_logits = readout._weights @ hidden + readout._biases
    scaled = expected_logits / temperature
    expected_probabilities = np.exp(scaled - np.max(scaled))
    expected_probabilities /= expected_probabilities.sum()

    decision = readout.select_greedy(hidden)

    assert decision.action_index == int(np.argmax(expected_probabilities))
    np.testing.assert_allclose(
        decision.probabilities, expected_probabilities, rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize(
    ("hidden_size", "learning_rate", "temperature"),
    [
        (0, 0.05, 1.0),
        (True, 0.05, 1.0),
        (2, 0.0, 1.0),
        (2, True, 1.0),
        (2, np.inf, 1.0),
        (2, 0.05, 0.0),
        (2, 0.05, True),
        (2, 0.05, np.nan),
    ],
)
def test_supervised_readout_constructor_matches_config_validation(
    hidden_size: object, learning_rate: object, temperature: object
) -> None:
    with pytest.raises(ValueError):
        diagnostics._DiagnosticSupervisedReadout(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            temperature=temperature,
        )


def test_supervised_label_update_matches_literal_softmax_gradient() -> None:
    learning_rate = 0.2
    temperature = 0.75
    readout = diagnostics._DiagnosticSupervisedReadout(
        hidden_size=3,
        learning_rate=learning_rate,
        temperature=temperature,
    )
    readout._weights[:] = [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]]
    readout._biases[:] = [0.15, -0.2]
    hidden = np.array([0.5, -1.5, 2.0], dtype=np.float64)
    label = 1
    before_weights = readout._weights.copy()
    before_biases = readout._biases.copy()
    logits = before_weights @ hidden + before_biases
    scaled = logits / temperature
    exponentials = np.exp(scaled - np.max(scaled))
    probabilities = exponentials / exponentials.sum()
    delta = np.eye(2, dtype=np.float64)[label] - probabilities
    expected_weights = before_weights + learning_rate * np.outer(delta, hidden)
    expected_biases = before_biases + learning_rate * delta

    observed = readout.observe_label(hidden, label)

    assert not observed.flags.writeable
    np.testing.assert_allclose(observed, probabilities, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        readout._weights, expected_weights, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(readout._biases, expected_biases, rtol=0.0, atol=1e-12)
    assert not hasattr(readout, "learn")
    assert not hasattr(readout, "select_for_training")


@pytest.mark.parametrize(
    "hidden",
    [
        np.array([1.0], dtype=np.float64),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([np.nan, 0.0], dtype=np.float64),
        np.array([np.inf, 0.0], dtype=np.float64),
    ],
)
def test_supervised_readout_rejects_invalid_hidden_without_mutation(
    hidden: np.ndarray,
) -> None:
    readout = diagnostics._DiagnosticSupervisedReadout(hidden_size=2)
    weights_before = readout._weights.copy()
    biases_before = readout._biases.copy()

    with pytest.raises(ValueError):
        readout.select_greedy(hidden)
    with pytest.raises(ValueError):
        readout.observe_label(hidden, 0)

    np.testing.assert_array_equal(readout._weights, weights_before)
    np.testing.assert_array_equal(readout._biases, biases_before)


@pytest.mark.parametrize("label", [True, -1, 2, np.int64(0), 0.0, "0"])
def test_supervised_readout_rejects_invalid_label_without_mutation(
    label: object,
) -> None:
    readout = diagnostics._DiagnosticSupervisedReadout(hidden_size=2)
    weights_before = readout._weights.copy()
    biases_before = readout._biases.copy()

    with pytest.raises(ValueError):
        readout.observe_label(np.array([1.0, -1.0], dtype=np.float64), label)

    np.testing.assert_array_equal(readout._weights, weights_before)
    np.testing.assert_array_equal(readout._biases, biases_before)


def test_supervised_diagnostic_checkpoints_are_read_only_and_label_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = np.array([[-1.0], [1.0], [-1.0], [1.0], [-1.0]] * 4)
    labels = np.array([0, 1, 0, 1, 0] * 4, dtype=np.int64)
    delays = np.tile(np.arange(1, 6, dtype=np.int64), 4)
    training = _hidden_dataset(states, labels, delays)
    evaluation = _hidden_dataset(states, labels, delays)
    config = diagnostics.LearningDiagnosticsConfig(
        hidden_size=1,
        training_episodes=20,
        evaluation_blocks=2,
        checkpoint_interval=10,
    )
    received: list[tuple[np.ndarray, int]] = []
    original_observe = diagnostics._DiagnosticSupervisedReadout.observe_label
    original_digest = diagnostics._DiagnosticSupervisedReadout.parameter_digest
    digests_before_evaluation: list[str] = []

    def observe_spy(
        self: object, hidden: np.ndarray, label: int
    ) -> np.ndarray:
        assert isinstance(hidden, np.ndarray)
        assert hidden.dtype == np.dtype(np.float64)
        assert hidden.ndim == 1
        assert not np.shares_memory(hidden, training.states)
        received.append((hidden.copy(), label))
        return original_observe(self, hidden, label)

    def greedy_spy(self: object, hidden: np.ndarray):
        digests_before_evaluation.append(original_digest(self))
        return original_greedy(self, hidden)

    original_greedy = diagnostics._DiagnosticSupervisedReadout.select_greedy
    monkeypatch.setattr(diagnostics._DiagnosticSupervisedReadout, "observe_label", observe_spy)
    monkeypatch.setattr(diagnostics._DiagnosticSupervisedReadout, "select_greedy", greedy_spy)
    monkeypatch.setattr(
        diagnostics,
        "RewardModulatedReadout",
        lambda *_args, **_kwargs: pytest.fail("supervised path must not use reward readout"),
        raising=False,
    )

    result = diagnostics._run_supervised_diagnostic(training, evaluation, config)

    assert tuple(checkpoint.episode for checkpoint in result.checkpoints) == (10, 20)
    assert tuple(checkpoint.parameter_digest for checkpoint in result.checkpoints) == (
        digests_before_evaluation[0],
        digests_before_evaluation[20],
    )
    assert len(received) == 20
    assert tuple(label for _, label in received) == tuple(training.labels.tolist())
    for hidden, expected in zip(received, training.states, strict=True):
        np.testing.assert_array_equal(hidden[0], expected)
    assert result.overall.total == 20
    assert tuple(delay for delay, _ in result.per_delay) == (1, 2, 3, 4, 5)
    assert tuple(count.total for _, count in result.per_delay) == (4, 4, 4, 4, 4)


@pytest.mark.parametrize(
    ("overall_correct", "per_delay_correct", "expected_passed"),
    [
        (180, (36, 36, 36, 36, 36), True),
        (179, (35, 36, 36, 36, 36), False),
        (194, (34, 40, 40, 40, 40), True),
        (193, (33, 40, 40, 40, 40), False),
    ],
    ids=(
        "overall-exact-boundary",
        "overall-one-below",
        "per-delay-exact-boundary",
        "per-delay-one-below",
    ),
)
def test_supervised_gate_uses_exact_overall_and_per_delay_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    overall_correct: int,
    per_delay_correct: tuple[int, int, int, int, int],
    expected_passed: bool,
) -> None:
    config = diagnostics.LearningDiagnosticsConfig(
        hidden_size=1,
        training_episodes=20,
        evaluation_blocks=20,
        checkpoint_interval=10,
    )
    training = _hidden_dataset(
        np.tile(np.array([[-1.0], [1.0]], dtype=np.float64), (10, 1)),
        np.tile(np.array([0, 1], dtype=np.int64), 10),
        np.tile(np.arange(1, 6, dtype=np.int64), 4),
    )
    evaluation = _hidden_dataset(
        np.tile(np.array([[-1.0], [1.0]], dtype=np.float64), (100, 1)),
        np.tile(np.array([0, 1], dtype=np.int64), 100),
        np.repeat(np.arange(1, 6, dtype=np.int64), 40),
    )
    expected_overall = diagnostics.AccuracyCount(overall_correct, 200)
    expected_per_delay = tuple(
        (delay, diagnostics.AccuracyCount(correct, 40))
        for delay, correct in enumerate(per_delay_correct, start=1)
    )

    monkeypatch.setattr(
        diagnostics,
        "_evaluate_supervised_readout",
        lambda *_args: (expected_overall, expected_per_delay),
    )

    result = diagnostics._run_supervised_diagnostic(training, evaluation, config)

    assert result.overall == expected_overall
    assert result.per_delay == expected_per_delay
    assert result.supervised_passed is expected_passed


def _trajectory_test_fixtures() -> tuple[
    diagnostics._DiagnosticFixtures,
    diagnostics.LearningDiagnosticsConfig,
]:
    task = DelayedCueTask()
    training_episodes = _build_fixtures(task, np.random.default_rng(123), 2)
    evaluation_episodes = _build_fixtures(task, np.random.default_rng(456), 2)
    training_labels = np.asarray(
        [episode.correct_action_index for episode in training_episodes], dtype=np.int64
    )
    training_delays = np.asarray(
        [episode.delay_steps for episode in training_episodes], dtype=np.int64
    )
    evaluation_labels = np.asarray(
        [episode.correct_action_index for episode in evaluation_episodes], dtype=np.int64
    )
    evaluation_delays = np.asarray(
        [episode.delay_steps for episode in evaluation_episodes], dtype=np.int64
    )
    training_states = np.linspace(
        -1.5, 1.5, num=40, dtype=np.float64
    ).reshape(20, 2)
    evaluation_states = np.linspace(
        -2.0, 2.0, num=40, dtype=np.float64
    ).reshape(20, 2)
    return (
        diagnostics._DiagnosticFixtures(
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
            training=_hidden_dataset(training_states, training_labels, training_delays),
            evaluation=_hidden_dataset(
                evaluation_states, evaluation_labels, evaluation_delays
            ),
        ),
        diagnostics.LearningDiagnosticsConfig(
            hidden_size=2,
            training_episodes=20,
            evaluation_blocks=2,
            checkpoint_interval=10,
        ),
    )


def _patch_trajectory_hidden_states(
    monkeypatch: pytest.MonkeyPatch,
    fixtures: diagnostics._DiagnosticFixtures,
) -> None:
    state_by_episode = {
        id(episode): state
        for episodes, states in (
            (fixtures.training_episodes, fixtures.training.states),
            (fixtures.evaluation_episodes, fixtures.evaluation.states),
        )
        for episode, state in zip(episodes, states, strict=True)
    }
    class FrozenPolicy:
        _input_weights = np.zeros((2, 4), dtype=np.float64)
        _recurrent_weights = np.zeros((2, 2), dtype=np.float64)
        _output_weights = np.zeros((2, 2), dtype=np.float64)

    monkeypatch.setattr(diagnostics, "_new_frozen_policy", lambda *_args: FrozenPolicy())
    monkeypatch.setattr(
        diagnostics,
        "_decision_hidden",
        lambda _policy, episode, **_kwargs: np.array(
            state_by_episode[id(episode)], dtype=np.float64, copy=True
        ),
    )


def _trajectory_readout(
    training_actions: tuple[int, ...],
    probability_rows: tuple[tuple[float, float], ...],
    *,
    mutate_on_greedy: bool = False,
    leave_pending: bool = False,
):
    class TrajectoryReadout:
        def __init__(self, *_args: object) -> None:
            self._actions = iter(training_actions)
            self._probabilities = iter(probability_rows)
            self._pending = False
            self._updates = 0
            self.received_rewards: list[float] = []

        @property
        def has_pending_feedback(self) -> bool:
            return self._pending

        def select_for_training(
            self,
            hidden: np.ndarray,
            legal_actions: tuple[int, int],
            rng: np.random.Generator,
        ) -> RewardReadoutDecision:
            assert hidden.dtype == np.dtype(np.float64)
            assert legal_actions == (0, 1)
            assert isinstance(rng, np.random.Generator)
            self._pending = True
            probabilities = np.asarray(next(self._probabilities), dtype=np.float64)
            probabilities.flags.writeable = False
            return RewardReadoutDecision(
                action_index=next(self._actions),
                logits=np.zeros(2, dtype=np.float64),
                probabilities=probabilities,
            )

        def learn(self, reward: float) -> None:
            assert self._pending
            assert type(reward) is float
            self.received_rewards.append(reward)
            self._updates += 1
            self._pending = leave_pending

        def select_greedy(
            self, hidden: np.ndarray, legal_actions: tuple[int, int]
        ) -> RewardReadoutDecision:
            assert hidden.dtype == np.dtype(np.float64)
            assert legal_actions == (0, 1)
            if mutate_on_greedy:
                self._updates += 1
            probabilities = np.array([0.5, 0.5], dtype=np.float64)
            probabilities.flags.writeable = False
            return RewardReadoutDecision(
                action_index=0,
                logits=np.zeros(2, dtype=np.float64),
                probabilities=probabilities,
            )

        def parameter_digest(self) -> str:
            return f"{self._updates:064x}"

    return TrajectoryReadout


def test_reward_trajectory_uses_post_selection_correct_probability_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures, config = _trajectory_test_fixtures()
    _patch_trajectory_hidden_states(monkeypatch, fixtures)
    probabilities = tuple(
        (0.2, 0.8) if index % 2 else (0.75, 0.25)
        for index in range(config.training_episodes)
    )
    readout_type = _trajectory_readout(
        tuple(index % 2 for index in range(config.training_episodes)), probabilities
    )
    monkeypatch.setattr(diagnostics, "RewardModulatedReadout", readout_type)

    trajectory = diagnostics._run_reward_trajectory(91, fixtures, config)

    labels = fixtures.training.labels
    correct_probabilities = np.asarray(
        [row[int(label)] for row, label in zip(probabilities, labels, strict=True)],
        dtype=np.float64,
    )
    assert tuple(checkpoint.episode for checkpoint in trajectory.checkpoints) == (10, 20)
    assert tuple(checkpoint.parameter_digest for checkpoint in trajectory.checkpoints) == (
        f"{10:064x}",
        f"{20:064x}",
    )
    final_checkpoint = trajectory.checkpoints[-1]
    assert final_checkpoint.mean_correct_action_probability == pytest.approx(
        float(np.mean(correct_probabilities)), abs=1e-12
    )
    assert final_checkpoint.percentile_10_correct_action_probability == pytest.approx(
        float(np.percentile(correct_probabilities, 10, method="linear")), abs=1e-12
    )
    assert final_checkpoint.correct_action_probabilities == tuple(
        correct_probabilities.tolist()
    )


def test_reward_trajectory_independently_records_gradients_and_block_cosines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures, config = _trajectory_test_fixtures()
    _patch_trajectory_hidden_states(monkeypatch, fixtures)
    probabilities = tuple(
        (0.2, 0.8) if index % 3 else (0.65, 0.35)
        for index in range(config.training_episodes)
    )
    actions = tuple((index + 1) % 2 for index in range(config.training_episodes))
    readout_type = _trajectory_readout(actions, probabilities)
    monkeypatch.setattr(diagnostics, "RewardModulatedReadout", readout_type)

    trajectory = diagnostics._run_reward_trajectory(92, fixtures, config)

    expected_supervised = []
    expected_sampled = []
    expected_ratios = []
    task = DelayedCueTask()
    for hidden, label, action, probabilities_row, episode in zip(
        fixtures.training.states,
        fixtures.training.labels,
        actions,
        probabilities,
        fixtures.training_episodes,
        strict=True,
    ):
        probability = np.asarray(probabilities_row, dtype=np.float64)
        augmented = np.concatenate((hidden, [1.0]))
        supervised = np.outer(np.eye(2, dtype=np.float64)[label] - probability, augmented)
        reward = task.reward(episode, action)
        sampled = reward * np.outer(
            np.eye(2, dtype=np.float64)[action] - probability, augmented
        )
        expected_supervised.append(tuple(supervised.ravel().tolist()))
        expected_sampled.append(tuple(sampled.ravel().tolist()))
        expected_bandit = sum(
            probability[candidate]
            * (1.0 if candidate == label else -1.0)
            * np.outer(
                np.eye(2, dtype=np.float64)[candidate] - probability,
                augmented,
            )
            for candidate in (0, 1)
        )
        expected_ratios.append(
            float(np.linalg.norm(expected_bandit) / np.linalg.norm(supervised))
        )

    first_block = trajectory.gradient_blocks[0]
    assert first_block.sampled_episode_gradients == tuple(expected_sampled[:10])
    assert first_block.supervised_episode_gradients == tuple(expected_supervised[:10])
    assert first_block.expected_bandit_to_supervised_norm_ratios == pytest.approx(
        expected_ratios[:10], abs=1e-12
    )
    summed_sampled = np.sum(np.asarray(expected_sampled[:10]), axis=0)
    summed_supervised = np.sum(np.asarray(expected_supervised[:10]), axis=0)
    expected_cosine = float(
        np.dot(summed_sampled, summed_supervised)
        / (np.linalg.norm(summed_sampled) * np.linalg.norm(summed_supervised))
    )
    assert first_block.sampled_gradient == pytest.approx(
        tuple(summed_sampled.tolist()), abs=1e-12
    )
    assert first_block.supervised_gradient == pytest.approx(
        tuple(summed_supervised.tolist()), abs=1e-12
    )
    assert first_block.cosine == pytest.approx(expected_cosine, abs=1e-12)
    assert first_block.mean_expected_bandit_to_supervised_norm_ratio == pytest.approx(
        float(np.mean(expected_ratios[:10])), abs=1e-12
    )


def test_reward_trajectory_reports_null_cosine_for_zero_norm_gradient_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures, config = _trajectory_test_fixtures()
    zero_states = np.zeros_like(fixtures.training.states)
    labels = np.tile(np.array([0, 1], dtype=np.int64), 10)
    delays = np.tile(np.arange(1, 6, dtype=np.int64), 4)
    fixtures = diagnostics._DiagnosticFixtures(
        training_episodes=fixtures.training_episodes,
        evaluation_episodes=fixtures.evaluation_episodes,
        training=_hidden_dataset(zero_states, labels, delays),
        evaluation=fixtures.evaluation,
    )
    _patch_trajectory_hidden_states(monkeypatch, fixtures)
    readout_type = _trajectory_readout(
        tuple(labels.tolist()), tuple((0.5, 0.5) for _ in labels)
    )
    monkeypatch.setattr(diagnostics, "RewardModulatedReadout", readout_type)

    trajectory = diagnostics._run_reward_trajectory(93, fixtures, config)

    assert tuple(block.cosine for block in trajectory.gradient_blocks) == (None, None)
    assert trajectory.zero_norm_block_count == 2
    assert trajectory.checkpoints[-1].zero_norm_block_count == 2


class _BoundaryEpisode:
    def __init__(self, label: int, delay: int, events: list[tuple[str, object]]) -> None:
        self._label = label
        self._delay = delay
        self._events = events

    @property
    def correct_action_index(self) -> int:
        assert any(kind == "select" for kind, _ in self._events)
        self._events.append(("label", self._label))
        return self._label

    @property
    def delay_steps(self) -> int:
        assert any(kind == "select" for kind, _ in self._events)
        self._events.append(("delay", self._delay))
        return self._delay


class _SelectionLockedVector(np.ndarray):
    def __new__(
        cls,
        values: np.ndarray,
        name: str,
        events: list[tuple[str, object]],
    ) -> Self:
        result = np.asarray(values).view(cls)
        result._name = name
        result._events = events
        return result

    def __array_finalize__(self, source: object) -> None:
        if source is not None:
            self._name = getattr(source, "_name", "metadata")
            self._events = getattr(source, "_events", [])

    def __getitem__(self, key: object):
        assert any(kind in {"select", "greedy"} for kind, _ in self._events)
        self._events.append((self._name, key))
        return super().__getitem__(key)


def test_reward_trajectory_preserves_reward_learner_boundary_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fixtures, config = _trajectory_test_fixtures()
    events: list[tuple[str, object]] = []
    training_episodes = tuple(
        _BoundaryEpisode(int(label), int(delay), events)
        for label, delay in zip(
            original_fixtures.training.labels,
            original_fixtures.training.delays,
            strict=True,
        )
    )
    evaluation_episodes = tuple(
        _BoundaryEpisode(int(label), int(delay), events)
        for label, delay in zip(
            original_fixtures.evaluation.labels,
            original_fixtures.evaluation.delays,
            strict=True,
        )
    )
    fixtures = diagnostics._DiagnosticFixtures(
        training_episodes=training_episodes,
        evaluation_episodes=evaluation_episodes,
        training=original_fixtures.training,
        evaluation=original_fixtures.evaluation,
    )
    object.__setattr__(
        fixtures.training,
        "labels",
        _SelectionLockedVector(fixtures.training.labels, "label", events),
    )
    object.__setattr__(
        fixtures.evaluation,
        "labels",
        _SelectionLockedVector(fixtures.evaluation.labels, "label", events),
    )
    object.__setattr__(
        fixtures.evaluation,
        "delays",
        _SelectionLockedVector(fixtures.evaluation.delays, "delay", events),
    )
    _patch_trajectory_hidden_states(monkeypatch, fixtures)

    class BoundaryTask:
        def reward(self, episode: _BoundaryEpisode, action: int) -> float:
            events.append(("reward", action))
            assert action == 1
            assert episode._label in (0, 1)
            return -1.0

    class BoundaryReadout:
        def __init__(self, *_args: object) -> None:
            self.pending = False
            self.learned: list[float] = []
            self.updates = 0

        @property
        def has_pending_feedback(self) -> bool:
            return self.pending

        def select_for_training(
            self, hidden: np.ndarray, legal_actions: tuple[int, int], rng: np.random.Generator
        ) -> RewardReadoutDecision:
            assert legal_actions == (0, 1)
            assert isinstance(rng, np.random.Generator)
            self.pending = True
            events.append(("select", 1))
            probabilities = np.array([0.4, 0.6], dtype=np.float64)
            probabilities.flags.writeable = False
            return RewardReadoutDecision(1, np.zeros(2), probabilities)

        def learn(self, reward: float) -> None:
            events.append(("learn", reward))
            assert reward == -1.0
            self.learned.append(reward)
            self.pending = False
            self.updates += 1

        def select_greedy(
            self, hidden: np.ndarray, legal_actions: tuple[int, int]
        ) -> RewardReadoutDecision:
            events.append(("greedy", 0))
            probabilities = np.array([0.5, 0.5], dtype=np.float64)
            probabilities.flags.writeable = False
            return RewardReadoutDecision(0, np.zeros(2), probabilities)

        def parameter_digest(self) -> str:
            return f"{self.updates:064x}"

    monkeypatch.setattr(diagnostics, "DelayedCueTask", BoundaryTask)
    monkeypatch.setattr(diagnostics, "RewardModulatedReadout", BoundaryReadout)

    diagnostics._run_reward_trajectory(94, fixtures, config)

    learner_events = [
        (kind, value) for kind, value in events if kind in {"select", "reward", "learn"}
    ]
    assert learner_events == [
        item
        for _ in range(config.training_episodes)
        for item in (("select", 1), ("reward", 1), ("learn", -1.0))
    ]
    first_select = next(index for index, item in enumerate(events) if item[0] == "select")
    assert all(
        index > first_select
        for index, (kind, _value) in enumerate(events)
        if kind in {"label", "delay"}
    )
    assert sum(kind == "label" for kind, _ in events) == 60
    assert sum(kind == "delay" for kind, _ in events) == 40


@pytest.mark.parametrize(
    ("mutate_on_greedy", "leave_pending", "error"),
    [
        (True, False, "checkpoint evaluation mutated parameters"),
        (False, True, "pending feedback"),
    ],
)
def test_reward_trajectory_rejects_checkpoint_mutation_and_pending_feedback(
    monkeypatch: pytest.MonkeyPatch,
    mutate_on_greedy: bool,
    leave_pending: bool,
    error: str,
) -> None:
    fixtures, config = _trajectory_test_fixtures()
    _patch_trajectory_hidden_states(monkeypatch, fixtures)
    readout_type = _trajectory_readout(
        tuple(0 for _ in range(config.training_episodes)),
        tuple((0.5, 0.5) for _ in range(config.training_episodes)),
        mutate_on_greedy=mutate_on_greedy,
        leave_pending=leave_pending,
    )
    monkeypatch.setattr(diagnostics, "RewardModulatedReadout", readout_type)

    with pytest.raises(RuntimeError, match=error):
        diagnostics._run_reward_trajectory(95, fixtures, config)


_PHASE_2B_EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "experiments"
    / "phase-2b-failure.json"
)


@pytest.mark.parametrize("seed", [7, 17, 29])
def test_reward_trajectory_matches_portable_phase_two_b_evidence_locally(seed: int) -> None:
    evidence = json.loads(_PHASE_2B_EVIDENCE.read_text(encoding="utf-8"))
    runtime = run_reward_learning_benchmark()
    expected_entry = next(item for item in evidence["results"] if item["seed"] == seed)
    runtime_entry = next(item for item in runtime["results"] if item["seed"] == seed)
    config = diagnostics.LearningDiagnosticsConfig()
    fixtures = diagnostics._build_diagnostic_fixtures(seed, config)

    first = diagnostics._run_reward_trajectory(seed, fixtures, config)
    second = diagnostics._run_reward_trajectory(seed, fixtures, config)

    assert diagnostics._match_phase_2b_evidence(seed, runtime_entry)
    assert first.overall == diagnostics.AccuracyCount(
        expected_entry["post_training"]["correct"], 200
    )
    assert first.per_delay == tuple(
        (item["delay"], diagnostics.AccuracyCount(item["correct"], item["total"]))
        for item in expected_entry["per_delay"]
    )
    assert first.total_training_reward == expected_entry["total_training_reward"]
    assert first.training_choice_digest == expected_entry["normal_training_choice_digest"]
    assert first.training_reward_digest == expected_entry["normal_training_reward_digest"]
    assert first.parameter_digest_before != first.parameter_digest_after
    assert first.matrix_digests_before == first.matrix_digests_after
    assert first.parameter_digest_before == second.parameter_digest_before
    assert first.parameter_digest_after == second.parameter_digest_after
    assert first.matrix_digests_before == second.matrix_digests_before
    assert first.matrix_digests_after == second.matrix_digests_after
    assert first == second
