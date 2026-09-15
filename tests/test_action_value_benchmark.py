from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest

import neural_state_machine
import neural_state_machine.action_value_benchmark as benchmark_module
from neural_state_machine.action_value import ActionValueUpdate, NormalizedActionValue
from neural_state_machine.action_value_benchmark import (
    ActionValueCheckpoint,
    ActionValueBenchmarkConfig,
    ActionValueExperimentResult,
    _build_fixture_bundle,
    _evaluate,
    _new_learner,
    _new_policy,
    _passes_acceptance,
    _permute_reward_blocks,
    _run_once,
    _train_normal,
    _train_shuffled,
    run_action_value_benchmark,
    run_action_value_experiment,
)
from neural_state_machine.memory_benchmark import AccuracyCount
from neural_state_machine.memory_task import Cue, DelayedCueEpisode, DelayedCueTask
from neural_state_machine.policy import RecurrentPolicy
from neural_state_machine.reward_learning import (
    _build_fixtures,
    _matrix_copies,
    _matrix_digests,
    _require_frozen,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"hidden_size": -1},
        {"hidden_size": True},
        {"hidden_size": 2.5},
        {"training_episodes": 0},
        {"training_episodes": 11},
        {"training_episodes": True},
        {"evaluation_blocks": 0},
        {"evaluation_blocks": -1},
        {"evaluation_blocks": True},
        {"checkpoint_interval": 0},
        {"checkpoint_interval": 11},
        {"checkpoint_interval": True},
        {"checkpoint_interval": 2.5},
        {"training_episodes": 20, "checkpoint_interval": 30},
        {"training_episodes": 30, "checkpoint_interval": 20},
        {"recurrent_radius": -0.1},
        {"recurrent_radius": 1.0},
        {"recurrent_radius": True},
        {"recurrent_radius": np.nan},
        {"recurrent_radius": np.inf},
        {"step_size": 0.0},
        {"step_size": 1.1},
        {"step_size": True},
        {"step_size": np.nan},
        {"step_size": np.inf},
    ],
)
def test_config_rejects_values_outside_the_frozen_protocol(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ActionValueBenchmarkConfig(**kwargs)


def test_config_has_only_preregistered_fields_and_is_frozen() -> None:
    config = ActionValueBenchmarkConfig()

    assert config == ActionValueBenchmarkConfig(
        hidden_size=64,
        recurrent_radius=0.9,
        step_size=0.1,
        training_episodes=2_000,
        evaluation_blocks=20,
        checkpoint_interval=100,
    )
    assert tuple(field.name for field in dataclasses.fields(config)) == (
        "hidden_size",
        "recurrent_radius",
        "step_size",
        "training_episodes",
        "evaluation_blocks",
        "checkpoint_interval",
    )
    with pytest.raises(FrozenInstanceError):
        config.hidden_size = 32


def _assert_episode_equal(
    actual: DelayedCueEpisode,
    expected: DelayedCueEpisode,
) -> None:
    assert actual.cue is expected.cue
    assert actual.delay_steps == expected.delay_steps
    assert actual.correct_action_index == expected.correct_action_index
    np.testing.assert_array_equal(actual.cue_stimulus, expected.cue_stimulus)
    assert len(actual.delay_stimuli) == len(expected.delay_stimuli)
    for actual_delay, expected_delay in zip(
        actual.delay_stimuli, expected.delay_stimuli, strict=True
    ):
        np.testing.assert_array_equal(actual_delay, expected_delay)
    np.testing.assert_array_equal(actual.decision_stimulus, expected.decision_stimulus)


@pytest.mark.parametrize("seed", [0, 7, 29])
def test_fixture_bundle_reconstructs_frozen_phase_two_b_lineages(seed: int) -> None:
    config = ActionValueBenchmarkConfig()
    first = _build_fixture_bundle(seed, config)
    second = _build_fixture_bundle(seed, config)
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

    assert len(first.training) == len(expected_training) == 2_000
    assert len(first.evaluation) == len(expected_evaluation) == 200
    for actual, expected in zip(first.training, expected_training, strict=True):
        _assert_episode_equal(actual, expected)
    for actual, expected in zip(first.evaluation, expected_evaluation, strict=True):
        _assert_episode_equal(actual, expected)
    for actual, reconstructed in zip(first.training, second.training, strict=True):
        _assert_episode_equal(actual, reconstructed)
    for actual, reconstructed in zip(first.evaluation, second.evaluation, strict=True):
        _assert_episode_equal(actual, reconstructed)
    assert first.training_fixture_digest == second.training_fixture_digest
    assert first.evaluation_fixture_digest == second.evaluation_fixture_digest
    assert len(first.training_fixture_digest) == 64
    assert len(first.evaluation_fixture_digest) == 64


def test_default_fixture_bundle_is_balanced_and_fully_immutable() -> None:
    bundle = _build_fixture_bundle(7, ActionValueBenchmarkConfig())
    cases = {(cue, delay) for cue in Cue for delay in range(1, 6)}

    assert all(
        {(episode.cue, episode.delay_steps) for episode in block} == cases
        for block in (
            bundle.training[start : start + 10] for start in range(0, len(bundle.training), 10)
        )
    )
    assert all(
        {(episode.cue, episode.delay_steps) for episode in block} == cases
        for block in (
            bundle.evaluation[start : start + 10] for start in range(0, len(bundle.evaluation), 10)
        )
    )
    assert sum(row.correct_action_index == 0 for row in bundle.evaluation) == 100
    assert sum(row.correct_action_index == 1 for row in bundle.evaluation) == 100
    assert {
        delay: sum(row.delay_steps == delay for row in bundle.evaluation) for delay in range(1, 6)
    } == {delay: 40 for delay in range(1, 6)}
    stimuli = [
        stimulus
        for episode in (*bundle.training, *bundle.evaluation)
        for stimulus in (
            episode.cue_stimulus,
            *episode.delay_stimuli,
            episode.decision_stimulus,
        )
    ]
    assert all(stimulus.dtype == np.float64 for stimulus in stimuli)
    assert all(stimulus.flags.writeable is False for stimulus in stimuli)


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray | None]] = []
        self._step = 0

    def reset_state(self) -> None:
        self.calls.append(("reset_state", None))
        self._step = 0

    def advance(self, stimulus: np.ndarray) -> np.ndarray:
        assert type(stimulus) is np.ndarray
        assert stimulus.dtype == np.float64
        self.calls.append(("advance", stimulus))
        self._step += 1
        return np.full(3, self._step, dtype=np.float64)


class _EvaluationOnlyLearner:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, tuple[int, int]]] = []

    def select_greedy(self, hidden: np.ndarray, legal: tuple[int, int]) -> SimpleNamespace:
        assert type(hidden) is np.ndarray
        assert hidden.dtype == np.float64
        assert hidden.flags.writeable is False
        assert legal == (0, 1)
        self.calls.append((hidden, legal))
        return SimpleNamespace(
            action_index=0,
            action_values=np.array([0.75, -0.25], dtype=np.float64),
        )

    def select_for_training(self, *args: object) -> None:
        raise AssertionError("evaluation must not select for training")

    def learn(self, *args: object) -> None:
        raise AssertionError("evaluation must not learn")


_SCORING_EVENTS: list[str] = []


class _ScoringAccessEpisode(DelayedCueEpisode):
    def __getattribute__(self, name: str) -> object:
        if name in {"cue", "correct_action_index", "delay_steps"}:
            _SCORING_EVENTS.append(name)
        return super().__getattribute__(name)


class _SelectionOrderLearner(_EvaluationOnlyLearner):
    def select_greedy(self, hidden: np.ndarray, legal: tuple[int, int]) -> SimpleNamespace:
        _SCORING_EVENTS.append("select_greedy")
        return super().select_greedy(hidden, legal)


def test_evaluation_passes_only_hidden_and_legal_indices_to_learner() -> None:
    fixtures = _build_fixtures(DelayedCueTask(), np.random.default_rng(101), 2)
    policy = _RecordingPolicy()
    learner = _EvaluationOnlyLearner()

    result = _evaluate(
        policy,
        learner,
        fixtures,
        reset_before_decision=False,
    )

    assert len(learner.calls) == 20
    assert result.actions == (0,) * 20
    assert result.overall == AccuracyCount(10, 20)
    assert [count for _, count in result.per_delay] == [AccuracyCount(2, 4)] * 5
    assert result.margin_mean == pytest.approx(0.0)
    assert result.margin_p10 == pytest.approx(-1.0)
    assert result.margin_minimum == pytest.approx(-1.0)
    assert {name for name, _ in policy.calls} == {"reset_state", "advance"}


def test_evaluation_reads_scoring_metadata_only_after_action_selection() -> None:
    originals = _build_fixtures(DelayedCueTask(), np.random.default_rng(103), 1)
    episodes = tuple(
        _ScoringAccessEpisode(
            cue=original.cue,
            delay_steps=original.delay_steps,
            cue_stimulus=original.cue_stimulus,
            delay_stimuli=original.delay_stimuli,
            decision_stimulus=original.decision_stimulus,
            correct_action_index=original.correct_action_index,
        )
        for original in originals
    )
    _SCORING_EVENTS.clear()

    _evaluate(
        _RecordingPolicy(),
        _SelectionOrderLearner(),
        episodes,
        reset_before_decision=False,
    )

    assert _SCORING_EVENTS == [
        event
        for _ in episodes
        for event in ("select_greedy", "correct_action_index", "delay_steps")
    ]


class _HiddenRecordingLearner:
    def __init__(self, learner: NormalizedActionValue) -> None:
        self.learner = learner
        self.hidden_states: list[np.ndarray] = []

    def select_greedy(self, hidden: np.ndarray, legal: tuple[int, int]) -> object:
        self.hidden_states.append(hidden.copy())
        return self.learner.select_greedy(hidden, legal)


def _sha256_actions(actions: tuple[int, ...]) -> str:
    return hashlib.sha256(bytes(actions)).hexdigest()


def test_real_evaluation_is_balanced_repeatable_and_mutation_free() -> None:
    config = ActionValueBenchmarkConfig()
    fixtures = _build_fixture_bundle(17, config).evaluation
    policy = RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=17,
        recurrent_radius=config.recurrent_radius,
    )
    learner = NormalizedActionValue(config.hidden_size, 2, step_size=config.step_size)
    learner_parameters = learner.parameter_snapshot()
    learner_digest = learner.parameter_digest()
    pending_before = learner.has_pending_feedback
    matrix_copies = _matrix_copies(policy)
    matrix_digests = _matrix_digests(policy)

    recording_learner = _HiddenRecordingLearner(learner)
    ordinary = _evaluate(policy, recording_learner, fixtures, reset_before_decision=False)
    reset = _evaluate(policy, recording_learner, fixtures, reset_before_decision=True)

    assert ordinary.actions == reset.actions == (0,) * 200
    assert ordinary.overall == reset.overall == AccuracyCount(100, 200)
    assert (
        ordinary.per_delay
        == reset.per_delay
        == tuple((delay, AccuracyCount(20, 40)) for delay in range(1, 6))
    )
    assert ordinary.action_digest == reset.action_digest == _sha256_actions((0,) * 200)
    assert reset.all_hidden_equal is True
    reset_hidden = recording_learner.hidden_states[200:]
    assert len(reset_hidden) == 200
    assert all(np.array_equal(reset_hidden[0], hidden) for hidden in reset_hidden[1:])
    assert ordinary.hidden_digest != reset.hidden_digest
    assert ordinary.margin_mean == reset.margin_mean == 0.0
    assert ordinary.margin_p10 == reset.margin_p10 == 0.0
    assert ordinary.margin_minimum == reset.margin_minimum == 0.0
    np.testing.assert_array_equal(learner.parameter_snapshot(), learner_parameters)
    assert learner.parameter_digest() == learner_digest
    assert learner.has_pending_feedback is pending_before is False
    _require_frozen(matrix_copies, policy)
    assert _matrix_digests(policy) == matrix_digests


def test_reset_evaluation_hidden_digest_is_seed_repeatable() -> None:
    config = ActionValueBenchmarkConfig()
    fixtures = _build_fixture_bundle(29, config).evaluation

    def evaluate_once() -> object:
        return _evaluate(
            RecurrentPolicy(
                input_size=4,
                action_count=2,
                hidden_size=config.hidden_size,
                seed=29,
                recurrent_radius=config.recurrent_radius,
            ),
            NormalizedActionValue(config.hidden_size, 2, config.step_size),
            fixtures,
            reset_before_decision=True,
        )

    first = evaluate_once()
    second = evaluate_once()
    assert first == second
    assert first.all_hidden_equal is True
    assert len(first.hidden_digest) == 64
    with pytest.raises(FrozenInstanceError):
        first.action_digest = "changed"


class _TrainingBoundaryLearner:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self._pending = False
        self._parameters = np.zeros((2, 4), dtype=np.float64)

    @property
    def has_pending_feedback(self) -> bool:
        return self._pending

    def select_for_training(
        self,
        hidden: np.ndarray,
        legal: tuple[int, int],
        rng: np.random.Generator,
    ) -> SimpleNamespace:
        assert type(hidden) is np.ndarray
        assert hidden.dtype == np.float64
        assert hidden.flags.writeable is False
        assert legal == (0, 1)
        assert isinstance(rng, np.random.Generator)
        self.events.append(("select", hidden.copy(), legal, rng))
        self._pending = True
        return SimpleNamespace(action_index=0)

    def learn(self, reward: float) -> ActionValueUpdate:
        assert type(reward) is float
        assert self._pending
        self.events.append(("learn", reward))
        self._pending = False
        return ActionValueUpdate(0, 0.0, reward, reward)

    def select_greedy(self, hidden: np.ndarray, legal: tuple[int, int]) -> SimpleNamespace:
        self.events.append(("greedy", hidden.copy(), legal))
        return SimpleNamespace(
            action_index=0,
            action_values=np.zeros(2, dtype=np.float64),
        )

    def parameter_snapshot(self) -> np.ndarray:
        snapshot = self._parameters.copy()
        snapshot.flags.writeable = False
        return snapshot

    def parameter_digest(self) -> str:
        return hashlib.sha256(self._parameters.tobytes()).hexdigest()


class _BoundaryTask:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events

    def reward(self, episode: DelayedCueEpisode, action: int) -> float:
        self.events.append(("reward", episode, action))
        return float(1.0 if action == episode.correct_action_index else -1.0)


_TRAINING_EVENTS: list[tuple[object, ...]] = []


class _BoundaryAccessEpisode(DelayedCueEpisode):
    def __getattribute__(self, name: str) -> object:
        if name in {"correct_action_index", "delay_steps"}:
            _TRAINING_EVENTS.append(("metadata", name))
        return super().__getattribute__(name)


class _TrainingOrderPolicy(_RecordingPolicy):
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        super().__init__()
        self.events = events

    def reset_state(self) -> None:
        self.events.append(("reset",))
        super().reset_state()

    def advance(self, stimulus: np.ndarray) -> np.ndarray:
        self.events.append(("advance", stimulus))
        return super().advance(stimulus)


def test_normal_training_preserves_scalar_feedback_boundary_and_order() -> None:
    config = ActionValueBenchmarkConfig(
        hidden_size=3,
        training_episodes=20,
        evaluation_blocks=1,
        checkpoint_interval=10,
    )
    originals = _build_fixture_bundle(13, config).training
    fixtures = tuple(
        _BoundaryAccessEpisode(
            cue=original.cue,
            delay_steps=original.delay_steps,
            cue_stimulus=original.cue_stimulus,
            delay_stimuli=original.delay_stimuli,
            decision_stimulus=original.decision_stimulus,
            correct_action_index=original.correct_action_index,
        )
        for original in originals
    )
    events = _TRAINING_EVENTS
    events.clear()
    rng = np.random.default_rng(51)

    trace = _train_normal(
        _TrainingOrderPolicy(events),
        _TrainingBoundaryLearner(events),
        _BoundaryTask(events),
        fixtures,
        rng,
        config,
    )

    assert len(trace.actions) == len(trace.rewards) == 20
    select_positions = [i for i, event in enumerate(events) if event[0] == "select"]
    reward_positions = [i for i, event in enumerate(events) if event[0] == "reward"]
    learn_positions = [i for i, event in enumerate(events) if event[0] == "learn"]
    assert len(select_positions) == len(reward_positions) == len(learn_positions) == 20
    assert all(
        select < reward < learn
        for select, reward, learn in zip(
            select_positions, reward_positions, learn_positions, strict=True
        )
    )
    assert all(type(events[position][1]) is float for position in learn_positions)
    assert all(
        any(event[0] == "advance" for event in events[previous + 1 : select])
        for previous, select in zip([-1, *learn_positions[:-1]], select_positions)
    )
    reset_positions = [i for i, event in enumerate(events) if event[0] == "reset"]
    for episode_index, (select, learn) in enumerate(
        zip(select_positions, learn_positions, strict=True)
    ):
        end = (
            reset_positions[episode_index + 1]
            if episode_index + 1 < len(reset_positions)
            else len(events)
        )
        metadata_positions = [
            position for position in range(select + 1, end) if events[position][0] == "metadata"
        ]
        assert metadata_positions
        assert all(position > select for position in metadata_positions)
        assert any(position < learn for position in metadata_positions)
        assert any(position > learn for position in metadata_positions)
    assert all(checkpoint.accuracy == AccuracyCount(5, 10) for checkpoint in trace.checkpoints)
    assert all(checkpoint.margin_mean == 0.0 for checkpoint in trace.checkpoints)
    assert all(checkpoint.margin_p10 == 0.0 for checkpoint in trace.checkpoints)
    assert all(checkpoint.margin_minimum == 0.0 for checkpoint in trace.checkpoints)
    assert all(checkpoint.td_error_abs_mean == 1.0 for checkpoint in trace.checkpoints)
    assert all(checkpoint.td_error_p90 == 1.0 for checkpoint in trace.checkpoints)
    assert all(checkpoint.td_error_maximum == 1.0 for checkpoint in trace.checkpoints)


def _train_with_action_lineage(
    seed: int, learner: NormalizedActionValue, fixtures: tuple[DelayedCueEpisode, ...]
) -> object:
    config = ActionValueBenchmarkConfig(
        hidden_size=learner.hidden_size,
        training_episodes=len(fixtures),
        evaluation_blocks=1,
        checkpoint_interval=10,
    )
    return _train_normal(
        _new_policy(seed, config),
        learner,
        DelayedCueTask(),
        fixtures,
        np.random.default_rng(np.random.SeedSequence([seed, 0x33414354])),
        config,
    )


def test_training_actions_depend_only_on_independent_same_lineage_generators() -> None:
    seed = 7
    config = ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=20,
        evaluation_blocks=1,
        checkpoint_interval=10,
    )
    fixtures = _build_fixture_bundle(seed, config).training
    zero = _new_learner(config)
    installed = _new_learner(config)
    installed._weights[:] = np.arange(18, dtype=np.float64).reshape(2, 9)

    first = _train_with_action_lineage(seed, zero, fixtures)
    second = _train_with_action_lineage(seed, installed, fixtures)

    assert first.actions == second.actions
    assert first.action_digest == second.action_digest


def test_literal_normal_rewards_are_permuted_only_within_each_block() -> None:
    rewards = tuple(float(value) for value in range(20))
    shuffled = _permute_reward_blocks(
        rewards,
        np.random.default_rng(np.random.SeedSequence([23, 0x33534846])),
        block_size=10,
    )

    assert len(shuffled) == len(rewards)
    assert all(
        sorted(rewards[start : start + 10]) == sorted(shuffled[start : start + 10])
        for start in range(0, len(rewards), 10)
    )
    assert any(
        rewards[start : start + 10] != shuffled[start : start + 10]
        for start in range(0, len(rewards), 10)
    )
    assert "task" not in inspect.signature(_train_shuffled).parameters


def test_normal_and_shuffled_training_are_schedule_matched_and_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = 17
    config = ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=20,
        evaluation_blocks=1,
        checkpoint_interval=10,
    )
    fixtures = _build_fixture_bundle(seed, config).training
    normal_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    shuffled_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    assert normal_rng is not shuffled_rng
    normal_state_before = normal_rng.bit_generator.state
    shuffled_state_before = shuffled_rng.bit_generator.state
    assert normal_state_before == shuffled_state_before
    normal = _train_normal(
        _new_policy(seed, config),
        _new_learner(config),
        DelayedCueTask(),
        fixtures,
        normal_rng,
        config,
    )
    assert normal.rewards == tuple(
        DelayedCueTask().reward(episode, action)
        for episode, action in zip(fixtures, normal.actions, strict=True)
    )
    assert normal.action_counts == tuple(
        (action, normal.actions.count(action)) for action in (0, 1)
    )
    assert normal.action_digest == _sha256_actions(normal.actions)
    normal_reward_hash = hashlib.sha256()
    normal_reward_hash.update(len(normal.rewards).to_bytes(8, "little"))
    normal_reward_hash.update(np.asarray(normal.rewards, dtype=np.float64).tobytes(order="C"))
    assert normal.reward_digest == normal_reward_hash.hexdigest()
    rewards = _permute_reward_blocks(
        normal.rewards,
        np.random.default_rng(np.random.SeedSequence([seed, 0x33534846])),
        block_size=10,
    )

    def reject_reward_query(*_: object) -> float:
        raise AssertionError("shuffled training must not query task reward")

    monkeypatch.setattr(DelayedCueTask, "reward", reject_reward_query)
    shuffled = _train_shuffled(
        _new_policy(seed, config),
        _new_learner(config),
        fixtures,
        shuffled_rng,
        rewards,
        config,
    )

    assert normal_rng.bit_generator.state == shuffled_rng.bit_generator.state
    assert normal.actions == shuffled.actions
    assert normal.action_digest == shuffled.action_digest
    assert normal.rewards != shuffled.rewards
    assert len(normal.reward_digest) == len(shuffled.reward_digest) == 64
    assert all(
        sorted(normal.rewards[start : start + 10]) == sorted(shuffled.rewards[start : start + 10])
        for start in range(0, config.training_episodes, 10)
    )
    assert normal.pending_feedback is shuffled.pending_feedback is False
    assert normal.final_parameter_digest != shuffled.final_parameter_digest


def test_checkpoints_cover_only_completed_intervals_without_mutation() -> None:
    seed = 29
    config = ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=20,
        evaluation_blocks=1,
        checkpoint_interval=10,
    )
    learner = _new_learner(config)
    trace = _train_normal(
        _new_policy(seed, config),
        learner,
        DelayedCueTask(),
        _build_fixture_bundle(seed, config).training,
        np.random.default_rng(np.random.SeedSequence([seed, 0x33414354])),
        config,
    )

    assert tuple(checkpoint.episode for checkpoint in trace.checkpoints) == (10, 20)
    assert all(isinstance(checkpoint, ActionValueCheckpoint) for checkpoint in trace.checkpoints)
    assert all(checkpoint.accuracy.total == 10 for checkpoint in trace.checkpoints)
    assert learner.parameter_digest() == trace.final_parameter_digest
    assert learner.has_pending_feedback is False
    assert all(
        np.isfinite(value)
        for checkpoint in trace.checkpoints
        for value in (
            checkpoint.margin_mean,
            checkpoint.margin_p10,
            checkpoint.margin_minimum,
            checkpoint.td_error_mean,
            checkpoint.td_error_abs_mean,
            checkpoint.td_error_p90,
            checkpoint.td_error_maximum,
        )
    )


class _DivergentCheckpointPolicy:
    def __init__(self) -> None:
        self._cue = 0

    def reset_state(self) -> None:
        self._cue = 0

    def advance(self, stimulus: np.ndarray) -> np.ndarray:
        if stimulus[0] == 1.0:
            self._cue = 0
        elif stimulus[1] == 1.0:
            self._cue = 1
        hidden = np.asarray([float(self._cue)], dtype=np.float64)
        hidden.flags.writeable = False
        return hidden


class _DivergentCheckpointLearner:
    def __init__(self) -> None:
        self._pending = False

    @property
    def has_pending_feedback(self) -> bool:
        return self._pending

    def select_for_training(
        self,
        hidden: np.ndarray,
        legal: tuple[int, int],
        rng: np.random.Generator,
    ) -> SimpleNamespace:
        assert legal == (0, 1)
        assert isinstance(rng, np.random.Generator)
        self._pending = True
        return SimpleNamespace(action_index=1 - int(hidden[0]))

    def learn(self, reward: float) -> ActionValueUpdate:
        assert type(reward) is float
        self._pending = False
        return ActionValueUpdate(0, 0.0, reward, reward)

    def parameter_snapshot(self) -> np.ndarray:
        snapshot = np.zeros((2, 2), dtype=np.float64)
        snapshot.flags.writeable = False
        return snapshot

    def parameter_digest(self) -> str:
        return "checkpoint-parameters"

    def select_greedy(
        self, hidden: np.ndarray, legal: tuple[int, int]
    ) -> SimpleNamespace:
        assert legal == (0, 1)
        correct = int(hidden[0])
        values = np.zeros(2, dtype=np.float64)
        values[correct] = 1.0
        return SimpleNamespace(action_index=correct, action_values=values)


def test_checkpoint_accuracy_uses_post_block_greedy_decisions() -> None:
    config = ActionValueBenchmarkConfig(
        hidden_size=1,
        training_episodes=10,
        evaluation_blocks=1,
        checkpoint_interval=10,
    )
    fixtures = _build_fixture_bundle(31, config).training

    trace = _train_normal(
        _DivergentCheckpointPolicy(),
        _DivergentCheckpointLearner(),
        DelayedCueTask(),
        fixtures,
        np.random.default_rng(37),
        config,
    )

    assert all(
        sampled != episode.correct_action_index
        for sampled, episode in zip(trace.actions, fixtures, strict=True)
    )
    assert trace.checkpoints[0].accuracy == AccuracyCount(10, 10)


def test_run_once_assembles_independent_frozen_fair_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = 7
    config = ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=20,
        evaluation_blocks=2,
        checkpoint_interval=10,
    )
    policies: list[RecurrentPolicy] = []
    learners: list[NormalizedActionValue] = []
    evaluation_fixture_ids: list[int] = []
    original_new_policy = benchmark_module._new_policy
    original_new_learner = benchmark_module._new_learner
    original_evaluate = benchmark_module._evaluate

    def capture_policy(
        captured_seed: int, captured_config: ActionValueBenchmarkConfig
    ) -> RecurrentPolicy:
        policy = original_new_policy(captured_seed, captured_config)
        policies.append(policy)
        return policy

    def capture_learner(
        captured_config: ActionValueBenchmarkConfig,
    ) -> NormalizedActionValue:
        learner = original_new_learner(captured_config)
        learners.append(learner)
        return learner

    def capture_evaluation(
        policy: object,
        learner: object,
        fixtures: tuple[DelayedCueEpisode, ...],
        *,
        reset_before_decision: bool,
    ) -> object:
        evaluation_fixture_ids.append(id(fixtures))
        return original_evaluate(
            policy,
            learner,
            fixtures,
            reset_before_decision=reset_before_decision,
        )

    monkeypatch.setattr(benchmark_module, "_new_policy", capture_policy)
    monkeypatch.setattr(benchmark_module, "_new_learner", capture_learner)
    monkeypatch.setattr(benchmark_module, "_evaluate", capture_evaluation)
    first = benchmark_module._run_once(seed, config)
    assert len(policies) == len(learners) == 2
    assert policies[0] is not policies[1]
    assert learners[0] is not learners[1]
    assert policies[0]._input_weights is not policies[1]._input_weights
    assert policies[0]._recurrent_weights is not policies[1]._recurrent_weights
    assert policies[0]._output_weights is not policies[1]._output_weights
    assert len(evaluation_fixture_ids) == 4
    assert len(set(evaluation_fixture_ids)) == 1

    monkeypatch.undo()
    second = _run_once(seed, config)

    assert first == second
    assert first.normal_training.actions == first.shuffled_training.actions
    assert first.normal_training.action_digest == first.shuffled_training.action_digest
    assert all(
        sorted(first.normal_training.rewards[start : start + 10])
        == sorted(first.shuffled_training.rewards[start : start + 10])
        for start in range(0, config.training_episodes, 10)
    )
    assert first.normal_parameter_digest_before == first.shuffled_parameter_digest_before
    assert first.normal_training.final_parameter_digest != first.normal_parameter_digest_before
    assert first.shuffled_training.final_parameter_digest != first.shuffled_parameter_digest_before
    assert first.normal_matrix_digests_before == first.normal_matrix_digests_after
    assert first.shuffled_matrix_digests_before == first.shuffled_matrix_digests_after
    assert first.normal_training.pending_feedback is False
    assert first.shuffled_training.pending_feedback is False
    assert (
        first.training_fixture_digest == _build_fixture_bundle(seed, config).training_fixture_digest
    )
    assert (
        first.evaluation_fixture_digest
        == _build_fixture_bundle(seed, config).evaluation_fixture_digest
    )
    assert first.pre_training.overall.total == first.post_training.overall.total == 20
    assert first.state_reset.overall.total == first.shuffled_control.overall.total == 20


def _accepted_result(seed: int, *, shuffled_correct: int = 100) -> ActionValueExperimentResult:
    reset_200 = AccuracyCount(100, 200)
    reset_40 = AccuracyCount(20, 40)
    return ActionValueExperimentResult(
        seed=seed,
        config=ActionValueBenchmarkConfig(),
        pre_training=reset_200,
        post_training=AccuracyCount(180, 200),
        state_reset=reset_200,
        shuffled_control=AccuracyCount(shuffled_correct, 200),
        per_delay=tuple((delay, AccuracyCount(36, 40)) for delay in range(1, 6)),
        reset_per_delay=tuple((delay, reset_40) for delay in range(1, 6)),
        shuffled_per_delay=tuple((delay, reset_40) for delay in range(1, 6)),
        normal_checkpoints=(),
        shuffled_checkpoints=(),
        normal_action_counts=((0, 1_000), (1, 1_000)),
        shuffled_action_counts=((0, 1_000), (1, 1_000)),
        normal_actions=(0,) * 1_000 + (1,) * 1_000,
        shuffled_actions=(0,) * 1_000 + (1,) * 1_000,
        normal_action_digest="normal-action",
        shuffled_action_digest="normal-action",
        normal_reward_digest="normal-reward",
        shuffled_reward_digest="shuffled-reward",
        action_sequences_equal=True,
        reward_block_multisets_equal=True,
        initial_parameter_digest="initial",
        normal_parameter_digest="normal",
        shuffled_parameter_digest="shuffled",
        normal_matrix_digests_before=("a", "b", "c"),
        normal_matrix_digests_after=("a", "b", "c"),
        shuffled_matrix_digests_before=("a", "b", "c"),
        shuffled_matrix_digests_after=("a", "b", "c"),
        training_fixture_digest="training",
        evaluation_fixture_digest="evaluation",
        decision_hidden_digest="decision",
        reset_hidden_digest="reset",
        post_margin_mean=1.0,
        post_margin_p10=0.5,
        post_margin_minimum=0.25,
        all_reset_hidden_equal=True,
        normal_pending_feedback=False,
        shuffled_pending_feedback=False,
        repeatable=True,
    )


def test_public_result_has_exact_frozen_tuple_contract() -> None:
    expected_fields = {
        "seed": int,
        "config": ActionValueBenchmarkConfig,
        "pre_training": AccuracyCount,
        "post_training": AccuracyCount,
        "state_reset": AccuracyCount,
        "shuffled_control": AccuracyCount,
        "per_delay": tuple[tuple[int, AccuracyCount], ...],
        "reset_per_delay": tuple[tuple[int, AccuracyCount], ...],
        "shuffled_per_delay": tuple[tuple[int, AccuracyCount], ...],
        "normal_checkpoints": tuple[ActionValueCheckpoint, ...],
        "shuffled_checkpoints": tuple[ActionValueCheckpoint, ...],
        "normal_action_counts": tuple[tuple[int, int], ...],
        "shuffled_action_counts": tuple[tuple[int, int], ...],
        "normal_actions": tuple[int, ...],
        "shuffled_actions": tuple[int, ...],
        "normal_action_digest": str,
        "shuffled_action_digest": str,
        "normal_reward_digest": str,
        "shuffled_reward_digest": str,
        "action_sequences_equal": bool,
        "reward_block_multisets_equal": bool,
        "initial_parameter_digest": str,
        "normal_parameter_digest": str,
        "shuffled_parameter_digest": str,
        "normal_matrix_digests_before": tuple[str, str, str],
        "normal_matrix_digests_after": tuple[str, str, str],
        "shuffled_matrix_digests_before": tuple[str, str, str],
        "shuffled_matrix_digests_after": tuple[str, str, str],
        "training_fixture_digest": str,
        "evaluation_fixture_digest": str,
        "decision_hidden_digest": str,
        "reset_hidden_digest": str,
        "post_margin_mean": float,
        "post_margin_p10": float,
        "post_margin_minimum": float,
        "all_reset_hidden_equal": bool,
        "normal_pending_feedback": bool,
        "shuffled_pending_feedback": bool,
        "repeatable": bool,
    }

    assert get_type_hints(ActionValueExperimentResult) == expected_fields
    result = _accepted_result(7)
    assert tuple(delay for delay, _ in result.per_delay) == (1, 2, 3, 4, 5)
    assert isinstance(result.normal_actions, tuple)
    assert isinstance(result.normal_action_counts, tuple)
    with pytest.raises(FrozenInstanceError):
        result.repeatable = False


@pytest.mark.parametrize("seed", [-1, True, 1.5, "7"])
def test_public_experiment_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError):
        run_action_value_experiment(seed)  # type: ignore[arg-type]


def test_public_experiment_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        run_action_value_experiment(7, object())  # type: ignore[arg-type]


def test_repeatability_compares_two_reconstructed_executions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=20,
        evaluation_blocks=1,
        checkpoint_interval=10,
    )
    baseline = _run_once(7, config)
    calls: list[tuple[int, ActionValueBenchmarkConfig]] = []
    returns = iter((baseline, dataclasses.replace(baseline, training_fixture_digest="different")))

    def reconstructed(seed: int, received: ActionValueBenchmarkConfig) -> object:
        calls.append((seed, received))
        return next(returns)

    monkeypatch.setattr(benchmark_module, "_run_once", reconstructed)
    result = run_action_value_experiment(7, config)

    assert calls == [(7, config), (7, config)]
    assert result.repeatable is False
    assert result.training_fixture_digest == baseline.training_fixture_digest


def test_public_experiment_is_equal_across_full_calls() -> None:
    config = ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=20,
        evaluation_blocks=1,
        checkpoint_interval=10,
    )
    assert run_action_value_experiment(17, config) == run_action_value_experiment(17, config)


def test_default_result_has_preregistered_checkpoint_and_nested_tuple_shape() -> None:
    result = run_action_value_experiment(7)

    assert len(result.normal_checkpoints) == len(result.shuffled_checkpoints) == 20
    assert tuple(row.episode for row in result.normal_checkpoints) == tuple(range(100, 2_001, 100))
    assert tuple(row.episode for row in result.shuffled_checkpoints) == tuple(
        range(100, 2_001, 100)
    )
    assert len(result.normal_actions) == len(result.shuffled_actions) == 2_000
    assert result.per_delay == tuple(result.per_delay)
    assert result.reset_per_delay == tuple(result.reset_per_delay)
    assert result.shuffled_per_delay == tuple(result.shuffled_per_delay)
    with pytest.raises(TypeError):
        result.normal_actions[0] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.normal_checkpoints[0].episode = 0


def test_phase_three_a_public_api_is_exported_from_package() -> None:
    expected = {
        "ActionValueBenchmarkConfig": ActionValueBenchmarkConfig,
        "ActionValueCheckpoint": ActionValueCheckpoint,
        "ActionValueExperimentResult": ActionValueExperimentResult,
        "run_action_value_experiment": run_action_value_experiment,
        "run_action_value_benchmark": run_action_value_benchmark,
    }

    assert {name: getattr(neural_state_machine, name) for name in expected} == expected
    assert expected.keys() <= set(neural_state_machine.__all__)


@pytest.mark.parametrize(
    "change",
    [
        {"repeatable": False},
        {"post_training": AccuracyCount(179, 200)},
        {"per_delay": ((1, AccuracyCount(33, 40)),) + tuple((d, AccuracyCount(36, 40)) for d in range(2, 6))},
        {"state_reset": AccuracyCount(99, 200)},
        {"reset_per_delay": ((1, AccuracyCount(19, 40)),) + tuple((d, AccuracyCount(20, 40)) for d in range(2, 6))},
        {"all_reset_hidden_equal": False},
        {"shuffled_control": AccuracyCount(150, 200)},
        {"action_sequences_equal": False},
        {"reward_block_multisets_equal": False},
        {"normal_parameter_digest": "initial"},
        {"shuffled_parameter_digest": "initial"},
        {"normal_matrix_digests_after": ("x", "b", "c")},
        {"shuffled_matrix_digests_after": ("x", "b", "c")},
        {"normal_pending_feedback": True},
        {"shuffled_pending_feedback": True},
    ],
)
def test_per_seed_acceptance_is_literal_and_conjunctive(change: dict[str, object]) -> None:
    assert _passes_acceptance(_accepted_result(7)) is True
    assert _passes_acceptance(dataclasses.replace(_accepted_result(7), **change)) is False


@pytest.mark.parametrize("correct", [80, 120])
def test_pooled_control_boundaries_are_inclusive(
    monkeypatch: pytest.MonkeyPatch, correct: int
) -> None:
    monkeypatch.setattr(
        benchmark_module,
        "run_action_value_experiment",
        lambda seed, config: _accepted_result(seed, shuffled_correct=correct),
    )

    assert run_action_value_benchmark((7, 17, 29))["all_passed"] is True


def test_failed_seed_cannot_be_compensated_by_strong_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = {
        7: _accepted_result(7, shuffled_correct=100),
        17: _accepted_result(17, shuffled_correct=100),
        29: dataclasses.replace(_accepted_result(29, shuffled_correct=100), repeatable=False),
    }
    monkeypatch.setattr(
        benchmark_module,
        "run_action_value_experiment",
        lambda seed, config: results[seed],
    )

    payload = run_action_value_benchmark((7, 17, 29))

    assert [row["passed"] for row in payload["results"]] == [True, True, False]
    assert payload["shuffled_pooled"] == {"accuracy": 0.5, "correct": 300, "total": 600}
    assert payload["all_passed"] is False


def test_benchmark_schema_is_complete_and_json_strict() -> None:
    config = ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=20,
        evaluation_blocks=1,
        checkpoint_interval=10,
    )
    payload = run_action_value_benchmark((7,), config)

    assert set(payload) == {
        "all_passed",
        "config",
        "evidence_schema_version",
        "frozen_evidence_sha256",
        "phase",
        "results",
        "rng_lineages",
        "seeds",
        "shuffled_pooled",
    }
    assert payload["phase"] == "3A"
    assert payload["evidence_schema_version"] == 1
    assert payload["seeds"] == [7]
    assert payload["rng_lineages"] == {
        "behavior_action": ["seed", 0x33414354],
        "evaluation_fixture": ["seed", 0x4556414C],
        "reward_shuffle": ["seed", 0x33534846],
        "training_fixture": ["seed", 0x54524149],
    }
    root = Path(__file__).resolve().parents[1]
    assert payload["frozen_evidence_sha256"] == {
        "phase_2b": hashlib.sha256(
            (root / "docs/experiments/phase-2b-failure.json").read_bytes()
        ).hexdigest(),
        "phase_2c": hashlib.sha256(
            (root / "docs/experiments/phase-2c-diagnostics.json").read_bytes()
        ).hexdigest(),
    }
    result = payload["results"][0]
    assert set(result) == {
        "action_sequences_equal",
        "all_reset_hidden_equal",
        "decision_hidden_digest",
        "evaluation_fixture_digest",
        "initial_parameter_digest",
        "matrix_controls",
        "normal_action_counts",
        "normal_action_digest",
        "normal_actions",
        "normal_checkpoints",
        "normal_parameter_digest",
        "normal_pending_feedback",
        "normal_reward_digest",
        "passed",
        "per_delay",
        "post_margin",
        "post_training",
        "pre_training",
        "repeatable",
        "reset_hidden_digest",
        "reset_per_delay",
        "reward_block_multisets_equal",
        "seed",
        "shuffled_action_counts",
        "shuffled_action_digest",
        "shuffled_actions",
        "shuffled_checkpoints",
        "shuffled_control",
        "shuffled_parameter_digest",
        "shuffled_pending_feedback",
        "shuffled_per_delay",
        "shuffled_reward_digest",
        "state_reset",
        "training_fixture_digest",
    }
    assert len(result["normal_actions"]) == len(result["shuffled_actions"]) == 20
    assert result["normal_checkpoints"][-1]["episode"] == 20
    assert set(result["normal_checkpoints"][-1]) == {
        "accuracy",
        "episode",
        "margin_mean",
        "margin_minimum",
        "margin_p10",
        "td_error_abs_mean",
        "td_error_maximum",
        "td_error_mean",
        "td_error_p90",
    }
    assert set(result["post_margin"]) == {"mean", "minimum", "p10"}
    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@pytest.mark.parametrize(
    "seeds",
    [(), [], (7, 7), (7, -1), (True,), (1.5,), None],
)
def test_benchmark_rejects_invalid_seed_sequences(seeds: object) -> None:
    with pytest.raises(ValueError):
        run_action_value_benchmark(seeds)  # type: ignore[arg-type]


def test_cli_is_byte_repeatable_and_preserves_truthful_exit_status() -> None:
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "scripts/benchmark_action_value.py"]

    first = subprocess.run(command, cwd=root, capture_output=True, check=False)
    second = subprocess.run(command, cwd=root, capture_output=True, check=False)

    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout
    assert first.stdout.count(b"\n") == 1
    payload = json.loads(first.stdout)
    assert payload["seeds"] == [7, 17, 29]
    assert payload["phase"] == "3A"
    assert payload["evidence_schema_version"] == 1
    expected_status = int(not payload["all_passed"])
    assert first.returncode == second.returncode == expected_status
    assert payload["all_passed"] is False


def _script_module(name: str) -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def default_benchmark_payload() -> dict[str, object]:
    return run_action_value_benchmark()


def test_evidence_source_commit_is_strict_and_does_not_change_live_payload(
    monkeypatch: pytest.MonkeyPatch,
    default_benchmark_payload: dict[str, object],
) -> None:
    writer = _script_module("benchmark_action_value")
    live_before = deepcopy(default_benchmark_payload)
    completed = subprocess.CompletedProcess(
        ["git", "rev-parse", "HEAD"], 0, stdout="a" * 40 + "\n", stderr=""
    )
    monkeypatch.setattr(writer.subprocess, "run", lambda *args, **kwargs: completed)

    assert writer._source_commit() == "a" * 40
    assert default_benchmark_payload == live_before
    assert "source_commit" not in default_benchmark_payload


@pytest.mark.parametrize(
    "completed",
    [
        subprocess.CompletedProcess(["git"], 1, stdout="", stderr="no repository"),
        subprocess.CompletedProcess(["git"], 0, stdout="", stderr=""),
        subprocess.CompletedProcess(["git"], 0, stdout="A" * 40 + "\n", stderr=""),
        subprocess.CompletedProcess(["git"], 0, stdout="g" * 40 + "\n", stderr=""),
        subprocess.CompletedProcess(["git"], 0, stdout="a" * 39 + "\n", stderr=""),
        subprocess.CompletedProcess(["git"], 0, stdout="a" * 41 + "\n", stderr=""),
    ],
)
def test_evidence_source_commit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    completed: subprocess.CompletedProcess[str],
) -> None:
    writer = _script_module("benchmark_action_value")
    monkeypatch.setattr(writer.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(RuntimeError, match="source commit"):
        writer._source_commit()


def test_evidence_source_commit_runs_git_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _script_module("benchmark_action_value")
    calls: list[tuple[object, object]] = []

    def capture(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="b" * 40 + "\n", stderr="")

    monkeypatch.setattr(writer.subprocess, "run", capture)

    assert writer._source_commit() == "b" * 40
    assert calls == [
        (
            ["git", "rev-parse", "HEAD"],
            {
                "cwd": writer._ROOT,
                "check": False,
                "capture_output": True,
                "text": True,
            },
        )
    ]


def test_action_value_evidence_writer_is_targeted_atomic_and_symlink_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _script_module("benchmark_action_value")
    approved = tmp_path / "phase-3a-action-value.json"
    frozen_2b = tmp_path / "phase-2b-failure.json"
    frozen_2c = tmp_path / "phase-2c-diagnostics.json"
    frozen_2b.write_text("phase 2b\n", encoding="utf-8")
    frozen_2c.write_text("phase 2c\n", encoding="utf-8")
    payload = {"all_passed": False, "source_commit": "c" * 40}
    forbidden = tmp_path / "other.json"
    forbidden.write_text("unchanged\n", encoding="utf-8")

    with pytest.raises(ValueError, match="approved Phase 3A"):
        writer._write_evidence(
            forbidden,
            payload,
            approved_evidence=approved,
            frozen_evidence=(frozen_2b, frozen_2c),
        )
    assert forbidden.read_text(encoding="utf-8") == "unchanged\n"

    for frozen in (frozen_2b, frozen_2c):
        before = frozen.read_bytes()
        with pytest.raises(ValueError, match="frozen Phase 2"):
            writer._write_evidence(
                frozen,
                payload,
                approved_evidence=approved,
                frozen_evidence=(frozen_2b, frozen_2c),
            )
        assert frozen.read_bytes() == before

    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text("must not change\n", encoding="utf-8")
    approved.symlink_to(unrelated)
    with pytest.raises(ValueError, match="symlink"):
        writer._write_evidence(
            approved,
            payload,
            approved_evidence=approved,
            frozen_evidence=(frozen_2b, frozen_2c),
        )
    assert unrelated.read_text(encoding="utf-8") == "must not change\n"
    approved.unlink()

    writer._write_evidence(
        approved,
        payload,
        approved_evidence=approved,
        frozen_evidence=(frozen_2b, frozen_2c),
    )
    rendered = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    assert approved.read_text(encoding="utf-8") == rendered
    original_stat = approved.stat()
    writer._write_evidence(
        approved,
        payload,
        approved_evidence=approved,
        frozen_evidence=(frozen_2b, frozen_2c),
    )
    assert approved.stat().st_mtime_ns == original_stat.st_mtime_ns

    monkeypatch.setattr(writer.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        writer._write_evidence(
            approved,
            {**payload, "all_passed": True},
            approved_evidence=approved,
            frozen_evidence=(frozen_2b, frozen_2c),
        )
    assert approved.read_text(encoding="utf-8") == rendered
    assert list(tmp_path.glob(f".{approved.name}.*.tmp")) == []


def _evidence_payload(payload: dict[str, object]) -> dict[str, object]:
    artifact = deepcopy(payload)
    artifact["source_commit"] = "d" * 40
    return artifact


def _mutate_evidence(payload: dict[str, object], mutation: str) -> None:
    results = payload["results"]
    assert isinstance(results, list)
    result = results[0]
    assert isinstance(result, dict)
    if mutation == "missing_key":
        del result["post_training"]
    elif mutation == "phase":
        payload["phase"] = "3B"
    elif mutation == "schema":
        payload["evidence_schema_version"] = True
    elif mutation == "config":
        assert isinstance(payload["config"], dict)
        payload["config"]["step_size"] = 0.2
    elif mutation == "seed_order":
        payload["seeds"] = [17, 7, 29]
    elif mutation == "duplicate_results":
        results[1] = deepcopy(results[0])
    elif mutation == "count":
        assert isinstance(result["post_training"], dict)
        result["post_training"]["correct"] = 201
    elif mutation == "accuracy":
        assert isinstance(result["post_training"], dict)
        result["post_training"]["accuracy"] = 0.123
    elif mutation == "digest":
        result["normal_action_digest"] = "not-a-digest"
    elif mutation == "local_digest":
        result["normal_parameter_digest"] = "not-a-digest"
    elif mutation == "local_matrix":
        assert isinstance(result["matrix_controls"], dict)
        result["matrix_controls"]["normal_after"] = ["f" * 64] * 3
    elif mutation == "checkpoint_sequence":
        assert isinstance(result["normal_checkpoints"], list)
        result["normal_checkpoints"][0]["episode"] = 101
    elif mutation == "checkpoint_nan":
        assert isinstance(result["normal_checkpoints"], list)
        result["normal_checkpoints"][0]["td_error_mean"] = float("nan")
    elif mutation == "frozen_hash":
        assert isinstance(payload["frozen_evidence_sha256"], dict)
        payload["frozen_evidence_sha256"]["phase_2b"] = "e" * 64
    elif mutation == "fairness":
        result["reward_block_multisets_equal"] = False
    elif mutation == "pending":
        result["normal_pending_feedback"] = True
    elif mutation == "repeatability":
        result["repeatable"] = False
    elif mutation == "pass_flag":
        result["passed"] = not result["passed"]
    elif mutation == "pooled":
        assert isinstance(payload["shuffled_pooled"], dict)
        payload["shuffled_pooled"]["correct"] += 1
    elif mutation == "all_passed":
        payload["all_passed"] = not payload["all_passed"]
    elif mutation == "source_commit":
        payload["source_commit"] = "D" * 40
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_key",
        "phase",
        "schema",
        "config",
        "seed_order",
        "duplicate_results",
        "count",
        "accuracy",
        "digest",
        "local_digest",
        "local_matrix",
        "checkpoint_sequence",
        "checkpoint_nan",
        "frozen_hash",
        "fairness",
        "pending",
        "repeatability",
        "pass_flag",
        "pooled",
        "all_passed",
        "source_commit",
    ],
)
def test_portable_evidence_validation_fails_closed(
    default_benchmark_payload: dict[str, object],
    mutation: str,
) -> None:
    verifier = _script_module("verify_action_value_evidence")
    artifact = _evidence_payload(default_benchmark_payload)
    _mutate_evidence(artifact, mutation)

    with pytest.raises(RuntimeError):
        verifier._portable_phase_3a_payload(artifact)


def test_portable_projection_excludes_only_environment_local_digests(
    default_benchmark_payload: dict[str, object],
) -> None:
    verifier = _script_module("verify_action_value_evidence")
    artifact = _evidence_payload(default_benchmark_payload)
    projected = verifier._portable_phase_3a_payload(artifact)

    assert set(projected) == {
        "all_passed",
        "config",
        "evidence_schema_version",
        "frozen_evidence_sha256",
        "phase",
        "results",
        "rng_lineages",
        "seeds",
        "shuffled_pooled",
    }
    omitted = {
        "decision_hidden_digest",
        "initial_parameter_digest",
        "matrix_controls",
        "normal_parameter_digest",
        "reset_hidden_digest",
        "shuffled_parameter_digest",
    }
    assert omitted.isdisjoint(projected["results"][0])
    assert set(projected["results"][0]) == set(artifact["results"][0]) - omitted
    assert "source_commit" not in projected


def test_local_float_integrity_is_strict(
    default_benchmark_payload: dict[str, object],
) -> None:
    verifier = _script_module("verify_action_value_evidence")
    verifier._verify_local_float_integrity(default_benchmark_payload)

    mutations = []
    for path, value in (
        (("matrix_controls", "normal_after"), ["f" * 64] * 3),
        (("normal_parameter_digest",), default_benchmark_payload["results"][0]["initial_parameter_digest"]),
        (("shuffled_parameter_digest",), default_benchmark_payload["results"][0]["initial_parameter_digest"]),
        (("action_sequences_equal",), False),
        (("reward_block_multisets_equal",), False),
        (("normal_pending_feedback",), True),
        (("shuffled_pending_feedback",), True),
    ):
        changed = deepcopy(default_benchmark_payload)
        row = changed["results"][0]
        assert isinstance(row, dict)
        if len(path) == 1:
            row[path[0]] = value
        else:
            assert isinstance(row[path[0]], dict)
            row[path[0]][path[1]] = value
        mutations.append(changed)

    for changed in mutations:
        with pytest.raises(RuntimeError):
            verifier._verify_local_float_integrity(changed)


def test_committed_evidence_loader_rejects_corrupted_json(tmp_path: Path) -> None:
    verifier = _script_module("verify_action_value_evidence")
    evidence = tmp_path / "phase-3a-action-value.json"
    evidence.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="load Phase 3A evidence"):
        verifier._load_committed_evidence(evidence)


def test_source_commit_must_be_an_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _script_module("verify_action_value_evidence")
    calls: list[tuple[object, object]] = []

    def capture(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", capture)
    verifier._verify_source_commit_ancestor("a" * 40)
    assert calls == [
        (
            ["git", "merge-base", "--is-ancestor", "a" * 40, "HEAD"],
            {"cwd": verifier._ROOT, "check": False, "capture_output": True, "text": True},
        )
    ]

    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )
    with pytest.raises(RuntimeError, match="not an ancestor"):
        verifier._verify_source_commit_ancestor("a" * 40)


def test_verifier_rejects_same_environment_divergence(
    monkeypatch: pytest.MonkeyPatch,
    default_benchmark_payload: dict[str, object],
) -> None:
    verifier = _script_module("verify_action_value_evidence")
    expected = _evidence_payload(default_benchmark_payload)
    changed = deepcopy(default_benchmark_payload)
    changed["all_passed"] = not changed["all_passed"]
    calls = iter((deepcopy(default_benchmark_payload), changed))
    monkeypatch.setattr(verifier, "_load_committed_evidence", lambda: expected)
    monkeypatch.setattr(verifier, "_verify_source_commit_ancestor", lambda source: None)
    monkeypatch.setattr(verifier, "run_action_value_benchmark", lambda: next(calls))

    with pytest.raises(RuntimeError, match="not byte-stable"):
        verifier.verify_action_value_evidence()


def test_verifier_accepts_truthful_failed_gate_without_rewriting_it(
    monkeypatch: pytest.MonkeyPatch,
    default_benchmark_payload: dict[str, object],
) -> None:
    verifier = _script_module("verify_action_value_evidence")
    expected = _evidence_payload(default_benchmark_payload)
    assert expected["all_passed"] is False
    monkeypatch.setattr(verifier, "_load_committed_evidence", lambda: deepcopy(expected))
    monkeypatch.setattr(verifier, "_verify_source_commit_ancestor", lambda source: None)
    monkeypatch.setattr(
        verifier,
        "run_action_value_benchmark",
        lambda: deepcopy(default_benchmark_payload),
    )

    projected = verifier.verify_action_value_evidence()

    assert projected["all_passed"] is False
    assert expected["all_passed"] is False
