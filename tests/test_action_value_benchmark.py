from __future__ import annotations

import dataclasses
import hashlib
import inspect
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

import neural_state_machine.action_value_benchmark as benchmark_module
from neural_state_machine.action_value import ActionValueUpdate, NormalizedActionValue
from neural_state_machine.action_value_benchmark import (
    ActionValueCheckpoint,
    ActionValueBenchmarkConfig,
    _build_fixture_bundle,
    _evaluate,
    _new_learner,
    _new_policy,
    _permute_reward_blocks,
    _run_once,
    _train_normal,
    _train_shuffled,
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
