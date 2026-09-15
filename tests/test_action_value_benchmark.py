from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from neural_state_machine.action_value import NormalizedActionValue
from neural_state_machine.action_value_benchmark import (
    ActionValueBenchmarkConfig,
    _build_fixture_bundle,
    _evaluate,
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
    np.testing.assert_array_equal(
        actual.decision_stimulus, expected.decision_stimulus
    )


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
    for actual, reconstructed in zip(
        first.evaluation, second.evaluation, strict=True
    ):
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
            bundle.training[start : start + 10]
            for start in range(0, len(bundle.training), 10)
        )
    )
    assert all(
        {(episode.cue, episode.delay_steps) for episode in block} == cases
        for block in (
            bundle.evaluation[start : start + 10]
            for start in range(0, len(bundle.evaluation), 10)
        )
    )
    assert sum(row.correct_action_index == 0 for row in bundle.evaluation) == 100
    assert sum(row.correct_action_index == 1 for row in bundle.evaluation) == 100
    assert {
        delay: sum(row.delay_steps == delay for row in bundle.evaluation)
        for delay in range(1, 6)
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

    def select_greedy(
        self, hidden: np.ndarray, legal: tuple[int, int]
    ) -> SimpleNamespace:
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
    def select_greedy(
        self, hidden: np.ndarray, legal: tuple[int, int]
    ) -> SimpleNamespace:
        _SCORING_EVENTS.append("select_greedy")
        return super().select_greedy(hidden, legal)


def test_evaluation_passes_only_hidden_and_legal_indices_to_learner() -> None:
    fixtures = _build_fixtures(
        DelayedCueTask(), np.random.default_rng(101), 2
    )
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
    assert [count for _, count in result.per_delay] == [
        AccuracyCount(2, 4)
    ] * 5
    assert result.margin_mean == pytest.approx(0.0)
    assert result.margin_p10 == pytest.approx(-1.0)
    assert result.margin_minimum == pytest.approx(-1.0)
    assert {name for name, _ in policy.calls} == {"reset_state", "advance"}


def test_evaluation_reads_scoring_metadata_only_after_action_selection() -> None:
    originals = _build_fixtures(
        DelayedCueTask(), np.random.default_rng(103), 1
    )
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

    def select_greedy(
        self, hidden: np.ndarray, legal: tuple[int, int]
    ) -> object:
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
    learner = NormalizedActionValue(
        config.hidden_size, 2, step_size=config.step_size
    )
    learner_parameters = learner.parameter_snapshot()
    learner_digest = learner.parameter_digest()
    pending_before = learner.has_pending_feedback
    matrix_copies = _matrix_copies(policy)
    matrix_digests = _matrix_digests(policy)

    recording_learner = _HiddenRecordingLearner(learner)
    ordinary = _evaluate(
        policy, recording_learner, fixtures, reset_before_decision=False
    )
    reset = _evaluate(
        policy, recording_learner, fixtures, reset_before_decision=True
    )

    assert ordinary.actions == reset.actions == (0,) * 200
    assert ordinary.overall == reset.overall == AccuracyCount(100, 200)
    assert ordinary.per_delay == reset.per_delay == tuple(
        (delay, AccuracyCount(20, 40)) for delay in range(1, 6)
    )
    assert ordinary.action_digest == reset.action_digest == _sha256_actions(
        (0,) * 200
    )
    assert reset.all_hidden_equal is True
    reset_hidden = recording_learner.hidden_states[200:]
    assert len(reset_hidden) == 200
    assert all(
        np.array_equal(reset_hidden[0], hidden)
        for hidden in reset_hidden[1:]
    )
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
