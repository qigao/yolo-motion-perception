from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from neural_state_machine.memory_benchmark import AccuracyCount
from neural_state_machine.memory_task import Cue, DelayedCueTask
from neural_state_machine.policy import RecurrentPolicy
from neural_state_machine.reward_learning import (
    RewardLearningConfig,
    _balanced_cases,
    _build_fixtures,
    _decision_hidden,
    _evaluate,
)
from neural_state_machine.reward_readout import RewardModulatedReadout


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"hidden_size": -1},
        {"hidden_size": True},
        {"hidden_size": 2.5},
        {"evaluation_blocks": 0},
        {"evaluation_blocks": True},
        {"training_episodes": 0},
        {"training_episodes": 11},
        {"training_episodes": True},
        {"recurrent_radius": -0.1},
        {"recurrent_radius": 1.0},
        {"recurrent_radius": True},
        {"recurrent_radius": np.nan},
        {"learning_rate": 0.0},
        {"learning_rate": True},
        {"learning_rate": np.inf},
        {"temperature": 0.0},
        {"temperature": True},
        {"temperature": np.nan},
    ],
)
def test_reward_learning_config_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        RewardLearningConfig(**kwargs)


def test_reward_learning_config_uses_preregistered_defaults() -> None:
    config = RewardLearningConfig()

    assert config == RewardLearningConfig(
        hidden_size=64,
        recurrent_radius=0.9,
        learning_rate=0.05,
        temperature=1.0,
        training_episodes=2_000,
        evaluation_blocks=20,
    )
    with pytest.raises(FrozenInstanceError):
        config.hidden_size = 32


def test_balanced_cases_are_complete_and_seed_repeatable() -> None:
    left_rng = np.random.default_rng(101)
    right_rng = np.random.default_rng(101)
    expected = {(cue, delay) for cue in Cue for delay in range(1, 6)}

    left_blocks = [tuple(_balanced_cases(left_rng)) for _ in range(5)]
    right_blocks = [tuple(_balanced_cases(right_rng)) for _ in range(5)]

    assert left_blocks == right_blocks
    assert all(len(block) == 10 for block in left_blocks)
    assert all(set(block) == expected for block in left_blocks)
    assert any(left_blocks[0] != block for block in left_blocks[1:])


def test_default_fixtures_are_balanced_fresh_and_immutable() -> None:
    task = DelayedCueTask()
    fixtures = _build_fixtures(
        task,
        np.random.default_rng(103),
        RewardLearningConfig().evaluation_blocks,
    )

    assert isinstance(fixtures, tuple)
    assert len(fixtures) == 200
    assert sum(episode.correct_action_index == 0 for episode in fixtures) == 100
    assert sum(episode.correct_action_index == 1 for episode in fixtures) == 100
    for delay in range(1, 6):
        assert sum(episode.delay_steps == delay for episode in fixtures) == 40

    distractors = [
        stimulus
        for episode in fixtures
        for stimulus in episode.delay_stimuli
    ]
    assert len({id(stimulus) for stimulus in distractors}) == len(distractors)
    assert all(stimulus.dtype == np.float64 for stimulus in distractors)
    assert all(stimulus.flags.writeable is False for stimulus in distractors)


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray | None]] = []
        self._step = 0

    def reset_state(self) -> None:
        self.calls.append(("reset_state", None))

    def advance(self, stimulus: np.ndarray) -> np.ndarray:
        self.calls.append(("advance", stimulus))
        self._step += 1
        return np.full(3, self._step, dtype=np.float64)


@pytest.mark.parametrize("reset_before_decision", [False, True])
def test_decision_hidden_uses_only_the_required_policy_sequence(
    reset_before_decision: bool,
) -> None:
    task = DelayedCueTask()
    episode = task.build_episode(
        Cue.RIGHT,
        3,
        np.random.default_rng(107),
    )
    policy = _RecordingPolicy()

    hidden = _decision_hidden(
        policy,
        episode,
        reset_before_decision=reset_before_decision,
    )

    expected_names = ["reset_state", "advance", "advance", "advance", "advance"]
    if reset_before_decision:
        expected_names.insert(-1, "reset_state")
    assert [name for name, _ in policy.calls] == expected_names
    advanced = [stimulus for name, stimulus in policy.calls if name == "advance"]
    expected_stimuli = [
        episode.cue_stimulus,
        *episode.delay_stimuli,
        episode.decision_stimulus,
    ]
    assert len(advanced) == len(expected_stimuli)
    assert all(
        actual is expected
        for actual, expected in zip(advanced, expected_stimuli, strict=True)
    )
    np.testing.assert_array_equal(hidden, np.full(3, 5))
    assert hidden.dtype == np.float64
    assert hidden.flags.writeable is False


class _EvaluationOnlyReadout:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, tuple[int, int]]] = []

    @property
    def has_pending_feedback(self) -> bool:
        return False

    def select_greedy(
        self,
        hidden: np.ndarray,
        legal: tuple[int, int],
    ) -> SimpleNamespace:
        self.calls.append((hidden, legal))
        return SimpleNamespace(action_index=len(self.calls) % 2)

    def select_for_training(self, *args: object) -> None:
        raise AssertionError("evaluation must not train")

    def learn(self, *args: object) -> None:
        raise AssertionError("evaluation must not learn")

    def parameter_digest(self) -> str:
        return "unchanged"


def test_evaluate_uses_only_greedy_hidden_and_legal_indices() -> None:
    fixtures = _build_fixtures(
        DelayedCueTask(),
        np.random.default_rng(109),
        2,
    )
    policy = _RecordingPolicy()
    readout = _EvaluationOnlyReadout()

    result = _evaluate(
        policy,
        readout,
        fixtures,
        reset_before_decision=False,
    )

    assert result.overall.total == 20
    assert len(result.choices) == 20
    assert len(readout.calls) == 20
    assert all(legal == (0, 1) for _, legal in readout.calls)
    assert all(
        isinstance(hidden, np.ndarray) and hidden.shape == (3,)
        for hidden, _ in readout.calls
    )


def test_reset_ablation_is_exactly_balanced_and_hidden_is_constant() -> None:
    config = RewardLearningConfig()
    fixtures = _build_fixtures(
        DelayedCueTask(),
        np.random.default_rng(113),
        config.evaluation_blocks,
    )
    policy = RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=7,
        recurrent_radius=config.recurrent_radius,
    )
    readout = RewardModulatedReadout(
        config.hidden_size,
        2,
        config.learning_rate,
        config.temperature,
    )

    result = _evaluate(
        policy,
        readout,
        fixtures,
        reset_before_decision=True,
    )

    assert result.overall == AccuracyCount(100, 200)
    assert result.per_delay == tuple(
        (delay, AccuracyCount(20, 40)) for delay in range(1, 6)
    )
    assert result.all_hidden_equal is True
    assert set(result.choices) == {0}


def test_evaluation_preserves_frozen_policy_and_readout() -> None:
    config = RewardLearningConfig(evaluation_blocks=2)
    fixtures = _build_fixtures(
        DelayedCueTask(),
        np.random.default_rng(127),
        config.evaluation_blocks,
    )
    policy = RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=17,
        recurrent_radius=config.recurrent_radius,
    )
    readout = RewardModulatedReadout(config.hidden_size, 2)
    matrices = tuple(
        values.copy()
        for values in (
            policy._input_weights,
            policy._recurrent_weights,
            policy._output_weights,
        )
    )
    digest = readout.parameter_digest()

    recurrent = _evaluate(
        policy,
        readout,
        fixtures,
        reset_before_decision=False,
    )
    reset = _evaluate(
        policy,
        readout,
        fixtures,
        reset_before_decision=True,
    )

    assert all(
        np.array_equal(before, after)
        for before, after in zip(
            matrices,
            (
                policy._input_weights,
                policy._recurrent_weights,
                policy._output_weights,
            ),
            strict=True,
        )
    )
    assert readout.parameter_digest() == digest
    assert readout.has_pending_feedback is False
    assert len(recurrent.choice_digest) == 64
    assert len(reset.choice_digest) == 64
    with pytest.raises(FrozenInstanceError):
        recurrent.overall = AccuracyCount(1, 1)
