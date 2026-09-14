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
    _run_once,
    _train_normal,
    _train_shuffled,
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

    expected_names = [
        "reset_state",
        "advance",
        "advance",
        "advance",
        "advance",
        "advance",
    ]
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



class _TrainingReadout:
    def __init__(self, actions: tuple[int, ...]) -> None:
        self._actions = iter(actions)
        self.selection_calls: list[tuple[np.ndarray, tuple[int, int], object]] = []
        self.rewards: list[float] = []
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
        self.selection_calls.append((hidden, legal, rng))
        self._pending = True
        return SimpleNamespace(action_index=next(self._actions))

    def learn(self, reward: float) -> None:
        assert self._pending is True
        self.rewards.append(reward)
        self._pending = False

    def parameter_digest(self) -> str:
        return "training-digest"


def test_training_fixture_tuple_is_balanced_and_rng_isolated() -> None:
    config = RewardLearningConfig(training_episodes=20, evaluation_blocks=2)
    seed = 131
    left_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x54524149])
    )
    right_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x54524149])
    )
    evaluation_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x4556414C])
    )
    evaluation_rng.normal(size=10_000)

    left = _build_fixtures(
        DelayedCueTask(),
        left_rng,
        config.training_episodes // 10,
    )
    right = _build_fixtures(
        DelayedCueTask(),
        right_rng,
        config.training_episodes // 10,
    )

    assert left == right
    assert len(left) == 20
    expected = {(cue, delay) for cue in Cue for delay in range(1, 6)}
    for offset in range(0, len(left), 10):
        assert {
            (episode.cue, episode.delay_steps)
            for episode in left[offset : offset + 10]
        } == expected
    distractors = [
        stimulus for episode in left for stimulus in episode.delay_stimuli
    ]
    assert len({id(value) for value in distractors}) == len(distractors)


def test_normal_training_boundary_passes_only_hidden_rng_and_scalar_reward() -> None:
    task = DelayedCueTask()
    fixtures = _build_fixtures(task, np.random.default_rng(137), 1)
    policy = _RecordingPolicy()
    actions = tuple(index % 2 for index in range(10))
    readout = _TrainingReadout(actions)
    action_rng = np.random.default_rng(139)

    result = _train_normal(policy, readout, task, fixtures, action_rng)

    assert len(readout.selection_calls) == 10
    assert all(legal == (0, 1) for _, legal, _ in readout.selection_calls)
    assert all(rng is action_rng for _, _, rng in readout.selection_calls)
    assert all(hidden.shape == (3,) for hidden, _, _ in readout.selection_calls)
    assert all(type(reward) is float for reward in readout.rewards)
    assert tuple(readout.rewards) == result.rewards
    assert result.actions == actions
    assert result.total_reward == sum(result.rewards)
    assert readout.has_pending_feedback is False


def test_normal_training_matches_an_independent_reference_loop() -> None:
    config = RewardLearningConfig(
        hidden_size=8,
        training_episodes=20,
        evaluation_blocks=1,
    )
    task = DelayedCueTask()
    fixtures = _build_fixtures(
        task,
        np.random.default_rng(np.random.SeedSequence([149, 0x54524149])),
        2,
    )
    implementation_policy = RecurrentPolicy(
        4,
        2,
        hidden_size=config.hidden_size,
        seed=149,
        recurrent_radius=config.recurrent_radius,
    )
    reference_policy = RecurrentPolicy(
        4,
        2,
        hidden_size=config.hidden_size,
        seed=149,
        recurrent_radius=config.recurrent_radius,
    )
    implementation_readout = RewardModulatedReadout(
        config.hidden_size,
        2,
        config.learning_rate,
        config.temperature,
    )
    reference_readout = RewardModulatedReadout(
        config.hidden_size,
        2,
        config.learning_rate,
        config.temperature,
    )
    implementation_rng = np.random.default_rng(
        np.random.SeedSequence([149, 0x4143544E])
    )
    reference_rng = np.random.default_rng(
        np.random.SeedSequence([149, 0x4143544E])
    )

    actual = _train_normal(
        implementation_policy,
        implementation_readout,
        task,
        fixtures,
        implementation_rng,
    )
    actions = []
    rewards = []
    for episode in fixtures:
        hidden = _decision_hidden(
            reference_policy,
            episode,
            reset_before_decision=False,
        )
        action = reference_readout.select_for_training(
            hidden,
            (0, 1),
            reference_rng,
        ).action_index
        reward = task.reward(episode, action)
        reference_readout.learn(reward)
        actions.append(action)
        rewards.append(reward)

    assert actual.actions == tuple(actions)
    assert actual.rewards == tuple(rewards)
    assert actual.total_reward == sum(rewards)
    assert actual.final_block == AccuracyCount(
        sum(reward > 0 for reward in rewards[-10:]),
        10,
    )
    assert actual.parameter_digest == reference_readout.parameter_digest()


class _RewardMustNotBeCalled:
    def reward(self, *args: object) -> float:
        raise AssertionError("shuffled control must not call task.reward")


def test_shuffled_training_uses_only_balanced_independent_rewards() -> None:
    fixtures = _build_fixtures(
        DelayedCueTask(),
        np.random.default_rng(151),
        3,
    )
    policy = _RecordingPolicy()
    readout = _TrainingReadout(tuple(index % 2 for index in range(30)))
    action_rng = np.random.default_rng(
        np.random.SeedSequence([151, 0x53414354])
    )
    reward_rng = np.random.default_rng(
        np.random.SeedSequence([151, 0x53485546])
    )

    result = _train_shuffled(
        policy,
        readout,
        _RewardMustNotBeCalled(),
        fixtures,
        action_rng,
        reward_rng,
    )

    assert tuple(readout.rewards) == result.rewards
    for offset in range(0, 30, 10):
        block = result.rewards[offset : offset + 10]
        assert block.count(1.0) == 5
        assert block.count(-1.0) == 5
    assert all(rng is action_rng for _, _, rng in readout.selection_calls)
    assert result.total_reward == 0
    assert readout.has_pending_feedback is False


def test_run_once_preserves_frozen_matrices_and_separates_learners() -> None:
    result = _run_once(
        157,
        RewardLearningConfig(
            hidden_size=8,
            training_episodes=20,
            evaluation_blocks=2,
        ),
    )

    assert result.normal_matrix_digests_before == (
        result.normal_matrix_digests_after
    )
    assert result.shuffled_matrix_digests_before == (
        result.shuffled_matrix_digests_after
    )
    assert result.normal_matrix_digests_before == (
        result.shuffled_matrix_digests_before
    )
    assert result.normal_readout_digest_before != (
        result.normal_readout_digest_after
    )
    assert result.shuffled_readout_digest_before != (
        result.shuffled_readout_digest_after
    )
    assert result.normal_readout_digest_after != (
        result.shuffled_readout_digest_after
    )
    assert result.normal_pending_feedback is False
    assert result.shuffled_pending_feedback is False
    assert result.pre_training.overall.total == 20
    assert result.post_training.overall.total == 20
    assert result.state_reset.overall == AccuracyCount(10, 20)
    assert result.shuffled_control.overall.total == 20
