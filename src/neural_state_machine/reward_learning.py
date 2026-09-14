"""Frozen recurrent-state evaluation protocol for Phase 2B reward learning."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .memory_benchmark import AccuracyCount
from .memory_task import Cue, DelayedCueEpisode, DelayedCueTask
from .policy import RecurrentPolicy
from .reward_readout import RewardModulatedReadout


@dataclass(frozen=True)
class RewardLearningConfig:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    learning_rate: float = 0.05
    temperature: float = 1.0
    training_episodes: int = 2_000
    evaluation_blocks: int = 20

    def __post_init__(self) -> None:
        for name in ("hidden_size", "evaluation_blocks", "training_episodes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_episodes % 10:
            raise ValueError("training_episodes must be a multiple of ten")
        if not _finite_number(self.recurrent_radius) or not (
            0.0 <= self.recurrent_radius < 1.0
        ):
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")
        for name in ("learning_rate", "temperature"):
            value = getattr(self, name)
            if not _finite_number(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class _Evaluation:
    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    choices: tuple[int, ...]
    choice_digest: str
    all_hidden_equal: bool



@dataclass(frozen=True)
class _TrainingResult:
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    total_reward: int
    final_block: AccuracyCount
    parameter_digest: str


@dataclass(frozen=True)
class _RunResult:
    pre_training: _Evaluation
    post_training: _Evaluation
    state_reset: _Evaluation
    shuffled_control: _Evaluation
    normal_training: _TrainingResult
    shuffled_training: _TrainingResult
    normal_readout_digest_before: str
    normal_readout_digest_after: str
    shuffled_readout_digest_before: str
    shuffled_readout_digest_after: str
    normal_matrix_digests_before: tuple[str, str, str]
    normal_matrix_digests_after: tuple[str, str, str]
    shuffled_matrix_digests_before: tuple[str, str, str]
    shuffled_matrix_digests_after: tuple[str, str, str]
    normal_pending_feedback: bool
    shuffled_pending_feedback: bool

def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except TypeError:
        return False


def _balanced_cases(rng: np.random.Generator) -> list[tuple[Cue, int]]:
    if not isinstance(rng, np.random.Generator):
        raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
            "rng must be a numpy.random.Generator"
        )
    cases = [
        (cue, delay)
        for cue in (Cue.LEFT, Cue.RIGHT)
        for delay in range(1, 6)
    ]
    rng.shuffle(cases)
    return cases


def _build_fixtures(
    task: DelayedCueTask,
    rng: np.random.Generator,
    blocks: int,
) -> tuple[DelayedCueEpisode, ...]:
    if not isinstance(task, DelayedCueTask):
        raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
            "task must be a DelayedCueTask"
        )
    if not isinstance(rng, np.random.Generator):
        raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
            "rng must be a numpy.random.Generator"
        )
    if type(blocks) is not int or blocks <= 0:
        raise ValueError("blocks must be a positive integer")
    return tuple(
        task.build_episode(cue, delay, rng)
        for _ in range(blocks)
        for cue, delay in _balanced_cases(rng)
    )


def _decision_hidden(
    policy: object,
    episode: DelayedCueEpisode,
    *,
    reset_before_decision: bool,
) -> np.ndarray:
    if not isinstance(episode, DelayedCueEpisode):
        raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
            "episode must be a DelayedCueEpisode"
        )
    if type(reset_before_decision) is not bool:
        raise ValueError("reset_before_decision must be a boolean")

    policy.reset_state()
    policy.advance(episode.cue_stimulus)
    for delay_stimulus in episode.delay_stimuli:
        policy.advance(delay_stimulus)
    if reset_before_decision:
        policy.reset_state()
    hidden = np.asarray(
        policy.advance(episode.decision_stimulus),
        dtype=np.float64,
    )
    if hidden.ndim != 1 or not np.all(np.isfinite(hidden)):
        raise ValueError("policy must return a finite rank-one hidden vector")
    return _readonly_copy(hidden)


def _evaluate(
    policy: object,
    readout: object,
    fixtures: tuple[DelayedCueEpisode, ...],
    *,
    reset_before_decision: bool,
) -> _Evaluation:
    correct = 0
    per_delay_correct = {delay: 0 for delay in range(1, 6)}
    per_delay_total = {delay: 0 for delay in range(1, 6)}
    choices: list[int] = []
    hidden_states: list[np.ndarray] = []

    for episode in fixtures:
        hidden = _decision_hidden(
            policy,
            episode,
            reset_before_decision=reset_before_decision,
        )
        decision = readout.select_greedy(hidden, (0, 1))
        action = decision.action_index
        matched = int(action == episode.correct_action_index)
        correct += matched
        per_delay_correct[episode.delay_steps] += matched
        per_delay_total[episode.delay_steps] += 1
        choices.append(action)
        hidden_states.append(hidden)

    choices_tuple = tuple(choices)
    first_hidden = hidden_states[0]
    return _Evaluation(
        overall=AccuracyCount(correct, len(fixtures)),
        per_delay=tuple(
            (delay, AccuracyCount(per_delay_correct[delay], per_delay_total[delay]))
            for delay in range(1, 6)
        ),
        choices=choices_tuple,
        choice_digest=hashlib.sha256(bytes(choices_tuple)).hexdigest(),
        all_hidden_equal=all(
            np.array_equal(first_hidden, hidden)
            for hidden in hidden_states[1:]
        ),
    )


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


def _train_normal(
    policy: object,
    readout: object,
    task: DelayedCueTask,
    fixtures: tuple[DelayedCueEpisode, ...],
    action_rng: np.random.Generator,
) -> _TrainingResult:
    actions: list[int] = []
    rewards: list[float] = []
    final_correct = 0
    final_start = len(fixtures) - 10
    for episode_index, episode in enumerate(fixtures):
        hidden = _decision_hidden(
            policy,
            episode,
            reset_before_decision=False,
        )
        action = readout.select_for_training(
            hidden,
            (0, 1),
            action_rng,
        ).action_index
        reward = task.reward(episode, action)
        readout.learn(reward)
        actions.append(action)
        rewards.append(reward)
        if episode_index >= final_start:
            final_correct += int(action == episode.correct_action_index)
    return _TrainingResult(
        actions=tuple(actions),
        rewards=tuple(rewards),
        total_reward=int(sum(rewards)),
        final_block=AccuracyCount(final_correct, 10),
        parameter_digest=readout.parameter_digest(),
    )


def _train_shuffled(
    policy: object,
    readout: object,
    task: object,
    fixtures: tuple[DelayedCueEpisode, ...],
    action_rng: np.random.Generator,
    shuffled_reward_rng: np.random.Generator,
) -> _TrainingResult:
    del task
    actions: list[int] = []
    rewards: list[float] = []
    final_correct = 0
    final_start = len(fixtures) - 10
    for block_start in range(0, len(fixtures), 10):
        block_rewards = np.array(
            [1.0] * 5 + [-1.0] * 5,
            dtype=np.float64,
        )
        shuffled_reward_rng.shuffle(block_rewards)
        for offset, episode in enumerate(fixtures[block_start : block_start + 10]):
            hidden = _decision_hidden(
                policy,
                episode,
                reset_before_decision=False,
            )
            action = readout.select_for_training(
                hidden,
                (0, 1),
                action_rng,
            ).action_index
            reward = float(block_rewards[offset])
            readout.learn(reward)
            actions.append(action)
            rewards.append(reward)
            if block_start + offset >= final_start:
                final_correct += int(action == episode.correct_action_index)
    return _TrainingResult(
        actions=tuple(actions),
        rewards=tuple(rewards),
        total_reward=int(sum(rewards)),
        final_block=AccuracyCount(final_correct, 10),
        parameter_digest=readout.parameter_digest(),
    )


def _run_once(seed: int, config: RewardLearningConfig) -> _RunResult:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    task = DelayedCueTask()
    training_fixture_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x54524149])
    )
    training_action_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x4143544E])
    )
    evaluation_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x4556414C])
    )
    shuffled_reward_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x53485546])
    )
    shuffled_action_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x53414354])
    )
    training_fixtures = _build_fixtures(
        task,
        training_fixture_rng,
        config.training_episodes // 10,
    )
    evaluation_fixtures = _build_fixtures(
        task,
        evaluation_rng,
        config.evaluation_blocks,
    )

    normal_policy = _new_policy(seed, config)
    shuffled_policy = _new_policy(seed, config)
    normal_readout = _new_readout(config)
    shuffled_readout = _new_readout(config)
    normal_arrays = _matrix_copies(normal_policy)
    shuffled_arrays = _matrix_copies(shuffled_policy)
    normal_matrix_before = _matrix_digests(normal_policy)
    shuffled_matrix_before = _matrix_digests(shuffled_policy)
    normal_digest_before = normal_readout.parameter_digest()
    shuffled_digest_before = shuffled_readout.parameter_digest()

    pre_training = _evaluate(
        normal_policy,
        normal_readout,
        evaluation_fixtures,
        reset_before_decision=False,
    )
    normal_training = _train_normal(
        normal_policy,
        normal_readout,
        task,
        training_fixtures,
        training_action_rng,
    )
    shuffled_training = _train_shuffled(
        shuffled_policy,
        shuffled_readout,
        task,
        training_fixtures,
        shuffled_action_rng,
        shuffled_reward_rng,
    )
    post_training = _evaluate(
        normal_policy,
        normal_readout,
        evaluation_fixtures,
        reset_before_decision=False,
    )
    state_reset = _evaluate(
        normal_policy,
        normal_readout,
        evaluation_fixtures,
        reset_before_decision=True,
    )
    shuffled_control = _evaluate(
        shuffled_policy,
        shuffled_readout,
        evaluation_fixtures,
        reset_before_decision=False,
    )

    _require_frozen(normal_arrays, normal_policy)
    _require_frozen(shuffled_arrays, shuffled_policy)
    return _RunResult(
        pre_training=pre_training,
        post_training=post_training,
        state_reset=state_reset,
        shuffled_control=shuffled_control,
        normal_training=normal_training,
        shuffled_training=shuffled_training,
        normal_readout_digest_before=normal_digest_before,
        normal_readout_digest_after=normal_readout.parameter_digest(),
        shuffled_readout_digest_before=shuffled_digest_before,
        shuffled_readout_digest_after=shuffled_readout.parameter_digest(),
        normal_matrix_digests_before=normal_matrix_before,
        normal_matrix_digests_after=_matrix_digests(normal_policy),
        shuffled_matrix_digests_before=shuffled_matrix_before,
        shuffled_matrix_digests_after=_matrix_digests(shuffled_policy),
        normal_pending_feedback=normal_readout.has_pending_feedback,
        shuffled_pending_feedback=shuffled_readout.has_pending_feedback,
    )


def _new_policy(seed: int, config: RewardLearningConfig) -> RecurrentPolicy:
    return RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=seed,
        recurrent_radius=config.recurrent_radius,
    )


def _new_readout(config: RewardLearningConfig) -> RewardModulatedReadout:
    return RewardModulatedReadout(
        config.hidden_size,
        2,
        config.learning_rate,
        config.temperature,
    )


def _matrix_copies(
    policy: RecurrentPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        policy._input_weights.copy(),
        policy._recurrent_weights.copy(),
        policy._output_weights.copy(),
    )


def _matrix_digests(policy: RecurrentPolicy) -> tuple[str, str, str]:
    return tuple(_array_digest(values) for values in _policy_matrices(policy))


def _policy_matrices(
    policy: RecurrentPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        policy._input_weights,
        policy._recurrent_weights,
        policy._output_weights,
    )


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _require_frozen(
    before: tuple[np.ndarray, np.ndarray, np.ndarray],
    policy: RecurrentPolicy,
) -> None:
    if not all(
        np.array_equal(expected, actual)
        for expected, actual in zip(
            before,
            _policy_matrices(policy),
            strict=True,
        )
    ):
        raise RuntimeError("frozen recurrent policy matrices changed")
