"""Fixed fixtures and mutation-free evaluation for Phase 3A."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .action_value import NormalizedActionValue
from .memory_benchmark import AccuracyCount
from .memory_task import DelayedCueEpisode, DelayedCueTask
from .policy import RecurrentPolicy
from .reward_learning import (
    _build_fixtures,
    _decision_hidden,
    _matrix_copies,
    _matrix_digests,
    _require_frozen,
)


@dataclass(frozen=True)
class ActionValueBenchmarkConfig:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    step_size: float = 0.1
    training_episodes: int = 2_000
    evaluation_blocks: int = 20
    checkpoint_interval: int = 100

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "training_episodes",
            "evaluation_blocks",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_episodes % 10:
            raise ValueError("training_episodes must be a multiple of ten")
        if self.checkpoint_interval % 10:
            raise ValueError("checkpoint_interval must be a multiple of ten")
        if self.training_episodes % self.checkpoint_interval:
            raise ValueError("training_episodes must be divisible by checkpoint_interval")
        if not _finite_number(self.recurrent_radius) or not (0.0 <= self.recurrent_radius < 1.0):
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")
        if not _finite_number(self.step_size) or not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be finite and in (0.0, 1.0]")


@dataclass(frozen=True)
class _FixtureBundle:
    training: tuple[DelayedCueEpisode, ...]
    evaluation: tuple[DelayedCueEpisode, ...]
    training_fixture_digest: str
    evaluation_fixture_digest: str


@dataclass(frozen=True)
class _Evaluation:
    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    actions: tuple[int, ...]
    action_digest: str
    hidden_digest: str
    all_hidden_equal: bool
    margin_mean: float
    margin_p10: float
    margin_minimum: float


@dataclass(frozen=True)
class ActionValueCheckpoint:
    episode: int
    accuracy: AccuracyCount
    margin_mean: float
    margin_p10: float
    margin_minimum: float
    td_error_mean: float
    td_error_abs_mean: float
    td_error_p90: float
    td_error_maximum: float


@dataclass(frozen=True)
class _TrainingTrace:
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    action_counts: tuple[tuple[int, int], ...]
    action_digest: str
    reward_digest: str
    checkpoints: tuple[ActionValueCheckpoint, ...]
    final_parameter_digest: str
    pending_feedback: bool


@dataclass(frozen=True)
class _RunResult:
    pre_training: _Evaluation
    post_training: _Evaluation
    state_reset: _Evaluation
    shuffled_control: _Evaluation
    normal_training: _TrainingTrace
    shuffled_training: _TrainingTrace
    normal_parameter_digest_before: str
    shuffled_parameter_digest_before: str
    normal_matrix_digests_before: tuple[str, str, str]
    normal_matrix_digests_after: tuple[str, str, str]
    shuffled_matrix_digests_before: tuple[str, str, str]
    shuffled_matrix_digests_after: tuple[str, str, str]
    training_fixture_digest: str
    evaluation_fixture_digest: str


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except TypeError:
        return False


def _build_fixture_bundle(seed: int, config: ActionValueBenchmarkConfig) -> _FixtureBundle:
    task = DelayedCueTask()
    training_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x54524149]))
    evaluation_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4556414C]))
    training = _build_fixtures(task, training_rng, config.training_episodes // 10)
    evaluation = _build_fixtures(task, evaluation_rng, config.evaluation_blocks)
    return _FixtureBundle(
        training=training,
        evaluation=evaluation,
        training_fixture_digest=_episode_digest(training),
        evaluation_fixture_digest=_episode_digest(evaluation),
    )


def _evaluate(
    policy: object,
    learner: object,
    fixtures: tuple[DelayedCueEpisode, ...],
    *,
    reset_before_decision: bool,
) -> _Evaluation:
    correct_count = 0
    per_delay_correct = {delay: 0 for delay in range(1, 6)}
    per_delay_total = {delay: 0 for delay in range(1, 6)}
    actions: list[int] = []
    hidden_states: list[np.ndarray] = []
    margins: list[float] = []

    for episode in fixtures:
        hidden = _decision_hidden(
            policy,
            episode,
            reset_before_decision=reset_before_decision,
        )
        decision = learner.select_greedy(hidden, (0, 1))

        action = decision.action_index
        correct = episode.correct_action_index
        other = 1 - correct
        matched = int(action == correct)
        margin = float(decision.action_values[correct] - decision.action_values[other])
        delay = episode.delay_steps

        correct_count += matched
        per_delay_correct[delay] += matched
        per_delay_total[delay] += 1
        actions.append(action)
        hidden_states.append(hidden)
        margins.append(margin)

    action_tuple = tuple(actions)
    margin_array = np.asarray(margins, dtype=np.float64)
    first_hidden = hidden_states[0]
    return _Evaluation(
        overall=AccuracyCount(correct_count, len(fixtures)),
        per_delay=tuple(
            (delay, AccuracyCount(per_delay_correct[delay], per_delay_total[delay]))
            for delay in range(1, 6)
        ),
        actions=action_tuple,
        action_digest=hashlib.sha256(bytes(action_tuple)).hexdigest(),
        hidden_digest=_array_sequence_digest(hidden_states),
        all_hidden_equal=all(np.array_equal(first_hidden, hidden) for hidden in hidden_states[1:]),
        margin_mean=float(np.mean(margin_array)),
        margin_p10=float(np.percentile(margin_array, 10)),
        margin_minimum=float(np.min(margin_array)),
    )


def _train_normal(
    policy: object,
    learner: object,
    task: DelayedCueTask,
    fixtures: tuple[DelayedCueEpisode, ...],
    action_rng: np.random.Generator,
    config: ActionValueBenchmarkConfig,
) -> _TrainingTrace:
    def reward_for(episode: DelayedCueEpisode, action: int, _: int) -> float:
        return float(task.reward(episode, action))

    return _train(
        policy,
        learner,
        fixtures,
        action_rng,
        config,
        reward_for,
    )


def _train_shuffled(
    policy: object,
    learner: object,
    fixtures: tuple[DelayedCueEpisode, ...],
    action_rng: np.random.Generator,
    shuffled_rewards: tuple[float, ...],
    config: ActionValueBenchmarkConfig,
) -> _TrainingTrace:
    rewards = tuple(float(reward) for reward in shuffled_rewards)
    if len(rewards) != len(fixtures) or not all(math.isfinite(reward) for reward in rewards):
        raise ValueError("shuffled_rewards must contain one finite scalar per fixture")

    def reward_for(_: DelayedCueEpisode, __: int, index: int) -> float:
        return rewards[index]

    return _train(
        policy,
        learner,
        fixtures,
        action_rng,
        config,
        reward_for,
    )


def _train(
    policy: object,
    learner: object,
    fixtures: tuple[DelayedCueEpisode, ...],
    action_rng: np.random.Generator,
    config: ActionValueBenchmarkConfig,
    reward_for: object,
) -> _TrainingTrace:
    if not isinstance(action_rng, np.random.Generator):
        raise ValueError("action_rng must be a numpy.random.Generator")
    if len(fixtures) != config.training_episodes:
        raise ValueError("fixtures must match config.training_episodes")

    actions: list[int] = []
    rewards: list[float] = []
    checkpoints: list[ActionValueCheckpoint] = []
    block_hidden: list[np.ndarray] = []
    block_correct: list[int] = []
    block_td_errors: list[float] = []

    for episode_index, episode in enumerate(fixtures, start=1):
        hidden = _decision_hidden(
            policy,
            episode,
            reset_before_decision=False,
        )
        decision = learner.select_for_training(hidden, (0, 1), action_rng)
        action = int(decision.action_index)
        reward = float(reward_for(episode, action, episode_index - 1))
        update = learner.learn(reward)

        # Scoring metadata is intentionally observed only after scalar feedback.
        correct_action = episode.correct_action_index
        hidden_copy = np.array(hidden, dtype=np.float64, copy=True)
        hidden_copy.flags.writeable = False
        actions.append(action)
        rewards.append(reward)
        block_hidden.append(hidden_copy)
        block_correct.append(correct_action)
        block_td_errors.append(float(update.td_error))

        if episode_index % config.checkpoint_interval == 0:
            checkpoints.append(
                _collect_checkpoint(
                    learner,
                    episode_index,
                    block_hidden,
                    block_correct,
                    block_td_errors,
                )
            )
            block_hidden = []
            block_correct = []
            block_td_errors = []

    if block_hidden or block_correct or block_td_errors:
        raise RuntimeError("training ended with an incomplete checkpoint interval")
    if learner.has_pending_feedback:
        raise RuntimeError("learner retained pending feedback after training")
    action_tuple = tuple(actions)
    reward_tuple = tuple(rewards)
    return _TrainingTrace(
        actions=action_tuple,
        rewards=reward_tuple,
        action_counts=tuple((action, action_tuple.count(action)) for action in (0, 1)),
        action_digest=hashlib.sha256(bytes(action_tuple)).hexdigest(),
        reward_digest=_reward_digest(reward_tuple),
        checkpoints=tuple(checkpoints),
        final_parameter_digest=learner.parameter_digest(),
        pending_feedback=bool(learner.has_pending_feedback),
    )


def _collect_checkpoint(
    learner: object,
    episode: int,
    hidden_states: list[np.ndarray],
    correct_actions: list[int],
    td_errors: list[float],
) -> ActionValueCheckpoint:
    if learner.has_pending_feedback:
        raise RuntimeError("checkpoint collection requires consumed feedback")
    parameters_before = learner.parameter_snapshot()
    digest_before = learner.parameter_digest()
    margins = []
    greedy_matches = []
    for hidden, correct in zip(hidden_states, correct_actions, strict=True):
        decision = learner.select_greedy(hidden, (0, 1))
        other = 1 - correct
        greedy_matches.append(int(decision.action_index == correct))
        margins.append(float(decision.action_values[correct] - decision.action_values[other]))
    parameters_after = learner.parameter_snapshot()
    if (
        learner.has_pending_feedback
        or learner.parameter_digest() != digest_before
        or not np.array_equal(parameters_before, parameters_after)
    ):
        raise RuntimeError("checkpoint collection mutated learner state")

    margin_array = np.asarray(margins, dtype=np.float64)
    td_array = np.asarray(td_errors, dtype=np.float64)
    return ActionValueCheckpoint(
        episode=episode,
        accuracy=AccuracyCount(sum(greedy_matches), len(greedy_matches)),
        margin_mean=float(np.mean(margin_array)),
        margin_p10=float(np.percentile(margin_array, 10)),
        margin_minimum=float(np.min(margin_array)),
        td_error_mean=float(np.mean(td_array)),
        td_error_abs_mean=float(np.mean(np.abs(td_array))),
        td_error_p90=float(np.percentile(td_array, 90)),
        td_error_maximum=float(np.max(td_array)),
    )


def _permute_reward_blocks(
    rewards: tuple[float, ...],
    rng: np.random.Generator,
    *,
    block_size: int,
) -> tuple[float, ...]:
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    values = tuple(float(reward) for reward in rewards)
    if len(values) % block_size:
        raise ValueError("rewards must contain complete blocks")
    if not all(math.isfinite(reward) for reward in values):
        raise ValueError("rewards must contain only finite scalars")
    shuffled: list[float] = []
    for start in range(0, len(values), block_size):
        block = np.asarray(values[start : start + block_size], dtype=np.float64)
        permutation = rng.permutation(block_size)
        shuffled.extend(float(block[index]) for index in permutation)
    return tuple(shuffled)


def _run_once(seed: int, config: ActionValueBenchmarkConfig) -> _RunResult:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(config, ActionValueBenchmarkConfig):
        raise ValueError("config must be an ActionValueBenchmarkConfig")

    task = DelayedCueTask()
    fixtures = _build_fixture_bundle(seed, config)
    normal_policy = _new_policy(seed, config)
    shuffled_policy = _new_policy(seed, config)
    normal_learner = _new_learner(config)
    shuffled_learner = _new_learner(config)
    normal_arrays = _matrix_copies(normal_policy)
    shuffled_arrays = _matrix_copies(shuffled_policy)
    normal_matrix_before = _matrix_digests(normal_policy)
    shuffled_matrix_before = _matrix_digests(shuffled_policy)
    normal_digest_before = normal_learner.parameter_digest()
    shuffled_digest_before = shuffled_learner.parameter_digest()
    normal_action_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    shuffled_action_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33534846]))

    pre_training = _evaluate(
        normal_policy,
        normal_learner,
        fixtures.evaluation,
        reset_before_decision=False,
    )
    normal_training = _train_normal(
        normal_policy,
        normal_learner,
        task,
        fixtures.training,
        normal_action_rng,
        config,
    )
    shuffled_rewards = _permute_reward_blocks(
        normal_training.rewards,
        shuffle_rng,
        block_size=10,
    )
    shuffled_training = _train_shuffled(
        shuffled_policy,
        shuffled_learner,
        fixtures.training,
        shuffled_action_rng,
        shuffled_rewards,
        config,
    )
    post_training = _evaluate(
        normal_policy,
        normal_learner,
        fixtures.evaluation,
        reset_before_decision=False,
    )
    state_reset = _evaluate(
        normal_policy,
        normal_learner,
        fixtures.evaluation,
        reset_before_decision=True,
    )
    shuffled_control = _evaluate(
        shuffled_policy,
        shuffled_learner,
        fixtures.evaluation,
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
        normal_parameter_digest_before=normal_digest_before,
        shuffled_parameter_digest_before=shuffled_digest_before,
        normal_matrix_digests_before=normal_matrix_before,
        normal_matrix_digests_after=_matrix_digests(normal_policy),
        shuffled_matrix_digests_before=shuffled_matrix_before,
        shuffled_matrix_digests_after=_matrix_digests(shuffled_policy),
        training_fixture_digest=fixtures.training_fixture_digest,
        evaluation_fixture_digest=fixtures.evaluation_fixture_digest,
    )


def _new_policy(seed: int, config: ActionValueBenchmarkConfig) -> RecurrentPolicy:
    return RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=seed,
        recurrent_radius=config.recurrent_radius,
    )


def _new_learner(config: ActionValueBenchmarkConfig) -> NormalizedActionValue:
    return NormalizedActionValue(
        config.hidden_size,
        2,
        step_size=config.step_size,
    )


def _reward_digest(rewards: tuple[float, ...]) -> str:
    values = np.ascontiguousarray(rewards, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(len(rewards).to_bytes(8, "little"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _episode_digest(episodes: tuple[DelayedCueEpisode, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(episodes).to_bytes(8, "little"))
    for episode in episodes:
        digest.update(int(episode.cue).to_bytes(1, "little"))
        digest.update(episode.delay_steps.to_bytes(1, "little"))
        digest.update(episode.correct_action_index.to_bytes(1, "little"))
        arrays = (
            episode.cue_stimulus,
            *episode.delay_stimuli,
            episode.decision_stimulus,
        )
        digest.update(len(arrays).to_bytes(1, "little"))
        _update_array_digest(digest, arrays)
    return digest.hexdigest()


def _array_sequence_digest(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(len(arrays).to_bytes(8, "little"))
    _update_array_digest(digest, arrays)
    return digest.hexdigest()


def _update_array_digest(
    digest: object,
    arrays: tuple[np.ndarray, ...] | list[np.ndarray],
) -> None:
    for values in arrays:
        contiguous = np.ascontiguousarray(values, dtype=np.float64)
        shape = tuple(int(size) for size in contiguous.shape)
        digest.update(len(shape).to_bytes(1, "little"))
        for size in shape:
            digest.update(size.to_bytes(8, "little"))
        digest.update(contiguous.tobytes(order="C"))
