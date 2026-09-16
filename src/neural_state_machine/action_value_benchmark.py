"""Fixed fixtures and mutation-free evaluation for Phase 3A."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
class ActionValueExperimentResult:
    seed: int
    config: ActionValueBenchmarkConfig
    pre_training: AccuracyCount
    post_training: AccuracyCount
    state_reset: AccuracyCount
    shuffled_control: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    reset_per_delay: tuple[tuple[int, AccuracyCount], ...]
    shuffled_per_delay: tuple[tuple[int, AccuracyCount], ...]
    normal_checkpoints: tuple[ActionValueCheckpoint, ...]
    shuffled_checkpoints: tuple[ActionValueCheckpoint, ...]
    normal_action_counts: tuple[tuple[int, int], ...]
    shuffled_action_counts: tuple[tuple[int, int], ...]
    normal_actions: tuple[int, ...]
    shuffled_actions: tuple[int, ...]
    normal_action_digest: str
    shuffled_action_digest: str
    normal_reward_digest: str
    shuffled_reward_digest: str
    action_sequences_equal: bool
    reward_block_multisets_equal: bool
    initial_parameter_digest: str
    normal_parameter_digest: str
    shuffled_parameter_digest: str
    normal_matrix_digests_before: tuple[str, str, str]
    normal_matrix_digests_after: tuple[str, str, str]
    shuffled_matrix_digests_before: tuple[str, str, str]
    shuffled_matrix_digests_after: tuple[str, str, str]
    training_fixture_digest: str
    evaluation_fixture_digest: str
    decision_hidden_digest: str
    reset_hidden_digest: str
    post_margin_mean: float
    post_margin_p10: float
    post_margin_minimum: float
    all_reset_hidden_equal: bool
    normal_pending_feedback: bool
    shuffled_pending_feedback: bool
    repeatable: bool


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


def run_action_value_experiment(
    seed: int = 7,
    config: ActionValueBenchmarkConfig | None = None,
) -> ActionValueExperimentResult:
    resolved_seed = _validated_seed(seed)
    resolved_config = _validated_config(config)
    first = _run_once(resolved_seed, resolved_config)
    second = _run_once(resolved_seed, resolved_config)
    return _public_result(
        resolved_seed,
        resolved_config,
        first,
        repeatable=first == second,
    )


def run_action_value_benchmark(
    seeds: Sequence[int] = (7, 17, 29),
    config: ActionValueBenchmarkConfig | None = None,
) -> dict[str, object]:
    resolved_seeds = _validated_seeds(seeds)
    resolved_config = _validated_config(config)
    results = tuple(
        run_action_value_experiment(seed, resolved_config) for seed in resolved_seeds
    )
    pooled_correct = sum(result.shuffled_control.correct for result in results)
    pooled_total = sum(result.shuffled_control.total for result in results)
    pooled_accuracy = pooled_correct / pooled_total
    per_seed_passed = tuple(_passes_acceptance(result) for result in results)
    return {
        "all_passed": all(per_seed_passed) and 0.40 <= pooled_accuracy <= 0.60,
        "config": _config_json(resolved_config),
        "evidence_schema_version": 1,
        "frozen_evidence_sha256": _frozen_evidence_sha256(),
        "phase": "3A",
        "results": [
            _result_json(result, passed)
            for result, passed in zip(results, per_seed_passed, strict=True)
        ],
        "rng_lineages": {
            "behavior_action": ["seed", 0x33414354],
            "evaluation_fixture": ["seed", 0x4556414C],
            "reward_shuffle": ["seed", 0x33534846],
            "training_fixture": ["seed", 0x54524149],
        },
        "seeds": list(resolved_seeds),
        "shuffled_pooled": {
            "accuracy": pooled_accuracy,
            "correct": pooled_correct,
            "total": pooled_total,
        },
    }


def _validated_seed(seed: object) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validated_seeds(seeds: object) -> tuple[int, ...]:
    try:
        resolved = tuple(seeds)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("seeds must be a non-empty sequence") from exc
    if not resolved:
        raise ValueError("seeds must be a non-empty sequence")
    if any(type(seed) is not int or seed < 0 for seed in resolved):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError("seeds must not contain duplicates")
    return resolved


def _validated_config(config: object) -> ActionValueBenchmarkConfig:
    if config is None:
        return ActionValueBenchmarkConfig()
    if not isinstance(config, ActionValueBenchmarkConfig):
        raise ValueError("config must be an ActionValueBenchmarkConfig")
    return config


def _public_result(
    seed: int,
    config: ActionValueBenchmarkConfig,
    run: _RunResult,
    *,
    repeatable: bool,
) -> ActionValueExperimentResult:
    return ActionValueExperimentResult(
        seed=seed,
        config=config,
        pre_training=run.pre_training.overall,
        post_training=run.post_training.overall,
        state_reset=run.state_reset.overall,
        shuffled_control=run.shuffled_control.overall,
        per_delay=run.post_training.per_delay,
        reset_per_delay=run.state_reset.per_delay,
        shuffled_per_delay=run.shuffled_control.per_delay,
        normal_checkpoints=run.normal_training.checkpoints,
        shuffled_checkpoints=run.shuffled_training.checkpoints,
        normal_action_counts=run.normal_training.action_counts,
        shuffled_action_counts=run.shuffled_training.action_counts,
        normal_actions=run.normal_training.actions,
        shuffled_actions=run.shuffled_training.actions,
        normal_action_digest=run.normal_training.action_digest,
        shuffled_action_digest=run.shuffled_training.action_digest,
        normal_reward_digest=run.normal_training.reward_digest,
        shuffled_reward_digest=run.shuffled_training.reward_digest,
        action_sequences_equal=(run.normal_training.actions == run.shuffled_training.actions),
        reward_block_multisets_equal=_reward_block_multisets_equal(
            run.normal_training.rewards,
            run.shuffled_training.rewards,
        ),
        initial_parameter_digest=run.normal_parameter_digest_before,
        normal_parameter_digest=run.normal_training.final_parameter_digest,
        shuffled_parameter_digest=run.shuffled_training.final_parameter_digest,
        normal_matrix_digests_before=run.normal_matrix_digests_before,
        normal_matrix_digests_after=run.normal_matrix_digests_after,
        shuffled_matrix_digests_before=run.shuffled_matrix_digests_before,
        shuffled_matrix_digests_after=run.shuffled_matrix_digests_after,
        training_fixture_digest=run.training_fixture_digest,
        evaluation_fixture_digest=run.evaluation_fixture_digest,
        decision_hidden_digest=run.post_training.hidden_digest,
        reset_hidden_digest=run.state_reset.hidden_digest,
        post_margin_mean=run.post_training.margin_mean,
        post_margin_p10=run.post_training.margin_p10,
        post_margin_minimum=run.post_training.margin_minimum,
        all_reset_hidden_equal=run.state_reset.all_hidden_equal,
        normal_pending_feedback=run.normal_training.pending_feedback,
        shuffled_pending_feedback=run.shuffled_training.pending_feedback,
        repeatable=repeatable,
    )


def _reward_block_multisets_equal(
    normal: tuple[float, ...],
    shuffled: tuple[float, ...],
) -> bool:
    return len(normal) == len(shuffled) and len(normal) % 10 == 0 and all(
        sorted(normal[start : start + 10]) == sorted(shuffled[start : start + 10])
        for start in range(0, len(normal), 10)
    )


def _passes_acceptance(result: ActionValueExperimentResult) -> bool:
    return (
        result.repeatable
        and result.post_training.total == 200
        and result.post_training.correct >= 180
        and all(
            score == AccuracyCount(score.correct, 40) and score.correct >= 34
            for _, score in result.per_delay
        )
        and result.state_reset == AccuracyCount(100, 200)
        and all(score == AccuracyCount(20, 40) for _, score in result.reset_per_delay)
        and result.all_reset_hidden_equal
        and result.shuffled_control.total == 200
        and result.shuffled_control.correct < 150
        and result.action_sequences_equal
        and result.reward_block_multisets_equal
        and result.normal_parameter_digest != result.initial_parameter_digest
        and result.shuffled_parameter_digest != result.initial_parameter_digest
        and result.normal_matrix_digests_before == result.normal_matrix_digests_after
        and result.shuffled_matrix_digests_before == result.shuffled_matrix_digests_after
        and not result.normal_pending_feedback
        and not result.shuffled_pending_feedback
    )


def _config_json(config: ActionValueBenchmarkConfig) -> dict[str, object]:
    return {
        "checkpoint_interval": config.checkpoint_interval,
        "evaluation_blocks": config.evaluation_blocks,
        "hidden_size": config.hidden_size,
        "recurrent_radius": config.recurrent_radius,
        "step_size": config.step_size,
        "training_episodes": config.training_episodes,
    }


def _result_json(
    result: ActionValueExperimentResult,
    passed: bool,
) -> dict[str, object]:
    return {
        "action_sequences_equal": result.action_sequences_equal,
        "all_reset_hidden_equal": result.all_reset_hidden_equal,
        "decision_hidden_digest": result.decision_hidden_digest,
        "evaluation_fixture_digest": result.evaluation_fixture_digest,
        "initial_parameter_digest": result.initial_parameter_digest,
        "matrix_controls": {
            "normal_after": list(result.normal_matrix_digests_after),
            "normal_before": list(result.normal_matrix_digests_before),
            "shuffled_after": list(result.shuffled_matrix_digests_after),
            "shuffled_before": list(result.shuffled_matrix_digests_before),
        },
        "normal_action_counts": _action_counts_json(result.normal_action_counts),
        "normal_action_digest": result.normal_action_digest,
        "normal_actions": list(result.normal_actions),
        "normal_checkpoints": _checkpoints_json(result.normal_checkpoints),
        "normal_parameter_digest": result.normal_parameter_digest,
        "normal_pending_feedback": result.normal_pending_feedback,
        "normal_reward_digest": result.normal_reward_digest,
        "passed": passed,
        "per_delay": _per_delay_json(result.per_delay),
        "post_margin": {
            "mean": result.post_margin_mean,
            "minimum": result.post_margin_minimum,
            "p10": result.post_margin_p10,
        },
        "post_training": _count_json(result.post_training),
        "pre_training": _count_json(result.pre_training),
        "repeatable": result.repeatable,
        "reset_hidden_digest": result.reset_hidden_digest,
        "reset_per_delay": _per_delay_json(result.reset_per_delay),
        "reward_block_multisets_equal": result.reward_block_multisets_equal,
        "seed": result.seed,
        "shuffled_action_counts": _action_counts_json(result.shuffled_action_counts),
        "shuffled_action_digest": result.shuffled_action_digest,
        "shuffled_actions": list(result.shuffled_actions),
        "shuffled_checkpoints": _checkpoints_json(result.shuffled_checkpoints),
        "shuffled_control": _count_json(result.shuffled_control),
        "shuffled_parameter_digest": result.shuffled_parameter_digest,
        "shuffled_pending_feedback": result.shuffled_pending_feedback,
        "shuffled_per_delay": _per_delay_json(result.shuffled_per_delay),
        "shuffled_reward_digest": result.shuffled_reward_digest,
        "state_reset": _count_json(result.state_reset),
        "training_fixture_digest": result.training_fixture_digest,
    }


def _count_json(count: AccuracyCount) -> dict[str, object]:
    return {
        "accuracy": count.accuracy,
        "correct": count.correct,
        "total": count.total,
    }


def _per_delay_json(
    rows: tuple[tuple[int, AccuracyCount], ...],
) -> list[dict[str, object]]:
    return [
        {
            "accuracy": score.accuracy,
            "correct": score.correct,
            "delay": delay,
            "total": score.total,
        }
        for delay, score in rows
    ]


def _action_counts_json(rows: tuple[tuple[int, int], ...]) -> list[dict[str, int]]:
    return [{"action": action, "count": count} for action, count in rows]


def _checkpoints_json(
    checkpoints: tuple[ActionValueCheckpoint, ...],
) -> list[dict[str, object]]:
    return [
        {
            "accuracy": _count_json(checkpoint.accuracy),
            "episode": checkpoint.episode,
            "margin_mean": checkpoint.margin_mean,
            "margin_minimum": checkpoint.margin_minimum,
            "margin_p10": checkpoint.margin_p10,
            "td_error_abs_mean": checkpoint.td_error_abs_mean,
            "td_error_maximum": checkpoint.td_error_maximum,
            "td_error_mean": checkpoint.td_error_mean,
            "td_error_p90": checkpoint.td_error_p90,
        }
        for checkpoint in checkpoints
    ]


def _frozen_evidence_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        "phase_2b": hashlib.sha256(
            (root / "docs/experiments/phase-2b-failure.json").read_bytes()
        ).hexdigest(),
        "phase_2c": hashlib.sha256(
            (root / "docs/experiments/phase-2c-diagnostics.json").read_bytes()
        ).hexdigest(),
    }
