"""Balanced delayed-cue training and matched recurrent/reset evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .memory_task import Cue, DelayedCueEpisode, DelayedCueTask
from .policy import RecurrentPolicy


@dataclass(frozen=True)
class MemoryExperimentConfig:
    hidden_size: int = 64
    learning_rate: float = 0.05
    recurrent_radius: float = 0.9
    training_episodes: int = 2_000
    epsilon_start: float = 0.25
    epsilon_end: float = 0.02
    evaluation_blocks: int = 20

    def __post_init__(self) -> None:
        for name in ("hidden_size", "evaluation_blocks", "training_episodes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_episodes % 10:
            raise ValueError("training_episodes must be a multiple of ten")
        if not _finite_number(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not _finite_number(self.recurrent_radius) or not 0.0 <= self.recurrent_radius < 1.0:
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")
        for name in ("epsilon_start", "epsilon_end"):
            value = getattr(self, name)
            if not _finite_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0.0, 1.0]")
        if self.epsilon_start < self.epsilon_end:
            raise ValueError("epsilon_start must be at least epsilon_end")


@dataclass(frozen=True)
class AccuracyCount:
    correct: int
    total: int

    def __post_init__(self) -> None:
        if type(self.correct) is not int or type(self.total) is not int:
            raise ValueError("correct and total must be integers")
        if self.total <= 0 or not 0 <= self.correct <= self.total:
            raise ValueError("counts must satisfy 0 <= correct <= total with positive total")

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True)
class _Evaluation:
    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    choices: tuple[int, ...]


@dataclass(frozen=True)
class _RunResult:
    pre_training: AccuracyCount
    post_training: AccuracyCount
    state_reset: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    total_training_reward: int
    final_block: AccuracyCount
    output_weight_digest: str
    recurrent_choice_digest: str
    reset_choice_digest: str


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except TypeError:
        return False


def _balanced_cases(rng: np.random.Generator) -> list[tuple[Cue, int]]:
    cases = [(cue, delay) for cue in (Cue.LEFT, Cue.RIGHT) for delay in range(1, 6)]
    rng.shuffle(cases)
    return cases


def _epsilon(episode_index: int, config: MemoryExperimentConfig) -> float:
    fraction = episode_index / (config.training_episodes - 1)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def _run_episode(
    policy: RecurrentPolicy,
    task: DelayedCueTask,
    episode: DelayedCueEpisode,
    *,
    explore_probability: float,
    rng: np.random.Generator | None,
    learn: bool,
    reset_before_decision: bool = False,
) -> tuple[int, float]:
    """Pass numeric frames to the policy; fixture labels stay in task scoring."""
    policy.reset_state()
    policy.advance(episode.cue_stimulus)
    for delay_stimulus in episode.delay_stimuli:
        policy.advance(delay_stimulus)
    if reset_before_decision:
        policy.reset_state()
    decision = policy.decide(
        episode.decision_stimulus,
        (int(Cue.LEFT), int(Cue.RIGHT)),
        explore_probability=explore_probability,
        rng=rng if explore_probability > 0.0 else None,
    )
    reward = task.reward(episode, decision.action_index)
    if learn:
        policy.learn(reward)
    return decision.action_index, reward


def _evaluation_fixtures(
    task: DelayedCueTask,
    rng: np.random.Generator,
    blocks: int,
) -> tuple[DelayedCueEpisode, ...]:
    return tuple(
        task.build_episode(cue, delay, rng)
        for _ in range(blocks)
        for cue, delay in _balanced_cases(rng)
    )


def _evaluate(
    policy: RecurrentPolicy,
    task: DelayedCueTask,
    fixtures: tuple[DelayedCueEpisode, ...],
    *,
    reset_before_decision: bool,
) -> _Evaluation:
    correct = 0
    per_delay_correct = {delay: 0 for delay in range(1, 6)}
    per_delay_total = {delay: 0 for delay in range(1, 6)}
    choices = []
    for episode in fixtures:
        action, _ = _run_episode(
            policy,
            task,
            episode,
            explore_probability=0.0,
            rng=None,
            learn=False,
            reset_before_decision=reset_before_decision,
        )
        matched = int(action == episode.correct_action_index)
        correct += matched
        per_delay_correct[episode.delay_steps] += matched
        per_delay_total[episode.delay_steps] += 1
        choices.append(action)
    return _Evaluation(
        overall=AccuracyCount(correct, len(fixtures)),
        per_delay=tuple(
            (delay, AccuracyCount(per_delay_correct[delay], per_delay_total[delay]))
            for delay in range(1, 6)
        ),
        choices=tuple(choices),
    )


def _train(
    policy: RecurrentPolicy,
    task: DelayedCueTask,
    rng: np.random.Generator,
    config: MemoryExperimentConfig,
) -> tuple[int, AccuracyCount]:
    total_reward = 0
    final_correct = 0
    episode_index = 0
    for _ in range(config.training_episodes // 10):
        for cue, delay in _balanced_cases(rng):
            episode = task.build_episode(cue, delay, rng)
            action, reward = _run_episode(
                policy,
                task,
                episode,
                explore_probability=_epsilon(episode_index, config),
                rng=rng,
                learn=True,
            )
            total_reward += int(reward)
            if episode_index >= config.training_episodes - 10:
                final_correct += int(action == episode.correct_action_index)
            episode_index += 1
    return total_reward, AccuracyCount(final_correct, 10)


def _run_once(seed: int, config: MemoryExperimentConfig) -> _RunResult:
    policy = RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=seed,
        learning_rate=config.learning_rate,
        recurrent_radius=config.recurrent_radius,
    )
    task = DelayedCueTask()
    train_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x54524149]))
    evaluation_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4556414C]))
    # Build once, then reuse these immutable episodes in all three evaluations.
    fixtures = _evaluation_fixtures(task, evaluation_rng, config.evaluation_blocks)
    pre_training = _evaluate(policy, task, fixtures, reset_before_decision=False)
    training_reward, final_block = _train(policy, task, train_rng, config)
    post_training = _evaluate(policy, task, fixtures, reset_before_decision=False)
    state_reset = _evaluate(policy, task, fixtures, reset_before_decision=True)
    return _RunResult(
        pre_training=pre_training.overall,
        post_training=post_training.overall,
        state_reset=state_reset.overall,
        per_delay=post_training.per_delay,
        total_training_reward=training_reward,
        final_block=final_block,
        output_weight_digest=policy.output_weight_digest(),
        recurrent_choice_digest=hashlib.sha256(bytes(post_training.choices)).hexdigest(),
        reset_choice_digest=hashlib.sha256(bytes(state_reset.choices)).hexdigest(),
    )
