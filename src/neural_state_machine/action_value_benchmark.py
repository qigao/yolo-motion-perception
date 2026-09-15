"""Fixed fixtures and mutation-free evaluation for Phase 3A."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .memory_benchmark import AccuracyCount
from .memory_task import DelayedCueEpisode, DelayedCueTask
from .reward_learning import _build_fixtures, _decision_hidden


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
            raise ValueError(
                "training_episodes must be divisible by checkpoint_interval"
            )
        if not _finite_number(self.recurrent_radius) or not (
            0.0 <= self.recurrent_radius < 1.0
        ):
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


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except TypeError:
        return False


def _build_fixture_bundle(
    seed: int, config: ActionValueBenchmarkConfig
) -> _FixtureBundle:
    task = DelayedCueTask()
    training_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x54524149])
    )
    evaluation_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x4556414C])
    )
    training = _build_fixtures(
        task, training_rng, config.training_episodes // 10
    )
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
        margin = float(
            decision.action_values[correct] - decision.action_values[other]
        )
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
        all_hidden_equal=all(
            np.array_equal(first_hidden, hidden)
            for hidden in hidden_states[1:]
        ),
        margin_mean=float(np.mean(margin_array)),
        margin_p10=float(np.percentile(margin_array, 10)),
        margin_minimum=float(np.min(margin_array)),
    )


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
