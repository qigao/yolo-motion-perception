"""Frozen recurrent-state evaluation protocol for Phase 2B reward learning."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .memory_benchmark import AccuracyCount
from .memory_task import Cue, DelayedCueEpisode, DelayedCueTask


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
