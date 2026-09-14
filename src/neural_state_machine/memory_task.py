"""Immutable fixtures and scoring for the delayed-cue experiment.

The task fixture contains cue identity and scoring metadata for experiment
bookkeeping.  Stimuli are the only values that a controller needs to receive.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Cue(IntEnum):
    LEFT = 0
    RIGHT = 1


_INPUT_SIZE = 4
_MIN_DELAY_STEPS = 1
_MAX_DELAY_STEPS = 5
_DECISION_STIMULUS = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_DECISION_STIMULUS.flags.writeable = False


@dataclass(frozen=True)
class DelayedCueEpisode:
    cue: Cue
    delay_steps: int
    cue_stimulus: np.ndarray
    delay_stimuli: tuple[np.ndarray, ...]
    decision_stimulus: np.ndarray
    correct_action_index: int

    def __post_init__(self) -> None:
        if type(self.cue) is not Cue:
            raise ValueError("cue must be a Cue member")
        _validate_delay_steps(self.delay_steps)
        if type(self.correct_action_index) is not int:
            raise ValueError("correct_action_index must be an integer")
        if self.correct_action_index != self.cue.value:
            raise ValueError("correct_action_index must match cue")

        cue_stimulus = _validated_vector(self.cue_stimulus, "cue_stimulus")
        decision_stimulus = _validated_vector(
            self.decision_stimulus, "decision_stimulus"
        )
        expected_cue = np.zeros(_INPUT_SIZE, dtype=np.float64)
        expected_cue[self.cue.value] = 1.0
        if not np.array_equal(cue_stimulus, expected_cue):
            raise ValueError("cue_stimulus must encode cue identity exactly")
        if not np.array_equal(decision_stimulus, _DECISION_STIMULUS):
            raise ValueError("decision_stimulus must be the shared decision literal")
        try:
            delay_stimuli = tuple(self.delay_stimuli)
        except TypeError as exc:
            raise ValueError("delay_stimuli must be an iterable of vectors") from exc
        if len(delay_stimuli) != self.delay_steps:
            raise ValueError("delay_stimuli must contain exactly delay_steps vectors")
        delay_copies = []
        for stimulus in delay_stimuli:
            validated = _validated_vector(stimulus, "delay_stimulus")
            if not np.all(validated[[0, 1, 3]] == 0.0) or not -0.25 <= validated[2] <= 0.25:
                raise ValueError(
                    "delay_stimulus must have only a bounded index-2 distractor"
                )
            delay_copies.append(_readonly_copy(validated))

        object.__setattr__(self, "cue_stimulus", _readonly_copy(cue_stimulus))
        object.__setattr__(self, "delay_stimuli", tuple(delay_copies))
        object.__setattr__(self, "decision_stimulus", _readonly_copy(decision_stimulus))


class DelayedCueTask:
    """Generate deterministic delayed-cue fixtures and score legal actions."""

    def build_episode(
        self, cue: Cue, delay_steps: int, rng: np.random.Generator
    ) -> DelayedCueEpisode:
        if type(cue) is not Cue:
            raise ValueError("cue must be a Cue member")
        _validate_delay_steps(delay_steps)
        if not isinstance(rng, np.random.Generator):
            raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
                "rng must be a numpy.random.Generator"
            )

        cue_stimulus = np.zeros(_INPUT_SIZE, dtype=np.float64)
        cue_stimulus[cue.value] = 1.0
        delay_stimuli = tuple(
            _delay_stimulus(float(rng.uniform(-0.25, 0.25)))
            for _ in range(delay_steps)
        )
        return DelayedCueEpisode(
            cue=cue,
            delay_steps=delay_steps,
            cue_stimulus=_readonly_copy(cue_stimulus),
            delay_stimuli=delay_stimuli,
            decision_stimulus=_fresh_decision_stimulus(),
            correct_action_index=cue.value,
        )

    def reward(self, episode: DelayedCueEpisode, action_index: int) -> float:
        if not isinstance(episode, DelayedCueEpisode):
            raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
                "episode must be a DelayedCueEpisode"
            )
        if type(action_index) is not int or action_index not in (0, 1):
            raise ValueError("action_index must be a legal action index")
        return 1.0 if action_index == episode.correct_action_index else -1.0


def _delay_stimulus(distractor: float) -> np.ndarray:
    values = np.array([0.0, 0.0, distractor, 0.0], dtype=np.float64)
    return _readonly_copy(values)


def _fresh_decision_stimulus() -> np.ndarray:
    return _readonly_copy(_DECISION_STIMULUS)


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


def _validated_vector(values: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if array.ndim != 1 or array.shape != (_INPUT_SIZE,):
        raise ValueError(f"{name} must have shape ({_INPUT_SIZE},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_delay_steps(delay_steps: object) -> None:
    if (
        type(delay_steps) is not int
        or not _MIN_DELAY_STEPS <= delay_steps <= _MAX_DELAY_STEPS
    ):
        raise ValueError("delay_steps must be an integer in [1, 5]")
