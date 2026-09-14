from __future__ import annotations

import math

import numpy as np

from .encoding import encode_observation
from .types import Action, GameObservation, NeuralDecision

_INPUT_SIZE = 9
_RECURRENT_RADIUS = 0.8


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


class RecurrentController:
    def __init__(
        self,
        hidden_size: int = 16,
        seed: int = 0,
        learning_rate: float = 0.1,
    ) -> None:
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")

        self.hidden_size = hidden_size
        self.learning_rate = float(learning_rate)
        rng = np.random.default_rng(seed)
        self._input_weights = rng.normal(
            0.0, 1.0 / math.sqrt(_INPUT_SIZE), size=(hidden_size, _INPUT_SIZE)
        )
        recurrent = rng.normal(
            0.0, 1.0 / math.sqrt(hidden_size), size=(hidden_size, hidden_size)
        )
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        self._recurrent_weights = recurrent * (_RECURRENT_RADIUS / radius)
        self._output_weights = rng.normal(
            0.0, 1.0 / math.sqrt(hidden_size), size=(len(Action), hidden_size)
        )
        self._hidden_state = np.zeros(hidden_size, dtype=np.float64)
        self._last_action_index: int | None = None
        self._last_hidden_state: np.ndarray | None = None

    def step(self, observation: GameObservation) -> NeuralDecision:
        encoded = encode_observation(observation)
        if encoded.shape != (_INPUT_SIZE,):
            raise ValueError(f"encoded observation must have shape ({_INPUT_SIZE},)")

        self._hidden_state = np.tanh(
            self._input_weights @ encoded + self._recurrent_weights @ self._hidden_state
        )
        logits = self._output_weights @ self._hidden_state
        action_index = int(np.argmax(logits))
        self._last_action_index = action_index
        self._last_hidden_state = self._hidden_state.copy()
        return NeuralDecision(
            action=Action(action_index),
            logits=_readonly_copy(logits),
            hidden_state=_readonly_copy(self._hidden_state),
        )

    def learn(self, reward: float) -> None:
        if self._last_action_index is None or self._last_hidden_state is None:
            raise RuntimeError("learning requires a preceding decision")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")

        clipped_reward = max(-1.0, min(1.0, float(reward)))
        self._output_weights[self._last_action_index] += (
            self.learning_rate * clipped_reward * self._last_hidden_state
        )

    def reset_state(self) -> None:
        self._hidden_state.fill(0.0)
        self._last_action_index = None
        self._last_hidden_state = None
