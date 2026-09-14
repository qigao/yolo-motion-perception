from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyDecision:
    action_index: int
    logits: np.ndarray
    hidden_state: np.ndarray


class RecurrentPolicy:
    def __init__(
        self,
        input_size: int,
        action_count: int,
        hidden_size: int = 16,
        seed: int = 0,
        learning_rate: float = 0.1,
        recurrent_radius: float = 0.8,
    ) -> None:
        if type(input_size) is not int or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if type(action_count) is not int or action_count < 2:
            raise ValueError("action_count must be an integer of at least two")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(learning_rate, bool) or not _is_finite_positive(learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if isinstance(recurrent_radius, bool) or not _is_valid_radius(recurrent_radius):
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")

        self.input_size = input_size
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.learning_rate = float(learning_rate)
        self.recurrent_radius = float(recurrent_radius)

        rng = np.random.default_rng(seed)
        self._input_weights = rng.normal(
            0.0, 1.0 / math.sqrt(input_size), size=(hidden_size, input_size)
        )
        recurrent = rng.normal(
            0.0, 1.0 / math.sqrt(hidden_size), size=(hidden_size, hidden_size)
        )
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        self._recurrent_weights = recurrent * (self.recurrent_radius / radius)
        self._output_weights = rng.normal(
            0.0, 1.0 / math.sqrt(hidden_size), size=(action_count, hidden_size)
        )
        if self.recurrent_radius == 0.0:
            self._recurrent_weights.fill(0.0)
        self._hidden_state = np.zeros(hidden_size, dtype=np.float64)
        self._last_action_index: int | None = None
        self._last_hidden_state: np.ndarray | None = None

    def _validated_stimulus(self, stimulus: np.ndarray) -> np.ndarray:
        try:
            values = np.asarray(stimulus, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("stimulus must be float64-compatible") from exc
        if values.ndim != 1 or values.shape != (self.input_size,):
            raise ValueError(f"stimulus must have shape ({self.input_size},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("stimulus must contain only finite values")
        return values

    def _evolve(self, stimulus: np.ndarray) -> None:
        values = self._validated_stimulus(stimulus)
        self._hidden_state = np.tanh(
            self._input_weights @ values + self._recurrent_weights @ self._hidden_state
        )

    def advance(self, stimulus: np.ndarray) -> np.ndarray:
        self._evolve(stimulus)
        return _readonly_copy(self._hidden_state)

    def decide(
        self,
        stimulus: np.ndarray,
        legal_action_indices: object,
        *,
        explore_probability: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> PolicyDecision:
        legal_indices = self._validated_legal_action_indices(legal_action_indices)
        probability = _validated_explore_probability(explore_probability)
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        if probability > 0.0 and rng is None:
            raise ValueError("rng is required when explore_probability is positive")
        self._evolve(stimulus)
        logits = self._output_weights @ self._hidden_state
        masked_logits = logits.copy()
        illegal = np.ones(self.action_count, dtype=bool)
        illegal[list(legal_indices)] = False
        masked_logits[illegal] = -np.inf
        if probability > 0.0 and rng.random() < probability:
            action_index = int(rng.choice(np.asarray(legal_indices, dtype=np.int64)))
        else:
            action_index = int(np.argmax(masked_logits))
        self._last_action_index = action_index
        self._last_hidden_state = self._hidden_state.copy()
        return PolicyDecision(
            action_index=action_index,
            logits=_readonly_copy(masked_logits),
            hidden_state=_readonly_copy(self._hidden_state),
        )

    def learn(self, reward: float) -> None:
        if self._last_action_index is None or self._last_hidden_state is None:
            raise RuntimeError("learning requires a preceding decision")
        try:
            reward_value = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError("reward must be finite") from exc
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite")

        clipped = max(-1.0, min(1.0, reward_value))
        self._output_weights[self._last_action_index] += (
            self.learning_rate * clipped * self._last_hidden_state
        )
        self._last_action_index = None
        self._last_hidden_state = None

    def reset_state(self) -> None:
        self._hidden_state.fill(0.0)
        self._last_action_index = None
        self._last_hidden_state = None

    def output_weight_digest(self) -> str:
        values = np.ascontiguousarray(self._output_weights, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    def _validated_legal_action_indices(self, legal_action_indices: object) -> tuple[int, ...]:
        try:
            values = tuple(legal_action_indices)
        except TypeError as exc:
            raise ValueError("legal_action_indices must be an iterable of action indices") from exc
        if not values:
            raise ValueError("legal_action_indices must not be empty")
        if any(type(index) is not int for index in values):
            raise ValueError("legal_action_indices must contain only integers")
        if len(set(values)) != len(values):
            raise ValueError("legal_action_indices must not contain duplicates")
        if any(index < 0 or index >= self.action_count for index in values):
            raise ValueError("legal_action_indices must be in range")
        return values


def _is_finite_positive(value: object) -> bool:
    try:
        return math.isfinite(value) and value > 0.0
    except TypeError:
        return False


def _is_valid_radius(value: object) -> bool:
    try:
        return math.isfinite(value) and 0.0 <= value < 1.0
    except TypeError:
        return False


def _validated_explore_probability(value: object) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("explore_probability must be finite and in [0.0, 1.0]") from exc
    if isinstance(value, bool) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("explore_probability must be finite and in [0.0, 1.0]")
    return probability


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied
