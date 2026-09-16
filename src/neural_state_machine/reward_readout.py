from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardReadoutDecision:
    action_index: int
    logits: np.ndarray
    probabilities: np.ndarray


class RewardModulatedReadout:
    def __init__(
        self,
        hidden_size: int,
        action_count: int,
        learning_rate: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if type(action_count) is not int or action_count < 2:
            raise ValueError("action_count must be an integer of at least two")
        if isinstance(learning_rate, bool) or not _is_finite_positive(learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if isinstance(temperature, bool) or not _is_finite_positive(temperature):
            raise ValueError("temperature must be finite and positive")

        self.hidden_size = hidden_size
        self.action_count = action_count
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)

        self._weights = np.zeros(
            (action_count, hidden_size),
            dtype=np.float64,
        )
        self._biases = np.zeros(action_count, dtype=np.float64)
        self._pending_weight_eligibility: np.ndarray | None = None
        self._pending_bias_eligibility: np.ndarray | None = None

    @property
    def has_pending_feedback(self) -> bool:
        return (
            self._pending_weight_eligibility is not None
            and self._pending_bias_eligibility is not None
        )

    def select_greedy(
        self,
        hidden_state: np.ndarray,
        legal_action_indices: object,
    ) -> RewardReadoutDecision:
        hidden = self._validated_hidden_state(hidden_state)
        legal_indices = self._validated_legal_action_indices(legal_action_indices)
        logits, probabilities = self._distribution(hidden, legal_indices)
        return RewardReadoutDecision(
            action_index=int(np.argmax(probabilities)),
            logits=_readonly_copy(logits),
            probabilities=_readonly_copy(probabilities),
        )

    def select_for_training(
        self,
        hidden_state: np.ndarray,
        legal_action_indices: object,
        rng: np.random.Generator,
    ) -> RewardReadoutDecision:
        if self.has_pending_feedback:
            raise RuntimeError("training feedback is already pending")
        if not isinstance(rng, np.random.Generator):
            raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
                "rng must be a numpy.random.Generator"
            )

        hidden = self._validated_hidden_state(hidden_state)
        legal_indices = self._validated_legal_action_indices(legal_action_indices)
        logits, probabilities = self._distribution(hidden, legal_indices)
        legal_array = np.asarray(legal_indices, dtype=np.int64)
        action_index = int(
            rng.choice(
                legal_array,
                p=probabilities[legal_array],
            )
        )
        one_hot = np.zeros(self.action_count, dtype=np.float64)
        one_hot[action_index] = 1.0
        eligibility = one_hot - probabilities
        self._pending_weight_eligibility = np.outer(eligibility, hidden)
        self._pending_bias_eligibility = eligibility.copy()
        return RewardReadoutDecision(
            action_index=action_index,
            logits=_readonly_copy(logits),
            probabilities=_readonly_copy(probabilities),
        )

    def learn(self, reward: float) -> None:
        if not self.has_pending_feedback:
            raise RuntimeError("learning requires pending training feedback")
        try:
            reward_value = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError("reward must be finite") from exc
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite")

        clipped_reward = max(-1.0, min(1.0, reward_value))
        self._weights += (
            self.learning_rate
            * clipped_reward
            * self._pending_weight_eligibility
        )
        self._biases += (
            self.learning_rate
            * clipped_reward
            * self._pending_bias_eligibility
        )
        self._pending_weight_eligibility = None
        self._pending_bias_eligibility = None

    def parameter_digest(self) -> str:
        weights = np.ascontiguousarray(self._weights, dtype=np.float64)
        biases = np.ascontiguousarray(self._biases, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(weights.shape).encode("ascii"))
        digest.update(weights.tobytes(order="C"))
        digest.update(str(biases.shape).encode("ascii"))
        digest.update(biases.tobytes(order="C"))
        return digest.hexdigest()

    def _distribution(
        self,
        hidden_state: np.ndarray,
        legal_action_indices: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_logits = self._weights @ hidden_state + self._biases
        masked_logits = np.full(self.action_count, -np.inf, dtype=np.float64)
        legal_array = np.asarray(legal_action_indices, dtype=np.int64)
        masked_logits[legal_array] = raw_logits[legal_array]

        scaled = raw_logits[legal_array] / self.temperature
        exponentials = np.exp(scaled - np.max(scaled))
        legal_probabilities = exponentials / np.sum(exponentials)
        probabilities = np.zeros(self.action_count, dtype=np.float64)
        probabilities[legal_array] = legal_probabilities
        return masked_logits, probabilities

    def _validated_hidden_state(self, hidden_state: object) -> np.ndarray:
        try:
            values = np.asarray(hidden_state, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("hidden_state must be float64-compatible") from exc
        if values.ndim != 1 or values.shape != (self.hidden_size,):
            raise ValueError(
                f"hidden_state must have shape ({self.hidden_size},)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("hidden_state must contain only finite values")
        return values

    def _validated_legal_action_indices(
        self,
        legal_action_indices: object,
    ) -> tuple[int, ...]:
        try:
            values = tuple(legal_action_indices)
        except TypeError as exc:
            raise ValueError(
                "legal_action_indices must be an iterable of action indices"
            ) from exc
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


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied
