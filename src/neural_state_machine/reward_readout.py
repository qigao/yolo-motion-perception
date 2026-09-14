from __future__ import annotations

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
