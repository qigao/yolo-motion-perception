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


def _is_finite_positive(value: object) -> bool:
    try:
        return math.isfinite(value) and value > 0.0
    except TypeError:
        return False
