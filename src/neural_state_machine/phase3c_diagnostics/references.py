from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RidgeReference:
    weights: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros((len(features), 2), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SourceDecision:
    source_step: int
    action: int
    feature: np.ndarray
    denominator: float
    prediction: float


class SourceVisibleDelayedReference:
    def __init__(self, hidden_size: int, *, step_size: float = 0.1) -> None:
        self.hidden_size = hidden_size
        self.step_size = step_size
        self._weights = np.zeros((2, hidden_size + 1), dtype=np.float64)
        self._records: dict[int, SourceDecision] = {}

    def record_decision(self, source_step: int, action: int, hidden: np.ndarray) -> float:
        return 0.0

    def deliver(self, source_steps: tuple[int, ...], rewards: tuple[float, ...]) -> None:
        return None

    def parameter_snapshot(self) -> np.ndarray:
        return self._weights.copy()


def fit_supervised_ridge(
    hidden: np.ndarray,
    correct_actions: tuple[int, ...],
    *,
    regularization: float = 1e-6,
) -> RidgeReference:
    return RidgeReference(np.zeros((hidden.shape[1] + 1, 2), dtype=np.float64))
