from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RidgeReference:
    weights: np.ndarray
    rank: int
    regularization: float

    def predict(self, hidden: np.ndarray) -> np.ndarray:
        values = np.asarray(hidden, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] + 1 != self.weights.shape[0]:
            raise ValueError("hidden matrix shape does not match ridge weights")
        if not np.all(np.isfinite(values)):
            raise ValueError("hidden matrix must be finite")
        design = np.column_stack((values, np.ones(values.shape[0], dtype=np.float64)))
        return design @ self.weights


@dataclass(frozen=True, slots=True)
class SourceDecision:
    source_step: int
    action: int
    feature: np.ndarray
    denominator: float
    prediction: float


class SourceVisibleDelayedReference:
    def __init__(self, hidden_size: int, *, step_size: float = 0.1) -> None:
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
            raise ValueError("step_size must be numeric")
        resolved_step = float(step_size)
        if not math.isfinite(resolved_step) or not 0.0 < resolved_step <= 1.0:
            raise ValueError("step_size must be finite and in (0, 1]")
        self.hidden_size = hidden_size
        self.step_size = resolved_step
        self._weights = np.zeros((2, hidden_size + 1), dtype=np.float64)
        self._records: dict[int, SourceDecision] = {}

    def record_decision(self, source_step: int, action: int, hidden: np.ndarray) -> float:
        if type(source_step) is not int or source_step < 0:
            raise ValueError("source_step must be a non-negative integer")
        if source_step in self._records:
            raise RuntimeError("source_step has already been recorded")
        if type(action) is not int or action not in (0, 1):
            raise ValueError("action must be 0 or 1")
        vector = np.asarray(hidden, dtype=np.float64)
        if vector.shape != (self.hidden_size,) or not np.all(np.isfinite(vector)):
            raise ValueError("hidden must be a finite vector matching hidden_size")
        feature = np.concatenate((vector, np.array([1.0], dtype=np.float64)))
        denominator = float(np.dot(feature, feature))
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("feature normalization is invalid")
        feature = np.array(feature, dtype=np.float64, copy=True)
        feature.flags.writeable = False
        prediction = float(self._weights[action] @ feature)
        self._records[source_step] = SourceDecision(
            source_step=source_step,
            action=action,
            feature=feature,
            denominator=denominator,
            prediction=prediction,
        )
        return prediction

    def deliver(self, source_steps: tuple[int, ...], rewards: tuple[float, ...]) -> None:
        if len(source_steps) != len(rewards):
            raise ValueError("source_steps and rewards lengths must match")
        if len(set(source_steps)) != len(source_steps):
            raise ValueError("source_steps must not contain duplicates")
        pairs = sorted(zip(source_steps, rewards, strict=True), key=lambda item: item[0])
        for source_step, reward in pairs:
            if type(source_step) is not int or source_step not in self._records:
                raise RuntimeError("delivery must reference a recorded source")
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                raise ValueError("reward must be numeric")
            reward_value = float(reward)
            if not math.isfinite(reward_value):
                raise ValueError("reward must be finite")
            record = self._records.pop(source_step)
            td_error = reward_value - record.prediction
            delta = self.step_size * td_error * record.feature / record.denominator
            candidate = self._weights[record.action] + delta
            if not np.all(np.isfinite(candidate)):
                raise ValueError("source-visible update would overflow")
            self._weights[record.action] = candidate

    @property
    def pending_count(self) -> int:
        return len(self._records)

    def parameter_snapshot(self) -> np.ndarray:
        copied = np.array(self._weights, dtype=np.float64, copy=True)
        copied.flags.writeable = False
        return copied


def fit_supervised_ridge(
    hidden: np.ndarray,
    correct_actions: tuple[int, ...],
    *,
    regularization: float = 1e-6,
) -> RidgeReference:
    values = np.asarray(hidden, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("hidden must be a non-empty finite matrix")
    if len(correct_actions) != values.shape[0]:
        raise ValueError("correct_actions must match the hidden row count")
    if any(type(action) is not int or action not in (0, 1) for action in correct_actions):
        raise ValueError("correct_actions must contain only 0 or 1")
    if isinstance(regularization, bool) or not isinstance(regularization, (int, float)):
        raise ValueError("regularization must be numeric")
    penalty = float(regularization)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("regularization must be finite and positive")

    design = np.column_stack((values, np.ones(values.shape[0], dtype=np.float64)))
    targets = -np.ones((values.shape[0], 2), dtype=np.float64)
    targets[np.arange(values.shape[0]), np.asarray(correct_actions, dtype=np.int64)] = 1.0
    dimension = design.shape[1]
    augmented_design = np.vstack((design, math.sqrt(penalty) * np.eye(dimension)))
    augmented_targets = np.vstack((targets, np.zeros((dimension, 2), dtype=np.float64)))
    weights, _, rank, _ = np.linalg.lstsq(
        augmented_design, augmented_targets, rcond=None
    )
    if not np.all(np.isfinite(weights)):
        raise ValueError("ridge solution must be finite")
    frozen = np.array(weights, dtype=np.float64, copy=True)
    frozen.flags.writeable = False
    return RidgeReference(weights=frozen, rank=int(rank), regularization=penalty)
