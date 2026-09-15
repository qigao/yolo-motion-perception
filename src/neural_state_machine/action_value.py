from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionValueDecision:
    action_index: int
    action_values: np.ndarray


@dataclass(frozen=True)
class _PendingCredit:
    action_index: int
    feature: np.ndarray
    denominator: float
    prediction: float


class NormalizedActionValue:
    def __init__(self, hidden_size: int, action_count: int, step_size: float = 0.1) -> None:
        self.hidden_size = validated_positive_integer(hidden_size, "hidden_size")
        self.action_count = validated_action_count(action_count)
        self.step_size = validated_step_size(step_size)
        self._weights = np.zeros(
            (self.action_count, self.hidden_size + 1), dtype=np.float64
        )
        self._pending: _PendingCredit | None = None

    @property
    def has_pending_feedback(self) -> bool:
        return self._pending is not None

    def parameter_snapshot(self) -> np.ndarray:
        return readonly_float64_copy(self._weights)

    def parameter_digest(self) -> str:
        values = np.ascontiguousarray(self._weights, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    def _feature(self, hidden_state: object) -> np.ndarray:
        try:
            hidden = np.asarray(hidden_state, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("hidden_state must be float64-compatible") from exc
        if hidden.ndim != 1 or hidden.shape != (self.hidden_size,):
            raise ValueError(f"hidden_state must have shape ({self.hidden_size},)")
        if not np.all(np.isfinite(hidden)):
            raise ValueError("hidden_state must contain only finite values")
        feature = np.concatenate((hidden, np.array([1.0], dtype=np.float64)))
        feature.flags.writeable = False
        return feature

    def _legal_actions(self, legal_action_indices: object) -> tuple[int, ...]:
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

    def select_greedy(
        self, hidden_state: object, legal_action_indices: object
    ) -> ActionValueDecision:
        feature = self._feature(hidden_state)
        legal = self._legal_actions(legal_action_indices)
        values = self._weights @ feature
        masked = np.full(self.action_count, -np.inf, dtype=np.float64)
        masked[list(legal)] = values[list(legal)]
        maximum = max(masked[index] for index in legal)
        selected = min(index for index in legal if masked[index] == maximum)
        return ActionValueDecision(selected, readonly_float64_copy(masked))

    def select_for_training(
        self,
        hidden_state: object,
        legal_action_indices: object,
        rng: np.random.Generator,
    ) -> ActionValueDecision:
        feature = self._feature(hidden_state)
        legal = self._legal_actions(legal_action_indices)
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        if self._pending is not None:
            raise RuntimeError("feedback is already pending")
        action_index = legal[int(rng.integers(len(legal)))]
        raw_values = self._weights @ feature
        masked = np.full(self.action_count, -np.inf, dtype=np.float64)
        masked[list(legal)] = raw_values[list(legal)]
        denominator = float(np.dot(feature, feature))
        self._pending = _PendingCredit(
            action_index=action_index,
            feature=readonly_float64_copy(feature),
            denominator=denominator,
            prediction=float(raw_values[action_index]),
        )
        return ActionValueDecision(action_index, readonly_float64_copy(masked))


def validated_positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validated_action_count(value: object) -> int:
    if type(value) is not int or value < 2:
        raise ValueError("action_count must be an integer of at least two")
    return value


def validated_step_size(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("step_size must be finite and in (0.0, 1.0]")
    try:
        step_size = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("step_size must be finite and in (0.0, 1.0]") from exc
    if not math.isfinite(step_size) or not 0.0 < step_size <= 1.0:
        raise ValueError("step_size must be finite and in (0.0, 1.0]")
    return step_size


def readonly_float64_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied
