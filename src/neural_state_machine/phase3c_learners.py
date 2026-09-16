"""Phase 3C learner arms for anonymous aggregate temporal feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .action_value import (
    ActionValueDecision,
    NormalizedActionValue,
    readonly_float64_copy,
    validated_finite_scalar,
)


@dataclass(frozen=True)
class AnonymousCreditUpdate:
    reward: float
    prediction: float
    td_error: float
    applied: bool


@dataclass(frozen=True)
class _CurrentCredit:
    action_index: int
    feature: np.ndarray
    denominator: float
    prediction: float


def _validated_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and in [0.0, 1.0]")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0]") from exc
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0]")
    return resolved


class AnonymousCurrentStepTD0(NormalizedActionValue):
    """Anonymous-feedback baseline that credits only the current real decision."""

    def __init__(self, hidden_size: int, action_count: int, step_size: float = 0.1) -> None:
        super().__init__(hidden_size, action_count, step_size)
        self._current: _CurrentCredit | None = None

    @property
    def has_pending_feedback(self) -> bool:
        return self._current is not None

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
        if self._current is not None:
            raise RuntimeError("feedback is already pending")
        action_index = legal[int(rng.integers(len(legal)))]
        raw_values = self._weights @ feature
        masked = np.full(self.action_count, -np.inf, dtype=np.float64)
        masked[list(legal)] = raw_values[list(legal)]
        denominator = float(np.dot(feature, feature))
        self._current = _CurrentCredit(
            action_index=action_index,
            feature=readonly_float64_copy(feature),
            denominator=denominator,
            prediction=float(raw_values[action_index]),
        )
        return ActionValueDecision(action_index, readonly_float64_copy(masked))

    def learn(self, aggregate_reward: object) -> AnonymousCreditUpdate:
        if self._current is None:
            raise RuntimeError("learning requires pending feedback")
        reward = validated_finite_scalar(aggregate_reward, "reward")
        current = self._current
        td_error = reward - current.prediction
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            delta = self.step_size * td_error * current.feature / current.denominator
            candidate = self._weights[current.action_index] + delta
        if not np.all(np.isfinite(candidate)):
            raise ValueError("finite reward would overflow action-value weights")
        self._weights[current.action_index] = candidate
        self._current = None
        return AnonymousCreditUpdate(
            reward=reward,
            prediction=current.prediction,
            td_error=td_error,
            applied=True,
        )

    def learn_drain(self, aggregate_reward: object) -> AnonymousCreditUpdate:
        if self._current is not None:
            raise RuntimeError("drain feedback requires no current decision")
        reward = validated_finite_scalar(aggregate_reward, "reward")
        return AnonymousCreditUpdate(
            reward=reward,
            prediction=0.0,
            td_error=0.0,
            applied=False,
        )


class NormalizedAnonymousEligibilityCredit(NormalizedActionValue):
    """Persistent normalized eligibility credit for anonymous aggregate feedback."""

    def __init__(
        self,
        hidden_size: int,
        action_count: int,
        step_size: float = 0.1,
        *,
        discount: float = 0.9,
        trace_decay: float = 0.8,
    ) -> None:
        super().__init__(hidden_size, action_count, step_size)
        self.discount = _validated_unit_interval(discount, "discount")
        self.trace_decay = _validated_unit_interval(trace_decay, "trace_decay")
        self._eligibility = np.zeros_like(self._weights)
        self._prediction_trace = 0.0
        self._current: _CurrentCredit | None = None
        self._trace_reset_count = 0
        self._run_ended = False

    @property
    def has_pending_feedback(self) -> bool:
        return self._current is not None

    @property
    def prediction_trace(self) -> float:
        return self._prediction_trace

    @property
    def trace_reset_count(self) -> int:
        return self._trace_reset_count

    @property
    def rho(self) -> float:
        return self.discount * self.trace_decay

    def eligibility_snapshot(self) -> np.ndarray:
        return readonly_float64_copy(self._eligibility)

    def select_for_training(
        self,
        hidden_state: object,
        legal_action_indices: object,
        rng: np.random.Generator,
    ) -> ActionValueDecision:
        if self._run_ended:
            raise RuntimeError("training run has ended")
        feature = self._feature(hidden_state)
        legal = self._legal_actions(legal_action_indices)
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        if self._current is not None:
            raise RuntimeError("feedback is already pending")
        action_index = legal[int(rng.integers(len(legal)))]
        raw_values = self._weights @ feature
        masked = np.full(self.action_count, -np.inf, dtype=np.float64)
        masked[list(legal)] = raw_values[list(legal)]
        denominator = float(np.dot(feature, feature))
        current = _CurrentCredit(
            action_index=action_index,
            feature=readonly_float64_copy(feature),
            denominator=denominator,
            prediction=float(raw_values[action_index]),
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            candidate_eligibility = self._eligibility * self.rho
            candidate_eligibility[current.action_index] += current.feature / current.denominator
            candidate_prediction = self.rho * self._prediction_trace + current.prediction
        if not np.all(np.isfinite(candidate_eligibility)) or not math.isfinite(candidate_prediction):
            raise ValueError("anonymous eligibility trace update would overflow")
        self._eligibility = candidate_eligibility
        self._prediction_trace = float(candidate_prediction)
        self._current = current
        return ActionValueDecision(action_index, readonly_float64_copy(masked))

    def _candidate_weights(
        self, reward: float, current: _CurrentCredit | None
    ) -> tuple[np.ndarray, float]:
        td_error = reward - self._prediction_trace
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self.rho == 0.0 and current is not None:
                candidate = self._weights.copy()
                delta = self.step_size * td_error * current.feature / current.denominator
                candidate[current.action_index] = self._weights[current.action_index] + delta
            else:
                candidate = self._weights + self.step_size * td_error * self._eligibility
        if not np.all(np.isfinite(candidate)):
            raise ValueError("finite reward would overflow anonymous eligibility weights")
        return candidate, td_error

    def learn(self, aggregate_reward: object) -> AnonymousCreditUpdate:
        if self._run_ended:
            raise RuntimeError("training run has ended")
        if self._current is None:
            raise RuntimeError("learning requires pending feedback")
        reward = validated_finite_scalar(aggregate_reward, "reward")
        current = self._current
        prediction = self._prediction_trace
        candidate, td_error = self._candidate_weights(reward, current)
        self._weights = candidate
        self._current = None
        return AnonymousCreditUpdate(
            reward=reward,
            prediction=prediction,
            td_error=td_error,
            applied=True,
        )

    def learn_drain(self, aggregate_reward: object) -> AnonymousCreditUpdate:
        if self._run_ended:
            raise RuntimeError("training run has ended")
        if self._current is not None:
            raise RuntimeError("drain feedback requires no current decision")
        reward = validated_finite_scalar(aggregate_reward, "reward")
        prediction = self._prediction_trace
        candidate, td_error = self._candidate_weights(reward, None)
        self._weights = candidate
        return AnonymousCreditUpdate(
            reward=reward,
            prediction=prediction,
            td_error=td_error,
            applied=True,
        )

    def end_run(self) -> None:
        if self._run_ended:
            raise RuntimeError("training run has already ended")
        if self._current is not None:
            raise RuntimeError("cannot end run with pending feedback")
        self._eligibility.fill(0.0)
        self._prediction_trace = 0.0
        self._trace_reset_count += 1
        self._run_ended = True
