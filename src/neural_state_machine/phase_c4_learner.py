"""Online delay-marginalized anonymous credit learner for Phase C4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_value import (
    ActionValueDecision,
    NormalizedActionValue,
    readonly_float64_copy,
    validated_finite_scalar,
)
from .phase_c4_delay_model import (
    DecisionCreditRow,
    DelayLaw,
    build_marginalized_features,
    current_weight_prediction,
)


@dataclass(frozen=True, slots=True)
class MarginalizedCreditUpdate:
    feedback_step: int
    reward: float
    prediction: float
    td_error: float
    applied: bool


class DelayMarginalizedAnonymousCredit(NormalizedActionValue):
    """Anonymous online learner using only the public finite delay law."""

    def __init__(
        self,
        hidden_size: int,
        action_count: int,
        step_size: float = 0.1,
        *,
        law: DelayLaw | None = None,
    ) -> None:
        super().__init__(hidden_size, action_count, step_size)
        resolved = DelayLaw.registered() if law is None else law
        if not isinstance(resolved, DelayLaw):
            raise ValueError("law must be a DelayLaw")
        if resolved not in (DelayLaw.registered(), DelayLaw.immediate()):
            raise ValueError("law must be the registered or immediate Phase C4 law")
        self.law = resolved
        self._current: DecisionCreditRow | None = None
        self._history: list[DecisionCreditRow] = []
        self._feedback_step = 0
        self._next_decision_index = 0
        self._draining = False

    @property
    def has_pending_feedback(self) -> bool:
        return self._current is not None

    def history_snapshot(self) -> tuple[DecisionCreditRow, ...]:
        return tuple(
            DecisionCreditRow(
                decision_index=row.decision_index,
                action_index=row.action_index,
                feature=row.feature,
                denominator=row.denominator,
            )
            for row in self._history
        )

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
        if self._draining:
            raise RuntimeError("cannot select a decision after terminal drain has begun")
        if self._current is not None:
            raise RuntimeError("feedback is already pending for the current decision")
        if self._next_decision_index != self._feedback_step:
            raise RuntimeError("decision and feedback clocks are not aligned")

        action_index = legal[int(rng.integers(len(legal)))]
        raw_values = self._weights @ feature
        masked = np.full(self.action_count, -np.inf, dtype=np.float64)
        masked[list(legal)] = raw_values[list(legal)]
        denominator = float(np.dot(feature, feature))
        self._current = DecisionCreditRow(
            decision_index=self._next_decision_index,
            action_index=action_index,
            feature=feature,
            denominator=denominator,
        )
        return ActionValueDecision(action_index, readonly_float64_copy(masked))

    def _rows_for_feedback(
        self, current: DecisionCreditRow | None
    ) -> tuple[DecisionCreditRow, ...]:
        if current is None:
            return tuple(self._history)
        return (*self._history, current)

    def _registered_candidate(
        self,
        reward: float,
        current: DecisionCreditRow | None,
    ) -> tuple[np.ndarray, float, float, bool]:
        rows = self._rows_for_feedback(current)
        if not rows:
            candidate = self._weights.copy()
            return candidate, 0.0, reward, False
        marginalized = build_marginalized_features(
            rows,
            feedback_step=self._feedback_step,
            action_count=self.action_count,
            law=self.law,
        )
        prediction = current_weight_prediction(
            self._weights,
            marginalized.expected_feature,
        )
        td_error = reward - prediction
        with np.errstate(over="ignore", invalid="ignore"):
            delta = self.step_size * td_error * marginalized.normalized_credit
            candidate = self._weights + delta
        if not np.all(np.isfinite(candidate)):
            raise ValueError("finite reward would overflow C4 marginalized weights")
        applied = bool(np.any(marginalized.normalized_credit != 0.0))
        return candidate, prediction, td_error, applied

    def _immediate_candidate(
        self,
        reward: float,
        current: DecisionCreditRow,
    ) -> tuple[np.ndarray, float, float]:
        raw_values = self._weights @ current.feature
        prediction = float(raw_values[current.action_index])
        td_error = reward - prediction
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            delta = self.step_size * td_error * current.feature / current.denominator
            candidate_row = self._weights[current.action_index] + delta
        if not np.all(np.isfinite(candidate_row)):
            raise ValueError("finite reward would overflow C4 immediate weights")
        candidate = self._weights.copy()
        candidate[current.action_index] = candidate_row
        return candidate, prediction, td_error

    def _evict_for_next_feedback(self) -> None:
        max_delay = max(self.law.support)
        minimum_index = max(0, self._feedback_step - max_delay)
        self._history = [
            row for row in self._history if row.decision_index >= minimum_index
        ]

    def learn(self, aggregate_reward: object) -> MarginalizedCreditUpdate:
        if self._draining:
            raise RuntimeError("real-decision feedback is not allowed during terminal drain")
        if self._current is None:
            raise RuntimeError("learning requires a current decision")
        reward = validated_finite_scalar(aggregate_reward, "reward")
        current = self._current

        if self.law == DelayLaw.immediate():
            candidate, prediction, td_error = self._immediate_candidate(reward, current)
            applied = True
        else:
            candidate, prediction, td_error, applied = self._registered_candidate(
                reward,
                current,
            )

        self._weights = candidate
        self._history.append(current)
        self._current = None
        self._next_decision_index += 1
        feedback_step = self._feedback_step
        self._feedback_step += 1
        self._evict_for_next_feedback()
        return MarginalizedCreditUpdate(
            feedback_step=feedback_step,
            reward=reward,
            prediction=prediction,
            td_error=td_error,
            applied=applied,
        )

    def learn_drain(self, aggregate_reward: object) -> MarginalizedCreditUpdate:
        if self._current is not None:
            raise RuntimeError("drain feedback requires no current decision")
        reward = validated_finite_scalar(aggregate_reward, "reward")
        candidate, prediction, td_error, applied = self._registered_candidate(
            reward,
            None,
        )

        self._weights = candidate
        self._draining = True
        feedback_step = self._feedback_step
        self._feedback_step += 1
        self._evict_for_next_feedback()
        return MarginalizedCreditUpdate(
            feedback_step=feedback_step,
            reward=reward,
            prediction=prediction,
            td_error=td_error,
            applied=applied,
        )
