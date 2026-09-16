"""Learner adapters used by the Phase 3B delayed-credit benchmark."""

from __future__ import annotations

from collections import deque

import numpy as np

from .action_value import (
    ActionValueDecision,
    ActionValueUpdate,
    NormalizedActionValue,
    _PendingCredit,
    readonly_float64_copy,
    validated_finite_scalar,
)
from .phase3a_credit_compare import EligibilityTraceActionValue


class DelayedTD0Adapter(NormalizedActionValue):
    """Phase 3B TD(0) adapter with FIFO multi-inflight credit."""

    def __init__(self, hidden_size: int, action_count: int, step_size: float = 0.1) -> None:
        super().__init__(hidden_size, action_count, step_size)
        self._credits: deque[_PendingCredit] = deque()

    @property
    def has_pending_feedback(self) -> bool:
        return bool(self._credits)

    @property
    def unresolved_credit_count(self) -> int:
        return len(self._credits)

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
        action_index = legal[int(rng.integers(len(legal)))]
        raw_values = self._weights @ feature
        masked = np.full(self.action_count, -np.inf, dtype=np.float64)
        masked[list(legal)] = raw_values[list(legal)]
        denominator = float(np.dot(feature, feature))
        self._credits.append(
            _PendingCredit(
                action_index=action_index,
                feature=readonly_float64_copy(feature),
                denominator=denominator,
                prediction=float(raw_values[action_index]),
            )
        )
        return ActionValueDecision(action_index, readonly_float64_copy(masked))

    def learn(self, reward: object) -> ActionValueUpdate:
        if not self._credits:
            raise RuntimeError("learning requires pending feedback")
        reward_value = validated_finite_scalar(reward, "reward")
        pending = self._credits[0]
        td_error = reward_value - pending.prediction
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            delta = self.step_size * td_error * pending.feature / pending.denominator
            candidate = self._weights[pending.action_index] + delta
        if not np.all(np.isfinite(candidate)):
            raise ValueError("finite reward would overflow action-value weights")
        self._weights[pending.action_index] = candidate
        self._credits.popleft()
        return ActionValueUpdate(
            action_index=pending.action_index,
            prediction_before=pending.prediction,
            reward=reward_value,
            td_error=td_error,
        )

    def reset_episode(self) -> None:
        if self.has_pending_feedback:
            raise RuntimeError("cannot reset an episode with pending feedback")


class EpisodeResetEligibilityTrace(EligibilityTraceActionValue):
    """Historical diagnostic TD(lambda); not an approved overlapping Phase 3B arm."""

    def reset_episode(self) -> None:
        if self.has_pending_feedback:
            raise RuntimeError("cannot reset an episode with pending feedback")
        self._eligibility.fill(0.0)
