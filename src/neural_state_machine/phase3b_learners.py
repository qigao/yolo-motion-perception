"""Learner adapters used by the Phase 3B delayed-credit benchmark."""

from __future__ import annotations

import numpy as np

from .action_value import ActionValueDecision, ActionValueUpdate, NormalizedActionValue
from .phase3a_credit_compare import EligibilityTraceActionValue


class DelayedTD0Adapter:
    """Thin protocol adapter that preserves the Phase 3A TD(0) learner."""

    def __init__(self, hidden_size: int, action_count: int, step_size: float = 0.1) -> None:
        self._learner = NormalizedActionValue(hidden_size, action_count, step_size)

    @property
    def has_pending_feedback(self) -> bool:
        return self._learner.has_pending_feedback

    def select_for_training(
        self,
        hidden_state: object,
        legal_action_indices: object,
        rng: np.random.Generator,
    ) -> ActionValueDecision:
        return self._learner.select_for_training(hidden_state, legal_action_indices, rng)

    def select_greedy(
        self, hidden_state: object, legal_action_indices: object
    ) -> ActionValueDecision:
        return self._learner.select_greedy(hidden_state, legal_action_indices)

    def learn(self, reward: object) -> ActionValueUpdate:
        return self._learner.learn(reward)

    def parameter_snapshot(self) -> np.ndarray:
        return self._learner.parameter_snapshot()

    def parameter_digest(self) -> str:
        return self._learner.parameter_digest()

    def reset_episode(self) -> None:
        if self.has_pending_feedback:
            raise RuntimeError("cannot reset an episode with pending feedback")


class EpisodeResetEligibilityTrace(EligibilityTraceActionValue):
    """TD(lambda) diagnostic arm with a terminal episode trace reset."""

    def reset_episode(self) -> None:
        if self.has_pending_feedback:
            raise RuntimeError("cannot reset an episode with pending feedback")
        self._eligibility.fill(0.0)
