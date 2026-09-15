"""Deterministic action-to-reward delivery for delayed-credit experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingReward:
    action_index: int
    reward: float
    due_step: int


@dataclass(frozen=True)
class RewardDelivery:
    action_index: int
    reward: float
    due_step: int


class DelayedRewardQueue:
    """A strict FIFO queue for exactly-once scalar reward delivery."""

    def __init__(self, max_delay: int) -> None:
        if type(max_delay) is not int or max_delay < 0:
            raise ValueError("max_delay must be a non-negative integer")
        self.max_delay = max_delay
        self._current_step = 0
        self._pending: list[PendingReward] = []

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def enqueue(self, action_index: object, reward: object, delay: object) -> PendingReward:
        if type(action_index) is not int or action_index < 0:
            raise ValueError("action_index must be a non-negative integer")
        if type(delay) is not int or delay < 0 or delay > self.max_delay:
            raise ValueError("delay must be a non-negative integer within max_delay")
        if isinstance(reward, bool):
            raise ValueError("reward must be a finite scalar")
        try:
            reward_value = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError("reward must be a finite scalar") from exc
        if not math.isfinite(reward_value):
            raise ValueError("reward must be a finite scalar")
        pending = PendingReward(
            action_index=action_index,
            reward=reward_value,
            due_step=self._current_step + delay,
        )
        self._pending.append(pending)
        return pending

    def advance(self) -> int:
        if any(item.due_step <= self._current_step for item in self._pending):
            raise RuntimeError("ready rewards must be delivered before advancing")
        self._current_step += 1
        return self._current_step

    def deliver_ready(self) -> tuple[RewardDelivery, ...]:
        stale = [item for item in self._pending if item.due_step < self._current_step]
        if stale:
            raise RuntimeError("a reward delivery is stale")
        ready = [item for item in self._pending if item.due_step == self._current_step]
        if not ready:
            return ()
        self._pending = [item for item in self._pending if item.due_step > self._current_step]
        return tuple(
            RewardDelivery(
                action_index=item.action_index,
                reward=item.reward,
                due_step=item.due_step,
            )
            for item in ready
        )

    def reset(self) -> None:
        self._current_step = 0
        self._pending.clear()
