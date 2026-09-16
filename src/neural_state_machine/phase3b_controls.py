"""Fail-closed integrity controls for delayed-credit experiments."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass

from .delayed_credit import RewardDelivery


@dataclass(frozen=True)
class TimelineAudit:
    action_count: int
    delivery_count: int
    terminal_drain_count: int
    max_pending_before_delivery: int
    max_pending_after_delivery: int
    decisions_with_prior_feedback_pending: int
    lag_histogram: tuple[tuple[int, int], ...]
    delivery_timeline_digest: str
    queue_pending_final: int
    learner_unresolved_final: int


def delivery_timeline_digest(deliveries: tuple[RewardDelivery, ...]) -> str:
    if not isinstance(deliveries, tuple):
        raise ValueError("deliveries must be a tuple")
    digest = hashlib.sha256()
    for delivery in deliveries:
        if not isinstance(delivery, RewardDelivery):
            raise ValueError("deliveries must contain RewardDelivery values")
        digest.update(
            f"{delivery.sequence}:{delivery.decision_step}:"
            f"{delivery.due_step}:{delivery.delivery_step}\n".encode("ascii")
        )
    return digest.hexdigest()


def validate_fixed_delay_timeline(audit: TimelineAudit, reward_delay: int) -> None:
    if not isinstance(audit, TimelineAudit):
        raise ValueError("audit must be a TimelineAudit")
    if type(reward_delay) is not int or reward_delay < 0:
        raise ValueError("reward_delay must be a non-negative integer")
    values = (
        audit.action_count,
        audit.delivery_count,
        audit.terminal_drain_count,
        audit.max_pending_before_delivery,
        audit.max_pending_after_delivery,
        audit.decisions_with_prior_feedback_pending,
        audit.queue_pending_final,
        audit.learner_unresolved_final,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("timeline counts must be non-negative integers")
    if audit.action_count <= 0:
        raise ValueError("action_count must be positive")
    if audit.delivery_count != audit.action_count:
        raise ValueError("delivery_count must equal action_count")
    if audit.queue_pending_final != 0:
        raise ValueError("queue must be empty after terminal drain")
    if audit.learner_unresolved_final != 0:
        raise ValueError("learner must have zero unresolved credits after terminal drain")

    count = audit.action_count
    if reward_delay == 0:
        if audit.terminal_drain_count != 0:
            raise ValueError("zero delay must not require terminal drain")
        if audit.max_pending_after_delivery != 0:
            raise ValueError("zero delay must leave no pending reward after delivery")
        if audit.lag_histogram != ((0, count),):
            raise ValueError("zero delay must have an exact zero-lag histogram")
        return

    if reward_delay >= count:
        raise ValueError("reward_delay must be smaller than action_count")
    if audit.terminal_drain_count != reward_delay:
        raise ValueError("terminal drain count must equal reward_delay")
    if audit.max_pending_before_delivery != reward_delay + 1:
        raise ValueError("max pending before delivery must equal reward_delay + 1")
    if audit.max_pending_after_delivery != reward_delay:
        raise ValueError("max pending after delivery must equal reward_delay")
    if audit.decisions_with_prior_feedback_pending <= 0:
        raise ValueError("nonzero delay must overlap later decisions")
    if audit.lag_histogram != ((reward_delay, count),):
        raise ValueError("delivery lag histogram must equal the registered fixed delay")


def action_sequences_equal(normal: tuple[int, ...], shuffled: tuple[int, ...]) -> bool:
    if not isinstance(normal, tuple) or not isinstance(shuffled, tuple):
        raise ValueError("action sequences must be tuples")
    if any(type(action) is not int or action < 0 for action in normal + shuffled):
        raise ValueError("actions must contain non-negative integers")
    return normal == shuffled


def reward_block_multisets_equal(
    normal: tuple[float, ...],
    shuffled: tuple[float, ...],
    *,
    block_size: int = 10,
) -> bool:
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(normal, tuple) or not isinstance(shuffled, tuple):
        raise ValueError("rewards must be tuples")
    if len(normal) != len(shuffled) or len(normal) % block_size:
        return False
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in normal + shuffled
    ):
        raise ValueError("rewards must contain finite scalars")
    return all(
        Counter(normal[start : start + block_size])
        == Counter(shuffled[start : start + block_size])
        for start in range(0, len(normal), block_size)
    )


def validate_queue_counts(
    action_count: int,
    delivered_count: int,
    pending_count: int,
) -> None:
    if any(
        type(value) is not int or value < 0
        for value in (action_count, delivered_count, pending_count)
    ):
        raise ValueError("queue counts must be non-negative integers")
    if delivered_count != action_count:
        raise ValueError("queue delivery count must equal action count")
    if pending_count != 0:
        raise ValueError("queue must be empty after terminal delivery")
