"""Fail-closed integrity controls for delayed-credit experiments."""

from __future__ import annotations

import math
from collections import Counter


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
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in normal + shuffled):
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
    if any(type(value) is not int or value < 0 for value in (action_count, delivered_count, pending_count)):
        raise ValueError("queue counts must be non-negative integers")
    if delivered_count != action_count:
        raise ValueError("queue delivery count must equal action count")
    if pending_count != 0:
        raise ValueError("queue must be empty after terminal delivery")
