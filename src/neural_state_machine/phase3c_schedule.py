"""Hidden variable-delay scheduling and anonymous reward aggregation for Phase 3C."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HiddenDelaySchedule:
    delays: tuple[int, ...]
    due_steps: tuple[int, ...]
    delay_digest: str
    due_step_digest: str
    inversion_count: int
    collision_step_count: int
    multiplicity_histogram: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LatentRewardRecord:
    source_step: int
    source_action: int
    reward: float
    delay: int
    due_step: int
    delivery_step: int


@dataclass(frozen=True)
class AggregateFeedback:
    delivery_step: int
    value: float
    multiplicity: int
    records: tuple[LatentRewardRecord, ...]


@dataclass(frozen=True)
class _PendingReward:
    source_step: int
    source_action: int
    reward: float
    delay: int
    due_step: int


def _integer_sequence_digest(values: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(values).to_bytes(8, "big", signed=False))
    for value in values:
        digest.update(int(value).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _validate_support(support: object) -> tuple[int, ...]:
    if not isinstance(support, tuple) or not support:
        raise ValueError("support must be a non-empty tuple")
    if any(type(delay) is not int or delay < 0 for delay in support):
        raise ValueError("support must contain non-negative integers")
    if len(set(support)) != len(support):
        raise ValueError("support must not contain duplicates")
    return support


def _inversion_count(due_steps: tuple[int, ...]) -> int:
    return sum(
        due_steps[j] < due_steps[i]
        for i in range(len(due_steps))
        for j in range(i + 1, len(due_steps))
    )


def build_hidden_delay_schedule(
    seed: int,
    decision_count: int,
    *,
    support: tuple[int, ...] = (1, 3, 5),
) -> HiddenDelaySchedule:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(decision_count) is not int or decision_count <= 0:
        raise ValueError("decision_count must be a positive integer")
    resolved_support = _validate_support(support)

    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x3343444C]))
    indices = rng.integers(len(resolved_support), size=decision_count)
    delays = tuple(resolved_support[int(index)] for index in indices)
    due_steps = tuple(step + delay for step, delay in enumerate(delays))
    multiplicities = Counter(due_steps)
    multiplicity_histogram = Counter(multiplicities.values())

    return HiddenDelaySchedule(
        delays=delays,
        due_steps=due_steps,
        delay_digest=_integer_sequence_digest(delays),
        due_step_digest=_integer_sequence_digest(due_steps),
        inversion_count=_inversion_count(due_steps),
        collision_step_count=sum(count >= 2 for count in multiplicities.values()),
        multiplicity_histogram=tuple(sorted(multiplicity_histogram.items())),
    )


def _finite_reward(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("reward must be a finite scalar")
    try:
        reward = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("reward must be a finite scalar") from exc
    if not math.isfinite(reward):
        raise ValueError("reward must be a finite scalar")
    return reward


class AnonymousRewardAggregator:
    def __init__(self, schedule: HiddenDelaySchedule) -> None:
        if not isinstance(schedule, HiddenDelaySchedule):
            raise ValueError("schedule must be a HiddenDelaySchedule")
        if len(schedule.delays) != len(schedule.due_steps):
            raise ValueError("schedule delay and due-step lengths must match")
        if any(
            due_step != source_step + delay
            for source_step, (delay, due_step) in enumerate(
                zip(schedule.delays, schedule.due_steps, strict=True)
            )
        ):
            raise ValueError("schedule due steps must equal source step plus delay")
        self._schedule = schedule
        self._pending: dict[int, list[_PendingReward]] = defaultdict(list)
        self._enqueued_sources: set[int] = set()
        self._delivered_steps: set[int] = set()

    @property
    def pending_count(self) -> int:
        return sum(len(bucket) for bucket in self._pending.values())

    def enqueue(self, source_step: int, source_action: int, reward: object) -> None:
        if type(source_step) is not int or not 0 <= source_step < len(self._schedule.delays):
            raise ValueError("source_step must index the registered schedule")
        if type(source_action) is not int or source_action < 0:
            raise ValueError("source_action must be a non-negative integer")
        if source_step in self._enqueued_sources:
            raise RuntimeError("source_step has already been enqueued")
        reward_value = _finite_reward(reward)
        delay = self._schedule.delays[source_step]
        due_step = self._schedule.due_steps[source_step]
        pending = _PendingReward(
            source_step=source_step,
            source_action=source_action,
            reward=reward_value,
            delay=delay,
            due_step=due_step,
        )
        self._pending[due_step].append(pending)
        self._enqueued_sources.add(source_step)

    def feedback_at(self, delivery_step: int) -> AggregateFeedback:
        if type(delivery_step) is not int or delivery_step < 0:
            raise ValueError("delivery_step must be a non-negative integer")
        if delivery_step in self._delivered_steps:
            raise RuntimeError("delivery_step has already been consumed")

        pending = tuple(self._pending.get(delivery_step, ()))
        value = sum(record.reward for record in pending)
        if not math.isfinite(value):
            raise ValueError("aggregate feedback must remain finite")

        records = tuple(
            LatentRewardRecord(
                source_step=record.source_step,
                source_action=record.source_action,
                reward=record.reward,
                delay=record.delay,
                due_step=record.due_step,
                delivery_step=delivery_step,
            )
            for record in pending
        )
        self._pending.pop(delivery_step, None)
        self._delivered_steps.add(delivery_step)
        return AggregateFeedback(
            delivery_step=delivery_step,
            value=float(value),
            multiplicity=len(records),
            records=records,
        )
