from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from neural_state_machine.phase3c_schedule import (
    AnonymousRewardAggregator,
    HiddenDelaySchedule,
    build_hidden_delay_schedule,
)


def _explicit_schedule(delays: tuple[int, ...]) -> HiddenDelaySchedule:
    due_steps = tuple(step + delay for step, delay in enumerate(delays))
    multiplicities = Counter(due_steps)
    histogram = Counter(multiplicities.values())
    inversion_count = sum(
        due_steps[j] < due_steps[i]
        for i in range(len(due_steps))
        for j in range(i + 1, len(due_steps))
    )
    return HiddenDelaySchedule(
        delays=delays,
        due_steps=due_steps,
        delay_digest="test-delay",
        due_step_digest="test-due",
        inversion_count=inversion_count,
        collision_step_count=sum(count >= 2 for count in multiplicities.values()),
        multiplicity_histogram=tuple(sorted(histogram.items())),
    )


@pytest.mark.parametrize("seed", [7, 17, 29])
def test_registered_schedule_is_deterministic_structurally_valid(seed: int) -> None:
    first = build_hidden_delay_schedule(seed, 2_000)
    second = build_hidden_delay_schedule(seed, 2_000)

    assert first == second
    assert set(first.delays) == {1, 3, 5}
    assert first.inversion_count > 0
    assert first.collision_step_count > 0
    assert len(first.delays) == 2_000
    assert len(first.due_steps) == 2_000


def test_schedule_is_independent_of_action_rng_consumption() -> None:
    untouched = build_hidden_delay_schedule(17, 200)
    action_rng = np.random.default_rng(np.random.SeedSequence([17, 0x33414354]))
    action_rng.integers(2, size=10_000)
    after_unrelated_activity = build_hidden_delay_schedule(17, 200)

    assert after_unrelated_activity == untouched


def test_explicit_schedule_has_exact_due_inversion_and_collision_structure() -> None:
    schedule = _explicit_schedule((5, 1, 3, 1))

    assert schedule.due_steps == (5, 2, 5, 4)
    assert schedule.inversion_count == 3
    assert schedule.collision_step_count == 1
    assert schedule.multiplicity_histogram == ((1, 2), (2, 1))


def test_aggregator_delivers_out_of_order_and_collides_anonymously() -> None:
    schedule = _explicit_schedule((3, 1, 1))
    aggregator = AnonymousRewardAggregator(schedule)

    aggregator.enqueue(0, 0, 1.0)
    empty0 = aggregator.feedback_at(0)
    aggregator.enqueue(1, 1, -1.0)
    empty1 = aggregator.feedback_at(1)
    aggregator.enqueue(2, 0, 1.0)
    source1 = aggregator.feedback_at(2)
    collision = aggregator.feedback_at(3)

    assert (empty0.value, empty0.multiplicity, empty0.records) == (0.0, 0, ())
    assert (empty1.value, empty1.multiplicity, empty1.records) == (0.0, 0, ())
    assert source1.value == -1.0
    assert tuple(record.source_step for record in source1.records) == (1,)
    assert collision.value == 2.0
    assert collision.multiplicity == 2
    assert tuple(record.source_step for record in collision.records) == (0, 2)
    assert all(record.delivery_step == 3 for record in collision.records)
    assert aggregator.pending_count == 0


def test_feedback_step_cannot_redeliver() -> None:
    aggregator = AnonymousRewardAggregator(_explicit_schedule((1,)))
    aggregator.enqueue(0, 0, 1.0)
    aggregator.feedback_at(1)

    with pytest.raises(RuntimeError):
        aggregator.feedback_at(1)


def test_invalid_reward_is_atomic() -> None:
    aggregator = AnonymousRewardAggregator(_explicit_schedule((1,)))

    with pytest.raises(ValueError):
        aggregator.enqueue(0, 0, float("nan"))

    assert aggregator.pending_count == 0
    aggregator.enqueue(0, 0, 1.0)
    assert aggregator.pending_count == 1


def test_terminal_drain_empties_all_records_without_synthetic_enqueue() -> None:
    aggregator = AnonymousRewardAggregator(_explicit_schedule((3, 1, 1)))
    for source_step, reward in enumerate((1.0, -1.0, 1.0)):
        aggregator.enqueue(source_step, source_step % 2, reward)
        aggregator.feedback_at(source_step)

    delivered = []
    delivery_step = 3
    while aggregator.pending_count:
        feedback = aggregator.feedback_at(delivery_step)
        delivered.extend(feedback.records)
        delivery_step += 1

    assert tuple(sorted(record.source_step for record in delivered)) == (0, 2)
    assert aggregator.pending_count == 0
