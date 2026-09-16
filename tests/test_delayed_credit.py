import numpy as np
import pytest

from neural_state_machine.delayed_credit import DelayedRewardQueue, RewardDelivery


def test_delivery_records_exact_decision_due_and_delivery_steps() -> None:
    queue = DelayedRewardQueue(max_delay=5)
    pending = queue.enqueue(action_index=1, reward=1.0, delay=3)
    assert pending.sequence == 0
    assert pending.decision_step == 0
    assert pending.due_step == 3

    for expected in (1, 2, 3):
        assert queue.advance() == expected
        if expected < 3:
            assert queue.deliver_ready() == ()

    assert queue.deliver_ready() == (
        RewardDelivery(
            sequence=0,
            action_index=1,
            reward=1.0,
            decision_step=0,
            due_step=3,
            delivery_step=3,
        ),
    )


def test_sequence_and_decision_step_follow_enqueue_order() -> None:
    queue = DelayedRewardQueue(max_delay=3)
    first = queue.enqueue(0, 1.0, 2)
    assert first.sequence == 0
    assert first.decision_step == 0
    assert queue.advance() == 1
    second = queue.enqueue(1, -1.0, 2)
    assert second.sequence == 1
    assert second.decision_step == 1

    assert queue.advance() == 2
    assert queue.deliver_ready() == (
        RewardDelivery(0, 0, 1.0, 0, 2, 2),
    )
    assert queue.advance() == 3
    assert queue.deliver_ready() == (
        RewardDelivery(1, 1, -1.0, 1, 3, 3),
    )


def test_reward_is_delivered_after_exact_registered_delay() -> None:
    queue = DelayedRewardQueue(max_delay=5)
    queue.enqueue(action_index=1, reward=1.0, delay=3)
    assert queue.deliver_ready() == ()
    assert queue.advance() == 1
    assert queue.deliver_ready() == ()
    assert queue.advance() == 2
    assert queue.deliver_ready() == ()
    assert queue.advance() == 3
    delivery = queue.deliver_ready()
    assert len(delivery) == 1
    assert delivery[0].action_index == 1
    assert delivery[0].reward == 1.0
    assert delivery[0].due_step == 3
    assert delivery[0].delivery_step == 3
    assert queue.pending_count == 0
    assert queue.deliver_ready() == ()


def test_zero_delay_is_ready_immediately_and_fifo_is_preserved() -> None:
    queue = DelayedRewardQueue(max_delay=2)
    first = queue.enqueue(2, 2.0, 0)
    second = queue.enqueue(1, -1.0, 0)
    assert queue.deliver_ready() == (
        RewardDelivery(first.sequence, 2, 2.0, 0, 0, 0),
        RewardDelivery(second.sequence, 1, -1.0, 0, 0, 0),
    )


@pytest.mark.parametrize(
    "delay",
    [-1, 3, True, 1.0],
)
def test_invalid_delays_are_rejected_atomically(delay: object) -> None:
    queue = DelayedRewardQueue(max_delay=2)
    with pytest.raises(ValueError, match="delay"):
        queue.enqueue(0, 1.0, delay)
    assert queue.pending_count == 0
    assert queue.current_step == 0


@pytest.mark.parametrize("reward", [None, True, float("nan"), float("inf")])
def test_invalid_rewards_are_rejected_atomically(reward: object) -> None:
    queue = DelayedRewardQueue(max_delay=2)
    with pytest.raises(ValueError, match="reward"):
        queue.enqueue(0, reward, 1)
    assert queue.pending_count == 0


def test_unconsumed_ready_reward_cannot_become_stale() -> None:
    queue = DelayedRewardQueue(max_delay=2)
    queue.enqueue(0, 1.0, 1)
    queue.advance()
    with pytest.raises(RuntimeError, match="ready"):
        queue.advance()
    assert queue.deliver_ready()[0].due_step == 1


def test_reset_discards_pending_rewards_and_restarts_step() -> None:
    queue = DelayedRewardQueue(max_delay=5)
    queue.enqueue(0, 1.0, 4)
    queue.advance()
    queue.reset()
    assert queue.current_step == 0
    assert queue.pending_count == 0
    assert queue.deliver_ready() == ()
    pending = queue.enqueue(1, 2.0, 0)
    assert pending.sequence == 0
    assert pending.decision_step == 0


def test_two_identical_call_sequences_have_identical_delivery_values() -> None:
    left = DelayedRewardQueue(max_delay=3)
    right = DelayedRewardQueue(max_delay=3)
    for queue in (left, right):
        queue.enqueue(0, np.float32(1.5), 1)
        queue.enqueue(1, -2.0, 2)
    assert left.advance() == right.advance() == 1
    assert left.deliver_ready() == right.deliver_ready()
    assert left.advance() == right.advance() == 2
    assert left.deliver_ready() == right.deliver_ready()
