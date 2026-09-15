from dataclasses import replace

import pytest

from neural_state_machine.delayed_credit import RewardDelivery
from neural_state_machine.phase3b_controls import (
    TimelineAudit,
    action_sequences_equal,
    delivery_timeline_digest,
    reward_block_multisets_equal,
    validate_fixed_delay_timeline,
    validate_queue_counts,
)


def test_fixed_delay_three_requires_real_overlap() -> None:
    audit = TimelineAudit(
        action_count=8,
        delivery_count=8,
        terminal_drain_count=3,
        max_pending_before_delivery=4,
        max_pending_after_delivery=3,
        decisions_with_prior_feedback_pending=7,
        lag_histogram=((3, 8),),
        delivery_timeline_digest="0" * 64,
        queue_pending_final=0,
        learner_unresolved_final=0,
    )
    validate_fixed_delay_timeline(audit, reward_delay=3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terminal_drain_count", 0),
        ("max_pending_before_delivery", 1),
        ("max_pending_after_delivery", 1),
        ("lag_histogram", ((0, 8),)),
        ("delivery_count", 7),
        ("queue_pending_final", 1),
        ("learner_unresolved_final", 1),
    ],
)
def test_fixed_delay_validator_rejects_broken_invariant(
    field: str, value: object
) -> None:
    audit = TimelineAudit(8, 8, 3, 4, 3, 7, ((3, 8),), "0" * 64, 0, 0)
    with pytest.raises(ValueError):
        validate_fixed_delay_timeline(replace(audit, **{field: value}), reward_delay=3)


def test_zero_delay_requires_empty_pipeline_after_each_delivery() -> None:
    audit = TimelineAudit(8, 8, 0, 1, 0, 0, ((0, 8),), "0" * 64, 0, 0)
    validate_fixed_delay_timeline(audit, reward_delay=0)


def test_timeline_digest_ignores_rewards_but_hashes_structure() -> None:
    left = (
        RewardDelivery(0, 0, 1.0, 0, 3, 3),
        RewardDelivery(1, 1, -1.0, 1, 4, 4),
    )
    right = (
        RewardDelivery(0, 0, -1.0, 0, 3, 3),
        RewardDelivery(1, 1, 1.0, 1, 4, 4),
    )
    changed = (
        RewardDelivery(0, 0, -1.0, 0, 3, 3),
        RewardDelivery(1, 1, 1.0, 1, 5, 5),
    )
    assert delivery_timeline_digest(left) == delivery_timeline_digest(right)
    assert delivery_timeline_digest(left) != delivery_timeline_digest(changed)


def test_action_lineage_control_requires_exact_equality() -> None:
    assert action_sequences_equal((0, 1, 0), (0, 1, 0)) is True
    assert action_sequences_equal((0, 1, 0), (0, 0, 1)) is False
    with pytest.raises(ValueError, match="non-negative"):
        action_sequences_equal((0, -1), (0, -1))


def test_reward_block_control_preserves_each_block_multiset() -> None:
    normal = (1.0, -1.0, 1.0, -1.0)
    shuffled = (-1.0, 1.0, -1.0, 1.0)
    assert reward_block_multisets_equal(normal, shuffled, block_size=2) is True
    assert (
        reward_block_multisets_equal(
            normal, (1.0, 1.0, 1.0, -1.0), block_size=2
        )
        is False
    )
    with pytest.raises(ValueError, match="block_size"):
        reward_block_multisets_equal(normal, shuffled, block_size=0)


def test_queue_count_control_is_fail_closed() -> None:
    validate_queue_counts(10, 10, 0)
    with pytest.raises(ValueError, match="delivery"):
        validate_queue_counts(10, 9, 0)
    with pytest.raises(ValueError, match="empty"):
        validate_queue_counts(10, 10, 1)
