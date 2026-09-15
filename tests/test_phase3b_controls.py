import pytest

from neural_state_machine.phase3b_controls import (
    action_sequences_equal,
    reward_block_multisets_equal,
    validate_queue_counts,
)


def test_action_lineage_control_requires_exact_equality() -> None:
    assert action_sequences_equal((0, 1, 0), (0, 1, 0)) is True
    assert action_sequences_equal((0, 1, 0), (0, 0, 1)) is False
    with pytest.raises(ValueError, match="non-negative"):
        action_sequences_equal((0, -1), (0, -1))


def test_reward_block_control_preserves_each_block_multiset() -> None:
    normal = (1.0, -1.0, 1.0, -1.0)
    shuffled = (-1.0, 1.0, -1.0, 1.0)
    assert reward_block_multisets_equal(normal, shuffled, block_size=2) is True
    assert reward_block_multisets_equal(normal, (1.0, 1.0, 1.0, -1.0), block_size=2) is False
    with pytest.raises(ValueError, match="block_size"):
        reward_block_multisets_equal(normal, shuffled, block_size=0)


def test_queue_count_control_is_fail_closed() -> None:
    validate_queue_counts(10, 10, 0)
    with pytest.raises(ValueError, match="delivery"):
        validate_queue_counts(10, 9, 0)
    with pytest.raises(ValueError, match="empty"):
        validate_queue_counts(10, 10, 1)
