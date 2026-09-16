from __future__ import annotations

import numpy as np

from neural_state_machine.action_value_benchmark import _permute_reward_blocks
from neural_state_machine.phase3c_diagnostics.provenance import (
    diagnostic_permutation,
    donor_rows,
    expected_current_donor_matches,
    original_block10_permutation,
)


def test_due_time_belongs_to_slot_not_donor() -> None:
    rows = donor_rows(permutation=(1, 0), due_steps=(1, 4))
    assert (rows[0].slot, rows[0].donor, rows[0].due_step) == (0, 1, 1)
    assert rows[0].donor == rows[0].due_step
    assert rows[1].due_step == 4


def test_original_permutation_reproduces_existing_block_shuffle() -> None:
    rewards = tuple(float(index) for index in range(20))
    permutation = original_block10_permutation(17, len(rewards))
    expected_rng = np.random.default_rng(np.random.SeedSequence([17, 0x33534846]))
    expected = _permute_reward_blocks(rewards, expected_rng, block_size=10)
    actual = tuple(rewards[index] for index in permutation)
    assert actual == expected
    assert all(index // 10 == permutation[index] // 10 for index in range(20))


def test_registered_diagnostic_permutation_is_repeatable_and_not_identity() -> None:
    first = diagnostic_permutation(17, 20, "global", 3)
    second = diagnostic_permutation(17, 20, "global", 3)
    assert first == second
    assert sorted(first) == list(range(20))
    assert first != tuple(range(20))


def test_block10_expected_match_formula_counts_only_same_block_due_slots() -> None:
    due_steps = (1, 3, 12, 13, 14, 15, 16, 17, 18, 19, 11, 12)
    eligible = sum(
        due < len(due_steps) and slot // 10 == due // 10
        for slot, due in enumerate(due_steps)
    )
    assert expected_current_donor_matches(due_steps, "block10") == eligible / 10
    assert expected_current_donor_matches(due_steps, "global") == sum(
        due < len(due_steps) for due in due_steps
    ) / len(due_steps)
