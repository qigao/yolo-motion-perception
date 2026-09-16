from __future__ import annotations

import math

import numpy as np
import pytest

from neural_state_machine.phase_c4_benchmark import PhaseC4Config
from neural_state_machine.phase_c4a_diagnostics.integrity import replay_registered_condition
from neural_state_machine.phase_c4a_diagnostics.model import DiagnosticConfig
from neural_state_machine.phase_c4a_diagnostics.permutations import (
    PermutationRow,
    permutation_indices,
    permute_rewards,
    run_permutation_grid_for_seed,
    summarize_permutation_rows,
)


def test_permutation_indices_use_exact_registered_rng_lineage():
    block10 = permutation_indices(7, "block10", 0, 20)
    global_ = permutation_indices(7, "global", 0, 20)

    np.testing.assert_array_equal(
        block10,
        np.asarray(
            [9, 5, 1, 2, 4, 6, 8, 0, 3, 7, 14, 18, 19, 11, 17, 10, 15, 16, 12, 13],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        global_,
        np.asarray(
            [18, 17, 5, 19, 14, 10, 3, 7, 9, 6, 16, 15, 2, 1, 13, 11, 4, 0, 12, 8],
            dtype=np.int64,
        ),
    )
    assert block10.flags.writeable is False
    assert global_.flags.writeable is False


def test_block10_is_block_local_and_fixed_points_are_not_rejected_or_resampled():
    indices = permutation_indices(7, "block10", 0, 20)

    assert indices[4] == 4
    for start in (0, 10):
        block = indices[start : start + 10]
        assert sorted(int(value) for value in block) == list(range(start, start + 10))


def test_global_permutation_covers_full_range_and_fixed_points_are_preserved():
    indices = permutation_indices(7, "global", 0, 20)

    assert sorted(int(value) for value in indices) == list(range(20))
    assert indices[7] == 7
    assert any(
        not (start <= int(value) < start + 10)
        for start in (0, 10)
        for value in indices[start : start + 10]
    )


def test_permute_rewards_is_only_an_indexed_value_reassignment():
    rewards = tuple(float(value) for value in range(10))
    indices = permutation_indices(17, "block10", 3, len(rewards))
    permuted = permute_rewards(rewards, indices)

    assert permuted == tuple(rewards[int(index)] for index in indices)
    assert sorted(permuted) == sorted(rewards)
    with pytest.raises(ValueError, match="permutation"):
        permute_rewards(rewards, np.asarray([0, 1, 1, 3, 4, 5, 6, 7, 8, 9]))


def _row(score: int, replicate: int) -> PermutationRow:
    return PermutationRow(
        seed=7,
        mode="block10",
        replicate=replicate,
        permutation_digest=f"{replicate:064x}",
        original_score=score,
        secondary_scores=(score,) * 8,
        target_parallel_ratio=0.5,
        target_residual_ratio=0.5,
        supervised_global_cosine=0.25,
        supervised_projection_magnitude=1.0,
    )


def test_summary_uses_registered_nearest_rank_percentiles_and_threshold_counts():
    rows = tuple(_row(100 + replicate, replicate) for replicate in range(32))
    summary = summarize_permutation_rows(rows, registered_shuffled_score=147)

    assert summary.seed == 7
    assert summary.mode == "block10"
    assert summary.count == 32
    assert summary.mean == pytest.approx(115.5)
    assert summary.median == pytest.approx(115.5)
    assert summary.minimum == 100
    assert summary.maximum == 131
    assert summary.p10 == 103
    assert summary.p90 == 128
    assert summary.count_lt_150 == 32
    assert summary.count_ge_150 == 0
    assert summary.count_ge_170 == 0
    assert summary.registered_shuffled_score == 147
    assert summary.registered_reference_is_p_value is False
    assert math.ceil(0.10 * 32) - 1 == 3
    assert math.ceil(0.90 * 32) - 1 == 28


def test_tiny_unregistered_grid_runs_exactly_two_replicates_per_mode():
    replay = replay_registered_condition(7, "normal")
    rows = run_permutation_grid_for_seed(
        replay,
        PhaseC4Config(),
        DiagnosticConfig.testing(2),
    )

    assert [(row.mode, row.replicate) for row in rows] == [
        ("block10", 0),
        ("block10", 1),
        ("global", 0),
        ("global", 1),
    ]
    assert all(row.seed == 7 for row in rows)
    assert all(0 <= row.original_score <= 200 for row in rows)
    assert all(len(row.secondary_scores) == 8 for row in rows)
    assert all(len(row.permutation_digest) == 64 for row in rows)
    assert all(0.0 <= row.target_parallel_ratio <= 1.0 + 1e-12 for row in rows)
    assert all(0.0 <= row.target_residual_ratio <= 1.0 + 1e-12 for row in rows)


def test_registered_config_remains_exactly_32_replicates_per_mode():
    config = DiagnosticConfig.registered()

    assert config.is_registered is True
    assert config.permutation_replicates == 32
    assert config.modes == ("block10", "global")
    assert config.permutation_lineage == 0x43344144
