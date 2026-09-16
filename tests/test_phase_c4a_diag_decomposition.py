from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.phase_c4a_diagnostics.decomposition import (
    compare_weights,
    decompose_readout_margin,
    summarize_margin_components,
)


def test_margin_components_exactly_recompose_full_correct_vs_other_margin():
    weights = np.asarray(
        [
            [0.5, -1.0, 0.25],
            [-0.75, 0.4, -0.5],
        ],
        dtype=np.float64,
    )
    feature = np.asarray([1.5, -0.25, 1.0], dtype=np.float64)

    for correct_action in (0, 1):
        components = decompose_readout_margin(weights, feature, correct_action)
        scores = weights @ feature
        other_action = 1 - correct_action
        expected_margin = float(scores[correct_action] - scores[other_action])

        assert components.full_margin == pytest.approx(expected_margin)
        assert components.hidden_contribution + components.bias_contribution == pytest.approx(
            expected_margin,
            rel=0.0,
            abs=1e-12,
        )
        assert (
            components.action_block_0_contribution
            + components.action_block_1_contribution
        ) == pytest.approx(expected_margin, rel=0.0, abs=1e-12)
        assert components.predicted_action == int(np.argmax(scores))
        assert components.is_correct is (int(np.argmax(scores)) == correct_action)


def test_recomputed_decisions_match_ordinary_argmax_for_every_fixture_row():
    weights = np.asarray(
        [
            [0.75, -0.25, 0.1],
            [-0.2, 0.8, -0.15],
        ],
        dtype=np.float64,
    )
    features = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, 1.0],
            [0.25, -0.75, 1.0],
        ],
        dtype=np.float64,
    )
    correct_actions = np.asarray([0, 1, 1, 1, 0], dtype=np.int64)

    for feature, correct_action in zip(features, correct_actions, strict=True):
        components = decompose_readout_margin(weights, feature, int(correct_action))
        expected = int(np.argmax(weights @ feature))
        assert components.predicted_action == expected
        assert components.is_correct is (expected == int(correct_action))


def test_weight_comparison_reports_registered_posthoc_geometry_only():
    normal = np.asarray(
        [
            [1.0, 0.0, 0.25],
            [0.0, 2.0, -0.5],
        ],
        dtype=np.float64,
    )
    shuffled = np.asarray(
        [
            [0.5, 0.5, 0.1],
            [-0.25, 1.5, -0.2],
        ],
        dtype=np.float64,
    )
    comparison = compare_weights(normal, shuffled)

    assert comparison.normal_norm == pytest.approx(float(np.linalg.norm(normal)))
    assert comparison.shuffled_norm == pytest.approx(float(np.linalg.norm(shuffled)))
    assert comparison.delta_norm == pytest.approx(float(np.linalg.norm(shuffled - normal)))
    expected_cosine = float(
        np.dot(normal.reshape(-1), shuffled.reshape(-1))
        / (np.linalg.norm(normal) * np.linalg.norm(shuffled))
    )
    assert comparison.global_cosine == pytest.approx(expected_cosine)
    assert comparison.normal_action_norms == pytest.approx(
        (float(np.linalg.norm(normal[0])), float(np.linalg.norm(normal[1])))
    )
    assert comparison.shuffled_action_norms == pytest.approx(
        (float(np.linalg.norm(shuffled[0])), float(np.linalg.norm(shuffled[1])))
    )
    assert comparison.normal_bias_terms == pytest.approx((0.25, -0.5))
    assert comparison.shuffled_bias_terms == pytest.approx((0.1, -0.2))


def test_margin_summary_separates_correct_incorrect_delay_and_near_zero():
    weights = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    features = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.25, 0.75, 1.0],
        ],
        dtype=np.float64,
    )
    correct_actions = (0, 1, 1, 0)
    cue_delays = (1, 1, 2, 2)
    rows = tuple(
        decompose_readout_margin(weights, feature, correct_action)
        for feature, correct_action in zip(features, correct_actions, strict=True)
    )
    summary = summarize_margin_components(rows, cue_delays)

    assert summary.overall.count == 4
    assert summary.correct.count + summary.incorrect.count == 4
    assert summary.overall.near_zero_count == 1
    assert tuple(delay for delay, _ in summary.by_delay) == (1, 2)
    assert tuple(cell.count for _, cell in summary.by_delay) == (2, 2)
