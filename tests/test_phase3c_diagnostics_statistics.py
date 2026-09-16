from __future__ import annotations

from neural_state_machine.phase3c_diagnostics.statistics import (
    lag_mean_product,
    nearest_rank,
    pearson_metric,
    summarize_scores,
)


def test_nearest_rank_uses_registered_definition() -> None:
    values = tuple(range(32))
    assert nearest_rank(values, 0.10) == 3
    assert nearest_rank(values, 0.90) == 28


def test_zero_variance_correlation_is_explicitly_undefined() -> None:
    metric = pearson_metric((1.0, 1.0, 1.0), (1.0, 2.0, 3.0))
    assert metric.value is None
    assert metric.reason == "zero_variance"
    assert metric.sample_count == 3


def test_lag_mean_product_uses_only_valid_real_decision_pairs() -> None:
    metric = lag_mean_product((2.0, 3.0, 4.0), (5.0, 7.0, 11.0), 1)
    assert metric.value == (3.0 * 5.0 + 4.0 * 7.0) / 2.0
    assert metric.reason is None
    assert metric.sample_count == 2


def test_score_summary_reports_fixed_threshold_counts() -> None:
    summary = summarize_scores(tuple(range(130, 162)))
    assert summary["minimum"] == 130
    assert summary["maximum"] == 161
    assert summary["count_at_least_150"] == 12
    assert summary["count_at_least_197"] == 0
    assert summary["p10"] == 133
    assert summary["p90"] == 158
