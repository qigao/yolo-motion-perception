from __future__ import annotations

from .contracts import NullableMetric


def nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    return 0


def pearson_metric(left: tuple[float, ...], right: tuple[float, ...]) -> NullableMetric:
    return NullableMetric(None, "unimplemented", len(left))


def lag_mean_product(feedback: tuple[float, ...], rewards: tuple[float, ...], lag: int) -> NullableMetric:
    return NullableMetric(None, "unimplemented", 0)


def summarize_scores(values: tuple[int, ...]) -> dict[str, int | float]:
    return {}
