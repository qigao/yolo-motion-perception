from __future__ import annotations

import math
import statistics

import numpy as np

from .contracts import NullableMetric


def _finite_tuple(values: tuple[float, ...], name: str) -> np.ndarray:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{name} must contain numeric values")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    if not isinstance(values, tuple) or not values:
        raise ValueError("values must be a non-empty tuple")
    if any(type(value) is not int for value in values):
        raise ValueError("values must contain integers")
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise ValueError("quantile must be numeric")
    q = float(quantile)
    if not math.isfinite(q) or not 0.0 < q <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def pearson_metric(
    left: tuple[float, ...], right: tuple[float, ...]
) -> NullableMetric:
    left_array = _finite_tuple(left, "left")
    right_array = _finite_tuple(right, "right")
    if left_array.shape != right_array.shape:
        raise ValueError("left and right lengths must match")
    left_centered = left_array - float(np.mean(left_array))
    right_centered = right_array - float(np.mean(right_array))
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return NullableMetric(None, "zero_variance", len(left))
    value = float(np.dot(left_centered, right_centered) / (left_norm * right_norm))
    return NullableMetric(value, None, len(left))


def lag_mean_product(
    feedback: tuple[float, ...], rewards: tuple[float, ...], lag: int
) -> NullableMetric:
    feedback_array = _finite_tuple(feedback, "feedback")
    rewards_array = _finite_tuple(rewards, "rewards")
    if feedback_array.shape != rewards_array.shape:
        raise ValueError("feedback and rewards lengths must match")
    if type(lag) is not int or lag < 0:
        raise ValueError("lag must be a non-negative integer")
    if lag >= len(feedback):
        return NullableMetric(None, "no_valid_pairs", 0)
    products = feedback_array[lag:] * rewards_array[: len(rewards) - lag]
    return NullableMetric(float(np.mean(products)), None, int(products.size))


def summarize_scores(values: tuple[int, ...]) -> dict[str, int | float]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("values must be a non-empty tuple")
    if any(type(value) is not int or value < 0 or value > 200 for value in values):
        raise ValueError("scores must be integer counts in [0, 200]")
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "p10": nearest_rank(values, 0.10),
        "p90": nearest_rank(values, 0.90),
        "count_at_least_150": sum(value >= 150 for value in values),
        "count_at_least_197": sum(value >= 197 for value in values),
    }
