from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contracts import NullableMetric


@dataclass(frozen=True, slots=True)
class ResidualTerms:
    feedback_minus_prediction: float
    feedback_minus_source_prediction: float
    source_minus_prediction: float
    squared_cross_term: float


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def residual_terms(
    feedback: float, prediction_trace: float, source_prediction: float
) -> ResidualTerms:
    feedback_value = _finite_scalar(feedback, "feedback")
    prediction_value = _finite_scalar(prediction_trace, "prediction_trace")
    source_value = _finite_scalar(source_prediction, "source_prediction")
    feedback_minus_source = feedback_value - source_value
    source_minus_prediction = source_value - prediction_value
    return ResidualTerms(
        feedback_minus_prediction=feedback_value - prediction_value,
        feedback_minus_source_prediction=feedback_minus_source,
        source_minus_prediction=source_minus_prediction,
        squared_cross_term=2.0 * feedback_minus_source * source_minus_prediction,
    )


def expected_drain_delta(
    rewards: tuple[float, ...],
    prediction: float,
    eligibility: np.ndarray,
    step_size: float,
) -> np.ndarray:
    if not isinstance(rewards, tuple):
        raise ValueError("rewards must be a tuple")
    reward_values = tuple(_finite_scalar(value, "reward") for value in rewards)
    prediction_value = _finite_scalar(prediction, "prediction")
    alpha = _finite_scalar(step_size, "step_size")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("step_size must be in (0, 1]")
    trace = np.asarray(eligibility, dtype=np.float64)
    if trace.ndim != 2 or not np.all(np.isfinite(trace)):
        raise ValueError("eligibility must be a finite matrix")
    scalar = alpha * (sum(reward_values) - len(reward_values) * prediction_value)
    result = scalar * trace
    if not np.all(np.isfinite(result)):
        raise ValueError("drain delta must be finite")
    return result


def cosine_metric(left: np.ndarray, right: np.ndarray) -> NullableMetric:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.size == 0:
        raise ValueError("cosine arrays must have the same non-empty shape")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError("cosine arrays must be finite")
    left_norm = float(np.linalg.norm(left_array.ravel()))
    right_norm = float(np.linalg.norm(right_array.ravel()))
    if left_norm == 0.0 or right_norm == 0.0:
        return NullableMetric(None, "zero_norm", 1)
    value = float(
        np.dot(left_array.ravel(), right_array.ravel()) / (left_norm * right_norm)
    )
    return NullableMetric(value, None, 1)


def accounting_tolerance(expected: float) -> float:
    return 1e-10 + 1e-12 * abs(_finite_scalar(expected, "expected"))


def maximum_accounting_residual(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape or actual_array.size == 0:
        raise ValueError("accounting arrays must have the same non-empty shape")
    if not np.all(np.isfinite(actual_array)) or not np.all(np.isfinite(expected_array)):
        raise ValueError("accounting arrays must be finite")
    residual = np.abs(actual_array - expected_array)
    allowed = 1e-10 + 1e-12 * np.abs(expected_array)
    maximum = float(np.max(residual))
    if np.any(residual > allowed):
        raise ValueError(f"accounting residual exceeds bound: max={maximum}")
    return maximum
