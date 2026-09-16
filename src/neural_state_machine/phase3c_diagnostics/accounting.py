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


def _history_direction(
    action: int,
    hidden: np.ndarray,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    if type(action) is not int or not 0 <= action < shape[0]:
        raise ValueError("history action is outside the eligibility action dimension")
    vector = np.asarray(hidden, dtype=np.float64)
    if vector.shape != (shape[1] - 1,) or not np.all(np.isfinite(vector)):
        raise ValueError("history hidden vector does not match eligibility shape")
    feature = np.concatenate((vector, np.array([1.0], dtype=np.float64)))
    denominator = float(np.dot(feature, feature))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("history feature normalization is invalid")
    result = np.zeros(shape, dtype=np.float64)
    result[action] = feature / denominator
    return result


def _frobenius_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64).ravel()))


def _inner_product(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(
            np.asarray(left, dtype=np.float64).ravel(),
            np.asarray(right, dtype=np.float64).ravel(),
        )
    )


def eligibility_history_split(
    *,
    history: tuple[tuple[int, np.ndarray], ...],
    step: int,
    source_steps: tuple[int, ...],
    captured_eligibility: np.ndarray,
    feedback: float,
    prediction_trace: float,
    step_size: float,
    rho: float,
    actual_update: np.ndarray,
    reference_direction: np.ndarray,
) -> dict[str, object]:
    """Reconstruct E_source/E_other and their registered Arm-B update components."""
    if not isinstance(history, tuple) or not history:
        raise ValueError("history must be a non-empty tuple")
    if type(step) is not int or not 0 <= step < len(history):
        raise ValueError("step must index the supplied history")
    if not isinstance(source_steps, tuple) or any(
        type(source) is not int or not 0 <= source <= step for source in source_steps
    ):
        raise ValueError("source_steps must be valid historical decision indices")
    if len(set(source_steps)) != len(source_steps):
        raise ValueError("source_steps must not contain duplicates")

    trace = np.asarray(captured_eligibility, dtype=np.float64)
    actual = np.asarray(actual_update, dtype=np.float64)
    reference = np.asarray(reference_direction, dtype=np.float64)
    if trace.ndim != 2 or trace.size == 0 or trace.shape[0] != 2:
        raise ValueError("captured eligibility must be a non-empty two-action matrix")
    if actual.shape != trace.shape or reference.shape != trace.shape:
        raise ValueError("actual/reference update shapes must match captured eligibility")
    if not (
        np.all(np.isfinite(trace))
        and np.all(np.isfinite(actual))
        and np.all(np.isfinite(reference))
    ):
        raise ValueError("D3 accounting arrays must be finite")

    resolved_rho = _finite_scalar(rho, "rho")
    if not 0.0 <= resolved_rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    alpha = _finite_scalar(step_size, "step_size")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("step_size must be in (0, 1]")
    feedback_value = _finite_scalar(feedback, "feedback")
    prediction_value = _finite_scalar(prediction_trace, "prediction_trace")

    source_set = set(source_steps)
    source = np.zeros_like(trace)
    other = np.zeros_like(trace)
    for source_index in range(step + 1):
        try:
            action, hidden = history[source_index]
        except (TypeError, ValueError) as exc:
            raise ValueError("history rows must be (action, hidden) pairs") from exc
        direction = _history_direction(action, hidden, shape=trace.shape)
        contribution = (resolved_rho ** (step - source_index)) * direction
        if source_index in source_set:
            source += contribution
        else:
            other += contribution

    reconstructed = source + other
    eligibility_residual = maximum_accounting_residual(trace, reconstructed)
    scalar = alpha * (feedback_value - prediction_value)
    source_update = scalar * source
    other_update = scalar * other
    reconstructed_update = source_update + other_update
    update_residual = maximum_accounting_residual(actual, reconstructed_update)

    return {
        "eligibility_source": source,
        "eligibility_other": other,
        "source_update": source_update,
        "other_update": other_update,
        "eligibility_reconstruction_max_residual": eligibility_residual,
        "update_reconstruction_max_residual": update_residual,
        "source_update_norm": _frobenius_norm(source_update),
        "other_update_norm": _frobenius_norm(other_update),
        "actual_update_norm": _frobenius_norm(actual),
        "reference_norm": _frobenius_norm(reference),
        "source_reference_inner_product": _inner_product(source_update, reference),
        "other_reference_inner_product": _inner_product(other_update, reference),
        "actual_reference_inner_product": _inner_product(actual, reference),
        "source_vs_reference_cosine": cosine_metric(source_update, reference).to_dict(),
        "other_vs_reference_cosine": cosine_metric(other_update, reference).to_dict(),
        "actual_vs_reference_cosine": cosine_metric(actual, reference).to_dict(),
    }
