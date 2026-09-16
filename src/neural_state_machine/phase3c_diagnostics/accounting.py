from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import NullableMetric


@dataclass(frozen=True, slots=True)
class ResidualTerms:
    feedback_minus_prediction: float
    feedback_minus_source_prediction: float
    source_minus_prediction: float
    squared_cross_term: float


def residual_terms(feedback: float, prediction_trace: float, source_prediction: float) -> ResidualTerms:
    return ResidualTerms(0.0, 0.0, 0.0, 0.0)


def expected_drain_delta(
    rewards: tuple[float, ...],
    prediction: float,
    eligibility: np.ndarray,
    step_size: float,
) -> np.ndarray:
    return np.zeros_like(eligibility)


def cosine_metric(left: np.ndarray, right: np.ndarray) -> NullableMetric:
    return NullableMetric(None, "unimplemented", 1)
