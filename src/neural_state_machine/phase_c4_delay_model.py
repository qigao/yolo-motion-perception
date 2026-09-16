"""Pure delay-marginalization primitives for Phase C4."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_PROBABILITY_ATOL = 1e-15


def _readonly_float64_copy(values: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    copied = np.ascontiguousarray(array, dtype=np.float64).copy()
    copied.flags.writeable = False
    return copied


@dataclass(frozen=True, slots=True)
class DelayLaw:
    support: tuple[int, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.support, tuple) or not self.support:
            raise ValueError("support must be a non-empty tuple")
        if any(type(delay) is not int or delay < 0 for delay in self.support):
            raise ValueError("support must contain non-negative integers")
        if len(set(self.support)) != len(self.support):
            raise ValueError("support must not contain duplicates")
        if not isinstance(self.probabilities, tuple):
            raise ValueError("probabilities must be a tuple")
        if len(self.probabilities) != len(self.support):
            raise ValueError("probabilities must match support length")
        if any(isinstance(value, bool) for value in self.probabilities):
            raise ValueError("probabilities must contain finite non-negative scalars")
        try:
            resolved = tuple(float(value) for value in self.probabilities)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "probabilities must contain finite non-negative scalars"
            ) from exc
        if any(not math.isfinite(value) or value < 0.0 for value in resolved):
            raise ValueError("probabilities must contain finite non-negative scalars")
        if not math.isclose(sum(resolved), 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_ATOL):
            raise ValueError("probabilities must sum to one")
        object.__setattr__(self, "probabilities", resolved)

    @classmethod
    def registered(cls) -> DelayLaw:
        one_third = 1.0 / 3.0
        return cls((1, 3, 5), (one_third, one_third, one_third))

    @classmethod
    def immediate(cls) -> DelayLaw:
        return cls((0,), (1.0,))


@dataclass(frozen=True, slots=True)
class DecisionCreditRow:
    decision_index: int
    action_index: int
    feature: np.ndarray
    denominator: float

    def __post_init__(self) -> None:
        if type(self.decision_index) is not int or self.decision_index < 0:
            raise ValueError("decision_index must be a non-negative integer")
        if type(self.action_index) is not int or self.action_index < 0:
            raise ValueError("action_index must be a non-negative integer")
        feature = _readonly_float64_copy(self.feature, name="feature")
        if isinstance(self.denominator, bool):
            raise ValueError("denominator must be finite and positive")
        try:
            denominator = float(self.denominator)
        except (TypeError, ValueError) as exc:
            raise ValueError("denominator must be finite and positive") from exc
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("denominator must be finite and positive")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "denominator", denominator)


@dataclass(frozen=True, slots=True)
class MarginalizedFeatures:
    expected_feature: np.ndarray
    normalized_credit: np.ndarray

    def __post_init__(self) -> None:
        expected = np.asarray(self.expected_feature, dtype=np.float64)
        credit = np.asarray(self.normalized_credit, dtype=np.float64)
        if expected.ndim != 2 or credit.ndim != 2 or expected.shape != credit.shape:
            raise ValueError("marginalized feature matrices must have the same 2D shape")
        if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(credit)):
            raise ValueError("marginalized feature matrices must be finite")
        expected_copy = np.ascontiguousarray(expected, dtype=np.float64).copy()
        credit_copy = np.ascontiguousarray(credit, dtype=np.float64).copy()
        expected_copy.flags.writeable = False
        credit_copy.flags.writeable = False
        object.__setattr__(self, "expected_feature", expected_copy)
        object.__setattr__(self, "normalized_credit", credit_copy)


def candidate_indices(
    feedback_step: int,
    decision_count: int,
    law: DelayLaw,
) -> tuple[int, ...]:
    if type(feedback_step) is not int or feedback_step < 0:
        raise ValueError("feedback_step must be a non-negative integer")
    if type(decision_count) is not int or decision_count < 0:
        raise ValueError("decision_count must be a non-negative integer")
    if not isinstance(law, DelayLaw):
        raise ValueError("law must be a DelayLaw")
    result: list[int] = []
    for delay in law.support:
        if delay > feedback_step:
            continue
        index = feedback_step - delay
        if index < decision_count:
            result.append(index)
    return tuple(result)


def build_marginalized_features(
    rows: tuple[DecisionCreditRow, ...],
    feedback_step: int,
    action_count: int,
    law: DelayLaw,
) -> MarginalizedFeatures:
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("rows must be a non-empty tuple of DecisionCreditRow")
    if any(not isinstance(row, DecisionCreditRow) for row in rows):
        raise ValueError("rows must contain only DecisionCreditRow values")
    if type(feedback_step) is not int or feedback_step < 0:
        raise ValueError("feedback_step must be a non-negative integer")
    if type(action_count) is not int or action_count < 2:
        raise ValueError("action_count must be an integer of at least two")
    if not isinstance(law, DelayLaw):
        raise ValueError("law must be a DelayLaw")

    feature_size = rows[0].feature.size
    if any(row.feature.size != feature_size for row in rows):
        raise ValueError("all decision rows must have the same feature size")
    if any(row.action_index >= action_count for row in rows):
        raise ValueError("decision row action_index must be in range")
    by_index = {row.decision_index: row for row in rows}
    if len(by_index) != len(rows):
        raise ValueError("decision rows must not contain duplicate indices")

    expected = np.zeros((action_count, feature_size), dtype=np.float64)
    credit = np.zeros_like(expected)
    for delay, probability in zip(law.support, law.probabilities, strict=True):
        if delay > feedback_step:
            continue
        row = by_index.get(feedback_step - delay)
        if row is None:
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            expected[row.action_index] += probability * row.feature
            credit[row.action_index] += probability * row.feature / row.denominator

    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(credit)):
        raise ValueError("marginalized feature construction overflowed float64")
    return MarginalizedFeatures(expected, credit)


def current_weight_prediction(weights: object, expected_feature: object) -> float:
    try:
        weight_array = np.asarray(weights, dtype=np.float64)
        feature_array = np.asarray(expected_feature, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights and expected_feature must be float64-compatible") from exc
    if weight_array.ndim != 2 or feature_array.ndim != 2:
        raise ValueError("weights and expected_feature must be two-dimensional")
    if weight_array.shape != feature_array.shape:
        raise ValueError("weights and expected_feature must have the same shape")
    if not np.all(np.isfinite(weight_array)) or not np.all(np.isfinite(feature_array)):
        raise ValueError("weights and expected_feature must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = float(np.sum(weight_array * feature_array, dtype=np.float64))
    if not math.isfinite(prediction):
        raise ValueError("current-weight prediction overflowed float64")
    return prediction
