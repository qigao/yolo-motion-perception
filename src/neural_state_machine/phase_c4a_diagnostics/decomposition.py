"""D3 post-hoc weight and evaluation-margin decomposition diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_NEAR_ZERO_MARGIN = 1e-9


@dataclass(frozen=True, slots=True)
class MarginComponents:
    correct_action: int
    other_action: int
    score0: float
    score1: float
    full_margin: float
    hidden_contribution: float
    bias_contribution: float
    action_block_0_contribution: float
    action_block_1_contribution: float
    predicted_action: int
    is_correct: bool
    near_zero: bool


@dataclass(frozen=True, slots=True)
class WeightComparison:
    normal_norm: float
    shuffled_norm: float
    delta_norm: float
    global_cosine: float | None
    per_action_cosines: tuple[float | None, float | None]
    normal_action_norms: tuple[float, float]
    shuffled_action_norms: tuple[float, float]
    normal_bias_terms: tuple[float, float]
    shuffled_bias_terms: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ContributionSlice:
    count: int
    margin_mean: float | None
    hidden_mean: float | None
    bias_mean: float | None
    action_block_0_mean: float | None
    action_block_1_mean: float | None
    near_zero_count: int


@dataclass(frozen=True, slots=True)
class MarginComponentSummary:
    overall: ContributionSlice
    correct: ContributionSlice
    incorrect: ContributionSlice
    by_delay: tuple[tuple[int, ContributionSlice], ...]


def _weights(value: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-action matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = float(
        np.dot(left.reshape(-1), right.reshape(-1)) / (left_norm * right_norm)
    )
    return max(-1.0, min(1.0, value))


def decompose_readout_margin(
    weights: object,
    hidden_with_bias: object,
    correct_action: int,
) -> MarginComponents:
    """Decompose one correct-vs-other margin without retraining the readout."""
    matrix = _weights(weights, "weights")
    try:
        feature = np.asarray(hidden_with_bias, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("hidden_with_bias must be float64-compatible") from exc
    if feature.ndim != 1 or feature.size != matrix.shape[1]:
        raise ValueError("hidden_with_bias must match the readout feature size")
    if not np.all(np.isfinite(feature)):
        raise ValueError("hidden_with_bias must contain only finite values")
    if type(correct_action) is not int or correct_action not in (0, 1):
        raise ValueError("correct_action must be 0 or 1")

    other_action = 1 - correct_action
    scores = matrix @ feature
    full_margin = float(scores[correct_action] - scores[other_action])
    hidden_contribution = float(
        np.dot(
            matrix[correct_action, :-1] - matrix[other_action, :-1],
            feature[:-1],
        )
    )
    bias_contribution = float(
        (matrix[correct_action, -1] - matrix[other_action, -1]) * feature[-1]
    )
    action_block_0 = float(scores[0] if correct_action == 0 else -scores[0])
    action_block_1 = float(scores[1] if correct_action == 1 else -scores[1])
    predicted_action = int(np.argmax(scores))

    return MarginComponents(
        correct_action=correct_action,
        other_action=other_action,
        score0=float(scores[0]),
        score1=float(scores[1]),
        full_margin=full_margin,
        hidden_contribution=hidden_contribution,
        bias_contribution=bias_contribution,
        action_block_0_contribution=action_block_0,
        action_block_1_contribution=action_block_1,
        predicted_action=predicted_action,
        is_correct=bool(predicted_action == correct_action),
        near_zero=bool(abs(full_margin) <= _NEAR_ZERO_MARGIN),
    )


def compare_weights(normal: object, shuffled: object) -> WeightComparison:
    """Compare frozen normal/shuffled readouts without fitting an ablation."""
    normal_matrix = _weights(normal, "normal")
    shuffled_matrix = _weights(shuffled, "shuffled")
    if normal_matrix.shape != shuffled_matrix.shape:
        raise ValueError("normal and shuffled weights must have identical shapes")

    return WeightComparison(
        normal_norm=float(np.linalg.norm(normal_matrix)),
        shuffled_norm=float(np.linalg.norm(shuffled_matrix)),
        delta_norm=float(np.linalg.norm(shuffled_matrix - normal_matrix)),
        global_cosine=_cosine(normal_matrix, shuffled_matrix),
        per_action_cosines=(
            _cosine(normal_matrix[0], shuffled_matrix[0]),
            _cosine(normal_matrix[1], shuffled_matrix[1]),
        ),
        normal_action_norms=(
            float(np.linalg.norm(normal_matrix[0])),
            float(np.linalg.norm(normal_matrix[1])),
        ),
        shuffled_action_norms=(
            float(np.linalg.norm(shuffled_matrix[0])),
            float(np.linalg.norm(shuffled_matrix[1])),
        ),
        normal_bias_terms=(float(normal_matrix[0, -1]), float(normal_matrix[1, -1])),
        shuffled_bias_terms=(
            float(shuffled_matrix[0, -1]),
            float(shuffled_matrix[1, -1]),
        ),
    )


def _summarize(rows: tuple[MarginComponents, ...]) -> ContributionSlice:
    if not rows:
        return ContributionSlice(0, None, None, None, None, None, 0)
    return ContributionSlice(
        count=len(rows),
        margin_mean=float(np.mean([row.full_margin for row in rows])),
        hidden_mean=float(np.mean([row.hidden_contribution for row in rows])),
        bias_mean=float(np.mean([row.bias_contribution for row in rows])),
        action_block_0_mean=float(
            np.mean([row.action_block_0_contribution for row in rows])
        ),
        action_block_1_mean=float(
            np.mean([row.action_block_1_contribution for row in rows])
        ),
        near_zero_count=sum(int(row.near_zero) for row in rows),
    )


def summarize_margin_components(
    components: object,
    cue_delays: object,
) -> MarginComponentSummary:
    """Summarize fixed decompositions by correctness and cue-delay cells."""
    rows = tuple(components)
    if any(not isinstance(row, MarginComponents) for row in rows):
        raise ValueError("components must contain only MarginComponents")
    if not rows:
        raise ValueError("components must be non-empty")
    try:
        delays = np.asarray(cue_delays)
    except (TypeError, ValueError) as exc:
        raise ValueError("cue_delays must be array-compatible") from exc
    if delays.ndim != 1 or delays.size != len(rows):
        raise ValueError("cue_delays must match the component count")
    if not np.issubdtype(delays.dtype, np.integer):
        raise ValueError("cue_delays must contain integers")
    delays = np.asarray(delays, dtype=np.int64)

    correct = tuple(row for row in rows if row.is_correct)
    incorrect = tuple(row for row in rows if not row.is_correct)
    by_delay: list[tuple[int, ContributionSlice]] = []
    for delay in np.unique(delays):
        selected = tuple(row for row, cell in zip(rows, delays, strict=True) if cell == delay)
        by_delay.append((int(delay), _summarize(selected)))

    return MarginComponentSummary(
        overall=_summarize(rows),
        correct=_summarize(correct),
        incorrect=_summarize(incorrect),
        by_delay=tuple(by_delay),
    )
