from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .flow_types import ArticulatedFlowEvidence

_LEFT_REGIONS = ("left_thigh", "left_calf", "left_foot")
_RIGHT_REGIONS = ("right_thigh", "right_calf", "right_foot")
_FUNDAMENTAL_PEAK_RATIO = 0.80


@dataclass(frozen=True)
class GaitConfig:
    history_seconds: float = 2.0
    min_samples: int = 12
    min_duration: float = 0.8
    min_quality: float = 0.55
    standing_energy_threshold: float = 0.008
    min_periodicity: float = 0.55
    min_bilateral_correlation: float = 0.45
    min_leg_support_fraction: float = 0.70
    walking_cadence_min_hz: float = 0.7
    walking_cadence_max_hz: float = 2.4
    running_cadence_min_hz: float = 2.2

    def __post_init__(self) -> None:
        if not math.isfinite(self.history_seconds) or self.history_seconds <= 0.0:
            raise ValueError("history seconds must be positive and finite")
        if self.min_samples < 2:
            raise ValueError("minimum samples must be at least two")
        if not math.isfinite(self.min_duration) or self.min_duration <= 0.0:
            raise ValueError("minimum duration must be positive and finite")
        for name, value in (
            ("minimum quality", self.min_quality),
            ("minimum periodicity", self.min_periodicity),
            ("minimum bilateral correlation", self.min_bilateral_correlation),
            ("minimum leg support fraction", self.min_leg_support_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            not math.isfinite(self.standing_energy_threshold)
            or self.standing_energy_threshold < 0.0
        ):
            raise ValueError("standing energy threshold must be finite and non-negative")
        if (
            not math.isfinite(self.walking_cadence_min_hz)
            or not math.isfinite(self.walking_cadence_max_hz)
            or self.walking_cadence_min_hz <= 0.0
            or self.walking_cadence_max_hz < self.walking_cadence_min_hz
        ):
            raise ValueError("walking cadence range is invalid")
        if not math.isfinite(self.running_cadence_min_hz) or self.running_cadence_min_hz <= 0.0:
            raise ValueError("running cadence minimum must be positive and finite")


@dataclass(frozen=True)
class GaitEvidence:
    sample_count: int
    duration: float
    left_support_fraction: float
    right_support_fraction: float
    left_energy: float
    right_energy: float
    periodicity: float
    bilateral_correlation: float
    phase_lag_seconds: float
    cadence_hz: float
    articulated_amplitude: float
    temporal_consistency: float
    quality: float

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample count must be non-negative")
        finite_values = (
            self.duration,
            self.left_support_fraction,
            self.right_support_fraction,
            self.left_energy,
            self.right_energy,
            self.periodicity,
            self.bilateral_correlation,
            self.phase_lag_seconds,
            self.cadence_hz,
            self.articulated_amplitude,
            self.temporal_consistency,
            self.quality,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("gait evidence values must be finite")
        if self.duration < 0.0 or self.left_energy < 0.0 or self.right_energy < 0.0:
            raise ValueError("gait duration and energies must be non-negative")
        if self.cadence_hz < 0.0 or self.articulated_amplitude < 0.0:
            raise ValueError("gait cadence and amplitude must be non-negative")
        for name, value in (
            ("left support fraction", self.left_support_fraction),
            ("right support fraction", self.right_support_fraction),
            ("periodicity", self.periodicity),
            ("bilateral correlation", self.bilateral_correlation),
            ("temporal consistency", self.temporal_consistency),
            ("quality", self.quality),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def _windowed(
    observations: Sequence[ArticulatedFlowEvidence],
    history_seconds: float,
) -> list[ArticulatedFlowEvidence]:
    values = list(observations)
    if not values:
        return []
    track_id = values[0].track_id
    previous_end = -math.inf
    for observation in values:
        if observation.track_id != track_id:
            raise ValueError("gait observations must belong to one track")
        if observation.start_timestamp < previous_end - 1e-9:
            raise ValueError("gait observations must be time ordered and non-overlapping")
        previous_end = observation.end_timestamp
    cutoff = values[-1].end_timestamp - history_seconds
    return [observation for observation in values if observation.end_timestamp > cutoff]


def _leg_vector(
    observation: ArticulatedFlowEvidence,
    names: tuple[str, ...],
) -> tuple[np.ndarray, bool]:
    vectors = [
        np.array((observation.region_flow[name].dx, observation.region_flow[name].dy), dtype=float)
        for name in names
        if name in observation.region_flow
    ]
    if not vectors:
        return np.zeros(2, dtype=float), False
    return np.mean(np.stack(vectors), axis=0), True


def _dominant_axis(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.array((1.0, 0.0), dtype=float)
    samples = np.stack(vectors)
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) <= 1e-12:
        return np.array((1.0, 0.0), dtype=float)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[0] < 0.0 or (abs(axis[0]) <= 1e-12 and axis[1] < 0.0):
        axis = -axis
    return axis


def _correlation(first: np.ndarray, second: np.ndarray, lag: int) -> float:
    if lag > 0:
        left = first[:-lag]
        right = second[lag:]
    elif lag < 0:
        left = first[-lag:]
        right = second[:lag]
    else:
        left = first
        right = second

    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(valid)) < 3:
        return 0.0
    left = left[valid]
    right = right[valid]
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _sample_interval(observations: list[ArticulatedFlowEvidence]) -> float:
    if len(observations) < 2:
        return 0.0
    times = np.array([observation.end_timestamp for observation in observations], dtype=float)
    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        return 0.0
    return float(np.median(intervals))


def _fundamental_peak(lag_scores: list[tuple[int, float]]) -> tuple[int, float]:
    strongest_lag, strongest_score = max(lag_scores, key=lambda item: item[1])
    if strongest_score <= 0.0:
        return strongest_lag, strongest_score

    minimum_strong_score = _FUNDAMENTAL_PEAK_RATIO * strongest_score
    strong_local_peaks: list[tuple[int, float]] = []
    for index, (lag, score) in enumerate(lag_scores):
        previous = lag_scores[index - 1][1] if index > 0 else -math.inf
        following = lag_scores[index + 1][1] if index + 1 < len(lag_scores) else -math.inf
        if score >= previous and score >= following and score >= minimum_strong_score:
            strong_local_peaks.append((lag, score))

    if not strong_local_peaks:
        return strongest_lag, strongest_score
    return min(strong_local_peaks, key=lambda item: item[0])


def _stride_cycle_and_periodicity(
    left: np.ndarray,
    right: np.ndarray,
    dt: float,
    config: GaitConfig,
) -> tuple[float, float]:
    if dt <= 0.0 or len(left) < 4:
        return 0.0, 0.0
    sample_rate = 1.0 / dt
    min_cycle_frequency = config.walking_cadence_min_hz / 2.0
    max_step_frequency = max(
        config.walking_cadence_max_hz,
        config.running_cadence_min_hz * 2.0,
    )
    max_cycle_frequency = min(max_step_frequency / 2.0, sample_rate * 0.45)
    if max_cycle_frequency <= min_cycle_frequency:
        return 0.0, 0.0

    min_lag = max(1, math.floor(sample_rate / max_cycle_frequency))
    max_lag = min(
        len(left) - config.min_samples,
        math.ceil(sample_rate / min_cycle_frequency),
    )
    if max_lag <= min_lag:
        return 0.0, 0.0

    lag_scores: list[tuple[int, float]] = []
    for lag in range(min_lag, max_lag + 1):
        correlations = []
        left_corr = _correlation(left, left, lag)
        right_corr = _correlation(right, right, lag)
        if np.isfinite(left).sum() >= 3:
            correlations.append(left_corr)
        if np.isfinite(right).sum() >= 3:
            correlations.append(right_corr)
        score = float(np.mean(correlations)) if correlations else 0.0
        lag_scores.append((lag, score))

    lag, score = _fundamental_peak(lag_scores)
    periodicity = float(np.clip(score, 0.0, 1.0))
    if periodicity <= 0.0:
        return 0.0, 0.0
    return sample_rate / lag, periodicity


def _bilateral_phase(
    left: np.ndarray,
    right: np.ndarray,
    stride_cycle_hz: float,
    dt: float,
) -> tuple[float, float]:
    if stride_cycle_hz <= 0.0 or dt <= 0.0:
        return 0.0, 0.0
    if np.isfinite(left).sum() < 3 or np.isfinite(right).sum() < 3:
        return 0.0, 0.0

    half_period = 0.5 / stride_cycle_hz
    min_lag = max(1, math.floor(0.60 * half_period / dt))
    max_lag = min(len(left) - 3, math.ceil(1.40 * half_period / dt))
    if max_lag < min_lag:
        return 0.0, 0.0

    candidates: list[tuple[int, float]] = []
    for lag in range(-max_lag, max_lag + 1):
        if abs(lag) < min_lag:
            continue
        candidates.append((lag, _correlation(left, right, lag)))
    if not candidates:
        return 0.0, 0.0
    best_lag, correlation = max(candidates, key=lambda item: item[1])
    return float(np.clip(correlation, 0.0, 1.0)), best_lag * dt


def estimate_gait(
    observations: Sequence[ArticulatedFlowEvidence],
    config: GaitConfig,
) -> GaitEvidence:
    window = _windowed(observations, config.history_seconds)
    if not window:
        return GaitEvidence(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    left_vectors: list[np.ndarray] = []
    right_vectors: list[np.ndarray] = []
    left_supported: list[bool] = []
    right_supported: list[bool] = []
    axis_vectors: list[np.ndarray] = []
    for observation in window:
        left_vector, has_left = _leg_vector(observation, _LEFT_REGIONS)
        right_vector, has_right = _leg_vector(observation, _RIGHT_REGIONS)
        left_vectors.append(left_vector)
        right_vectors.append(right_vector)
        left_supported.append(has_left)
        right_supported.append(has_right)
        if has_left:
            axis_vectors.append(left_vector)
        if has_right:
            axis_vectors.append(right_vector)

    axis = _dominant_axis(axis_vectors)
    left_signal = np.array(
        [
            float(vector @ axis) if supported else np.nan
            for vector, supported in zip(left_vectors, left_supported)
        ],
        dtype=float,
    )
    right_signal = np.array(
        [
            float(vector @ axis) if supported else np.nan
            for vector, supported in zip(right_vectors, right_supported)
        ],
        dtype=float,
    )

    left_norms = [
        float(np.linalg.norm(vector))
        for vector, supported in zip(left_vectors, left_supported)
        if supported
    ]
    right_norms = [
        float(np.linalg.norm(vector))
        for vector, supported in zip(right_vectors, right_supported)
        if supported
    ]
    left_energy = float(np.mean(left_norms)) if left_norms else 0.0
    right_energy = float(np.mean(right_norms)) if right_norms else 0.0

    dt = _sample_interval(window)
    stride_cycle_hz, periodicity = _stride_cycle_and_periodicity(
        left_signal,
        right_signal,
        dt,
        config,
    )
    bilateral_correlation, phase_lag_seconds = _bilateral_phase(
        left_signal,
        right_signal,
        stride_cycle_hz,
        dt,
    )
    cadence_hz = 2.0 * stride_cycle_hz

    finite_signal = np.concatenate(
        (left_signal[np.isfinite(left_signal)], right_signal[np.isfinite(right_signal)])
    )
    articulated_amplitude = (
        float(np.sqrt(np.mean(np.square(finite_signal)))) if finite_signal.size else 0.0
    )

    if len(window) >= 2:
        times = np.array([observation.end_timestamp for observation in window], dtype=float)
        intervals = np.diff(times)
        mean_interval = float(np.mean(intervals))
        if mean_interval > 0.0:
            temporal_consistency = float(
                np.clip(1.0 - float(np.std(intervals)) / mean_interval, 0.0, 1.0)
            )
        else:
            temporal_consistency = 0.0
    else:
        temporal_consistency = 0.0

    return GaitEvidence(
        sample_count=len(window),
        duration=window[-1].end_timestamp - window[0].start_timestamp,
        left_support_fraction=float(np.mean(left_supported)),
        right_support_fraction=float(np.mean(right_supported)),
        left_energy=left_energy,
        right_energy=right_energy,
        periodicity=periodicity,
        bilateral_correlation=bilateral_correlation,
        phase_lag_seconds=phase_lag_seconds,
        cadence_hz=cadence_hz,
        articulated_amplitude=articulated_amplitude,
        temporal_consistency=temporal_consistency,
        quality=float(np.mean([observation.quality for observation in window])),
    )
