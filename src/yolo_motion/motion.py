from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .types import MotionEvidence, TrackObservation


@dataclass(frozen=True)
class MotionConfig:
    history_seconds: float = 1.0
    min_samples: int = 5
    min_duration: float = 0.3
    stationary_speed_threshold: float = 0.02
    radial_rate_threshold: float = 0.15
    min_radial_confidence: float = 0.6
    min_trend_consistency: float = 0.75

    def __post_init__(self) -> None:
        if self.history_seconds <= 0.0:
            raise ValueError("history_seconds must be positive")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if self.min_duration < 0.0:
            raise ValueError("min_duration must be non-negative")
        if self.stationary_speed_threshold < 0.0:
            raise ValueError("stationary_speed_threshold must be non-negative")
        if self.radial_rate_threshold <= 0.0:
            raise ValueError("radial_rate_threshold must be positive")
        if not 0.0 <= self.min_radial_confidence <= 1.0:
            raise ValueError("min_radial_confidence must be in [0, 1]")
        if not 0.0 <= self.min_trend_consistency <= 1.0:
            raise ValueError("min_trend_consistency must be in [0, 1]")


def _linear_fit(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    centered_t = times - times.mean()
    denom = float(np.dot(centered_t, centered_t))
    if denom <= 1e-15:
        return 0.0, 0.0
    centered_v = values - values.mean()
    slope = float(np.dot(centered_t, centered_v) / denom)
    fitted = values.mean() + slope * centered_t
    residual = values - fitted
    sse = float(np.dot(residual, residual))
    sst = float(np.dot(centered_v, centered_v))
    if sst <= 1e-15:
        quality = 1.0 if sse <= 1e-15 else 0.0
    else:
        quality = max(0.0, min(1.0, 1.0 - sse / sst))
    return slope, quality


def _trend_consistency(values: np.ndarray, slope: float) -> float:
    if len(values) < 2:
        return 0.0
    if abs(slope) <= 1e-12:
        return 1.0
    deltas = np.diff(values)
    direction = 1.0 if slope > 0.0 else -1.0
    agreeing = np.count_nonzero(deltas * direction > 0.0)
    flat = np.count_nonzero(np.isclose(deltas, 0.0, atol=1e-12))
    return float((agreeing + 0.5 * flat) / len(deltas))


def estimate_motion(
    observations: Sequence[TrackObservation], config: MotionConfig
) -> MotionEvidence:
    if not observations:
        return MotionEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

    times = np.asarray([obs.timestamp for obs in observations], dtype=float)
    xs = np.asarray([obs.cx for obs in observations], dtype=float)
    ys = np.asarray([obs.cy for obs in observations], dtype=float)
    log_areas = np.asarray([obs.log_area for obs in observations], dtype=float)

    times = times - times[0]
    duration = float(times[-1]) if len(times) > 1 else 0.0
    vx, _ = _linear_fit(times, xs)
    vy, _ = _linear_fit(times, ys)
    expansion_rate, fit_quality = _linear_fit(times, log_areas)
    consistency = _trend_consistency(log_areas, expansion_rate)
    speed = math.hypot(vx, vy)

    enough_history = len(observations) >= config.min_samples and duration >= config.min_duration
    if enough_history:
        magnitude = min(1.0, abs(expansion_rate) / config.radial_rate_threshold)
        radial_confidence = max(0.0, min(1.0, magnitude * consistency * fit_quality))
    else:
        radial_confidence = 0.0

    approach_confidence = radial_confidence if expansion_rate > 0.0 else 0.0
    recede_confidence = radial_confidence if expansion_rate < 0.0 else 0.0

    return MotionEvidence(
        vx=vx,
        vy=vy,
        speed=speed,
        expansion_rate=expansion_rate,
        trend_consistency=consistency,
        fit_quality=fit_quality,
        sample_count=len(observations),
        duration=duration,
        approach_confidence=approach_confidence,
        recede_confidence=recede_confidence,
    )
