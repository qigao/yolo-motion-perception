from __future__ import annotations

import json
import math

from .motion import MotionConfig
from .pipeline import MotionPipeline, TrackMotionResult
from .types import TrackObservation


def _sequence(
    *,
    vx: float = 0.0,
    expansion_rate: float = 0.0,
    scales: list[float] | None = None,
    count: int = 8,
    dt: float = 0.1,
) -> list[TrackObservation]:
    observations: list[TrackObservation] = []
    for i in range(count):
        t = i * dt
        if scales is None:
            side_scale = math.exp(expansion_rate * t / 2.0)
        else:
            side_scale = math.sqrt(scales[i])
        observations.append(
            TrackObservation(
                track_id=1,
                timestamp=t,
                class_id=0,
                confidence=0.95,
                cx=0.4 + vx * t,
                cy=0.5,
                width=0.1 * side_scale,
                height=0.2 * side_scale,
            )
        )
    return observations


def _run_sequence(observations: list[TrackObservation], config: MotionConfig) -> TrackMotionResult:
    pipeline = MotionPipeline(config)
    result = None
    for observation in observations:
        result = pipeline.update(observation)
    if result is None:
        raise RuntimeError("synthetic sequence did not produce enough history")
    return result


def _summarize(result: TrackMotionResult) -> dict[str, object]:
    return {
        "lateral": result.state.lateral.value,
        "radial": result.state.radial.value,
        "vx": round(result.evidence.vx, 6),
        "vy": round(result.evidence.vy, 6),
        "speed": round(result.evidence.speed, 6),
        "expansion_rate": round(result.evidence.expansion_rate, 6),
        "trend_consistency": round(result.evidence.trend_consistency, 6),
        "fit_quality": round(result.evidence.fit_quality, 6),
        "approach_confidence": round(result.evidence.approach_confidence, 6),
        "recede_confidence": round(result.evidence.recede_confidence, 6),
    }


def run_benchmark() -> dict[str, dict[str, object]]:
    config = MotionConfig()
    scenarios = {
        "stationary": _sequence(),
        "lateral": _sequence(vx=0.08),
        "approaching": _sequence(expansion_rate=0.5),
        "receding": _sequence(expansion_rate=-0.5),
        "scale_spike": _sequence(scales=[1.0, 1.0, 1.0, 1.8, 1.0, 1.0, 1.0, 1.0]),
    }
    return {
        name: _summarize(_run_sequence(observations, config))
        for name, observations in scenarios.items()
    }


def main() -> int:
    summary = run_benchmark()
    print(json.dumps(summary, indent=2, sort_keys=True))

    expected = {
        "stationary": ("stationary", "stable"),
        "lateral": ("moving", "stable"),
        "approaching": ("stationary", "approaching"),
        "receding": ("stationary", "receding"),
        "scale_spike": ("stationary", "stable"),
    }
    failures = []
    for name, (lateral, radial) in expected.items():
        actual = summary[name]
        if (actual["lateral"], actual["radial"]) != (lateral, radial):
            failures.append(
                f"{name}: expected {lateral}/{radial}, got "
                f"{actual['lateral']}/{actual['radial']}"
            )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0
