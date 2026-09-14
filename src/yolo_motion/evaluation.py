from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .types import LateralState, RadialState


class EvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class AnnotationInterval:
    track_id: int
    start: float
    end: float
    lateral: str | None = None
    radial: str | None = None


@dataclass(frozen=True)
class AnnotationSet:
    video: str | None
    intervals: tuple[AnnotationInterval, ...]


@dataclass(frozen=True)
class Prediction:
    track_id: int
    timestamp: float
    lateral: str
    radial: str


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{context} must be a mapping")
    return value


def _validate_state(value: Any, *, dimension: str, allowed: set[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        raise EvaluationError(f"invalid {dimension} state: {value!r}")
    return value


def load_annotations(path: str | Path) -> AnnotationSet:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = _require_mapping(raw, context="annotation root")
    raw_intervals = root.get("intervals", [])
    if not isinstance(raw_intervals, list):
        raise EvaluationError("intervals must be a list")

    lateral_states = {state.value for state in LateralState}
    radial_states = {state.value for state in RadialState}
    intervals: list[AnnotationInterval] = []
    for index, raw_interval in enumerate(raw_intervals):
        row = _require_mapping(raw_interval, context=f"intervals[{index}]")
        try:
            track_id = int(row["track_id"])
            start = float(row["start"])
            end = float(row["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(
                f"intervals[{index}] requires numeric track_id/start/end"
            ) from exc
        if end <= start:
            raise EvaluationError(f"intervals[{index}] end must be greater than start")
        lateral = _validate_state(
            row.get("lateral"), dimension="lateral", allowed=lateral_states
        )
        radial = _validate_state(
            row.get("radial"), dimension="radial", allowed=radial_states
        )
        if lateral is None and radial is None:
            raise EvaluationError(f"intervals[{index}] must label lateral and/or radial")
        intervals.append(
            AnnotationInterval(
                track_id=track_id,
                start=start,
                end=end,
                lateral=lateral,
                radial=radial,
            )
        )

    video = root.get("video")
    if video is not None and not isinstance(video, str):
        raise EvaluationError("video must be a string when provided")
    return AnnotationSet(video=video, intervals=tuple(intervals))


def load_predictions(path: str | Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"invalid JSON at line {line_number}") from exc
            row = _require_mapping(row, context=f"prediction line {line_number}")
            try:
                prediction = Prediction(
                    track_id=int(row["track_id"]),
                    timestamp=float(row["timestamp"]),
                    lateral=str(row["lateral"]),
                    radial=str(row["radial"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationError(
                    f"prediction line {line_number} requires track_id/timestamp/lateral/radial"
                ) from exc
            predictions.append(prediction)
    return predictions


def _empty_metric() -> dict[str, Any]:
    return {"samples": 0, "correct": 0, "accuracy": None, "confusion": {}}


def _score(
    metric: dict[str, Any],
    *,
    expected: str,
    actual: str,
) -> None:
    metric["samples"] += 1
    if actual == expected:
        metric["correct"] += 1
    expected_row = metric["confusion"].setdefault(expected, {})
    expected_row[actual] = expected_row.get(actual, 0) + 1


def _finish_metric(metric: dict[str, Any]) -> None:
    samples = metric["samples"]
    metric["accuracy"] = metric["correct"] / samples if samples else None


def evaluate_predictions(
    predictions: list[Prediction], annotations: AnnotationSet
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "lateral": _empty_metric(),
        "radial": _empty_metric(),
        "intervals": [],
    }

    by_track: dict[int, list[Prediction]] = {}
    for prediction in predictions:
        by_track.setdefault(prediction.track_id, []).append(prediction)
    for track_predictions in by_track.values():
        track_predictions.sort(key=lambda item: item.timestamp)

    for interval in annotations.intervals:
        selected = [
            prediction
            for prediction in by_track.get(interval.track_id, [])
            if interval.start <= prediction.timestamp <= interval.end
        ]
        first_correct: dict[str, float | None] = {"lateral": None, "radial": None}
        for prediction in selected:
            if interval.lateral is not None:
                _score(
                    report["lateral"],
                    expected=interval.lateral,
                    actual=prediction.lateral,
                )
                if (
                    first_correct["lateral"] is None
                    and prediction.lateral == interval.lateral
                ):
                    first_correct["lateral"] = prediction.timestamp - interval.start
            if interval.radial is not None:
                _score(
                    report["radial"],
                    expected=interval.radial,
                    actual=prediction.radial,
                )
                if first_correct["radial"] is None and prediction.radial == interval.radial:
                    first_correct["radial"] = prediction.timestamp - interval.start

        report["intervals"].append(
            {
                "track_id": interval.track_id,
                "start": interval.start,
                "end": interval.end,
                "samples": len(selected),
                "first_correct_latency": first_correct,
            }
        )

    _finish_metric(report["lateral"])
    _finish_metric(report["radial"])
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-jsonl",
        description="Score yolo-motion JSONL output against labeled time intervals.",
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_predictions(
        load_predictions(args.predictions), load_annotations(args.annotations)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
