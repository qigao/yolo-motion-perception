import json
from pathlib import Path

import pytest

from yolo_motion.evaluation import (
    EvaluationError,
    evaluate_predictions,
    load_annotations,
    load_predictions,
    main,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_annotations_validates_intervals(tmp_path: Path):
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        """
video: sample.mp4
intervals:
  - track_id: 7
    start: 1.0
    end: 2.0
    lateral: moving
    radial: approaching
""".strip(),
        encoding="utf-8",
    )

    dataset = load_annotations(labels)

    assert dataset.video == "sample.mp4"
    assert len(dataset.intervals) == 1
    interval = dataset.intervals[0]
    assert interval.track_id == 7
    assert interval.start == 1.0
    assert interval.end == 2.0
    assert interval.lateral == "moving"
    assert interval.radial == "approaching"


def test_load_annotations_rejects_invalid_state(tmp_path: Path):
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        """
intervals:
  - track_id: 1
    start: 0.0
    end: 1.0
    radial: closing-fast
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="radial"):
        load_annotations(labels)


def test_evaluate_predictions_scores_only_labeled_dimensions(tmp_path: Path):
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {"track_id": 7, "timestamp": 1.1, "lateral": "moving", "radial": "stable"},
            {"track_id": 7, "timestamp": 1.2, "lateral": "moving", "radial": "approaching"},
            {"track_id": 7, "timestamp": 1.3, "lateral": "stationary", "radial": "approaching"},
            {"track_id": 99, "timestamp": 1.2, "lateral": "moving", "radial": "approaching"},
        ],
    )
    predictions = load_predictions(predictions_path)
    annotations = load_annotations_from_text(
        tmp_path,
        """
intervals:
  - track_id: 7
    start: 1.0
    end: 1.4
    lateral: moving
    radial: approaching
""",
    )

    report = evaluate_predictions(predictions, annotations)

    assert report["lateral"]["samples"] == 3
    assert report["lateral"]["correct"] == 2
    assert report["lateral"]["accuracy"] == pytest.approx(2 / 3)
    assert report["radial"]["samples"] == 3
    assert report["radial"]["correct"] == 2
    assert report["radial"]["accuracy"] == pytest.approx(2 / 3)
    assert report["intervals"][0]["first_correct_latency"]["lateral"] == pytest.approx(0.1)
    assert report["intervals"][0]["first_correct_latency"]["radial"] == pytest.approx(0.2)


def test_omitted_dimension_is_not_scored(tmp_path: Path):
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [{"track_id": 3, "timestamp": 0.5, "lateral": "moving", "radial": "receding"}],
    )
    annotations = load_annotations_from_text(
        tmp_path,
        """
intervals:
  - track_id: 3
    start: 0.0
    end: 1.0
    radial: receding
""",
    )

    report = evaluate_predictions(load_predictions(predictions_path), annotations)

    assert report["lateral"]["samples"] == 0
    assert report["lateral"]["accuracy"] is None
    assert report["radial"]["accuracy"] == 1.0


def test_interval_without_matching_prediction_reports_no_latency(tmp_path: Path):
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [{"track_id": 4, "timestamp": 0.2, "lateral": "stationary", "radial": "stable"}],
    )
    annotations = load_annotations_from_text(
        tmp_path,
        """
intervals:
  - track_id: 4
    start: 1.0
    end: 2.0
    radial: approaching
""",
    )

    report = evaluate_predictions(load_predictions(predictions_path), annotations)

    assert report["radial"]["samples"] == 0
    assert report["intervals"][0]["first_correct_latency"]["radial"] is None


def load_annotations_from_text(tmp_path: Path, text: str):
    path = tmp_path / "labels.yaml"
    path.write_text(text.strip(), encoding="utf-8")
    return load_annotations(path)


def test_main_prints_machine_readable_report(tmp_path: Path, capsys):
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [{"track_id": 5, "timestamp": 0.5, "lateral": "stationary", "radial": "approaching"}],
    )
    labels_path = tmp_path / "labels.yaml"
    labels_path.write_text(
        """
intervals:
  - track_id: 5
    start: 0.0
    end: 1.0
    radial: approaching
""".strip(),
        encoding="utf-8",
    )

    assert main(["--predictions", str(predictions_path), "--annotations", str(labels_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["radial"]["accuracy"] == 1.0
