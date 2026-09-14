from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_motion.track_summary import (
    TrackSummaryError,
    load_track_records,
    main,
    render_annotation_draft,
    summarize_tracks,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_summarize_tracks_groups_sorts_and_counts_states(tmp_path: Path) -> None:
    path = tmp_path / "motion.jsonl"
    _write_jsonl(
        path,
        [
            {"track_id": 1, "timestamp": 2.0, "class_id": 0, "lateral": "moving", "radial": "approaching"},
            {"track_id": 2, "timestamp": 1.5, "class_id": 0, "lateral": "stationary", "radial": "stable"},
            {"track_id": 1, "timestamp": 1.0, "class_id": 0, "lateral": "stationary", "radial": "stable"},
            {"track_id": 1, "timestamp": 3.0, "class_id": 0, "lateral": "moving", "radial": "approaching"},
        ],
    )

    summaries = summarize_tracks(load_track_records(path))

    assert [summary.track_id for summary in summaries] == [1, 2]
    first = summaries[0]
    assert first.class_id == 0
    assert first.start == 1.0
    assert first.end == 3.0
    assert first.samples == 3
    assert first.lateral_counts == {"moving": 2, "stationary": 1}
    assert first.radial_counts == {"approaching": 2, "stable": 1}


def test_render_annotation_draft_keeps_labels_for_human_review(tmp_path: Path) -> None:
    path = tmp_path / "motion.jsonl"
    _write_jsonl(
        path,
        [
            {"track_id": 7, "timestamp": 0.42, "class_id": 0, "lateral": "moving", "radial": "approaching"},
            {"track_id": 7, "timestamp": 7.91, "class_id": 0, "lateral": "moving", "radial": "approaching"},
        ],
    )
    summary = summarize_tracks(load_track_records(path))

    draft = render_annotation_draft("benchmarks/videos/approaching.mp4", summary)

    assert "video: benchmarks/videos/approaching.mp4" in draft
    assert "track_id: 7" in draft
    assert "start: 0.42" in draft
    assert "end: 7.91" in draft
    assert "# lateral:" in draft
    assert "# radial:" in draft
    assert "approaching" not in draft.split("# radial:", 1)[1]


def test_empty_prediction_file_returns_no_tracks(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert summarize_tracks(load_track_records(path)) == []


def test_invalid_json_raises_track_summary_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(TrackSummaryError, match="invalid JSON"):
        load_track_records(path)


def test_missing_required_fields_raise_track_summary_error(tmp_path: Path) -> None:
    path = tmp_path / "bad-row.jsonl"
    _write_jsonl(path, [{"track_id": 1, "timestamp": 1.0}])

    with pytest.raises(TrackSummaryError, match="requires"):
        load_track_records(path)


def test_cli_prints_summary_and_writes_annotation_draft(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    predictions = tmp_path / "approaching.jsonl"
    draft = tmp_path / "approaching.yaml"
    _write_jsonl(
        predictions,
        [
            {"track_id": 1, "timestamp": 1.0, "class_id": 0, "lateral": "stationary", "radial": "approaching"},
            {"track_id": 1, "timestamp": 2.0, "class_id": 0, "lateral": "stationary", "radial": "approaching"},
        ],
    )

    assert main(["--predictions", str(predictions), "--annotation-draft", str(draft)]) == 0

    output = capsys.readouterr().out
    assert "track_id" in output
    assert "approaching" in output
    text = draft.read_text(encoding="utf-8")
    assert "video: benchmarks/videos/approaching.mp4" in text
    assert "track_id: 1" in text
