import csv
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "r1_e3a_review_csv_to_json.py"
SPEC = importlib.util.spec_from_file_location("r1_e3a_review_csv_to_json", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def manifest() -> dict:
    return {
        "science_head_sha": "9" * 40,
        "videos": [
            {
                "source_video_id": "v1",
                "split": "train",
                "source_window_component_id": "component-a",
                "frame_count": 300,
            },
            {
                "source_video_id": "v2",
                "split": "train",
                "source_window_component_id": "component-a",
                "frame_count": 300,
            },
        ],
    }


def test_convert_human_csv_to_canonical_review(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    completion = tmp_path / "completion.csv"

    write_csv(
        annotations,
        CONVERTER.ANNOTATION_COLUMNS,
        [
            {
                "episode_id": "e1",
                "physical_event_id": "p1",
                "split": "train",
                "source_window_component_id": "component-a",
                "registered_video_id": "v1",
                "context_video_ids": "v2",
                "start_frame_inclusive": "10",
                "end_frame_exclusive": "50",
                "label": "pick_up",
                "actor_description": "person in dark shirt",
                "target_description": "bag",
                "target_category_hint": "handbag",
                "annotation_revision": "1",
                "annotator": "annotator-a",
                "review_status": "accepted",
                "reviewer": "reviewer-b",
                "notes": "",
            }
        ],
    )
    write_csv(
        completion,
        CONVERTER.COMPLETION_COLUMNS,
        [
            {
                "source_video_id": "v1",
                "full_range_reviewed": "true",
                "reviewed_candidate_count": "1",
                "reviewer": "reviewer-b",
            },
            {
                "source_video_id": "v2",
                "full_range_reviewed": "true",
                "reviewed_candidate_count": "1",
                "reviewer": "reviewer-b",
            },
        ],
    )

    result = CONVERTER.convert_review(
        annotations,
        completion,
        manifest(),
        source_manifest_sha256="1" * 64,
        handbook_sha256="2" * 64,
        sampling_protocol_sha256="3" * 64,
    )

    assert result["schema"] == "r1-e3a-semantic-annotation-review-v1"
    assert result["status"] == "reviewed"
    assert result["source_manifest_sha256"] == "1" * 64
    assert result["handbook_sha256"] == "2" * 64
    assert result["sampling_protocol_sha256"] == "3" * 64
    assert result["video_reviews"][0]["candidate_count"] == 1
    assert result["records"][0]["context_video_ids"] == ["v2"]
    assert result["records"][0]["start_frame_inclusive"] == 10
    assert result["records"][0]["annotation_revision"] == 1


def test_converter_rejects_split_mismatch(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    completion = tmp_path / "completion.csv"
    row = {column: "" for column in CONVERTER.ANNOTATION_COLUMNS}
    row.update(
        {
            "episode_id": "e1",
            "physical_event_id": "p1",
            "split": "eval",
            "source_window_component_id": "component-a",
            "registered_video_id": "v1",
            "start_frame_inclusive": "10",
            "end_frame_exclusive": "50",
            "label": "approach",
            "annotation_revision": "1",
            "annotator": "a",
            "review_status": "accepted",
            "reviewer": "b",
        }
    )
    write_csv(annotations, CONVERTER.ANNOTATION_COLUMNS, [row])
    write_csv(
        completion,
        CONVERTER.COMPLETION_COLUMNS,
        [
            {
                "source_video_id": "v1",
                "full_range_reviewed": "true",
                "reviewed_candidate_count": "1",
                "reviewer": "b",
            },
            {
                "source_video_id": "v2",
                "full_range_reviewed": "true",
                "reviewed_candidate_count": "0",
                "reviewer": "b",
            },
        ],
    )

    with pytest.raises(ValueError, match="split mismatch"):
        CONVERTER.convert_review(
            annotations,
            completion,
            manifest(),
            source_manifest_sha256="1" * 64,
            handbook_sha256="2" * 64,
            sampling_protocol_sha256="3" * 64,
        )


def test_converter_requires_completion_for_every_manifest_video(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    completion = tmp_path / "completion.csv"
    write_csv(annotations, CONVERTER.ANNOTATION_COLUMNS, [])
    write_csv(
        completion,
        CONVERTER.COMPLETION_COLUMNS,
        [
            {
                "source_video_id": "v1",
                "full_range_reviewed": "true",
                "reviewed_candidate_count": "0",
                "reviewer": "b",
            }
        ],
    )

    with pytest.raises(ValueError, match="completion coverage mismatch"):
        CONVERTER.convert_review(
            annotations,
            completion,
            manifest(),
            source_manifest_sha256="1" * 64,
            handbook_sha256="2" * 64,
            sampling_protocol_sha256="3" * 64,
        )
