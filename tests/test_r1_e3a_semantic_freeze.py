import csv
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "r1_e3a_semantic_freeze.py"
SPEC = importlib.util.spec_from_file_location("r1_e3a_semantic_freeze", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
SEMANTIC_FREEZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEMANTIC_FREEZE)


def source_manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema": SEMANTIC_FREEZE.SOURCE_SCHEMA,
        "handbook_sha256": "a" * 64,
        "science_head_sha": "b" * 40,
        "videos": [],
    }
    videos = manifest["videos"]
    assert isinstance(videos, list)
    for video_id, camera in (("v1", "G299"), ("v2", "G330")):
        videos.append(
            {
                "source_video_id": video_id,
                "camera": camera,
                "split": "train",
                "source_window_component_id": "component-a",
                "anchor_capture_group_id": "anchor-a",
                "sha256": ("1" if video_id == "v1" else "2") * 64,
                "frame_count": 9000,
            }
        )
    return manifest


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEMANTIC_FREEZE.REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def annotation_row(**changes: str) -> dict[str, str]:
    value = {
        "episode_id": "e1",
        "physical_event_id": "p1",
        "split": "train",
        "source_window_component_id": "component-a",
        "registered_video_id": "v1",
        "context_video_ids": "v2",
        "start_frame_inclusive": "100",
        "end_frame_exclusive": "400",
        "label": "pick_up",
        "actor_description": "person in jacket",
        "target_description": "backpack",
        "target_category_hint": "backpack",
        "annotation_revision": "v1",
        "annotator": "annotator-a",
        "review_status": "accepted",
        "reviewer": "reviewer-b",
        "notes": "",
    }
    value.update(changes)
    return value


def test_valid_reviewed_annotation_reports_counts(tmp_path: Path) -> None:
    path = tmp_path / "annotations.csv"
    write_csv(path, [annotation_row()])

    summary = SEMANTIC_FREEZE.validate_annotations(source_manifest(), path)

    assert summary["accepted_counts"]["train"]["pick_up"] == 1
    assert summary["size_gate_pass"] is False
    assert summary["multiview_physical_event_deduplication_asserted"] is True


def test_duplicate_physical_event_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "annotations.csv"
    write_csv(path, [annotation_row(), annotation_row(episode_id="e2")])

    with pytest.raises(ValueError, match="duplicate physical_event_id"):
        SEMANTIC_FREEZE.validate_annotations(source_manifest(), path)


def test_unresolved_review_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "annotations.csv"
    write_csv(path, [annotation_row(review_status="needs_resolution")])

    with pytest.raises(ValueError, match="unresolved annotation"):
        SEMANTIC_FREEZE.validate_annotations(source_manifest(), path)


def test_context_cannot_cross_component(tmp_path: Path) -> None:
    manifest = source_manifest()
    videos = manifest["videos"]
    assert isinstance(videos, list)
    assert isinstance(videos[1], dict)
    videos[1]["source_window_component_id"] = "other"

    path = tmp_path / "annotations.csv"
    write_csv(path, [annotation_row()])

    with pytest.raises(ValueError, match="context video crosses component"):
        SEMANTIC_FREEZE.validate_annotations(manifest, path)
