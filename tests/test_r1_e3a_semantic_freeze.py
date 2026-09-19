import csv
import importlib.util
import json
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


def registration_manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema": SEMANTIC_FREEZE.SOURCE_SCHEMA,
        "handbook_sha256": "a" * 64,
        "science_head_sha": "b" * 40,
        "videos": [],
    }
    videos = manifest["videos"]
    assert isinstance(videos, list)
    for video_id, split, component, camera, digest_char in (
        ("t1", "train", "train-a", "G101", "1"),
        ("t2", "train", "train-b", "G102", "2"),
        ("e1", "eval", "eval-a", "G201", "3"),
        ("e2", "eval", "eval-b", "G202", "4"),
    ):
        videos.append(
            {
                "source_video_id": video_id,
                "camera": camera,
                "split": split,
                "source_window_component_id": component,
                "anchor_capture_group_id": component,
                "sha256": digest_char * 64,
                "frame_count": 9000,
            }
        )
    return manifest


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEMANTIC_FREEZE.REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_completion(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SEMANTIC_FREEZE.REVIEW_COMPLETION_COLUMNS,
        )
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


def completion_row(
    video_id: str,
    *,
    count: int = 1,
    complete: bool = True,
) -> dict[str, str]:
    return {
        "source_video_id": video_id,
        "full_range_reviewed": "true" if complete else "false",
        "reviewed_candidate_count": str(count),
        "reviewer": "reviewer-b",
    }


def registration_rows(
    *,
    use_two_videos_and_components: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    serial = 0
    for split, minimum in (("train", 20), ("eval", 10)):
        for label in SEMANTIC_FREEZE.REGISTERED_LABELS:
            for index in range(minimum):
                serial += 1
                if split == "train":
                    video = (
                        ("t1", "train-a")
                        if not use_two_videos_and_components or index % 2 == 0
                        else ("t2", "train-b")
                    )
                else:
                    video = (
                        ("e1", "eval-a")
                        if not use_two_videos_and_components or index % 2 == 0
                        else ("e2", "eval-b")
                    )
                rows.append(
                    {
                        **annotation_row(),
                        "episode_id": f"episode-{serial}",
                        "physical_event_id": f"physical-{serial}",
                        "split": split,
                        "source_window_component_id": video[1],
                        "registered_video_id": video[0],
                        "context_video_ids": "",
                        "label": label,
                    }
                )
    return rows


def test_valid_reviewed_annotation_reports_counts(tmp_path: Path) -> None:
    path = tmp_path / "annotations.csv"
    write_csv(path, [annotation_row()])

    summary = SEMANTIC_FREEZE.validate_annotations(source_manifest(), path)

    assert summary["accepted_counts"]["train"]["pick_up"] == 1
    assert summary["reviewed_candidate_counts_by_video"] == {"v1": 1, "v2": 1}
    assert summary["accepted_video_counts"]["train"] == 1
    assert summary["accepted_component_counts"]["train"] == 1
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


def test_review_completion_requires_full_range_for_every_source_video(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations.csv"
    completion = tmp_path / "completion.csv"
    write_csv(annotations, [annotation_row()])
    summary = SEMANTIC_FREEZE.validate_annotations(source_manifest(), annotations)
    write_completion(
        completion,
        [
            completion_row("v1"),
            completion_row("v2", complete=False),
        ],
    )

    with pytest.raises(ValueError, match="full-range review"):
        SEMANTIC_FREEZE.validate_review_completion(
            source_manifest(),
            summary,
            completion,
        )


def test_review_completion_counts_registered_and_context_views(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations.csv"
    completion = tmp_path / "completion.csv"
    write_csv(annotations, [annotation_row()])
    summary = SEMANTIC_FREEZE.validate_annotations(source_manifest(), annotations)
    write_completion(
        completion,
        [
            completion_row("v1"),
            completion_row("v2"),
        ],
    )

    evidence = SEMANTIC_FREEZE.validate_review_completion(
        source_manifest(),
        summary,
        completion,
    )

    assert evidence["complete_source_videos"] == 2
    assert evidence["reviewed_candidate_counts_by_video"] == {"v1": 1, "v2": 1}


def test_review_completion_rejects_candidate_count_mismatch(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    completion = tmp_path / "completion.csv"
    write_csv(annotations, [annotation_row()])
    summary = SEMANTIC_FREEZE.validate_annotations(source_manifest(), annotations)
    write_completion(
        completion,
        [
            completion_row("v1"),
            completion_row("v2", count=0),
        ],
    )

    with pytest.raises(ValueError, match="candidate count mismatch"):
        SEMANTIC_FREEZE.validate_review_completion(
            source_manifest(),
            summary,
            completion,
        )


def test_size_gate_requires_two_accepted_videos_and_components_per_split(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotations.csv"
    write_csv(path, registration_rows(use_two_videos_and_components=False))

    summary = SEMANTIC_FREEZE.validate_annotations(registration_manifest(), path)

    assert summary["accepted_counts"]["train"]["approach"] == 20
    assert summary["accepted_counts"]["eval"]["approach"] == 10
    assert summary["accepted_video_counts"] == {"train": 1, "eval": 1}
    assert summary["accepted_component_counts"] == {"train": 1, "eval": 1}
    assert summary["size_gate_pass"] is False

    write_csv(path, registration_rows(use_two_videos_and_components=True))
    summary = SEMANTIC_FREEZE.validate_annotations(registration_manifest(), path)

    assert summary["accepted_video_counts"] == {"train": 2, "eval": 2}
    assert summary["accepted_component_counts"] == {"train": 2, "eval": 2}
    assert summary["size_gate_pass"] is True


def test_freeze_binds_sampling_protocol_and_completion_evidence(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "source.json"
    annotations = tmp_path / "annotations.csv"
    completion = tmp_path / "completion.csv"
    sampling = tmp_path / "sampling.md"
    manifest_path.write_text(json.dumps(source_manifest()) + "\n", encoding="utf-8")
    write_csv(annotations, [annotation_row()])
    write_completion(
        completion,
        [
            completion_row("v1"),
            completion_row("v2"),
        ],
    )
    sampling.write_text("frozen sampling protocol\n", encoding="utf-8")

    freeze = SEMANTIC_FREEZE.build_freeze(
        manifest_path,
        annotations,
        completion,
        sampling,
    )

    assert freeze["sampling_protocol_sha256"] == SEMANTIC_FREEZE.sha256_file(sampling)
    assert freeze["review_completion_csv_sha256"] == SEMANTIC_FREEZE.sha256_file(
        completion
    )
    assert freeze["constraints"]["all_activated_source_videos_fully_reviewed"] is True
