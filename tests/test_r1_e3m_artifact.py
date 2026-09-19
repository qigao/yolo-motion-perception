import hashlib
import json
from pathlib import Path

import pytest

from neural_state_machine.r1_e3m_artifact import (
    ArtifactInvalid,
    artifact_root_digest,
    load_track_artifact,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_artifact(
    root: Path,
    *,
    cross_component: bool = False,
    unsorted_tracks: bool = False,
    semantic_file: bool = False,
) -> str:
    root.mkdir()
    source_manifest = {
        "schema": "r1-e3a-source-video-manifest-v1",
        "science_head_sha": "9" * 40,
        "videos": [
            {
                "source_video_id": "train-a",
                "sha256": "1" * 64,
                "split": "train",
                "fps": 30.0,
                "frame_count": 300,
                "duration_seconds": 10.0,
                "width": 640,
                "height": 480,
                "source_window_component_id": "component-a",
            },
            {
                "source_video_id": "eval-a",
                "sha256": "2" * 64,
                "split": "eval",
                "fps": 30.0,
                "frame_count": 300,
                "duration_seconds": 10.0,
                "width": 640,
                "height": 480,
                "source_window_component_id": (
                    "component-a" if cross_component else "component-b"
                ),
            },
        ],
    }
    source_path = root / "source-manifest.json"
    _write_json(source_path, source_manifest)

    rows = [
        {
            "video_id": "train-a",
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "track_id": 1,
            "class_id": 0,
            "confidence": 0.9,
            "x1": 10.0,
            "y1": 20.0,
            "x2": 110.0,
            "y2": 220.0,
            "cx_norm": 0.09375,
            "cy_norm": 0.25,
            "w_norm": 0.15625,
            "h_norm": 0.4166666667,
        },
        {
            "video_id": "train-a",
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "track_id": 2,
            "class_id": 24,
            "confidence": 0.8,
            "x1": 200.0,
            "y1": 100.0,
            "x2": 260.0,
            "y2": 180.0,
            "cx_norm": 0.359375,
            "cy_norm": 0.2916666667,
            "w_norm": 0.09375,
            "h_norm": 0.1666666667,
        },
    ]
    if unsorted_tracks:
        rows.reverse()
    tracks_path = root / "tracks.jsonl"
    tracks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    provenance_path = root / "extraction-provenance.json"
    _write_json(
        provenance_path,
        {
            "schema": "r1-e3m-extraction-provenance-v1",
            "constraints": {
                "no_semantic_labels": True,
                "no_human_event_boundaries": True,
            },
        },
    )

    if semantic_file:
        _write_json(root / "episodes.json", [{"label": "pick_up"}])

    files = {}
    for name in (
        "source-manifest.json",
        "tracks.jsonl",
        "extraction-provenance.json",
    ):
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    artifact_manifest = {
        "schema": "r1-e3m-track-artifact-v1",
        "source_manifest_sha256": files["source-manifest.json"],
        "files": files,
    }
    _write_json(root / "track-artifact-manifest.json", artifact_manifest)
    return files["source-manifest.json"]


def test_valid_minimal_artifact_has_stable_root_digest(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    source_sha = _write_artifact(root)

    artifact = load_track_artifact(root, expected_source_manifest_sha256=source_sha)

    assert len(artifact.videos) == 2
    assert len(artifact.tracks) == 2
    first = artifact_root_digest(
        root,
        expected_source_manifest_sha256=source_sha,
    )
    second = artifact_root_digest(
        root,
        expected_source_manifest_sha256=source_sha,
    )
    assert first == second
    assert len(first) == 64


def test_wrong_source_manifest_binding_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_artifact(root)

    with pytest.raises(ArtifactInvalid, match="source manifest sha256 mismatch"):
        load_track_artifact(
            root,
            expected_source_manifest_sha256="f" * 64,
        )


def test_source_window_component_cannot_cross_splits(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    source_sha = _write_artifact(root, cross_component=True)

    with pytest.raises(ArtifactInvalid, match="source-window component overlap"):
        load_track_artifact(root, expected_source_manifest_sha256=source_sha)


def test_track_rows_must_be_canonically_sorted(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    source_sha = _write_artifact(root, unsorted_tracks=True)

    with pytest.raises(ArtifactInvalid, match="sorted"):
        load_track_artifact(root, expected_source_manifest_sha256=source_sha)


def test_semantic_episode_payload_is_forbidden(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    source_sha = _write_artifact(root, semantic_file=True)

    with pytest.raises(ArtifactInvalid, match="semantic"):
        load_track_artifact(root, expected_source_manifest_sha256=source_sha)
