from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _api():
    from neural_state_machine.r1_e3_artifact import (
        ArtifactInvalid,
        artifact_root_digest,
        load_artifact,
        validate_artifact_structure,
    )

    return ArtifactInvalid, artifact_root_digest, load_artifact, validate_artifact_structure


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_signed_json(root: Path, name: str, payload: object) -> str:
    data = _canonical_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    (root / name).write_bytes(data)
    (root / f"{name.removesuffix('.json')}.sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )
    return digest


def _write_signed_jsonl(root: Path, name: str, rows: list[dict[str, object]]) -> str:
    data = b"".join(_canonical_bytes(row) for row in rows)
    digest = hashlib.sha256(data).hexdigest()
    (root / name).write_bytes(data)
    (root / f"{name.removesuffix('.jsonl')}.sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )
    return digest


def _minimal_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    root.mkdir()

    videos = [
        {
            "video_id": "train-video",
            "sha256": "1" * 64,
            "split": "training",
            "fps": 30.0,
            "frame_count": 300,
            "width": 1280,
            "height": 720,
        },
        {
            "video_id": "eval-video",
            "sha256": "2" * 64,
            "split": "evaluation",
            "fps": 30.0,
            "frame_count": 300,
            "width": 1280,
            "height": 720,
        },
    ]
    episodes = [
        {
            "episode_id": "train-approach-001",
            "video_id": "train-video",
            "start_time_seconds": 0.0,
            "end_time_seconds": 1.0,
            "label": "approach",
            "actor_track_id": 10,
            "target_track_id": 20,
            "annotation_revision": "r1",
            "reviewer": "reviewer-a",
        },
        {
            "episode_id": "eval-pass-by-001",
            "video_id": "eval-video",
            "start_time_seconds": 0.0,
            "end_time_seconds": 1.0,
            "label": "pass_by",
            "actor_track_id": 30,
            "target_track_id": 40,
            "annotation_revision": "r1",
            "reviewer": "reviewer-b",
        },
    ]
    tracks = [
        {
            "video_id": "eval-video",
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "track_id": 30,
            "class_id": 0,
            "confidence": 0.90,
            "x1": 100.0,
            "y1": 100.0,
            "x2": 200.0,
            "y2": 300.0,
            "cx_norm": 0.1171875,
            "cy_norm": 0.2777777777777778,
            "w_norm": 0.078125,
            "h_norm": 0.2777777777777778,
        },
        {
            "video_id": "eval-video",
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "track_id": 40,
            "class_id": 1,
            "confidence": 0.88,
            "x1": 500.0,
            "y1": 200.0,
            "x2": 700.0,
            "y2": 500.0,
            "cx_norm": 0.46875,
            "cy_norm": 0.4861111111111111,
            "w_norm": 0.15625,
            "h_norm": 0.4166666666666667,
        },
        {
            "video_id": "train-video",
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "track_id": 10,
            "class_id": 0,
            "confidence": 0.91,
            "x1": 120.0,
            "y1": 90.0,
            "x2": 220.0,
            "y2": 290.0,
            "cx_norm": 0.1328125,
            "cy_norm": 0.2638888888888889,
            "w_norm": 0.078125,
            "h_norm": 0.2777777777777778,
        },
        {
            "video_id": "train-video",
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "track_id": 20,
            "class_id": 1,
            "confidence": 0.89,
            "x1": 520.0,
            "y1": 210.0,
            "x2": 720.0,
            "y2": 510.0,
            "cx_norm": 0.484375,
            "cy_norm": 0.5,
            "w_norm": 0.15625,
            "h_norm": 0.4166666666666667,
        },
    ]
    provenance = {
        "schema": "r1-e3-extraction-provenance-v1",
        "extractor_commit": "a" * 40,
        "yolo": {
            "package_version": "fixed-test-version",
            "weights_sha256": "3" * 64,
        },
        "botsort": {
            "config_sha256": "4" * 64,
        },
        "runtime": {
            "python": "3.12.14",
            "lock_sha256": "5" * 64,
        },
    }

    file_digests = {
        "videos.json": _write_signed_json(root, "videos.json", videos),
        "episodes.json": _write_signed_json(root, "episodes.json", episodes),
        "tracks.jsonl": _write_signed_jsonl(root, "tracks.jsonl", tracks),
        "extraction-provenance.json": _write_signed_json(
            root,
            "extraction-provenance.json",
            provenance,
        ),
    }
    _write_signed_json(
        root,
        "artifact-manifest.json",
        {
            "schema": "r1-e3-artifact-v1",
            "files": file_digests,
        },
    )
    return root


def test_minimal_artifact_loads_as_frozen_records(tmp_path: Path) -> None:
    _, _, load_artifact, validate = _api()
    root = _minimal_artifact(tmp_path)

    validated = validate(root)
    artifact = load_artifact(root)

    assert validated.valid is True
    assert validated.video_count == 2
    assert validated.episode_count == 2
    assert validated.track_row_count == 4
    assert tuple(video.video_id for video in artifact.videos) == (
        "train-video",
        "eval-video",
    )
    assert tuple(episode.label for episode in artifact.episodes) == (
        "approach",
        "pass_by",
    )


def test_artifact_rejects_checksum_mismatch(tmp_path: Path) -> None:
    ArtifactInvalid, _, _, validate = _api()
    root = _minimal_artifact(tmp_path)
    (root / "videos.json").write_text("{}\n", encoding="utf-8")

    try:
        validate(root)
    except ArtifactInvalid as exc:
        assert "sha256" in str(exc)
    else:
        raise AssertionError("checksum mismatch was accepted")


def test_artifact_rejects_video_hash_overlap_between_splits(tmp_path: Path) -> None:
    ArtifactInvalid, _, _, validate = _api()
    root = _minimal_artifact(tmp_path)
    videos = json.loads((root / "videos.json").read_text(encoding="utf-8"))
    videos[1]["sha256"] = videos[0]["sha256"]
    digest = _write_signed_json(root, "videos.json", videos)

    manifest = json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["videos.json"] = digest
    _write_signed_json(root, "artifact-manifest.json", manifest)

    try:
        validate(root)
    except ArtifactInvalid as exc:
        assert "video hash overlap" in str(exc)
    else:
        raise AssertionError("cross-split video hash overlap was accepted")


def test_tracks_must_be_sorted_by_video_frame_track(tmp_path: Path) -> None:
    ArtifactInvalid, _, _, validate = _api()
    root = _minimal_artifact(tmp_path)
    rows = [
        json.loads(line)
        for line in (root / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[0], rows[1] = rows[1], rows[0]
    digest = _write_signed_jsonl(root, "tracks.jsonl", rows)

    manifest = json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["tracks.jsonl"] = digest
    _write_signed_json(root, "artifact-manifest.json", manifest)

    try:
        validate(root)
    except ArtifactInvalid as exc:
        assert "sorted" in str(exc)
    else:
        raise AssertionError("unsorted track rows were accepted")


def test_duplicate_episode_id_is_rejected(tmp_path: Path) -> None:
    ArtifactInvalid, _, _, validate = _api()
    root = _minimal_artifact(tmp_path)
    episodes = json.loads((root / "episodes.json").read_text(encoding="utf-8"))
    episodes[1]["episode_id"] = episodes[0]["episode_id"]
    digest = _write_signed_json(root, "episodes.json", episodes)

    manifest = json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["episodes.json"] = digest
    _write_signed_json(root, "artifact-manifest.json", manifest)

    try:
        validate(root)
    except ArtifactInvalid as exc:
        assert "episode_id" in str(exc)
    else:
        raise AssertionError("duplicate episode_id was accepted")


def test_artifact_root_digest_is_deterministic_and_content_addressed(
    tmp_path: Path,
) -> None:
    _, artifact_root_digest, _, _ = _api()
    root = _minimal_artifact(tmp_path)

    first = artifact_root_digest(root)
    second = artifact_root_digest(root)

    assert first == second
    assert len(first) == 64

    provenance = json.loads(
        (root / "extraction-provenance.json").read_text(encoding="utf-8")
    )
    provenance["runtime"]["python"] = "3.12.15"
    digest = _write_signed_json(root, "extraction-provenance.json", provenance)
    manifest = json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["extraction-provenance.json"] = digest
    _write_signed_json(root, "artifact-manifest.json", manifest)

    assert artifact_root_digest(root) != first
