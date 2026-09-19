from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


TRACK_ARTIFACT_SCHEMA = "r1-e3m-track-artifact-v1"
SOURCE_MANIFEST_SCHEMA = "r1-e3a-source-video-manifest-v1"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "1db8cc200ba9eaab2c14ed260b06ed582ed75d54f57c3036e1312cd2f184f4cf"
)
_REQUIRED_FILES = (
    "source-manifest.json",
    "tracks.jsonl",
    "extraction-provenance.json",
)
_FORBIDDEN_SEMANTIC_FILES = (
    "episodes.json",
    "annotations.json",
    "semantic-annotations.json",
    "semantic-review.json",
)
_ROOT_DIGEST_VERSION = b"r1-e3m-track-artifact-root-v1\0"
_FORBIDDEN_TRACK_KEYS = {
    "episode_id",
    "label",
    "review_status",
    "reviewer",
    "physical_event_id",
}


class ArtifactInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceVideo:
    video_id: str
    sha256: str
    split: str
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int
    source_window_component_id: str


@dataclass(frozen=True)
class TrackRow:
    video_id: str
    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float


@dataclass(frozen=True)
class TrackArtifact:
    videos: tuple[SourceVideo, ...]
    tracks: tuple[TrackRow, ...]
    provenance: Mapping[str, object]
    artifact_manifest: Mapping[str, object]


def load_track_artifact(
    root: Path | str,
    *,
    expected_source_manifest_sha256: str = FROZEN_SOURCE_MANIFEST_SHA256,
) -> TrackArtifact:
    root_path = Path(root)
    _reject_semantic_payloads(root_path)

    manifest = _read_json_object(
        root_path / "track-artifact-manifest.json",
        "track-artifact-manifest.json",
    )
    if manifest.get("schema") != TRACK_ARTIFACT_SCHEMA:
        raise ArtifactInvalid("track artifact schema mismatch")

    declared_source_sha = _require_digest(
        manifest.get("source_manifest_sha256"),
        "source manifest sha256",
    )
    expected_source_sha = _require_digest(
        expected_source_manifest_sha256,
        "expected source manifest sha256",
    )
    if declared_source_sha != expected_source_sha:
        raise ArtifactInvalid("source manifest sha256 mismatch")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactInvalid("track artifact files must be an object")
    if set(files) != set(_REQUIRED_FILES):
        raise ArtifactInvalid(
            "track artifact files must contain exactly: "
            + ", ".join(_REQUIRED_FILES)
        )
    for name, expected_digest in files.items():
        if not isinstance(name, str):
            raise ArtifactInvalid("artifact file name must be a string")
        digest = _require_digest(expected_digest, f"{name} sha256")
        path = root_path / name
        if not path.is_file():
            raise ArtifactInvalid(f"missing artifact file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ArtifactInvalid(f"{name} sha256 mismatch")

    source_path = root_path / "source-manifest.json"
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha != expected_source_sha:
        raise ArtifactInvalid("source manifest sha256 mismatch")

    source_manifest = _read_json_object(source_path, "source-manifest.json")
    if source_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ArtifactInvalid("source manifest schema mismatch")

    videos_payload = source_manifest.get("videos")
    if not isinstance(videos_payload, list) or not videos_payload:
        raise ArtifactInvalid("source manifest videos must be a non-empty list")
    videos = tuple(_source_video(item) for item in videos_payload)
    _validate_video_identity_and_split(videos)

    tracks = tuple(_read_tracks(root_path / "tracks.jsonl"))
    if not tracks:
        raise ArtifactInvalid("tracks.jsonl must contain at least one row")
    _validate_tracks(videos, tracks)

    provenance = _read_json_object(
        root_path / "extraction-provenance.json",
        "extraction-provenance.json",
    )
    constraints = provenance.get("constraints")
    if not isinstance(constraints, dict):
        raise ArtifactInvalid("extraction provenance constraints must be an object")
    if constraints.get("no_semantic_labels") is not True:
        raise ArtifactInvalid("extraction must assert no semantic labels")
    if constraints.get("no_human_event_boundaries") is not True:
        raise ArtifactInvalid("extraction must assert no human event boundaries")

    return TrackArtifact(
        videos=videos,
        tracks=tracks,
        provenance=provenance,
        artifact_manifest=manifest,
    )


def artifact_root_digest(
    root: Path | str,
    *,
    expected_source_manifest_sha256: str = FROZEN_SOURCE_MANIFEST_SHA256,
) -> str:
    root_path = Path(root)
    artifact = load_track_artifact(
        root_path,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    files = artifact.artifact_manifest["files"]
    assert isinstance(files, dict)

    digest = hashlib.sha256()
    digest.update(_ROOT_DIGEST_VERSION)

    manifest_path = root_path / "track-artifact-manifest.json"
    digest.update(b"track-artifact-manifest.json\0")
    digest.update(hashlib.sha256(manifest_path.read_bytes()).hexdigest().encode("ascii"))
    digest.update(b"\0")

    for name in sorted(files):
        value = files[name]
        assert isinstance(name, str)
        assert isinstance(value, str)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _reject_semantic_payloads(root: Path) -> None:
    for name in _FORBIDDEN_SEMANTIC_FILES:
        if (root / name).exists():
            raise ArtifactInvalid(f"semantic payload is forbidden: {name}")


def _source_video(value: object) -> SourceVideo:
    item = _as_object(value, "source video")
    video_id = _nonempty_string(item.get("source_video_id"), "source_video_id")
    sha256 = _require_digest(item.get("sha256"), "source video sha256")
    split = item.get("split")
    if split not in ("train", "eval"):
        raise ArtifactInvalid("source video split must be train or eval")
    fps = _positive_float(item.get("fps"), "source video fps")
    frame_count = _positive_int(item.get("frame_count"), "source video frame_count")
    duration = _positive_float(
        item.get("duration_seconds"),
        "source video duration_seconds",
    )
    width = _positive_int(item.get("width"), "source video width")
    height = _positive_int(item.get("height"), "source video height")
    component = _nonempty_string(
        item.get("source_window_component_id"),
        "source_window_component_id",
    )
    return SourceVideo(
        video_id=video_id,
        sha256=sha256,
        split=split,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        width=width,
        height=height,
        source_window_component_id=component,
    )


def _validate_video_identity_and_split(videos: tuple[SourceVideo, ...]) -> None:
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    component_splits: dict[str, str] = {}
    for video in videos:
        if video.video_id in seen_ids:
            raise ArtifactInvalid("duplicate source video id")
        seen_ids.add(video.video_id)
        if video.sha256 in seen_hashes:
            raise ArtifactInvalid("duplicate source video sha256")
        seen_hashes.add(video.sha256)

        prior_split = component_splits.get(video.source_window_component_id)
        if prior_split is not None and prior_split != video.split:
            raise ArtifactInvalid("source-window component overlap between splits")
        component_splits[video.source_window_component_id] = video.split


def _read_tracks(path: Path) -> list[TrackRow]:
    if not path.is_file():
        raise ArtifactInvalid("missing tracks.jsonl")
    rows: list[TrackRow] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ArtifactInvalid("invalid tracks.jsonl") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ArtifactInvalid(f"blank tracks.jsonl row at line {line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactInvalid(
                f"invalid tracks.jsonl row at line {line_number}"
            ) from exc
        item = _as_object(payload, f"track row {line_number}")
        semantic_keys = _FORBIDDEN_TRACK_KEYS.intersection(item)
        if semantic_keys:
            raise ArtifactInvalid(
                "semantic fields are forbidden in track rows: "
                + ", ".join(sorted(semantic_keys))
            )
        rows.append(_track_row(item))
    return rows


def _track_row(item: Mapping[str, object]) -> TrackRow:
    video_id = _nonempty_string(item.get("video_id"), "track video_id")
    frame_index = _nonnegative_int(item.get("frame_index"), "frame_index")
    timestamp = _nonnegative_float(
        item.get("timestamp_seconds"),
        "timestamp_seconds",
    )
    track_id = _nonnegative_int(item.get("track_id"), "track_id")
    class_id = _nonnegative_int(item.get("class_id"), "class_id")
    confidence = _unit_float(item.get("confidence"), "confidence")
    x1 = _finite_float(item.get("x1"), "x1")
    y1 = _finite_float(item.get("y1"), "y1")
    x2 = _finite_float(item.get("x2"), "x2")
    y2 = _finite_float(item.get("y2"), "y2")
    if x2 <= x1 or y2 <= y1:
        raise ArtifactInvalid("track bbox must have positive width and height")
    cx_norm = _unit_float(item.get("cx_norm"), "cx_norm")
    cy_norm = _unit_float(item.get("cy_norm"), "cy_norm")
    w_norm = _unit_float(item.get("w_norm"), "w_norm")
    h_norm = _unit_float(item.get("h_norm"), "h_norm")
    if w_norm <= 0.0 or h_norm <= 0.0:
        raise ArtifactInvalid("normalized bbox size must be positive")
    return TrackRow(
        video_id=video_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        track_id=track_id,
        class_id=class_id,
        confidence=confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        cx_norm=cx_norm,
        cy_norm=cy_norm,
        w_norm=w_norm,
        h_norm=h_norm,
    )


def _validate_tracks(
    videos: tuple[SourceVideo, ...],
    tracks: tuple[TrackRow, ...],
) -> None:
    by_id = {video.video_id: video for video in videos}
    previous_key: tuple[str, int, int] | None = None

    for row in tracks:
        video = by_id.get(row.video_id)
        if video is None:
            raise ArtifactInvalid("track row references unknown video")
        if row.frame_index >= video.frame_count:
            raise ArtifactInvalid("track frame_index exceeds source video")
        if row.timestamp_seconds > video.duration_seconds + (1.0 / video.fps):
            raise ArtifactInvalid("track timestamp exceeds source video")
        if not 0.0 <= row.x1 < row.x2 <= float(video.width):
            raise ArtifactInvalid("track x coordinates exceed source frame")
        if not 0.0 <= row.y1 < row.y2 <= float(video.height):
            raise ArtifactInvalid("track y coordinates exceed source frame")

        key = (row.video_id, row.frame_index, row.track_id)
        if previous_key is not None and key <= previous_key:
            raise ArtifactInvalid(
                "track rows must be sorted by video_id, frame_index, track_id"
            )
        previous_key = key


def _read_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactInvalid(f"invalid {name}") from exc
    if not isinstance(payload, dict):
        raise ArtifactInvalid(f"{name} must be an object")
    return payload


def _as_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactInvalid(f"{name} must be an object")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactInvalid(f"{name} must be a non-empty string")
    return value


def _require_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ArtifactInvalid(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactInvalid(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactInvalid(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ArtifactInvalid(f"{name} must be positive")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ArtifactInvalid(f"{name} must be non-negative")
    return result


def _unit_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ArtifactInvalid(f"{name} must be in [0, 1]")
    return result


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ArtifactInvalid(f"{name} must be a positive Python integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactInvalid(f"{name} must be a non-negative Python integer")
    return value
