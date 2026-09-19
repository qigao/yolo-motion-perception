from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ARTIFACT_SCHEMA = "r1-e3-artifact-v1"
REGISTERED_LABELS = ("approach", "touch", "pick_up", "pass_by")
_REQUIRED_PAYLOAD_FILES = (
    "videos.json",
    "episodes.json",
    "tracks.jsonl",
    "extraction-provenance.json",
)
_ROOT_DIGEST_VERSION = b"r1-e3-artifact-root-v1\0"


class ArtifactInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceVideoRecord:
    video_id: str
    sha256: str
    split: str
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class RawTrackRow:
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
class EpisodeAnnotation:
    episode_id: str
    video_id: str
    start_time_seconds: float
    end_time_seconds: float
    label: str
    actor_track_id: int
    target_track_id: int
    annotation_revision: str
    reviewer: str


@dataclass(frozen=True)
class FrozenTrackArtifact:
    videos: tuple[SourceVideoRecord, ...]
    episodes: tuple[EpisodeAnnotation, ...]
    tracks: tuple[RawTrackRow, ...]
    provenance: Mapping[str, object]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class ArtifactValidationResult:
    valid: bool
    video_count: int
    episode_count: int
    track_row_count: int
    root_digest: str


def load_artifact(root: Path | str) -> FrozenTrackArtifact:
    root_path = Path(root)
    manifest = _read_manifest(root_path)
    _verify_declared_files(root_path, manifest)

    videos_payload = _read_json(root_path / "videos.json", "videos.json")
    episodes_payload = _read_json(root_path / "episodes.json", "episodes.json")
    provenance = _read_json(
        root_path / "extraction-provenance.json",
        "extraction-provenance.json",
    )
    tracks_payload = _read_jsonl(root_path / "tracks.jsonl")

    if not isinstance(videos_payload, list) or not videos_payload:
        raise ArtifactInvalid("videos.json must contain a non-empty array")
    if not isinstance(episodes_payload, list) or not episodes_payload:
        raise ArtifactInvalid("episodes.json must contain a non-empty array")
    if not isinstance(provenance, dict):
        raise ArtifactInvalid("extraction provenance must be an object")
    if not tracks_payload:
        raise ArtifactInvalid("tracks.jsonl must contain at least one row")

    videos = tuple(_video_record(item) for item in videos_payload)
    episodes = tuple(_episode_record(item) for item in episodes_payload)
    tracks = tuple(_track_row(item) for item in tracks_payload)

    _validate_video_identity(videos)
    _validate_track_rows(videos, tracks)
    _validate_episode_identity(videos, tracks, episodes)

    return FrozenTrackArtifact(
        videos=videos,
        episodes=episodes,
        tracks=tracks,
        provenance=provenance,
        manifest=manifest,
    )


def validate_artifact_structure(root: Path | str) -> ArtifactValidationResult:
    artifact = load_artifact(root)
    return ArtifactValidationResult(
        valid=True,
        video_count=len(artifact.videos),
        episode_count=len(artifact.episodes),
        track_row_count=len(artifact.tracks),
        root_digest=_root_digest(Path(root), artifact.manifest),
    )


def artifact_root_digest(root: Path | str) -> str:
    root_path = Path(root)
    artifact = load_artifact(root_path)
    return _root_digest(root_path, artifact.manifest)


def _read_manifest(root: Path) -> dict[str, object]:
    manifest = _read_verified_json(root, "artifact-manifest.json")
    if not isinstance(manifest, dict):
        raise ArtifactInvalid("artifact manifest must be an object")
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ArtifactInvalid("artifact manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactInvalid("artifact manifest files must be an object")
    missing = set(_REQUIRED_PAYLOAD_FILES) - set(files)
    if missing:
        raise ArtifactInvalid(
            "artifact manifest is missing required files: "
            + ", ".join(sorted(missing))
        )
    for name, digest in files.items():
        if not isinstance(name, str) or not name:
            raise ArtifactInvalid("artifact manifest file names must be non-empty strings")
        _require_digest(digest, f"manifest digest for {name}")
    return manifest


def _verify_declared_files(root: Path, manifest: Mapping[str, object]) -> None:
    files = manifest["files"]
    assert isinstance(files, dict)
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ArtifactInvalid("artifact manifest file entry is invalid")
        path = root / name
        if not path.is_file():
            raise ArtifactInvalid(f"missing artifact file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar_name = _sidecar_name(name)
        sidecar = root / sidecar_name
        if not sidecar.is_file():
            raise ArtifactInvalid(f"missing {sidecar_name}")
        sidecar_digest = sidecar.read_text(encoding="utf-8").strip()
        _require_digest(sidecar_digest, sidecar_name)
        if actual != sidecar_digest or actual != expected:
            raise ArtifactInvalid(f"{name} sha256 mismatch")


def _read_verified_json(root: Path, name: str) -> object:
    path = root / name
    sidecar = root / _sidecar_name(name)
    if not path.is_file() or not sidecar.is_file():
        raise ArtifactInvalid(f"missing {name} or checksum")
    data = path.read_bytes()
    expected = sidecar.read_text(encoding="utf-8").strip()
    _require_digest(expected, _sidecar_name(name))
    if hashlib.sha256(data).hexdigest() != expected:
        raise ArtifactInvalid(f"{name} sha256 mismatch")
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactInvalid(f"invalid {name}") from exc


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactInvalid(f"invalid {name}") from exc


def _read_jsonl(path: Path) -> list[object]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ArtifactInvalid("invalid tracks.jsonl") from exc
    rows: list[object] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ArtifactInvalid(
                f"tracks.jsonl contains a blank row at line {line_number}"
            )
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ArtifactInvalid(
                f"invalid tracks.jsonl row at line {line_number}"
            ) from exc
    return rows


def _video_record(value: object) -> SourceVideoRecord:
    item = _object(value, "video")
    video_id = _nonempty_string(item.get("video_id"), "video_id")
    sha256 = _require_digest(item.get("sha256"), "video sha256")
    split = item.get("split")
    if split not in ("training", "evaluation"):
        raise ArtifactInvalid("video split must be training or evaluation")
    fps = _finite_float(item.get("fps"), "video fps")
    if fps <= 0.0:
        raise ArtifactInvalid("video fps must be positive")
    frame_count = _positive_int(item.get("frame_count"), "video frame_count")
    width = _positive_int(item.get("width"), "video width")
    height = _positive_int(item.get("height"), "video height")
    return SourceVideoRecord(
        video_id=video_id,
        sha256=sha256,
        split=split,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def _episode_record(value: object) -> EpisodeAnnotation:
    item = _object(value, "episode")
    episode_id = _nonempty_string(item.get("episode_id"), "episode_id")
    video_id = _nonempty_string(item.get("video_id"), "episode video_id")
    start = _finite_float(item.get("start_time_seconds"), "episode start_time_seconds")
    end = _finite_float(item.get("end_time_seconds"), "episode end_time_seconds")
    if start < 0.0 or end <= start:
        raise ArtifactInvalid("episode time range must be positive and increasing")
    label = item.get("label")
    if label not in REGISTERED_LABELS:
        raise ArtifactInvalid("episode label is not registered")
    actor_track_id = _nonnegative_int(item.get("actor_track_id"), "actor_track_id")
    target_track_id = _nonnegative_int(item.get("target_track_id"), "target_track_id")
    revision = _nonempty_string(
        item.get("annotation_revision"),
        "annotation_revision",
    )
    reviewer = _nonempty_string(item.get("reviewer"), "reviewer")
    return EpisodeAnnotation(
        episode_id=episode_id,
        video_id=video_id,
        start_time_seconds=start,
        end_time_seconds=end,
        label=label,
        actor_track_id=actor_track_id,
        target_track_id=target_track_id,
        annotation_revision=revision,
        reviewer=reviewer,
    )


def _track_row(value: object) -> RawTrackRow:
    item = _object(value, "track row")
    video_id = _nonempty_string(item.get("video_id"), "track video_id")
    frame_index = _nonnegative_int(item.get("frame_index"), "frame_index")
    timestamp = _finite_float(item.get("timestamp_seconds"), "timestamp_seconds")
    if timestamp < 0.0:
        raise ArtifactInvalid("timestamp_seconds must be non-negative")
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
    return RawTrackRow(
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


def _validate_video_identity(videos: tuple[SourceVideoRecord, ...]) -> None:
    ids: set[str] = set()
    hashes: dict[str, str] = {}
    for video in videos:
        if video.video_id in ids:
            raise ArtifactInvalid("duplicate video_id")
        ids.add(video.video_id)
        previous_split = hashes.get(video.sha256)
        if previous_split is not None:
            if previous_split != video.split:
                raise ArtifactInvalid("video hash overlap between splits")
            raise ArtifactInvalid("duplicate video sha256")
        hashes[video.sha256] = video.split


def _validate_track_rows(
    videos: tuple[SourceVideoRecord, ...],
    tracks: tuple[RawTrackRow, ...],
) -> None:
    by_id = {video.video_id: video for video in videos}
    previous_key: tuple[str, int, int] | None = None
    for track in tracks:
        video = by_id.get(track.video_id)
        if video is None:
            raise ArtifactInvalid("track row references unknown video_id")
        if track.frame_index >= video.frame_count:
            raise ArtifactInvalid("track frame_index exceeds source video")
        if not (0.0 <= track.x1 < track.x2 <= float(video.width)):
            raise ArtifactInvalid("track x coordinates exceed image bounds")
        if not (0.0 <= track.y1 < track.y2 <= float(video.height)):
            raise ArtifactInvalid("track y coordinates exceed image bounds")
        key = (track.video_id, track.frame_index, track.track_id)
        if previous_key is not None and key <= previous_key:
            raise ArtifactInvalid(
                "track rows must be sorted by video_id, frame_index, track_id"
            )
        previous_key = key


def _validate_episode_identity(
    videos: tuple[SourceVideoRecord, ...],
    tracks: tuple[RawTrackRow, ...],
    episodes: tuple[EpisodeAnnotation, ...],
) -> None:
    video_ids = {video.video_id for video in videos}
    track_ids_by_video: dict[str, set[int]] = {}
    for row in tracks:
        track_ids_by_video.setdefault(row.video_id, set()).add(row.track_id)

    episode_ids: set[str] = set()
    for episode in episodes:
        if episode.episode_id in episode_ids:
            raise ArtifactInvalid("duplicate episode_id")
        episode_ids.add(episode.episode_id)
        if episode.video_id not in video_ids:
            raise ArtifactInvalid("episode references unknown video_id")
        known_tracks = track_ids_by_video.get(episode.video_id, set())
        if episode.actor_track_id not in known_tracks:
            raise ArtifactInvalid("episode actor_track_id is unknown")
        if episode.target_track_id not in known_tracks:
            raise ArtifactInvalid("episode target_track_id is unknown")


def _root_digest(root: Path, manifest: Mapping[str, object]) -> str:
    files = manifest["files"]
    assert isinstance(files, dict)
    digest = hashlib.sha256()
    digest.update(_ROOT_DIGEST_VERSION)

    names = ["artifact-manifest.json", *sorted(str(name) for name in files)]
    for name in names:
        path = root / name
        if not path.is_file():
            raise ArtifactInvalid(f"missing artifact file: {name}")
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sidecar_name(name: str) -> str:
    if name.endswith(".jsonl"):
        return f"{name[:-6]}.sha256"
    if name.endswith(".json"):
        return f"{name[:-5]}.sha256"
    return f"{name}.sha256"


def _object(value: object, name: str) -> dict[str, object]:
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
