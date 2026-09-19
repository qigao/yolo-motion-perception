from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


WINDOW_SHAPE = (20, 6)
WINDOW_DURATION_SECONDS = 2.0
_SCHEMA = "r1-e3m-mechanism-artifact-v1"
_DIGEST_VERSION = b"r1-e3m-mechanism-artifact-v1\0"


class MechanismArtifactInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class MechanismVideoRecord:
    video_id: str
    sha256: str
    split: str
    fps: float
    frame_count: int
    width: int
    height: int
    source_window_component_id: str

    def __post_init__(self) -> None:
        _nonempty(self.video_id, "video_id")
        _digest(self.sha256, "video sha256")
        _split(self.split)
        _positive_float(self.fps, "fps")
        _positive_int(self.frame_count, "frame_count")
        _positive_int(self.width, "width")
        _positive_int(self.height, "height")
        _nonempty(
            self.source_window_component_id,
            "source_window_component_id",
        )


@dataclass(frozen=True)
class TrackWindow:
    window_id: str
    video_id: str
    split: str
    track_id: int
    source_start_seconds: float
    source_end_seconds: float
    tensor: np.ndarray

    def __post_init__(self) -> None:
        _digest(self.window_id, "window_id")
        _nonempty(self.video_id, "video_id")
        _split(self.split)
        _nonnegative_int(self.track_id, "track_id")
        start = _finite_float(
            self.source_start_seconds,
            "source_start_seconds",
        )
        end = _finite_float(
            self.source_end_seconds,
            "source_end_seconds",
        )
        if start < 0.0 or end <= start:
            raise MechanismArtifactInvalid(
                "window source bounds must be non-negative and increasing"
            )

        tensor = self.tensor
        if not isinstance(tensor, np.ndarray):
            raise MechanismArtifactInvalid("window tensor must be a NumPy array")
        if tensor.shape != WINDOW_SHAPE:
            raise MechanismArtifactInvalid("window tensor must have shape 20 x 6")
        if tensor.dtype != np.float64:
            raise MechanismArtifactInvalid("window tensor must use float64")
        if not np.isfinite(tensor).all():
            raise MechanismArtifactInvalid("window tensor must be finite")

        frozen = np.array(tensor, dtype=np.float64, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "tensor", frozen)


@dataclass(frozen=True)
class MechanismArtifact:
    source_manifest_sha256: str
    raw_track_sha256: str
    extraction_provenance_sha256: str
    videos: tuple[MechanismVideoRecord, ...]
    windows: tuple[TrackWindow, ...]

    def __post_init__(self) -> None:
        _digest(self.source_manifest_sha256, "source_manifest_sha256")
        _digest(self.raw_track_sha256, "raw_track_sha256")
        _digest(
            self.extraction_provenance_sha256,
            "extraction_provenance_sha256",
        )
        if not isinstance(self.videos, tuple) or not self.videos:
            raise MechanismArtifactInvalid("videos must be a non-empty tuple")
        if not isinstance(self.windows, tuple) or not self.windows:
            raise MechanismArtifactInvalid("windows must be a non-empty tuple")


def track_window_id(
    video_id: str,
    track_id: int,
    source_start_seconds: float,
    source_end_seconds: float,
) -> str:
    _nonempty(video_id, "video_id")
    _nonnegative_int(track_id, "track_id")
    start = _finite_float(source_start_seconds, "source_start_seconds")
    end = _finite_float(source_end_seconds, "source_end_seconds")
    if start < 0.0 or end <= start:
        raise MechanismArtifactInvalid(
            "window source bounds must be non-negative and increasing"
        )
    payload = (
        f"r1-e3m-window-v1\0{video_id}\0{track_id}\0"
        f"{start:.9f}\0{end:.9f}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def window_tensor_digest(window: TrackWindow) -> str:
    digest = hashlib.sha256()
    digest.update(b"r1-e3m-window-tensor-v1\0")
    digest.update(str(window.tensor.shape).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(window.tensor.tobytes(order="C"))
    return digest.hexdigest()


def validate_mechanism_artifact(
    artifact: MechanismArtifact,
) -> MechanismArtifact:
    videos = _validate_videos(artifact.videos)
    _validate_windows(artifact.windows, videos)
    return artifact


def mechanism_artifact_digest(artifact: MechanismArtifact) -> str:
    validate_mechanism_artifact(artifact)
    payload = {
        "schema": _SCHEMA,
        "source_manifest_sha256": artifact.source_manifest_sha256,
        "raw_track_sha256": artifact.raw_track_sha256,
        "extraction_provenance_sha256":
            artifact.extraction_provenance_sha256,
        "videos": [
            {
                "video_id": video.video_id,
                "sha256": video.sha256,
                "split": video.split,
                "fps": video.fps,
                "frame_count": video.frame_count,
                "width": video.width,
                "height": video.height,
                "source_window_component_id":
                    video.source_window_component_id,
            }
            for video in artifact.videos
        ],
        "windows": [
            {
                "window_id": window.window_id,
                "video_id": window.video_id,
                "split": window.split,
                "track_id": window.track_id,
                "source_start_seconds": window.source_start_seconds,
                "source_end_seconds": window.source_end_seconds,
                "tensor_sha256": window_tensor_digest(window),
            }
            for window in artifact.windows
        ],
    }
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_DIGEST_VERSION)
    digest.update(encoded)
    return digest.hexdigest()


def _validate_videos(
    videos: Iterable[MechanismVideoRecord],
) -> dict[str, MechanismVideoRecord]:
    by_id: dict[str, MechanismVideoRecord] = {}
    hash_splits: dict[str, str] = {}
    for video in videos:
        if video.video_id in by_id:
            raise MechanismArtifactInvalid("duplicate video_id")
        by_id[video.video_id] = video

        previous_split = hash_splits.get(video.sha256)
        if previous_split is not None:
            if previous_split != video.split:
                raise MechanismArtifactInvalid(
                    "video hash overlap between train/eval splits"
                )
            raise MechanismArtifactInvalid("duplicate source video sha256")
        hash_splits[video.sha256] = video.split
    return by_id


def _validate_windows(
    windows: Iterable[TrackWindow],
    videos: dict[str, MechanismVideoRecord],
) -> None:
    window_ids: set[str] = set()
    split_counts = {"train": 0, "eval": 0}

    for window in windows:
        if window.window_id in window_ids:
            raise MechanismArtifactInvalid("duplicate window_id")
        window_ids.add(window.window_id)

        video = videos.get(window.video_id)
        if video is None:
            raise MechanismArtifactInvalid(
                "window references unknown source video"
            )
        if window.split != video.split:
            raise MechanismArtifactInvalid(
                f"window split mismatch for {window.video_id}"
            )
        expected_id = track_window_id(
            window.video_id,
            window.track_id,
            window.source_start_seconds,
            window.source_end_seconds,
        )
        if window.window_id != expected_id:
            raise MechanismArtifactInvalid(
                "window_id does not match frozen window identity"
            )
        split_counts[window.split] += 1

    if split_counts["train"] == 0 or split_counts["eval"] == 0:
        raise MechanismArtifactInvalid(
            "artifact must contain train and eval windows"
        )


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MechanismArtifactInvalid(f"{name} must be a non-empty string")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise MechanismArtifactInvalid(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _split(value: object) -> str:
    if value not in ("train", "eval"):
        raise MechanismArtifactInvalid("split must be train or eval")
    assert isinstance(value, str)
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanismArtifactInvalid(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MechanismArtifactInvalid(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise MechanismArtifactInvalid(f"{name} must be positive")
    return result


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise MechanismArtifactInvalid(
            f"{name} must be a positive Python integer"
        )
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise MechanismArtifactInvalid(
            f"{name} must be a non-negative Python integer"
        )
    return value
