from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .r1_e3_artifact import (
    EpisodeAnnotation,
    FrozenTrackArtifact,
    RawTrackRow,
)


_BIN_COUNT = 20
_MAX_INTERPOLATION_GAP_SECONDS = 0.5
_TENSOR_DIGEST_VERSION = b"r1-e3-normalized-episode-v1\0"


@dataclass(frozen=True)
class NormalizedEpisode:
    episode_id: str
    video_id: str
    label: str
    tensor: np.ndarray
    tensor_digest: str

    def __post_init__(self) -> None:
        values = np.asarray(self.tensor, dtype=np.float64)
        if values.shape != (_BIN_COUNT, 14):
            raise ValueError("normalized episode tensor must have shape (20, 14)")
        if not np.all(np.isfinite(values)):
            raise ValueError("normalized episode tensor must be finite")
        copied = np.array(values, dtype=np.float64, copy=True, order="C")
        copied.flags.writeable = False
        object.__setattr__(self, "tensor", copied)


def normalize_episode(
    artifact: FrozenTrackArtifact,
    annotation: EpisodeAnnotation,
) -> NormalizedEpisode:
    if not isinstance(artifact, FrozenTrackArtifact):
        raise ValueError("artifact must be FrozenTrackArtifact")
    if not isinstance(annotation, EpisodeAnnotation):
        raise ValueError("annotation must be EpisodeAnnotation")

    actor_rows = _rows_for_track(
        artifact,
        annotation.video_id,
        annotation.actor_track_id,
    )
    target_rows = _rows_for_track(
        artifact,
        annotation.video_id,
        annotation.target_track_id,
    )

    timestamps = np.linspace(
        annotation.start_time_seconds,
        annotation.end_time_seconds,
        _BIN_COUNT,
        dtype=np.float64,
    )
    tensor = np.zeros((_BIN_COUNT, 14), dtype=np.float64)

    for index, timestamp in enumerate(timestamps):
        actor = _sample_track(actor_rows, float(timestamp))
        target = _sample_track(target_rows, float(timestamp))

        if actor is not None:
            tensor[index, 0:5] = actor
            tensor[index, 5] = 1.0
        if target is not None:
            tensor[index, 6:11] = target
            tensor[index, 11] = 1.0
        if actor is not None and target is not None:
            tensor[index, 12] = math.hypot(
                float(actor[0] - target[0]),
                float(actor[1] - target[1]),
            ) / math.sqrt(2.0)
            tensor[index, 13] = _box_iou(actor, target)

    tensor = np.clip(tensor, 0.0, 1.0)
    tensor.flags.writeable = False
    return NormalizedEpisode(
        episode_id=annotation.episode_id,
        video_id=annotation.video_id,
        label=annotation.label,
        tensor=tensor,
        tensor_digest=_tensor_digest(tensor),
    )


def normalize_registered_episodes(
    artifact: FrozenTrackArtifact,
) -> tuple[NormalizedEpisode, ...]:
    if not isinstance(artifact, FrozenTrackArtifact):
        raise ValueError("artifact must be FrozenTrackArtifact")
    return tuple(normalize_episode(artifact, episode) for episode in artifact.episodes)


def _rows_for_track(
    artifact: FrozenTrackArtifact,
    video_id: str,
    track_id: int,
) -> tuple[RawTrackRow, ...]:
    rows = tuple(
        row
        for row in artifact.tracks
        if row.video_id == video_id and row.track_id == track_id
    )
    if not rows:
        raise ValueError("episode references a track with no observations")
    return tuple(sorted(rows, key=lambda row: row.timestamp_seconds))


def _sample_track(
    rows: tuple[RawTrackRow, ...],
    timestamp: float,
) -> np.ndarray | None:
    times = [row.timestamp_seconds for row in rows]
    index = bisect.bisect_left(times, timestamp)

    if index < len(rows) and times[index] == timestamp:
        return _track_vector(rows[index])
    if index == 0 or index == len(rows):
        return None

    left = rows[index - 1]
    right = rows[index]
    gap = right.timestamp_seconds - left.timestamp_seconds
    if gap <= 0.0 or gap > _MAX_INTERPOLATION_GAP_SECONDS:
        return None

    alpha = (timestamp - left.timestamp_seconds) / gap
    left_values = _track_vector(left)
    right_values = _track_vector(right)
    result = left_values + alpha * (right_values - left_values)
    return np.clip(result, 0.0, 1.0)


def _track_vector(row: RawTrackRow) -> np.ndarray:
    return np.asarray(
        (
            row.cx_norm,
            row.cy_norm,
            row.w_norm,
            row.h_norm,
            row.confidence,
        ),
        dtype=np.float64,
    )


def _box_iou(actor: np.ndarray, target: np.ndarray) -> float:
    ax1 = float(actor[0] - actor[2] / 2.0)
    ay1 = float(actor[1] - actor[3] / 2.0)
    ax2 = float(actor[0] + actor[2] / 2.0)
    ay2 = float(actor[1] + actor[3] / 2.0)

    bx1 = float(target[0] - target[2] / 2.0)
    by1 = float(target[1] - target[3] / 2.0)
    bx2 = float(target[0] + target[2] / 2.0)
    by2 = float(target[1] + target[3] / 2.0)

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    actor_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    target_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = actor_area + target_area - intersection
    if union <= 0.0:
        return 0.0
    return min(1.0, max(0.0, intersection / union))


def _tensor_digest(tensor: np.ndarray) -> str:
    values = np.ascontiguousarray(tensor, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(_TENSOR_DIGEST_VERSION)
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()
