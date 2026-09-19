from __future__ import annotations

import bisect
import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .r1_e3m_artifact import TrackArtifact, TrackRow


WINDOW_SECONDS = 4.0
BIN_COUNT = 20
BIN_SECONDS = 0.2
MIN_PRESENT_BINS = 16
MAX_INTERPOLATION_GAP_SECONDS = 0.5
_SEQUENCE_ID_VERSION = b"r1-e3m-sequence-id-v1\0"
_TENSOR_DIGEST_VERSION = b"r1-e3m-window-tensor-v1\0"


@dataclass(frozen=True)
class UnlabeledWindow:
    sequence_id: str
    video_id: str
    split: str
    source_window_component_id: str
    start_seconds: float
    end_seconds: float
    actor_track_id: int
    target_track_id: int
    tensor: np.ndarray
    tensor_digest: str

    def __post_init__(self) -> None:
        values = np.asarray(self.tensor, dtype=np.float64)
        if values.shape != (BIN_COUNT, 14):
            raise ValueError("R1-E3M tensor must have shape (20, 14)")
        if not np.all(np.isfinite(values)):
            raise ValueError("R1-E3M tensor must be finite")
        copied = np.array(values, dtype=np.float64, copy=True, order="C")
        copied.flags.writeable = False
        object.__setattr__(self, "tensor", copied)


def build_pair_windows(
    artifact: TrackArtifact,
    *,
    artifact_root_digest: str,
) -> tuple[UnlabeledWindow, ...]:
    if not isinstance(artifact, TrackArtifact):
        raise ValueError("artifact must be TrackArtifact")
    _require_digest(artifact_root_digest)

    rows_by_video_track: dict[str, dict[int, tuple[TrackRow, ...]]] = {}
    for video in artifact.videos:
        grouped: dict[int, list[TrackRow]] = defaultdict(list)
        for row in artifact.tracks:
            if row.video_id == video.video_id:
                grouped[row.track_id].append(row)
        rows_by_video_track[video.video_id] = {
            track_id: tuple(sorted(rows, key=lambda item: item.timestamp_seconds))
            for track_id, rows in grouped.items()
        }

    windows: list[UnlabeledWindow] = []
    for video in artifact.videos:
        track_rows = rows_by_video_track[video.video_id]
        full_window_count = int(math.floor(video.duration_seconds / WINDOW_SECONDS))
        for window_index in range(full_window_count):
            start = float(window_index) * WINDOW_SECONDS
            end = start + WINDOW_SECONDS
            timestamps = tuple(
                start + (index + 0.5) * BIN_SECONDS
                for index in range(BIN_COUNT)
            )

            actors = [
                track_id
                for track_id, rows in track_rows.items()
                if _modal_class(rows, start, end) == 0
            ]
            for actor_track_id in sorted(actors):
                actor_samples = _sample_series(
                    track_rows[actor_track_id],
                    timestamps,
                )
                if _presence_count(actor_samples) < MIN_PRESENT_BINS:
                    continue

                for target_track_id in sorted(track_rows):
                    if target_track_id == actor_track_id:
                        continue
                    target_samples = _sample_series(
                        track_rows[target_track_id],
                        timestamps,
                    )
                    if _presence_count(target_samples) < MIN_PRESENT_BINS:
                        continue

                    tensor = _pair_tensor(actor_samples, target_samples)
                    sequence_id = _sequence_id(
                        artifact_root_digest=artifact_root_digest,
                        video_id=video.video_id,
                        start_seconds=start,
                        actor_track_id=actor_track_id,
                        target_track_id=target_track_id,
                    )
                    windows.append(
                        UnlabeledWindow(
                            sequence_id=sequence_id,
                            video_id=video.video_id,
                            split=video.split,
                            source_window_component_id=(
                                video.source_window_component_id
                            ),
                            start_seconds=start,
                            end_seconds=end,
                            actor_track_id=actor_track_id,
                            target_track_id=target_track_id,
                            tensor=tensor,
                            tensor_digest=_tensor_digest(tensor),
                        )
                    )

    return tuple(
        sorted(
            windows,
            key=lambda item: (
                item.video_id,
                item.start_seconds,
                item.actor_track_id,
                item.target_track_id,
            ),
        )
    )


def _modal_class(
    rows: tuple[TrackRow, ...],
    start: float,
    end: float,
) -> int | None:
    counts = Counter(
        row.class_id
        for row in rows
        if start <= row.timestamp_seconds < end
    )
    if not counts:
        return None
    max_count = max(counts.values())
    return min(class_id for class_id, count in counts.items() if count == max_count)


def _sample_series(
    rows: tuple[TrackRow, ...],
    timestamps: tuple[float, ...],
) -> tuple[np.ndarray | None, ...]:
    return tuple(_sample_track(rows, timestamp) for timestamp in timestamps)


def _sample_track(
    rows: tuple[TrackRow, ...],
    timestamp: float,
) -> np.ndarray | None:
    times = [row.timestamp_seconds for row in rows]
    index = bisect.bisect_left(times, timestamp)

    if index < len(rows) and math.isclose(
        times[index],
        timestamp,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return _track_vector(rows[index])
    if index == 0 or index == len(rows):
        return None

    left = rows[index - 1]
    right = rows[index]
    gap = right.timestamp_seconds - left.timestamp_seconds
    if gap <= 0.0 or gap > MAX_INTERPOLATION_GAP_SECONDS:
        return None

    alpha = (timestamp - left.timestamp_seconds) / gap
    result = _track_vector(left) + alpha * (
        _track_vector(right) - _track_vector(left)
    )
    return np.clip(result, 0.0, 1.0)


def _track_vector(row: TrackRow) -> np.ndarray:
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


def _presence_count(
    samples: tuple[np.ndarray | None, ...],
) -> int:
    return sum(sample is not None for sample in samples)


def _pair_tensor(
    actor_samples: tuple[np.ndarray | None, ...],
    target_samples: tuple[np.ndarray | None, ...],
) -> np.ndarray:
    tensor = np.zeros((BIN_COUNT, 14), dtype=np.float64)

    for index, (actor, target) in enumerate(zip(actor_samples, target_samples)):
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
    return tensor


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


def _sequence_id(
    *,
    artifact_root_digest: str,
    video_id: str,
    start_seconds: float,
    actor_track_id: int,
    target_track_id: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(_SEQUENCE_ID_VERSION)
    for value in (
        artifact_root_digest,
        video_id,
        f"{start_seconds:.3f}",
        str(actor_track_id),
        str(target_track_id),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _tensor_digest(tensor: np.ndarray) -> str:
    values = np.ascontiguousarray(tensor, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(_TENSOR_DIGEST_VERSION)
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("artifact_root_digest must be a lowercase SHA-256 digest")
    return value
