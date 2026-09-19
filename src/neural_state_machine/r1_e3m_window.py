from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from typing import Iterable, Protocol

import numpy as np

from neural_state_machine.r1_e3m_artifact import (
    MechanismArtifactInvalid,
    MechanismVideoRecord,
    TrackWindow,
    track_window_id,
)


WINDOW_DURATION_SECONDS = 2.0
BIN_COUNT = 20
BIN_WIDTH_SECONDS = WINDOW_DURATION_SECONDS / BIN_COUNT
MAX_OBSERVATION_AGE_SECONDS = 0.5
MIN_PRESENT_BINS = 16


class RawTrackLike(Protocol):
    video_id: str
    timestamp_seconds: float
    track_id: int
    confidence: float
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float


def build_track_windows(
    raw_rows: Iterable[RawTrackLike],
    videos: tuple[MechanismVideoRecord, ...],
) -> tuple[TrackWindow, ...]:
    videos_by_id = {video.video_id: video for video in videos}
    if len(videos_by_id) != len(videos):
        raise MechanismArtifactInvalid("duplicate video_id")

    grouped: dict[tuple[str, int], list[RawTrackLike]] = defaultdict(list)
    for row in raw_rows:
        video = videos_by_id.get(row.video_id)
        if video is None:
            raise MechanismArtifactInvalid(
                f"raw track row references unknown video: {row.video_id}"
            )
        _validate_row(row)
        grouped[(row.video_id, row.track_id)].append(row)

    windows: list[TrackWindow] = []
    for (video_id, track_id), rows in sorted(grouped.items()):
        rows = sorted(
            rows,
            key=lambda row: (
                float(row.timestamp_seconds),
                int(getattr(row, "frame_index", 0)),
            ),
        )
        video = videos_by_id[video_id]
        timestamps = [float(row.timestamp_seconds) for row in rows]
        first_window = int(math.floor(timestamps[0] / WINDOW_DURATION_SECONDS))
        last_window = int(math.floor(timestamps[-1] / WINDOW_DURATION_SECONDS))

        for window_number in range(first_window, last_window + 1):
            start = window_number * WINDOW_DURATION_SECONDS
            end = start + WINDOW_DURATION_SECONDS
            tensor = _window_tensor(rows, timestamps, start)
            if int(tensor[:, 5].sum()) < MIN_PRESENT_BINS:
                continue
            windows.append(
                TrackWindow(
                    window_id=track_window_id(
                        video_id,
                        track_id,
                        start,
                        end,
                    ),
                    video_id=video_id,
                    split=video.split,
                    track_id=track_id,
                    source_start_seconds=start,
                    source_end_seconds=end,
                    tensor=tensor,
                )
            )

    return tuple(
        sorted(
            windows,
            key=lambda window: (
                window.video_id,
                window.track_id,
                window.source_start_seconds,
            ),
        )
    )


def _window_tensor(
    rows: list[RawTrackLike],
    timestamps: list[float],
    start: float,
) -> np.ndarray:
    tensor = np.zeros((BIN_COUNT, 6), dtype=np.float64)

    for index in range(BIN_COUNT):
        center = start + (index + 0.5) * BIN_WIDTH_SECONDS
        row_index = bisect_right(timestamps, center) - 1
        if row_index < 0:
            continue

        row = rows[row_index]
        age = center - float(row.timestamp_seconds)
        if age < -1e-12:
            raise MechanismArtifactInvalid(
                "causal window builder selected a future observation"
            )
        if age > MAX_OBSERVATION_AGE_SECONDS + 1e-12:
            continue

        tensor[index, 0] = float(row.cx_norm)
        tensor[index, 1] = float(row.cy_norm)
        tensor[index, 2] = float(row.w_norm)
        tensor[index, 3] = float(row.h_norm)
        tensor[index, 4] = float(row.confidence)
        tensor[index, 5] = 1.0

    return tensor


def _validate_row(row: RawTrackLike) -> None:
    if not isinstance(row.video_id, str) or not row.video_id:
        raise MechanismArtifactInvalid("raw track video_id must be non-empty")
    if type(row.track_id) is not int or row.track_id < 0:
        raise MechanismArtifactInvalid(
            "raw track track_id must be a non-negative integer"
        )

    timestamp = _finite(row.timestamp_seconds, "timestamp_seconds")
    if timestamp < 0.0:
        raise MechanismArtifactInvalid(
            "raw track timestamp_seconds must be non-negative"
        )

    for name in ("confidence", "cx_norm", "cy_norm", "w_norm", "h_norm"):
        value = _finite(getattr(row, name), name)
        if not 0.0 <= value <= 1.0:
            raise MechanismArtifactInvalid(
                f"raw track {name} must be in [0, 1]"
            )

    if float(row.w_norm) <= 0.0 or float(row.h_norm) <= 0.0:
        raise MechanismArtifactInvalid(
            "raw track normalized bbox size must be positive"
        )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanismArtifactInvalid(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MechanismArtifactInvalid(f"{name} must be finite")
    return result
