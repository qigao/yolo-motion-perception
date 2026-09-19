from __future__ import annotations

from inspect import signature

import numpy as np

from neural_state_machine.r1_e3_artifact import RawTrackRow
from neural_state_machine.r1_e3m_artifact import MechanismVideoRecord


def _api():
    from neural_state_machine.r1_e3m_window import build_track_windows

    return build_track_windows


def _video(video_id: str = "video-a", split: str = "train"):
    return MechanismVideoRecord(
        video_id=video_id,
        sha256=("1" if split == "train" else "2") * 64,
        split=split,
        fps=30.0,
        frame_count=600,
        width=1280,
        height=720,
        source_window_component_id=f"{split}-component",
    )


def _row(
    timestamp: float,
    *,
    video_id: str = "video-a",
    track_id: int = 7,
    frame_index: int | None = None,
) -> RawTrackRow:
    if frame_index is None:
        frame_index = int(round(timestamp * 30.0))
    cx = 0.1 + timestamp * 0.01
    return RawTrackRow(
        video_id=video_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        track_id=track_id,
        class_id=0,
        confidence=0.9,
        x1=100.0,
        y1=100.0,
        x2=200.0,
        y2=300.0,
        cx_norm=cx,
        cy_norm=0.3,
        w_norm=0.1,
        h_norm=0.2,
    )


def _dense_rows(
    start: float,
    end_inclusive: float,
    *,
    step: float = 0.1,
    video_id: str = "video-a",
    track_id: int = 7,
) -> list[RawTrackRow]:
    count = int(round((end_inclusive - start) / step)) + 1
    return [
        _row(
            round(start + index * step, 9),
            video_id=video_id,
            track_id=track_id,
        )
        for index in range(count)
    ]


def test_builder_has_no_semantic_annotation_parameter() -> None:
    build_track_windows = _api()

    names = set(signature(build_track_windows).parameters)

    assert "annotations" not in names
    assert "episodes" not in names
    assert "labels" not in names


def test_dense_track_produces_exact_20_by_6_window() -> None:
    build_track_windows = _api()
    rows = _dense_rows(0.0, 1.9)

    windows = build_track_windows(rows, (_video(),))

    assert len(windows) == 1
    window = windows[0]
    assert window.source_start_seconds == 0.0
    assert window.source_end_seconds == 2.0
    assert window.tensor.shape == (20, 6)
    assert window.tensor.dtype == np.float64
    assert int(window.tensor[:, 5].sum()) == 20


def test_windows_are_anchored_to_source_video_time_not_track_start() -> None:
    build_track_windows = _api()
    rows = _dense_rows(2.1, 3.9)

    windows = build_track_windows(rows, (_video(),))

    assert len(windows) == 1
    assert windows[0].source_start_seconds == 2.0
    assert windows[0].source_end_seconds == 4.0


def test_future_observation_never_populates_earlier_bin() -> None:
    build_track_windows = _api()
    rows = _dense_rows(0.1, 1.9)

    windows = build_track_windows(rows, (_video(),))

    assert len(windows) == 1
    assert windows[0].tensor[0, 5] == 0.0
    assert np.all(windows[0].tensor[0, :5] == 0.0)
    assert windows[0].tensor[1, 5] == 1.0


def test_observation_older_than_half_second_is_missing() -> None:
    build_track_windows = _api()
    rows = [_row(0.0), _row(0.1), _row(0.2), _row(1.0)]

    windows = build_track_windows(rows, (_video(),))

    assert windows == ()


def test_exactly_sixteen_present_bins_is_accepted() -> None:
    build_track_windows = _api()
    rows = _dense_rows(0.0, 1.1)

    windows = build_track_windows(rows, (_video(),))

    assert len(windows) == 1
    assert int(windows[0].tensor[:, 5].sum()) == 16


def test_fifteen_present_bins_is_rejected() -> None:
    build_track_windows = _api()
    rows = _dense_rows(0.0, 1.0)

    windows = build_track_windows(rows, (_video(),))

    assert windows == ()


def test_track_ids_are_never_bridged() -> None:
    build_track_windows = _api()
    rows = [
        *_dense_rows(0.0, 0.9, track_id=7),
        *_dense_rows(1.0, 1.9, track_id=8),
    ]

    windows = build_track_windows(rows, (_video(),))

    assert windows == ()


def test_builder_is_repeatable_and_sorted() -> None:
    build_track_windows = _api()
    videos = (
        _video("video-a", "train"),
        _video("video-b", "eval"),
    )
    rows = [
        *_dense_rows(0.0, 1.9, video_id="video-b", track_id=4),
        *_dense_rows(2.0, 3.9, video_id="video-a", track_id=9),
        *_dense_rows(0.0, 1.9, video_id="video-a", track_id=3),
    ]

    first = build_track_windows(rows, videos)
    second = build_track_windows(list(reversed(rows)), videos)

    assert [window.window_id for window in first] == [
        window.window_id for window in second
    ]
    assert [
        (
            window.video_id,
            window.track_id,
            window.source_start_seconds,
        )
        for window in first
    ] == [
        ("video-a", 3, 0.0),
        ("video-a", 9, 2.0),
        ("video-b", 4, 0.0),
    ]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.tensor, right.tensor)
