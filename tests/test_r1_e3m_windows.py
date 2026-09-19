import numpy as np

from neural_state_machine.r1_e3m_artifact import (
    SourceVideo,
    TrackArtifact,
    TrackRow,
)
from neural_state_machine.r1_e3m_windows import build_pair_windows


def _row(
    *,
    timestamp: float,
    track_id: int,
    class_id: int,
    frame_index: int,
    cx: float,
) -> TrackRow:
    return TrackRow(
        video_id="video-a",
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        track_id=track_id,
        class_id=class_id,
        confidence=0.9,
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=220.0,
        cx_norm=cx,
        cy_norm=0.5,
        w_norm=0.2,
        h_norm=0.4,
    )


def _artifact(rows: list[TrackRow], *, duration: float = 4.0) -> TrackArtifact:
    return TrackArtifact(
        videos=(
            SourceVideo(
                video_id="video-a",
                sha256="1" * 64,
                split="train",
                fps=30.0,
                frame_count=1000,
                duration_seconds=duration,
                width=640,
                height=480,
                source_window_component_id="component-a",
            ),
        ),
        tracks=tuple(rows),
        provenance={},
        artifact_manifest={},
    )


def _dense_pair(*, target_bins: int = 20, actor_classes=None) -> list[TrackRow]:
    rows = []
    actor_classes = actor_classes or [0] * 20
    for index in range(20):
        timestamp = 0.1 + 0.2 * index
        rows.append(
            _row(
                timestamp=timestamp,
                track_id=1,
                class_id=actor_classes[index],
                frame_index=index * 2,
                cx=0.2 + 0.01 * index,
            )
        )
        if index < target_bins:
            rows.append(
                _row(
                    timestamp=timestamp,
                    track_id=2,
                    class_id=24,
                    frame_index=index * 2,
                    cx=0.7 - 0.01 * index,
                )
            )
    return sorted(rows, key=lambda row: (row.frame_index, row.track_id))


def test_fixed_window_has_twenty_center_samples_and_fourteen_channels():
    windows = build_pair_windows(
        _artifact(_dense_pair()),
        artifact_root_digest="a" * 64,
    )

    assert len(windows) == 1
    window = windows[0]
    assert window.start_seconds == 0.0
    assert window.end_seconds == 4.0
    assert window.tensor.shape == (20, 14)
    assert window.tensor.dtype == np.float64
    assert np.all(window.tensor[:, 5] == 1.0)
    assert np.all(window.tensor[:, 11] == 1.0)


def test_partial_end_window_is_not_created():
    rows = _dense_pair()
    second = [
        TrackRow(
            **{
                **row.__dict__,
                "timestamp_seconds": row.timestamp_seconds + 4.0,
                "frame_index": row.frame_index + 200,
            }
        )
        for row in rows
        if row.timestamp_seconds + 4.0 < 6.0
    ]

    windows = build_pair_windows(
        _artifact(rows + second, duration=6.0),
        artifact_root_digest="a" * 64,
    )

    assert {window.start_seconds for window in windows} == {0.0}


def test_presence_gate_accepts_sixteen_and_rejects_fifteen():
    accepted = build_pair_windows(
        _artifact(_dense_pair(target_bins=16)),
        artifact_root_digest="a" * 64,
    )
    rejected = build_pair_windows(
        _artifact(_dense_pair(target_bins=15)),
        artifact_root_digest="a" * 64,
    )

    assert len(accepted) == 1
    assert rejected == ()


def test_actor_modal_class_tie_uses_lower_class_id():
    classes = [0, 1] * 10
    windows = build_pair_windows(
        _artifact(_dense_pair(actor_classes=classes)),
        artifact_root_digest="a" * 64,
    )

    assert len(windows) == 1
    assert windows[0].actor_track_id == 1


def test_interpolation_gap_above_half_second_yields_missing_presence():
    rows = _dense_pair()
    rows = [
        row
        for row in rows
        if not (
            row.track_id == 2
            and 0.1 < row.timestamp_seconds < 0.9
        )
    ]

    windows = build_pair_windows(
        _artifact(rows),
        artifact_root_digest="a" * 64,
    )

    window = windows[0]
    assert window.tensor[1, 11] == 0.0


def test_sequence_identity_and_tensor_digest_are_deterministic():
    artifact = _artifact(_dense_pair())

    first = build_pair_windows(artifact, artifact_root_digest="a" * 64)
    second = build_pair_windows(artifact, artifact_root_digest="a" * 64)

    assert first[0].sequence_id == second[0].sequence_id
    assert first[0].tensor_digest == second[0].tensor_digest
    assert np.array_equal(first[0].tensor, second[0].tensor)
