from __future__ import annotations

import math

import numpy as np
import pytest

from neural_state_machine.r1_e3_artifact import (
    EpisodeAnnotation,
    FrozenTrackArtifact,
    RawTrackRow,
    SourceVideoRecord,
)


def _api():
    from neural_state_machine.r1_e3_normalize import (
        normalize_episode,
        normalize_registered_episodes,
    )

    return normalize_episode, normalize_registered_episodes


def _row(
    *,
    timestamp: float,
    track_id: int,
    cx: float,
    cy: float,
    width: float,
    height: float,
    confidence: float,
) -> RawTrackRow:
    frame_index = int(round(timestamp * 100.0))
    return RawTrackRow(
        video_id="video-1",
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        track_id=track_id,
        class_id=0 if track_id == 10 else 1,
        confidence=confidence,
        x1=(cx - width / 2.0) * 1000.0,
        y1=(cy - height / 2.0) * 1000.0,
        x2=(cx + width / 2.0) * 1000.0,
        y2=(cy + height / 2.0) * 1000.0,
        cx_norm=cx,
        cy_norm=cy,
        w_norm=width,
        h_norm=height,
    )


def _artifact(
    *,
    actor_times: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    target_times: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[FrozenTrackArtifact, EpisodeAnnotation]:
    actor_rows = tuple(
        _row(
            timestamp=t,
            track_id=10,
            cx=0.20 + 0.20 * t,
            cy=0.30,
            width=0.10,
            height=0.20,
            confidence=0.80 + 0.10 * t,
        )
        for t in actor_times
    )
    target_rows = tuple(
        _row(
            timestamp=t,
            track_id=20,
            cx=0.70,
            cy=0.30 + 0.10 * t,
            width=0.20,
            height=0.20,
            confidence=0.90,
        )
        for t in target_times
    )
    tracks = tuple(
        sorted(
            (*actor_rows, *target_rows),
            key=lambda row: (row.video_id, row.frame_index, row.track_id),
        )
    )
    episode = EpisodeAnnotation(
        episode_id="episode-1",
        video_id="video-1",
        start_time_seconds=0.0,
        end_time_seconds=1.0,
        label="approach",
        actor_track_id=10,
        target_track_id=20,
        annotation_revision="r1",
        reviewer="reviewer-a",
    )
    artifact = FrozenTrackArtifact(
        videos=(
            SourceVideoRecord(
                video_id="video-1",
                sha256="1" * 64,
                split="training",
                fps=100.0,
                frame_count=101,
                width=1000,
                height=1000,
            ),
        ),
        episodes=(episode,),
        tracks=tracks,
        provenance={},
        manifest={},
    )
    return artifact, episode


def test_normalizer_returns_exact_20_by_14_float64_readonly_tensor() -> None:
    normalize_episode, _ = _api()
    artifact, episode = _artifact()

    normalized = normalize_episode(artifact, episode)

    assert normalized.episode_id == "episode-1"
    assert normalized.video_id == "video-1"
    assert normalized.label == "approach"
    assert normalized.tensor.shape == (20, 14)
    assert normalized.tensor.dtype == np.float64
    assert not normalized.tensor.flags.writeable
    with pytest.raises(ValueError):
        normalized.tensor[0, 0] = 9.0


def test_interpolation_is_timestamp_linear_with_fixed_20_bin_grid() -> None:
    normalize_episode, _ = _api()
    artifact, episode = _artifact()

    normalized = normalize_episode(artifact, episode)

    timestamp = 10.0 / 19.0
    assert normalized.tensor[10, 0] == pytest.approx(0.20 + 0.20 * timestamp)
    assert normalized.tensor[10, 4] == pytest.approx(0.80 + 0.10 * timestamp)
    assert normalized.tensor[10, 5] == 1.0
    assert normalized.tensor[10, 11] == 1.0


def test_interpolation_is_forbidden_when_bracketing_gap_exceeds_half_second() -> None:
    normalize_episode, _ = _api()
    artifact, episode = _artifact(actor_times=(0.0, 0.6, 1.0))

    normalized = normalize_episode(artifact, episode)

    # Bin 6 is at 6/19 ~= 0.316, bracketed by actor rows at 0.0 and 0.6.
    assert np.array_equal(normalized.tensor[6, 0:6], np.zeros(6, dtype=np.float64))
    assert normalized.tensor[6, 12] == 0.0
    assert normalized.tensor[6, 13] == 0.0


def test_pair_features_match_registered_distance_and_iou_formulas() -> None:
    normalize_episode, _ = _api()
    artifact, episode = _artifact()

    normalized = normalize_episode(artifact, episode)
    first = normalized.tensor[0]

    expected_distance = math.hypot(0.20 - 0.70, 0.30 - 0.30) / math.sqrt(2.0)
    # Actor box [0.15, 0.20, 0.25, 0.40], target [0.60, 0.20, 0.80, 0.40]
    # are disjoint at t=0.
    assert first[12] == pytest.approx(expected_distance)
    assert first[13] == 0.0


def test_normalized_episode_digest_is_repeatable_and_tensor_sensitive() -> None:
    normalize_episode, _ = _api()
    artifact, episode = _artifact()
    changed_artifact, changed_episode = _artifact(
        target_times=(0.0, 0.20, 0.40, 0.60, 0.80, 1.0)
    )

    left = normalize_episode(artifact, episode)
    same = normalize_episode(artifact, episode)
    changed = normalize_episode(changed_artifact, changed_episode)

    assert left.tensor_digest == same.tensor_digest
    assert len(left.tensor_digest) == 64
    assert changed.tensor_digest != left.tensor_digest


def test_normalize_registered_episodes_preserves_frozen_episode_order() -> None:
    _, normalize_all = _api()
    artifact, episode = _artifact()
    second = EpisodeAnnotation(
        episode_id="episode-2",
        video_id=episode.video_id,
        start_time_seconds=episode.start_time_seconds,
        end_time_seconds=episode.end_time_seconds,
        label="pass_by",
        actor_track_id=episode.actor_track_id,
        target_track_id=episode.target_track_id,
        annotation_revision="r1",
        reviewer="reviewer-b",
    )
    artifact = FrozenTrackArtifact(
        videos=artifact.videos,
        episodes=(episode, second),
        tracks=artifact.tracks,
        provenance=artifact.provenance,
        manifest=artifact.manifest,
    )

    normalized = normalize_all(artifact)

    assert tuple(item.episode_id for item in normalized) == ("episode-1", "episode-2")
