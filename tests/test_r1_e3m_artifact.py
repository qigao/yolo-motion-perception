from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest


def _api():
    from neural_state_machine.r1_e3m_artifact import (
        MechanismArtifact,
        MechanismArtifactInvalid,
        MechanismVideoRecord,
        TrackWindow,
        mechanism_artifact_digest,
        track_window_id,
        validate_mechanism_artifact,
    )

    return (
        MechanismArtifact,
        MechanismArtifactInvalid,
        MechanismVideoRecord,
        TrackWindow,
        mechanism_artifact_digest,
        track_window_id,
        validate_mechanism_artifact,
    )


def _video(video_id: str, digest: str, split: str):
    _, _, MechanismVideoRecord, _, _, _, _ = _api()
    return MechanismVideoRecord(
        video_id=video_id,
        sha256=digest,
        split=split,
        fps=30.0,
        frame_count=300,
        width=1280,
        height=720,
        source_window_component_id=f"{split}-component",
    )


def _window(video_id: str, split: str, track_id: int, start: float):
    _, _, _, TrackWindow, _, track_window_id, _ = _api()
    tensor = np.zeros((20, 6), dtype=np.float64)
    tensor[:, 0] = np.linspace(0.1, 0.2, 20)
    tensor[:, 1] = 0.3
    tensor[:, 2] = 0.1
    tensor[:, 3] = 0.2
    tensor[:, 4] = 0.9
    tensor[:16, 5] = 1.0
    end = start + 2.0
    return TrackWindow(
        window_id=track_window_id(video_id, track_id, start, end),
        video_id=video_id,
        split=split,
        track_id=track_id,
        source_start_seconds=start,
        source_end_seconds=end,
        tensor=tensor,
    )


def _artifact():
    MechanismArtifact, _, _, _, _, _, _ = _api()
    videos = (
        _video("train-video", "1" * 64, "train"),
        _video("eval-video", "2" * 64, "eval"),
    )
    windows = (
        _window("train-video", "train", 10, 0.0),
        _window("eval-video", "eval", 20, 0.0),
    )
    return MechanismArtifact(
        source_manifest_sha256="3" * 64,
        raw_track_sha256="4" * 64,
        extraction_provenance_sha256="5" * 64,
        videos=videos,
        windows=windows,
    )


def test_track_window_is_exact_20_by_6_float64_and_read_only() -> None:
    window = _window("train-video", "train", 10, 0.0)

    assert window.tensor.shape == (20, 6)
    assert window.tensor.dtype == np.float64
    assert window.tensor.flags.writeable is False


def test_track_window_id_is_deterministic_from_identity() -> None:
    *_, track_window_id, _ = _api()

    first = track_window_id("video-a", 7, 2.0, 4.0)
    second = track_window_id("video-a", 7, 2.0, 4.0)

    assert first == second
    assert len(first) == 64
    assert first != track_window_id("video-a", 8, 2.0, 4.0)


def test_schema_has_no_semantic_label_or_episode_fields() -> None:
    _, _, _, TrackWindow, _, _, _ = _api()

    names = {field.name for field in fields(TrackWindow)}

    assert "label" not in names
    assert "episode_id" not in names
    assert "actor_track_id" not in names
    assert "target_track_id" not in names


def test_invalid_tensor_shape_fails_closed() -> None:
    _, MechanismArtifactInvalid, _, TrackWindow, _, track_window_id, _ = _api()

    with pytest.raises(MechanismArtifactInvalid, match="20 x 6"):
        TrackWindow(
            window_id=track_window_id("video", 1, 0.0, 2.0),
            video_id="video",
            split="train",
            track_id=1,
            source_start_seconds=0.0,
            source_end_seconds=2.0,
            tensor=np.zeros((20, 5), dtype=np.float64),
        )


def test_non_float64_tensor_fails_closed() -> None:
    _, MechanismArtifactInvalid, _, TrackWindow, _, track_window_id, _ = _api()

    with pytest.raises(MechanismArtifactInvalid, match="float64"):
        TrackWindow(
            window_id=track_window_id("video", 1, 0.0, 2.0),
            video_id="video",
            split="train",
            track_id=1,
            source_start_seconds=0.0,
            source_end_seconds=2.0,
            tensor=np.zeros((20, 6), dtype=np.float32),
        )


def test_duplicate_window_id_fails_closed() -> None:
    (
        MechanismArtifact,
        MechanismArtifactInvalid,
        _,
        _,
        _,
        _,
        validate,
    ) = _api()
    base = _artifact()

    duplicate = MechanismArtifact(
        source_manifest_sha256=base.source_manifest_sha256,
        raw_track_sha256=base.raw_track_sha256,
        extraction_provenance_sha256=base.extraction_provenance_sha256,
        videos=base.videos,
        windows=(base.windows[0], base.windows[0], base.windows[1]),
    )

    with pytest.raises(MechanismArtifactInvalid, match="duplicate window_id"):
        validate(duplicate)


def test_window_split_must_match_source_video() -> None:
    (
        MechanismArtifact,
        MechanismArtifactInvalid,
        _,
        TrackWindow,
        _,
        track_window_id,
        validate,
    ) = _api()
    base = _artifact()
    bad = TrackWindow(
        window_id=track_window_id("train-video", 11, 2.0, 4.0),
        video_id="train-video",
        split="eval",
        track_id=11,
        source_start_seconds=2.0,
        source_end_seconds=4.0,
        tensor=base.windows[0].tensor,
    )
    artifact = MechanismArtifact(
        source_manifest_sha256=base.source_manifest_sha256,
        raw_track_sha256=base.raw_track_sha256,
        extraction_provenance_sha256=base.extraction_provenance_sha256,
        videos=base.videos,
        windows=base.windows + (bad,),
    )

    with pytest.raises(MechanismArtifactInvalid, match="split mismatch"):
        validate(artifact)


def test_video_hash_overlap_between_splits_fails_closed() -> None:
    (
        MechanismArtifact,
        MechanismArtifactInvalid,
        _,
        _,
        _,
        _,
        validate,
    ) = _api()
    base = _artifact()
    videos = (
        _video("train-video", "a" * 64, "train"),
        _video("eval-video", "a" * 64, "eval"),
    )
    artifact = MechanismArtifact(
        source_manifest_sha256=base.source_manifest_sha256,
        raw_track_sha256=base.raw_track_sha256,
        extraction_provenance_sha256=base.extraction_provenance_sha256,
        videos=videos,
        windows=base.windows,
    )

    with pytest.raises(MechanismArtifactInvalid, match="hash overlap"):
        validate(artifact)


def test_artifact_digest_is_deterministic_and_content_addressed() -> None:
    *_, mechanism_artifact_digest, _, validate = _api()
    artifact = _artifact()
    validate(artifact)

    first = mechanism_artifact_digest(artifact)
    second = mechanism_artifact_digest(artifact)

    assert first == second
    assert len(first) == 64

    changed = _artifact()
    tensor = changed.windows[0].tensor.copy()
    tensor.setflags(write=True)
    tensor[0, 0] += 0.01
    replacement = type(changed.windows[0])(
        window_id=changed.windows[0].window_id,
        video_id=changed.windows[0].video_id,
        split=changed.windows[0].split,
        track_id=changed.windows[0].track_id,
        source_start_seconds=changed.windows[0].source_start_seconds,
        source_end_seconds=changed.windows[0].source_end_seconds,
        tensor=tensor,
    )
    changed = type(changed)(
        source_manifest_sha256=changed.source_manifest_sha256,
        raw_track_sha256=changed.raw_track_sha256,
        extraction_provenance_sha256=changed.extraction_provenance_sha256,
        videos=changed.videos,
        windows=(replacement, changed.windows[1]),
    )

    assert mechanism_artifact_digest(changed) != first
