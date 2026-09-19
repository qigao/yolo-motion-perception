from __future__ import annotations

from pathlib import Path

import pytest


def _api():
    from neural_state_machine.r1_e3m_extraction import (
        EXPECTED_SOURCE_MANIFEST_SHA256,
        ExtractionContractInvalid,
        extraction_settings_payload,
        mechanism_videos_from_source_manifest,
        verify_window_video_coverage,
    )

    return (
        EXPECTED_SOURCE_MANIFEST_SHA256,
        ExtractionContractInvalid,
        extraction_settings_payload,
        mechanism_videos_from_source_manifest,
        verify_window_video_coverage,
    )


def _manifest() -> dict[str, object]:
    return {
        "schema": "r1-e3a-source-video-manifest-v1",
        "science_head_sha": "9" * 40,
        "videos": [
            {
                "source_video_id": "train-a",
                "sha256": "1" * 64,
                "split": "train",
                "fps": 30.0,
                "frame_count": 9000,
                "width": 1920,
                "height": 1080,
                "source_window_component_id": "train-component",
            },
            {
                "source_video_id": "eval-a",
                "sha256": "2" * 64,
                "split": "eval",
                "fps": 30.0,
                "frame_count": 9000,
                "width": 1920,
                "height": 1080,
                "source_window_component_id": "eval-component",
            },
        ],
    }


def test_extraction_settings_are_frozen_before_scoring() -> None:
    expected_sha, _, settings, _, _ = _api()

    payload = settings()

    assert (
        expected_sha
        == "1db8cc200ba9eaab2c14ed260b06ed582ed75d54f57c3036e1312cd2f184f4cf"
    )
    assert payload == {
        "ultralytics_version": "8.4.155",
        "detector_model": "yolo11n.pt",
        "class_filter": [0],
        "confidence_threshold": 0.25,
        "iou_threshold": 0.70,
        "imgsz": 640,
        "device": "cpu",
        "half": False,
        "native_frame_stride": 6,
        "effective_fps_for_30fps_source": 5.0,
        "tracker_config": "configs/r1_e3m_botsort.yaml",
        "predicted_only_rows_allowed": False,
    }


def test_source_manifest_converts_without_semantic_fields() -> None:
    _, _, _, convert, _ = _api()

    videos = convert(_manifest())

    assert [video.video_id for video in videos] == ["eval-a", "train-a"]
    assert [video.split for video in videos] == ["eval", "train"]
    assert all(video.fps == 30.0 for video in videos)


def test_source_manifest_rejects_wrong_schema() -> None:
    _, Invalid, _, convert, _ = _api()
    payload = _manifest()
    payload["schema"] = "wrong"

    with pytest.raises(Invalid, match="schema"):
        convert(payload)


def test_all_frozen_source_videos_must_have_eligible_windows() -> None:
    _, Invalid, _, convert, verify_coverage = _api()
    videos = convert(_manifest())

    with pytest.raises(Invalid, match="missing eligible windows"):
        verify_coverage(videos, {"train-a": 3})


def test_tracker_config_contains_frozen_botsort_values() -> None:
    path = Path("configs/r1_e3m_botsort.yaml")
    text = path.read_text(encoding="utf-8")

    for required in (
        "tracker_type: botsort",
        "track_high_thresh: 0.25",
        "track_low_thresh: 0.1",
        "new_track_thresh: 0.25",
        "track_buffer: 30",
        "match_thresh: 0.8",
        "gmc_method: sparseOptFlow",
        "with_reid: false",
    ):
        assert required in text


def test_extraction_module_contains_no_semantic_label_dependency() -> None:
    path = Path(
        "src/neural_state_machine/r1_e3m_extraction.py"
    )
    text = path.read_text(encoding="utf-8")

    forbidden = (
        "batch1-semantic-review",
        "semantic-review",
        '"label"',
        "episode_id",
        "actor_track_id",
        "target_track_id",
    )
    assert not any(value in text for value in forbidden)
