from __future__ import annotations

from collections.abc import Mapping

from neural_state_machine.r1_e3m_artifact import MechanismVideoRecord


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "1db8cc200ba9eaab2c14ed260b06ed582ed75d54f57c3036e1312cd2f184f4cf"
)
SOURCE_MANIFEST_SCHEMA = "r1-e3a-source-video-manifest-v1"
ULTRALYTICS_VERSION = "8.4.155"
DETECTOR_MODEL = "yolo11n.pt"
CLASS_FILTER = (0,)
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.70
IMGSZ = 640
DEVICE = "cpu"
HALF = False
NATIVE_FRAME_STRIDE = 6
TRACKER_CONFIG = "configs/r1_e3m_botsort.yaml"
PREDICTED_ONLY_ROWS_ALLOWED = False


class ExtractionContractInvalid(RuntimeError):
    pass


def extraction_settings_payload() -> dict[str, object]:
    return {
        "ultralytics_version": ULTRALYTICS_VERSION,
        "detector_model": DETECTOR_MODEL,
        "class_filter": list(CLASS_FILTER),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "imgsz": IMGSZ,
        "device": DEVICE,
        "half": HALF,
        "native_frame_stride": NATIVE_FRAME_STRIDE,
        "effective_fps_for_30fps_source": 5.0,
        "tracker_config": TRACKER_CONFIG,
        "predicted_only_rows_allowed": PREDICTED_ONLY_ROWS_ALLOWED,
    }


def mechanism_videos_from_source_manifest(
    payload: Mapping[str, object],
) -> tuple[MechanismVideoRecord, ...]:
    if payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ExtractionContractInvalid(
            "source manifest schema mismatch"
        )
    videos_raw = payload.get("videos")
    if not isinstance(videos_raw, list) or not videos_raw:
        raise ExtractionContractInvalid(
            "source manifest videos must be a non-empty array"
        )

    videos: list[MechanismVideoRecord] = []
    seen_ids: set[str] = set()
    for value in videos_raw:
        if not isinstance(value, dict):
            raise ExtractionContractInvalid(
                "source manifest video must be an object"
            )
        video_id = _text(
            value.get("source_video_id"),
            "source_video_id",
        )
        if video_id in seen_ids:
            raise ExtractionContractInvalid(
                f"duplicate source_video_id: {video_id}"
            )
        seen_ids.add(video_id)

        split = value.get("split")
        if split not in ("train", "eval"):
            raise ExtractionContractInvalid(
                f"invalid source split for {video_id}"
            )

        videos.append(
            MechanismVideoRecord(
                video_id=video_id,
                sha256=_digest(
                    value.get("sha256"),
                    f"{video_id}.sha256",
                ),
                split=split,
                fps=_positive_number(
                    value.get("fps"),
                    f"{video_id}.fps",
                ),
                frame_count=_positive_int(
                    value.get("frame_count"),
                    f"{video_id}.frame_count",
                ),
                width=_positive_int(
                    value.get("width"),
                    f"{video_id}.width",
                ),
                height=_positive_int(
                    value.get("height"),
                    f"{video_id}.height",
                ),
                source_window_component_id=_text(
                    value.get("source_window_component_id"),
                    f"{video_id}.source_window_component_id",
                ),
            )
        )

    return tuple(
        sorted(videos, key=lambda video: video.video_id)
    )


def verify_window_video_coverage(
    videos: tuple[MechanismVideoRecord, ...],
    window_counts_by_video: Mapping[str, int],
) -> None:
    expected = {video.video_id for video in videos}
    extra = sorted(set(window_counts_by_video) - expected)
    if extra:
        raise ExtractionContractInvalid(
            f"window counts reference unknown videos: {extra}"
        )

    missing = sorted(
        video_id
        for video_id in expected
        if int(window_counts_by_video.get(video_id, 0)) <= 0
    )
    if missing:
        raise ExtractionContractInvalid(
            f"missing eligible windows for frozen videos: {missing}"
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExtractionContractInvalid(
            f"{name} must be a non-empty string"
        )
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ExtractionContractInvalid(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ExtractionContractInvalid(
            f"{name} must be numeric"
        )
    number = float(value)
    if number <= 0.0:
        raise ExtractionContractInvalid(
            f"{name} must be positive"
        )
    return number


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ExtractionContractInvalid(
            f"{name} must be a positive integer"
        )
    return value
