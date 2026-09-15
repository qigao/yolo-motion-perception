from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pose_types import PoseLayout, PoseObservation
from .types import TrackObservation

_MIN_MEAN_KEYPOINT_CONFIDENCE = 0.05


@dataclass(frozen=True)
class RtmwAdapterConfig:
    model_config: str
    weights: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.model_config.strip():
            raise ValueError("RTMW model config must not be empty")
        if not self.weights.strip():
            raise ValueError("RTMW weights must not be empty")
        if not self.device.strip():
            raise ValueError("RTMW device must not be empty")


class RtmwPoseAdapter:
    def __init__(self, inferencer: object):
        if not callable(inferencer):
            raise TypeError("RTMW inferencer must be callable")
        self._inferencer = inferencer

    def infer_track(
        self,
        frame: np.ndarray,
        track: TrackObservation,
        timestamp: float,
    ) -> PoseObservation | None:
        if not math.isfinite(timestamp):
            raise ValueError("pose timestamp must be finite")

        image = np.asarray(frame)
        if image.ndim not in (2, 3):
            raise ValueError("frame must be a grayscale or color image array")
        frame_height, frame_width = image.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")

        x1, y1, x2, y2 = _track_crop(track, frame_width, frame_height)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        result_generator = self._inferencer(crop, return_vis=False, show=False)
        result = next(iter(result_generator), None)
        sample = _first_prediction(result)
        if sample is None:
            return None

        keypoints, confidence = _prediction_arrays(sample, track.track_id)
        if float(np.mean(confidence)) < _MIN_MEAN_KEYPOINT_CONFIDENCE:
            return None

        full_frame_xy = np.array(keypoints, dtype=float, copy=True)
        full_frame_xy[:, 0] = (full_frame_xy[:, 0] + x1) / frame_width
        full_frame_xy[:, 1] = (full_frame_xy[:, 1] + y1) / frame_height
        np.clip(full_frame_xy, 0.0, 1.0, out=full_frame_xy)

        return PoseObservation(
            track_id=track.track_id,
            timestamp=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
            xy=full_frame_xy,
            confidence=confidence,
            layout=PoseLayout.COCO_WHOLEBODY_133,
        )


def load_mmpose_rtmw(config: RtmwAdapterConfig) -> RtmwPoseAdapter:
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError as exc:
        raise RuntimeError(
            "MMPose is required only for the RTMW runtime; install yolo-motion-perception[rtmw]"
        ) from exc

    inferencer = MMPoseInferencer(
        pose2d=config.model_config,
        pose2d_weights=config.weights,
        device=config.device,
        det_model="whole_image",
    )
    return RtmwPoseAdapter(inferencer)


def _pixel_floor(value: float) -> int:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest)
    return math.floor(value)


def _pixel_ceil(value: float) -> int:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest)
    return math.ceil(value)


def _track_crop(
    track: TrackObservation,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    left = (track.cx - track.width / 2.0) * frame_width
    right = (track.cx + track.width / 2.0) * frame_width
    top = (track.cy - track.height / 2.0) * frame_height
    bottom = (track.cy + track.height / 2.0) * frame_height

    x1 = max(0, min(frame_width, _pixel_floor(left)))
    y1 = max(0, min(frame_height, _pixel_floor(top)))
    x2 = max(0, min(frame_width, _pixel_ceil(right)))
    y2 = max(0, min(frame_height, _pixel_ceil(bottom)))
    return x1, y1, x2, y2


def _first_prediction(result: Any) -> Any | None:
    if result is None:
        return None
    if not isinstance(result, dict):
        raise ValueError("RTMW inferencer result must be a mapping")

    predictions = result.get("predictions")
    if predictions is None:
        return None
    while isinstance(predictions, (list, tuple)) and len(predictions) == 1 and isinstance(
        predictions[0], (list, tuple)
    ):
        predictions = predictions[0]
    if not isinstance(predictions, (list, tuple)) or not predictions:
        return None
    return predictions[0]


def _prediction_arrays(sample: Any, track_id: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(sample, dict):
        raise ValueError(f"RTMW prediction for track {track_id} must be a mapping")

    keypoints = np.asarray(sample.get("keypoints"), dtype=float)
    if keypoints.ndim == 3 and keypoints.shape[0] == 1:
        keypoints = keypoints[0]
    if keypoints.shape != (133, 2):
        count = keypoints.shape[-2] if keypoints.ndim >= 2 else 0
        raise ValueError(
            f"RTMW prediction for track {track_id} must contain 133 keypoints; got {count}"
        )

    raw_scores = sample.get("keypoint_scores")
    if raw_scores is None:
        raw_scores = sample.get("keypoints_visible")
    confidence = np.asarray(raw_scores, dtype=float)
    if confidence.ndim == 2 and confidence.shape[0] == 1:
        confidence = confidence[0]
    if confidence.shape != (133,):
        raise ValueError(
            f"RTMW prediction for track {track_id} must contain 133 keypoint scores"
        )
    if not np.isfinite(keypoints).all() or not np.isfinite(confidence).all():
        raise ValueError(f"RTMW prediction for track {track_id} contains non-finite values")

    confidence = np.clip(confidence, 0.0, 1.0)
    return keypoints, confidence
