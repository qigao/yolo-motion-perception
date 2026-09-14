from __future__ import annotations

from typing import Any

import numpy as np

from .types import TrackObservation


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def observations_from_result(
    result: Any,
    timestamp: float,
    frame_shape: tuple[int, int],
) -> list[TrackObservation]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return []

    frame_height, frame_width = frame_shape
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_shape must contain positive height and width")

    track_ids = _to_numpy(boxes.id).reshape(-1)
    xywh = _to_numpy(boxes.xywh).reshape(-1, 4)
    confidences = _to_numpy(boxes.conf).reshape(-1)
    classes = _to_numpy(boxes.cls).reshape(-1)

    if not (len(track_ids) == len(xywh) == len(confidences) == len(classes)):
        raise ValueError("tracker result arrays must have matching lengths")

    observations: list[TrackObservation] = []
    for track_id, (cx, cy, width, height), confidence, class_id in zip(
        track_ids, xywh, confidences, classes, strict=True
    ):
        observations.append(
            TrackObservation(
                track_id=int(track_id),
                timestamp=float(timestamp),
                class_id=int(class_id),
                confidence=float(confidence),
                cx=float(cx) / frame_width,
                cy=float(cy) / frame_height,
                width=float(width) / frame_width,
                height=float(height) / frame_height,
            )
        )
    return observations
