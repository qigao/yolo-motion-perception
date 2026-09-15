from __future__ import annotations

from typing import Protocol

import numpy as np

from .flow_types import FlowObservation


class FlowBackend(Protocol):
    def compute(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        start_timestamp: float,
        end_timestamp: float,
    ) -> FlowObservation: ...


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by optional runtime only
        raise RuntimeError(
            "OpenCV is required for optical flow; install yolo-motion-perception[optical]"
        ) from exc
    return cv2


def _as_gray(frame: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    array = np.asarray(frame)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] in {3, 4}:
        code = cv2.COLOR_BGR2GRAY if array.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(array, code)
    else:
        raise ValueError("frame must be a 2D grayscale or 3/4-channel image")

    if not np.issubdtype(gray.dtype, np.number):
        raise ValueError("frame values must be numeric")
    if not np.isfinite(gray).all():
        raise ValueError("frame values must be finite")
    if gray.dtype != np.uint8:
        clipped = np.clip(gray, 0, 255)
        gray = clipped.astype(np.uint8)
    return np.ascontiguousarray(gray)


class OpenCvFarnebackBackend:
    def __init__(
        self,
        *,
        backward_check: bool = False,
        fb_max_error: float = 1.5,
    ) -> None:
        if fb_max_error <= 0.0 or not np.isfinite(fb_max_error):
            raise ValueError("forward/backward error threshold must be positive and finite")
        self.backward_check = backward_check
        self.fb_max_error = float(fb_max_error)

    @staticmethod
    def _dense_flow(previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
        cv2 = _import_cv2()
        return cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

    def compute(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        start_timestamp: float,
        end_timestamp: float,
    ) -> FlowObservation:
        previous_gray = _as_gray(previous_frame)
        current_gray = _as_gray(current_frame)
        if previous_gray.shape != current_gray.shape:
            raise ValueError("previous and current frame shape must match")

        forward = self._dense_flow(previous_gray, current_gray)
        if not np.isfinite(forward).all():
            raise ValueError("OpenCV produced non-finite optical flow")

        valid = np.ones(previous_gray.shape, dtype=bool)
        if self.backward_check:
            cv2 = _import_cv2()
            backward = self._dense_flow(current_gray, previous_gray)
            height, width = previous_gray.shape
            yy, xx = np.mgrid[:height, :width].astype(np.float32)
            map_x = (xx + forward[..., 0]).astype(np.float32)
            map_y = (yy + forward[..., 1]).astype(np.float32)
            sampled_backward_x = cv2.remap(
                backward[..., 0], map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
            )
            sampled_backward_y = cv2.remap(
                backward[..., 1], map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
            )
            error = np.hypot(
                forward[..., 0] + sampled_backward_x,
                forward[..., 1] + sampled_backward_y,
            )
            in_bounds = (
                (map_x >= 0.0)
                & (map_x <= width - 1)
                & (map_y >= 0.0)
                & (map_y <= height - 1)
            )
            valid = in_bounds & np.isfinite(error) & (error <= self.fb_max_error)

        return FlowObservation(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            dx=forward[..., 0],
            dy=forward[..., 1],
            valid=valid,
            backend="opencv-farneback",
        )
