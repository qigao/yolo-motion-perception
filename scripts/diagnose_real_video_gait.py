from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from rtmlib import RTMPose
from ultralytics import YOLO
from validate_real_video_gait import (
    KTH_BASE,
    RTMW_MODEL,
    SCENARIOS,
    RtmLibCropInferencer,
    _download,
    _largest_person,
)

from yolo_motion.articulated_flow import ArticulatedFlowConfig, estimate_articulated_flow
from yolo_motion.camera_motion import CameraMotionConfig, compensate_flow, estimate_camera_motion
from yolo_motion.flow_backend import OpenCvFarnebackBackend
from yolo_motion.pose_regions import PoseRegionConfig, build_body_regions
from yolo_motion.rtmw_adapter import RtmwPoseAdapter
from yolo_motion.ultralytics_adapter import observations_from_result

OUTPUT_DIR = Path("runs/real-video-gait")
_POSE_DIAGNOSTIC_INDICES = (5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
_POSE_FLOW_NEIGHBORHOOD_RATIO = 0.02
_POSE_FLOW_ERROR_QUANTILE = 0.25


def _articulated_to_dict(articulated) -> dict[str, object]:
    return {
        "track_id": articulated.track_id,
        "start_timestamp": articulated.start_timestamp,
        "end_timestamp": articulated.end_timestamp,
        "torso_dx": articulated.torso_dx,
        "torso_dy": articulated.torso_dy,
        "normalized_torso_dx": articulated.normalized_torso_dx,
        "normalized_torso_dy": articulated.normalized_torso_dy,
        "region_flow": {
            name: asdict(region) for name, region in articulated.region_flow.items()
        },
        "pose_flow_agreement": articulated.pose_flow_agreement,
        "person_height_px": articulated.person_height_px,
        "quality": articulated.quality,
    }


def _pose_to_dict(pose) -> dict[str, object]:
    return {
        str(index): {
            "x": float(pose.xy[index, 0]),
            "y": float(pose.xy[index, 1]),
            "confidence": float(pose.confidence[index]),
        }
        for index in _POSE_DIAGNOSTIC_INDICES
    }


def _pose_flow_errors(previous_pose, current_pose, flow, person_height_px: float):
    height, width = flow.dx.shape
    radius_px = max(1, round(_POSE_FLOW_NEIGHBORHOOD_RATIO * person_height_px))
    output: dict[str, object] = {}

    for index in _POSE_DIAGNOSTIC_INDICES:
        if previous_pose.confidence[index] <= 0.0 or current_pose.confidence[index] <= 0.0:
            continue

        previous_x = float(previous_pose.xy[index, 0] * previous_pose.frame_width)
        previous_y = float(previous_pose.xy[index, 1] * previous_pose.frame_height)
        current_x = float(current_pose.xy[index, 0] * current_pose.frame_width)
        current_y = float(current_pose.xy[index, 1] * current_pose.frame_height)
        pose_dx = current_x - previous_x
        pose_dy = current_y - previous_y

        cx = int(np.clip(round(previous_x), 0, width - 1))
        cy = int(np.clip(round(previous_y), 0, height - 1))
        x0 = max(0, cx - radius_px)
        x1 = min(width, cx + radius_px + 1)
        y0 = max(0, cy - radius_px)
        y1 = min(height, cy + radius_px + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        local_mask = (
            (xx - previous_x) ** 2 + (yy - previous_y) ** 2
            <= radius_px * radius_px
        )
        if flow.valid is not None:
            local_mask &= flow.valid[y0:y1, x0:x1]
        if not np.any(local_mask):
            continue

        local_dx = flow.dx[y0:y1, x0:x1][local_mask]
        local_dy = flow.dy[y0:y1, x0:x1][local_mask]
        errors = np.hypot(local_dx - pose_dx, local_dy - pose_dy)
        if errors.size == 0 or not np.isfinite(errors).all():
            continue

        raw_error = float(np.quantile(errors, _POSE_FLOW_ERROR_QUANTILE))
        output[str(index)] = {
            "normalized_error": raw_error / person_height_px,
            "raw_error_px": raw_error,
            "pose_dx": pose_dx,
            "pose_dy": pose_dy,
            "previous_confidence": float(previous_pose.confidence[index]),
            "current_confidence": float(current_pose.confidence[index]),
        }

    return output


def _diagnose_scenario(
    name: str,
    video_path: Path,
    end_frame: int,
    adapter: RtmwPoseAdapter,
) -> None:
    yolo = YOLO("yolo11n.pt")
    flow_backend = OpenCvFarnebackBackend(backward_check=False)
    camera_config = CameraMotionConfig(mode="fixed")
    articulated_config = ArticulatedFlowConfig()
    region_config = PoseRegionConfig()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open KTH video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0

    previous_samples: dict[int, tuple[object, object]] = {}
    segment_samples: dict[int, int] = {}
    frame_number = 0
    output = OUTPUT_DIR / f"{name}-articulated.jsonl"

    with output.open("w", encoding="utf-8") as handle:
        try:
            while True:
                ok, raw_frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                if frame_number > end_frame:
                    break

                frame = cv2.resize(raw_frame, (640, 480), interpolation=cv2.INTER_CUBIC)
                timestamp = (frame_number - 1) / fps
                tracked = yolo.track(
                    frame,
                    persist=True,
                    tracker="botsort.yaml",
                    classes=[0],
                    conf=0.10,
                    imgsz=640,
                    verbose=False,
                )[0]
                observation = _largest_person(
                    observations_from_result(tracked, timestamp, frame.shape[:2])
                )
                if observation is None:
                    handle.write(
                        json.dumps(
                            {
                                "kind": "miss",
                                "frame": frame_number,
                                "timestamp": timestamp,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    previous_samples.clear()
                    segment_samples.clear()
                    continue

                current_pose = adapter.infer_track(frame, observation, timestamp)
                if current_pose is None:
                    handle.write(
                        json.dumps(
                            {
                                "kind": "pose-miss",
                                "frame": frame_number,
                                "timestamp": timestamp,
                                "track_id": observation.track_id,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    previous_samples.pop(observation.track_id, None)
                    segment_samples.pop(observation.track_id, None)
                    continue

                active_id = observation.track_id
                for stale_id in set(previous_samples) - {active_id}:
                    del previous_samples[stale_id]
                    segment_samples.pop(stale_id, None)

                previous = previous_samples.get(active_id)
                if previous is not None:
                    previous_frame, previous_pose = previous
                    flow = flow_backend.compute(
                        previous_frame,
                        frame,
                        previous_pose.timestamp,
                        current_pose.timestamp,
                    )
                    camera = estimate_camera_motion(
                        previous_frame,
                        frame,
                        [],
                        camera_config,
                    )
                    compensated = compensate_flow(flow, camera)
                    person_height_px = observation.height * current_pose.frame_height
                    regions = build_body_regions(
                        previous_pose,
                        person_height_px,
                        region_config,
                    )
                    articulated = estimate_articulated_flow(
                        observation,
                        previous_pose,
                        current_pose,
                        regions,
                        compensated,
                        articulated_config,
                    )
                    if articulated is not None:
                        segment_samples[active_id] = segment_samples.get(active_id, 0) + 1
                        handle.write(
                            json.dumps(
                                {
                                    "kind": "articulated",
                                    "frame": frame_number,
                                    "timestamp": timestamp,
                                    "track_id": active_id,
                                    "segment_sample": segment_samples[active_id],
                                    "track": {
                                        "cx": observation.cx,
                                        "cy": observation.cy,
                                        "width": observation.width,
                                        "height": observation.height,
                                        "confidence": observation.confidence,
                                    },
                                    "pose": _pose_to_dict(current_pose),
                                    "pose_flow_errors": _pose_flow_errors(
                                        previous_pose,
                                        current_pose,
                                        compensated,
                                        person_height_px,
                                    ),
                                    "evidence": _articulated_to_dict(articulated),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )

                previous_samples[active_id] = (frame.copy(), current_pose)
        finally:
            capture.release()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos_dir = OUTPUT_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    pose_model = RTMPose(
        onnx_model=RTMW_MODEL,
        model_input_size=(192, 256),
        backend="onnxruntime",
        device="cpu",
    )
    adapter = RtmwPoseAdapter(RtmLibCropInferencer(pose_model))

    for name in ("walking", "running"):
        scenario = SCENARIOS[name]
        filename = str(scenario["filename"])
        video_path = videos_dir / filename
        _download(f"{KTH_BASE}/{filename}", video_path)
        _diagnose_scenario(name, video_path, int(scenario["end_frame"]), adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
