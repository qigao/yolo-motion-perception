from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
import yaml
from rtmlib import RTMPose
from ultralytics import YOLO

from yolo_motion.articulated_flow import ArticulatedFlowConfig, estimate_articulated_flow
from yolo_motion.camera_motion import CameraMotionConfig, compensate_flow, estimate_camera_motion
from yolo_motion.flow_backend import OpenCvFarnebackBackend
from yolo_motion.gait import GaitConfig
from yolo_motion.gait_pipeline import OpticalGaitPipeline
from yolo_motion.motion import MotionConfig
from yolo_motion.pipeline import MotionPipeline
from yolo_motion.pose_regions import PoseRegionConfig, build_body_regions
from yolo_motion.rtmw_adapter import RtmwPoseAdapter
from yolo_motion.ultralytics_adapter import observations_from_result

KTH_BASE = "https://www.csc.kth.se/cvap/actions"
RTMW_MODEL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip"
)

SCENARIOS = {
    "standing": {
        "filename": "person15_handclapping_d1_uncomp.avi",
        "end_frame": 65,
        "expected": "standing",
    },
    "walking": {
        "filename": "person15_walking_d1_uncomp.avi",
        "end_frame": 105,
        "expected": "walking",
    },
    "running": {
        "filename": "person15_running_d1_uncomp.avi",
        "end_frame": 50,
        "expected": "running",
    },
}


class RtmLibCropInferencer:
    def __init__(self, pose_model: RTMPose):
        self.pose_model = pose_model

    def __call__(self, crop: np.ndarray, **_kwargs):
        keypoints, scores = self.pose_model(crop)
        if len(keypoints) == 0:
            yield {"predictions": [[]]}
            return
        sample = {
            "keypoints": np.asarray(keypoints[0], dtype=float).tolist(),
            "keypoint_scores": np.asarray(scores[0], dtype=float).reshape(-1).tolist(),
        }
        yield {"predictions": [[sample]]}


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "yolo-motion-perception-ci"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    if destination.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file from {url}")


def _load_gait_config(path: Path) -> GaitConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("gait config must be a YAML mapping")
    return GaitConfig(**raw)


def _largest_person(observations):
    people = [observation for observation in observations if observation.class_id == 0]
    if not people:
        return None
    return max(people, key=lambda observation: observation.area)


def _intervals(predictions: list[dict[str, object]], expected: str) -> list[dict[str, object]]:
    by_track: dict[int, list[float]] = defaultdict(list)
    for prediction in predictions:
        by_track[int(prediction["track_id"])].append(float(prediction["timestamp"]))
    return [
        {
            "track_id": track_id,
            "start": min(times),
            "end": max(times),
            "label": expected,
        }
        for track_id, times in sorted(by_track.items())
    ]


def _summarize(predictions: list[dict[str, object]], expected: str) -> dict[str, object]:
    counts = Counter(str(item["locomotion"]) for item in predictions)
    known_counts = Counter({state: count for state, count in counts.items() if state != "unknown"})
    known_total = sum(known_counts.values())
    dominant_known = known_counts.most_common(1)[0][0] if known_counts else None
    expected_known_fraction = (
        known_counts.get(expected, 0) / known_total if known_total else 0.0
    )
    passed = (
        len(predictions) >= 5
        and dominant_known == expected
        and expected_known_fraction >= 0.50
    )
    return {
        "expected": expected,
        "prediction_count": len(predictions),
        "state_counts": dict(sorted(counts.items())),
        "known_prediction_count": known_total,
        "unknown_prediction_count": counts.get("unknown", 0),
        "dominant_known": dominant_known,
        "expected_known_fraction": round(expected_known_fraction, 6),
        "passed": passed,
        "labeled_track_intervals": _intervals(predictions, expected),
    }


def _run_scenario(
    *,
    name: str,
    video_path: Path,
    end_frame: int,
    expected: str,
    rtmw_adapter: RtmwPoseAdapter,
    gait_config: GaitConfig,
    output_dir: Path,
    stride: int,
) -> dict[str, object]:
    yolo = YOLO("yolo11n.pt")
    flow_backend = OpenCvFarnebackBackend(backward_check=False)
    camera_config = CameraMotionConfig(mode="fixed")
    articulated_config = ArticulatedFlowConfig()
    region_config = PoseRegionConfig()
    gait_pipeline = OpticalGaitPipeline(gait_config)
    motion_pipeline = MotionPipeline(MotionConfig())

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open KTH video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 25.0

    previous_samples: dict[int, tuple[np.ndarray, object]] = {}
    predictions: list[dict[str, object]] = []
    processed_frames = 0
    detected_frames = 0
    pose_frames = 0
    evidence_frames = 0
    frame_number = 0

    jsonl_path = output_dir / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        try:
            while True:
                ok, raw_frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                if frame_number > end_frame:
                    break
                if (frame_number - 1) % stride != 0:
                    continue

                processed_frames += 1
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
                    gait_pipeline.drop_stale(set())
                    motion_pipeline.drop_stale(set())
                    previous_samples.clear()
                    continue

                detected_frames += 1
                active_ids = {observation.track_id}
                gait_pipeline.drop_stale(active_ids)
                motion_pipeline.drop_stale(active_ids)
                for stale_id in set(previous_samples) - active_ids:
                    del previous_samples[stale_id]

                motion_result = motion_pipeline.update(observation)
                current_pose = rtmw_adapter.infer_track(frame, observation, timestamp)
                if current_pose is None:
                    continue
                pose_frames += 1

                previous = previous_samples.get(observation.track_id)
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
                    regions = build_body_regions(
                        previous_pose,
                        observation.height * current_pose.frame_height,
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
                        evidence_frames += 1
                        gait_result = gait_pipeline.update(articulated)
                        if gait_result is not None:
                            payload = {
                                "scenario": name,
                                "expected": expected,
                                "frame": frame_number,
                                "timestamp": timestamp,
                                "track_id": observation.track_id,
                                "locomotion": gait_result.state.value,
                                "gait": asdict(gait_result.evidence),
                                "lateral": (
                                    motion_result.state.lateral.value
                                    if motion_result is not None
                                    else None
                                ),
                                "radial": (
                                    motion_result.state.radial.value
                                    if motion_result is not None
                                    else None
                                ),
                            }
                            predictions.append(payload)
                            jsonl.write(json.dumps(payload, sort_keys=True) + "\n")

                previous_samples[observation.track_id] = (frame.copy(), current_pose)
        finally:
            capture.release()

    summary = _summarize(predictions, expected)
    summary.update(
        {
            "video": video_path.name,
            "source_url": f"{KTH_BASE}/{video_path.name}",
            "annotated_frame_range": [1, end_frame],
            "fps": fps,
            "stride": stride,
            "processed_frames": processed_frames,
            "detected_frames": detected_frames,
            "pose_frames": pose_frames,
            "articulated_evidence_frames": evidence_frames,
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/real-video-gait"))
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    gait_config = _load_gait_config(root / "configs/gait.yaml")
    pose_model = RTMPose(
        onnx_model=RTMW_MODEL,
        model_input_size=(192, 256),
        backend="onnxruntime",
        device="cpu",
    )
    rtmw_adapter = RtmwPoseAdapter(RtmLibCropInferencer(pose_model))

    provenance = {
        "git_sha": os.environ.get("GITHUB_SHA"),
        "tracker": "Ultralytics YOLO11n + BoT-SORT",
        "detector_model": "yolo11n.pt",
        "rtmw_runtime": "rtmlib RTMPose / ONNX Runtime CPU",
        "rtmw_model": RTMW_MODEL,
        "flow_backend": "opencv-farneback forward",
        "camera_mode": "fixed",
        "gait_config": asdict(gait_config),
        "articulated_config": asdict(ArticulatedFlowConfig()),
        "pose_region_config": asdict(PoseRegionConfig()),
        "processing_resolution": [640, 480],
        "frame_stride": args.stride,
        "dataset": "KTH human actions sample sequences",
        "dataset_home": KTH_BASE,
        "dataset_note": "KTH page states public availability for non-commercial use",
        "versions": {
            "ultralytics": metadata.version("ultralytics"),
            "rtmlib": metadata.version("rtmlib"),
            "onnxruntime": metadata.version("onnxruntime"),
            "opencv-python": metadata.version("opencv-python"),
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results: dict[str, object] = {}
    failures: list[str] = []
    for name, scenario in SCENARIOS.items():
        filename = str(scenario["filename"])
        video_path = videos_dir / filename
        _download(f"{KTH_BASE}/{filename}", video_path)
        result = _run_scenario(
            name=name,
            video_path=video_path,
            end_frame=int(scenario["end_frame"]),
            expected=str(scenario["expected"]),
            rtmw_adapter=rtmw_adapter,
            gait_config=gait_config,
            output_dir=output_dir,
            stride=args.stride,
        )
        results[name] = result
        if not bool(result["passed"]):
            failures.append(
                f"{name}: expected={result['expected']} dominant={result['dominant_known']} "
                f"known_fraction={result['expected_known_fraction']} "
                f"counts={result['state_counts']}"
            )

    report = {
        "provenance": provenance,
        "scenarios": results,
        "passed": not failures,
        "failures": failures,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
