from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import yaml

from .motion import MotionConfig
from .pipeline import MotionPipeline, TrackMotionResult
from .types import TrackObservation
from .ultralytics_adapter import observations_from_result


def load_motion_config(path: str | Path) -> MotionConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("motion config must be a YAML mapping")
    return MotionConfig(**raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo-motion",
        description="Estimate motion state from YOLO multi-object tracks.",
    )
    parser.add_argument("--source", required=True, help="Video path or camera index such as 0")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path/name")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker config")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Motion YAML config")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence")
    parser.add_argument("--jsonl", type=Path, help="Optional motion-result JSONL output path")
    parser.add_argument(
        "--observations-jsonl",
        type=Path,
        help="Optional raw YOLO/BoT-SORT TrackObservation JSONL output path",
    )
    parser.add_argument("--show", action="store_true", help="Display annotated video")
    return parser


def _coerce_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def _observation_to_dict(
    observation: TrackObservation, *, frame_index: int
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "track_id": observation.track_id,
        "timestamp": observation.timestamp,
        "class_id": observation.class_id,
        "detection_confidence": observation.confidence,
        "cx": observation.cx,
        "cy": observation.cy,
        "width": observation.width,
        "height": observation.height,
    }


def _result_to_dict(result: TrackMotionResult) -> dict[str, object]:
    return {
        "track_id": result.track_id,
        "timestamp": result.observation.timestamp,
        "class_id": result.observation.class_id,
        "detection_confidence": result.observation.confidence,
        "lateral": result.state.lateral.value,
        "radial": result.state.radial.value,
        "evidence": asdict(result.evidence),
    }


def _write_payload(handle: TextIO | None, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"))
    if handle is None:
        print(encoded)
    else:
        handle.write(encoded + "\n")
        handle.flush()


def _write_result(handle: TextIO | None, result: TrackMotionResult) -> None:
    _write_payload(handle, _result_to_dict(result))


def _write_observation(
    handle: TextIO, observation: TrackObservation, *, frame_index: int
) -> None:
    _write_payload(handle, _observation_to_dict(observation, frame_index=frame_index))


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Runtime vision dependencies are missing. Install with: "
            "pip install -e '.[vision]'"
        ) from exc

    config = load_motion_config(args.config)
    pipeline = MotionPipeline(config)
    model = YOLO(args.model)
    capture = cv2.VideoCapture(_coerce_source(args.source))
    if not capture.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_index = 0
    output: TextIO | None = None
    observations_output: TextIO | None = None
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        output = args.jsonl.open("w", encoding="utf-8")
    if args.observations_jsonl is not None:
        args.observations_jsonl.parent.mkdir(parents=True, exist_ok=True)
        observations_output = args.observations_jsonl.open("w", encoding="utf-8")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms > 0.0:
                timestamp = timestamp_ms / 1000.0
            elif fps > 0.0:
                timestamp = frame_index / fps
            else:
                timestamp = frame_index / 30.0

            tracked = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                verbose=False,
            )
            result = tracked[0]
            observations = observations_from_result(result, timestamp, frame.shape[:2])
            if observations_output is not None:
                for observation in observations:
                    _write_observation(
                        observations_output,
                        observation,
                        frame_index=frame_index,
                    )

            active_ids = {observation.track_id for observation in observations}
            motion_results = []
            for observation in observations:
                motion_result = pipeline.update(observation)
                if motion_result is not None:
                    motion_results.append(motion_result)
                    _write_result(output, motion_result)
            pipeline.drop_stale(active_ids)

            if args.show:
                annotated = result.plot()
                for row, motion_result in enumerate(motion_results):
                    label = (
                        f"#{motion_result.track_id} "
                        f"{motion_result.state.lateral.value}/"
                        f"{motion_result.state.radial.value} "
                        f"e={motion_result.evidence.expansion_rate:+.3f}/s"
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (10, 25 + row * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow("yolo-motion-perception", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if output is not None:
            output.close()
        if observations_output is not None:
            observations_output.close()
        if args.show:
            cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
