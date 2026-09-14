from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import yaml

from .motion import MotionConfig
from .pipeline import MotionPipeline, TrackMotionResult
from .ultralytics_adapter import observations_from_result


def load_motion_config(path: str | Path) -> MotionConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("motion config must be a YAML mapping")
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
    parser.add_argument("--jsonl", type=Path, help="Optional JSONL output path")
    parser.add_argument("--show", action="store_true", help="Display annotated video")
    return parser


def _coerce_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


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


def _write_result(handle: TextIO | None, result: TrackMotionResult) -> None:
    payload = json.dumps(_result_to_dict(result), separators=(",", ":"))
    if handle is None:
        print(payload)
    else:
        handle.write(payload + "\n")
        handle.flush()


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
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        output = args.jsonl.open("w", encoding="utf-8")

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
        if args.show:
            cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
