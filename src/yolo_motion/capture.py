from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_SCENARIOS = frozenset(
    {
        "stationary",
        "lateral_crossing",
        "approaching",
        "pose_change",
    }
)


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptureConfig:
    scenario: str
    camera: int = 0
    duration: float = 8.0
    countdown: float = 3.0
    output_dir: Path = Path("benchmarks/videos")
    fallback_fps: float = 30.0

    def __post_init__(self) -> None:
        if self.scenario not in SUPPORTED_SCENARIOS:
            allowed = ", ".join(sorted(SUPPORTED_SCENARIOS))
            raise ValueError(f"unsupported scenario {self.scenario!r}; choose one of: {allowed}")
        if self.duration <= 0:
            raise ValueError("duration must be greater than zero")
        if self.countdown < 0:
            raise ValueError("countdown must be zero or greater")
        if self.fallback_fps <= 0:
            raise ValueError("fallback_fps must be greater than zero")


@dataclass(frozen=True)
class CaptureResult:
    scenario: str
    camera: int
    started_at: str
    requested_duration: float
    actual_duration: float
    countdown: float
    fps: float
    width: int
    height: int
    frame_count: int
    video: str


def output_paths(config: CaptureConfig) -> tuple[Path, Path]:
    return (
        config.output_dir / f"{config.scenario}.mp4",
        config.output_dir / f"{config.scenario}.json",
    )


def _default_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def capture_scenario(
    config: CaptureConfig,
    *,
    overwrite: bool = False,
    cv2_module: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    utc_now: Callable[[], str] = _default_utc_now,
    notify: Callable[[str], None] = print,
) -> CaptureResult:
    video_path, metadata_path = output_paths(config)
    existing = [path for path in (video_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise CaptureError(f"output already exists: {joined}; use --overwrite to replace it")

    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise CaptureError(
                "OpenCV is required for capture. Install with: pip install -e '.[vision]'"
            ) from exc

    config.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2_module.VideoCapture(config.camera)
    writer = None
    if not capture.isOpened():
        capture.release()
        raise CaptureError(f"cannot open camera: {config.camera}")

    width = round(float(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH)))
    height = round(float(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT)))
    reported_fps = float(capture.get(cv2_module.CAP_PROP_FPS))
    fps = reported_fps if reported_fps > 0 else config.fallback_fps
    if width <= 0 or height <= 0:
        capture.release()
        raise CaptureError(f"camera returned invalid frame size: {width}x{height}")

    fourcc = cv2_module.VideoWriter_fourcc(*"mp4v")
    writer = cv2_module.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise CaptureError(f"cannot open video writer: {video_path}")

    remaining = config.countdown
    while remaining > 0:
        notify(f"Recording starts in {math.ceil(remaining)}...")
        delay = min(1.0, remaining)
        sleep(delay)
        remaining -= delay
    notify("Recording...")

    started_at = utc_now()
    started = monotonic()
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frame_count += 1
            if monotonic() - started >= config.duration:
                break
        actual_duration = max(0.0, monotonic() - started)
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        if video_path.exists():
            video_path.unlink()
        raise CaptureError("capture produced no frames")

    result = CaptureResult(
        scenario=config.scenario,
        camera=config.camera,
        started_at=started_at,
        requested_duration=config.duration,
        actual_duration=actual_duration,
        countdown=config.countdown,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        video=video_path.name,
    )
    metadata_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capture-scenario",
        description="Record one fixed-camera real-video benchmark scenario.",
    )
    parser.add_argument("--scenario", required=True, choices=sorted(SUPPORTED_SCENARIOS))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/videos"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    capture_fn: Callable[..., CaptureResult] = capture_scenario,
) -> int:
    args = build_parser().parse_args(argv)
    config = CaptureConfig(
        scenario=args.scenario,
        camera=args.camera,
        duration=args.duration,
        countdown=args.countdown,
        output_dir=args.output_dir,
    )
    try:
        result = capture_fn(config, overwrite=args.overwrite)
    except CaptureError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0
