import json
from pathlib import Path

import pytest

from yolo_motion.capture import (
    CaptureConfig,
    CaptureError,
    capture_scenario,
    output_paths,
)


class FakeCapture:
    def __init__(self, frames: list[object], *, opened: bool = True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {
            1: 640,
            2: 480,
            3: 20.0,
        }[prop]

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, *, opened: bool = True):
        self.opened = opened
        self.frames: list[object] = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3

    def __init__(self, capture: FakeCapture, writer: FakeWriter):
        self.capture = capture
        self.writer = writer
        self.capture_source = None
        self.writer_args = None

    def VideoCapture(self, source):
        self.capture_source = source
        return self.capture

    @staticmethod
    def VideoWriter_fourcc(*chars):
        return tuple(chars)

    def VideoWriter(self, *args):
        self.writer_args = args
        return self.writer


def test_capture_config_rejects_unsupported_scenario_and_bad_timing():
    with pytest.raises(ValueError, match="unsupported scenario"):
        CaptureConfig(scenario="receding")
    with pytest.raises(ValueError, match="duration"):
        CaptureConfig(scenario="approaching", duration=0)
    with pytest.raises(ValueError, match="countdown"):
        CaptureConfig(scenario="approaching", countdown=-1)


def test_output_paths_are_deterministic(tmp_path: Path):
    config = CaptureConfig(scenario="approaching", output_dir=tmp_path)

    video, metadata = output_paths(config)

    assert video == tmp_path / "approaching.mp4"
    assert metadata == tmp_path / "approaching.json"


def test_capture_refuses_to_overwrite_existing_output(tmp_path: Path):
    video = tmp_path / "approaching.mp4"
    video.write_bytes(b"existing")
    config = CaptureConfig(scenario="approaching", output_dir=tmp_path)
    cv2 = FakeCv2(FakeCapture([object()]), FakeWriter())

    with pytest.raises(CaptureError, match="already exists"):
        capture_scenario(config, cv2_module=cv2, sleep=lambda _: None)

    assert cv2.capture_source is None


def test_capture_writes_video_metadata_and_releases_resources(tmp_path: Path):
    capture = FakeCapture(["f1", "f2", "f3"])
    writer = FakeWriter()
    cv2 = FakeCv2(capture, writer)
    config = CaptureConfig(
        scenario="stationary",
        camera=2,
        duration=0.1,
        countdown=0,
        output_dir=tmp_path,
    )
    ticks = iter([10.0, 10.0, 10.04, 10.08, 10.12])

    result = capture_scenario(
        config,
        overwrite=True,
        cv2_module=cv2,
        sleep=lambda _: None,
        monotonic=lambda: next(ticks),
        utc_now=lambda: "2026-09-14T10:00:00+00:00",
    )

    assert cv2.capture_source == 2
    assert writer.frames == ["f1", "f2", "f3"]
    assert capture.released is True
    assert writer.released is True
    assert result.frame_count == 3
    assert result.width == 640
    assert result.height == 480
    assert result.fps == 20.0
    assert result.started_at == "2026-09-14T10:00:00+00:00"
    payload = json.loads((tmp_path / "stationary.json").read_text(encoding="utf-8"))
    assert payload["scenario"] == "stationary"
    assert payload["camera"] == 2
    assert payload["frame_count"] == 3
    assert payload["video"] == "stationary.mp4"


def test_capture_uses_fallback_fps_and_releases_on_empty_stream(tmp_path: Path):
    capture = FakeCapture([])
    writer = FakeWriter()
    cv2 = FakeCv2(capture, writer)
    capture.get = lambda prop: {1: 640, 2: 480, 3: 0.0}[prop]
    config = CaptureConfig(scenario="pose_change", countdown=0, output_dir=tmp_path)

    with pytest.raises(CaptureError, match="no frames"):
        capture_scenario(
            config,
            overwrite=True,
            cv2_module=cv2,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )

    assert capture.released is True
    assert writer.released is True
    assert cv2.writer_args[1] == ("m", "p", "4", "v")
    assert cv2.writer_args[2] == 30.0


def test_capture_overwrite_allows_existing_video_and_metadata(tmp_path: Path):
    (tmp_path / "lateral_crossing.mp4").write_bytes(b"old")
    (tmp_path / "lateral_crossing.json").write_text("{}", encoding="utf-8")
    capture = FakeCapture(["frame"])
    writer = FakeWriter()
    cv2 = FakeCv2(capture, writer)
    config = CaptureConfig(
        scenario="lateral_crossing",
        duration=0.01,
        countdown=0,
        output_dir=tmp_path,
    )
    ticks = iter([0.0, 0.0, 0.02])

    result = capture_scenario(
        config,
        overwrite=True,
        cv2_module=cv2,
        sleep=lambda _: None,
        monotonic=lambda: next(ticks),
    )

    assert result.frame_count == 1
    assert json.loads((tmp_path / "lateral_crossing.json").read_text())["frame_count"] == 1


def test_capture_emits_countdown_before_recording(tmp_path: Path):
    capture = FakeCapture(["frame"])
    writer = FakeWriter()
    cv2 = FakeCv2(capture, writer)
    config = CaptureConfig(
        scenario="approaching",
        duration=0.01,
        countdown=2,
        output_dir=tmp_path,
    )
    ticks = iter([0.0, 0.0, 0.02])
    messages: list[str] = []
    sleeps: list[float] = []

    capture_scenario(
        config,
        overwrite=True,
        cv2_module=cv2,
        sleep=sleeps.append,
        monotonic=lambda: next(ticks),
        notify=messages.append,
    )

    assert messages == [
        "Recording starts in 2...",
        "Recording starts in 1...",
        "Recording...",
    ]
    assert sleeps == [1.0, 1.0]


def test_capture_cli_parser_accepts_approved_scenario_and_defaults():
    from yolo_motion.capture import build_parser

    args = build_parser().parse_args(["--scenario", "approaching"])

    assert args.scenario == "approaching"
    assert args.camera == 0
    assert args.duration == 8.0
    assert args.countdown == 3.0
    assert args.output_dir == Path("benchmarks/videos")
    assert args.overwrite is False


def test_capture_main_prints_metadata_from_injected_capture(tmp_path: Path, capsys):
    from yolo_motion.capture import CaptureResult, main

    seen = {}

    def fake_capture(config, *, overwrite=False):
        seen["config"] = config
        seen["overwrite"] = overwrite
        return CaptureResult(
            scenario=config.scenario,
            camera=config.camera,
            started_at="2026-09-14T10:00:00+00:00",
            requested_duration=config.duration,
            actual_duration=1.0,
            countdown=config.countdown,
            fps=30.0,
            width=640,
            height=480,
            frame_count=30,
            video=f"{config.scenario}.mp4",
        )

    rc = main(
        [
            "--scenario",
            "approaching",
            "--camera",
            "3",
            "--duration",
            "5",
            "--countdown",
            "0",
            "--output-dir",
            str(tmp_path),
            "--overwrite",
        ],
        capture_fn=fake_capture,
    )

    assert rc == 0
    assert seen["config"].camera == 3
    assert seen["config"].duration == 5.0
    assert seen["config"].output_dir == tmp_path
    assert seen["overwrite"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"] == "approaching"
    assert payload["frame_count"] == 30
