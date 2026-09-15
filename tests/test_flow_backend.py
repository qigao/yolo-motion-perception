import numpy as np
import pytest

from yolo_motion.flow_backend import OpenCvFarnebackBackend


def textured_frame(height=96, width=128):
    rng = np.random.default_rng(20260915)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


def test_farneback_backend_returns_dense_repository_owned_flow():
    frame = textured_frame()
    backend = OpenCvFarnebackBackend()
    flow = backend.compute(frame, frame.copy(), 1.0, 1.05)

    assert flow.backend == "opencv-farneback"
    assert flow.dx.shape == frame.shape
    assert flow.dy.shape == frame.shape
    assert flow.valid is not None
    assert flow.valid.shape == frame.shape
    assert bool(np.all(flow.valid)) is True
    assert float(np.median(np.hypot(flow.dx, flow.dy))) < 0.05


def test_farneback_backend_detects_nonzero_translation():
    previous = textured_frame()
    current = np.roll(previous, shift=(1, 2), axis=(0, 1))
    flow = OpenCvFarnebackBackend().compute(previous, current, 2.0, 2.05)

    central = np.s_[12:-12, 12:-12]
    assert float(np.median(flow.dx[central])) > 0.75
    assert float(np.median(flow.dy[central])) > 0.25


def test_farneback_backend_accepts_rgb_frames():
    gray = textured_frame()
    rgb = np.repeat(gray[..., None], 3, axis=2)
    flow = OpenCvFarnebackBackend().compute(rgb, rgb.copy(), 0.0, 0.1)
    assert flow.dx.shape == gray.shape


def test_farneback_backend_rejects_mismatched_frames():
    with pytest.raises(ValueError, match="shape"):
        OpenCvFarnebackBackend().compute(
            np.zeros((64, 64), dtype=np.uint8),
            np.zeros((63, 64), dtype=np.uint8),
            0.0,
            0.1,
        )
