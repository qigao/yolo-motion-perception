from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from neural_state_machine.r1_e3m_artifact import (
    MechanismArtifact,
    MechanismVideoRecord,
    TrackWindow,
    track_window_id,
)
from neural_state_machine.r1_e3m_artifact_evidence import (
    write_mechanism_candidate,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_r1_e3m_artifact.py"


def _artifact() -> MechanismArtifact:
    videos = (
        MechanismVideoRecord(
            video_id="train",
            sha256="1" * 64,
            split="train",
            fps=30.0,
            frame_count=300,
            width=1280,
            height=720,
            source_window_component_id="train-component",
        ),
        MechanismVideoRecord(
            video_id="eval",
            sha256="2" * 64,
            split="eval",
            fps=30.0,
            frame_count=300,
            width=1280,
            height=720,
            source_window_component_id="eval-component",
        ),
    )

    windows = []
    for video_id, split, track_id in (
        ("train", "train", 7),
        ("eval", "eval", 9),
    ):
        tensor = np.zeros((20, 6), dtype=np.float64)
        tensor[:, :5] = (0.2, 0.3, 0.1, 0.2, 0.9)
        tensor[:, 5] = 1.0
        windows.append(
            TrackWindow(
                window_id=track_window_id(video_id, track_id, 0.0, 2.0),
                video_id=video_id,
                split=split,
                track_id=track_id,
                source_start_seconds=0.0,
                source_end_seconds=2.0,
                tensor=tensor,
            )
        )

    return MechanismArtifact(
        source_manifest_sha256="a" * 64,
        raw_track_sha256="b" * 64,
        extraction_provenance_sha256="c" * 64,
        videos=videos,
        windows=tuple(windows),
    )


def test_freeze_and_verify_cli_roundtrip(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    frozen = tmp_path / "frozen"
    write_mechanism_candidate(_artifact(), candidate)

    freeze = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "freeze",
            "--candidate",
            str(candidate),
            "--output",
            str(frozen),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    frozen_payload = json.loads(freeze.stdout)
    assert frozen_payload["valid"] is True
    assert frozen_payload["window_count"] == 2
    assert len(frozen_payload["root_digest"]) == 64

    verify = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            "--root",
            str(frozen),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    verified_payload = json.loads(verify.stdout)
    assert verified_payload == frozen_payload


def test_packaging_cli_has_no_detector_semantic_or_measurement_runtime() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()

    assert "ultralytics" not in source
    assert "botsort" not in source
    assert "yolo(" not in source
    assert "semantic-review" not in source
    assert "batch1-semantic-review" not in source
    assert '"measure"' not in source
