from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_state_machine.r1_e3m_artifact import (
    MechanismArtifact,
    MechanismVideoRecord,
    TrackWindow,
    mechanism_artifact_digest,
    track_window_id,
)


def _api():
    from neural_state_machine.r1_e3m_artifact_evidence import (
        FrozenMechanismArtifactInvalid,
        freeze_mechanism_artifact,
        load_frozen_mechanism_artifact,
        verify_frozen_mechanism_artifact,
    )

    return (
        FrozenMechanismArtifactInvalid,
        freeze_mechanism_artifact,
        load_frozen_mechanism_artifact,
        verify_frozen_mechanism_artifact,
    )


def _video(video_id: str, split: str, digit: str) -> MechanismVideoRecord:
    return MechanismVideoRecord(
        video_id=video_id,
        sha256=digit * 64,
        split=split,
        fps=30.0,
        frame_count=300,
        width=1280,
        height=720,
        source_window_component_id=f"{split}-component",
    )


def _window(video_id: str, split: str, track_id: int) -> TrackWindow:
    tensor = np.zeros((20, 6), dtype=np.float64)
    tensor[:, 0] = np.linspace(0.1, 0.2, 20)
    tensor[:, 1:5] = (0.3, 0.1, 0.2, 0.9)
    tensor[:, 5] = 1.0
    return TrackWindow(
        window_id=track_window_id(video_id, track_id, 0.0, 2.0),
        video_id=video_id,
        split=split,
        track_id=track_id,
        source_start_seconds=0.0,
        source_end_seconds=2.0,
        tensor=tensor,
    )


def _artifact() -> MechanismArtifact:
    return MechanismArtifact(
        source_manifest_sha256="a" * 64,
        raw_track_sha256="b" * 64,
        extraction_provenance_sha256="c" * 64,
        videos=(
            _video("train-video", "train", "1"),
            _video("eval-video", "eval", "2"),
        ),
        windows=(
            _window("train-video", "train", 7),
            _window("eval-video", "eval", 9),
        ),
    )


def test_freeze_roundtrip_preserves_exact_artifact_digest(tmp_path: Path) -> None:
    _, freeze, load, verify = _api()
    artifact = _artifact()
    root = tmp_path / "frozen"

    frozen = freeze(artifact, root)
    loaded = load(root)
    verified = verify(root)

    assert frozen.root_digest == mechanism_artifact_digest(artifact)
    assert verified.root_digest == frozen.root_digest
    assert verified.video_count == 2
    assert verified.window_count == 2
    assert mechanism_artifact_digest(loaded) == frozen.root_digest
    np.testing.assert_array_equal(
        loaded.windows[0].tensor,
        artifact.windows[0].tensor,
    )


def test_freeze_is_write_once(tmp_path: Path) -> None:
    _, freeze, _, _ = _api()
    root = tmp_path / "frozen"

    freeze(_artifact(), root)

    with pytest.raises(FileExistsError):
        freeze(_artifact(), root)


def test_manifest_binds_all_payload_files(tmp_path: Path) -> None:
    _, freeze, _, _ = _api()
    root = tmp_path / "frozen"

    freeze(_artifact(), root)
    manifest = json.loads(
        (root / "artifact-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema"] == "r1-e3m-frozen-artifact-v1"
    assert set(manifest["files"]) == {
        "videos.json",
        "windows.json",
        "window-tensors.npy",
    }
    assert manifest["source_manifest_sha256"] == "a" * 64
    assert manifest["raw_track_sha256"] == "b" * 64
    assert manifest["extraction_provenance_sha256"] == "c" * 64


def test_corrupted_tensor_file_fails_closed(tmp_path: Path) -> None:
    Invalid, freeze, _, verify = _api()
    root = tmp_path / "frozen"
    freeze(_artifact(), root)

    with (root / "window-tensors.npy").open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(Invalid, match="sha256"):
        verify(root)


def test_corrupted_window_metadata_fails_closed(tmp_path: Path) -> None:
    Invalid, freeze, _, verify = _api()
    root = tmp_path / "frozen"
    freeze(_artifact(), root)

    windows = json.loads((root / "windows.json").read_text(encoding="utf-8"))
    windows[0]["split"] = "eval"
    (root / "windows.json").write_text(
        json.dumps(windows, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Invalid, match="sha256"):
        verify(root)


def test_frozen_metadata_contains_no_semantic_labels(tmp_path: Path) -> None:
    _, freeze, _, _ = _api()
    root = tmp_path / "frozen"
    freeze(_artifact(), root)

    text = "\n".join(
        [
            (root / "artifact-manifest.json").read_text(encoding="utf-8"),
            (root / "videos.json").read_text(encoding="utf-8"),
            (root / "windows.json").read_text(encoding="utf-8"),
        ]
    )

    assert '"label"' not in text
    assert '"episode_id"' not in text
    assert '"actor_track_id"' not in text
    assert '"target_track_id"' not in text
