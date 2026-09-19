from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


LABELS = ("approach", "touch", "pick_up", "pass_by")


def _api():
    from neural_state_machine.r1_e3_artifact_evidence import (
        ArtifactInvalid,
        freeze_artifact,
        verify_frozen_artifact,
    )

    return ArtifactInvalid, freeze_artifact, verify_frozen_artifact


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_signed_json(root: Path, name: str, payload: object) -> str:
    data = _canonical_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    (root / name).write_bytes(data)
    (root / f"{name.removesuffix('.json')}.sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )
    return digest


def _write_signed_jsonl(root: Path, name: str, rows: list[dict[str, object]]) -> str:
    data = b"".join(_canonical_bytes(row) for row in rows)
    digest = hashlib.sha256(data).hexdigest()
    (root / name).write_bytes(data)
    (root / f"{name.removesuffix('.jsonl')}.sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )
    return digest


def _candidate_artifact(
    tmp_path: Path,
    *,
    train_per_class: int = 20,
    eval_per_class: int = 10,
) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()

    videos = []
    for split, prefix in (("training", "train"), ("evaluation", "eval")):
        for index in range(2):
            videos.append(
                {
                    "video_id": f"{prefix}-{index}",
                    "sha256": hashlib.sha256(f"{prefix}-{index}".encode()).hexdigest(),
                    "split": split,
                    "fps": 20.0,
                    "frame_count": 40,
                    "width": 1000,
                    "height": 1000,
                }
            )

    tracks: list[dict[str, object]] = []
    for video in sorted(videos, key=lambda item: str(item["video_id"])):
        for frame_index, timestamp in ((0, 0.0), (20, 1.0)):
            for track_id, x in ((10, 0.2), (20, 0.6)):
                tracks.append(
                    {
                        "video_id": video["video_id"],
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "track_id": track_id,
                        "class_id": 0 if track_id == 10 else 1,
                        "confidence": 0.9,
                        "x1": (x - 0.05) * 1000,
                        "y1": 300.0,
                        "x2": (x + 0.05) * 1000,
                        "y2": 500.0,
                        "cx_norm": x,
                        "cy_norm": 0.4,
                        "w_norm": 0.1,
                        "h_norm": 0.2,
                    }
                )

    episodes: list[dict[str, object]] = []
    for split, prefix, count in (
        ("training", "train", train_per_class),
        ("evaluation", "eval", eval_per_class),
    ):
        for label_index, label in enumerate(LABELS):
            for index in range(count):
                video_id = f"{prefix}-{index % 2}"
                episodes.append(
                    {
                        "episode_id": f"{prefix}-{label}-{index:03d}",
                        "video_id": video_id,
                        "start_time_seconds": 0.0,
                        "end_time_seconds": 1.0,
                        "label": label,
                        "actor_track_id": 10,
                        "target_track_id": 20,
                        "annotation_revision": "r1",
                        "reviewer": "reviewer",
                    }
                )

    provenance = {
        "schema": "r1-e3-extraction-provenance-v1",
        "extractor_commit": "a" * 40,
        "yolo": {
            "package_version": "fixed-test-version",
            "weights_sha256": "3" * 64,
        },
        "botsort": {"config_sha256": "4" * 64},
        "runtime": {
            "python": "3.12.14",
            "lock_sha256": "5" * 64,
        },
    }

    digests = {
        "videos.json": _write_signed_json(root, "videos.json", videos),
        "episodes.json": _write_signed_json(root, "episodes.json", episodes),
        "tracks.jsonl": _write_signed_jsonl(root, "tracks.jsonl", tracks),
        "extraction-provenance.json": _write_signed_json(
            root,
            "extraction-provenance.json",
            provenance,
        ),
    }
    _write_signed_json(
        root,
        "artifact-manifest.json",
        {"schema": "r1-e3-artifact-v1", "files": digests},
    )
    return root


def test_freeze_rejects_dataset_below_registered_class_counts(tmp_path: Path) -> None:
    ArtifactInvalid, freeze, _ = _api()
    candidate = _candidate_artifact(tmp_path, train_per_class=19, eval_per_class=10)

    with pytest.raises(ArtifactInvalid, match="20 training episodes per class"):
        freeze(candidate, tmp_path / "frozen")


def test_freeze_requires_two_source_videos_per_split(tmp_path: Path) -> None:
    ArtifactInvalid, freeze, _ = _api()
    candidate = _candidate_artifact(tmp_path)
    episodes = json.loads((candidate / "episodes.json").read_text())
    for episode in episodes:
        if episode["video_id"].startswith("eval-"):
            episode["video_id"] = "eval-0"
    digest = _write_signed_json(candidate, "episodes.json", episodes)
    manifest = json.loads((candidate / "artifact-manifest.json").read_text())
    manifest["files"]["episodes.json"] = digest
    _write_signed_json(candidate, "artifact-manifest.json", manifest)

    with pytest.raises(ArtifactInvalid, match="two distinct evaluation source videos"):
        freeze(candidate, tmp_path / "frozen")


def test_freeze_is_write_once_and_writes_normalized_episode_artifact(
    tmp_path: Path,
) -> None:
    _, freeze, verify = _api()
    candidate = _candidate_artifact(tmp_path)
    frozen = tmp_path / "frozen"

    written = freeze(candidate, frozen)

    assert len(written["artifact_root_digest"]) == 64
    assert (frozen / "normalized-episodes.jsonl").is_file()
    assert (frozen / "normalized-episodes.sha256").is_file()
    assert (frozen / "frozen-artifact.json").is_file()
    assert (frozen / "frozen-artifact.sha256").is_file()
    verified = verify(frozen)
    assert verified["valid"] is True
    assert verified["artifact_root_digest"] == written["artifact_root_digest"]
    assert verified["training_episode_count"] == 80
    assert verified["evaluation_episode_count"] == 40

    with pytest.raises(FileExistsError):
        freeze(candidate, frozen)


def test_frozen_normalized_rows_have_exact_shape_label_and_digest(tmp_path: Path) -> None:
    _, freeze, _ = _api()
    frozen = tmp_path / "frozen"
    freeze(_candidate_artifact(tmp_path), frozen)

    rows = [
        json.loads(line)
        for line in (frozen / "normalized-episodes.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 120
    first = rows[0]
    assert first["label"] in LABELS
    assert len(first["tensor"]) == 20
    assert all(len(row) == 14 for row in first["tensor"])
    assert len(first["tensor_digest"]) == 64


def test_verify_fails_closed_on_normalized_episode_tamper(tmp_path: Path) -> None:
    ArtifactInvalid, freeze, verify = _api()
    frozen = tmp_path / "frozen"
    freeze(_candidate_artifact(tmp_path), frozen)
    path = frozen / "normalized-episodes.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")

    with pytest.raises(ArtifactInvalid, match="normalized-episodes"):
        verify(frozen)
