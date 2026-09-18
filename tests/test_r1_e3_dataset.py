from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from neural_state_machine.r1_e3_artifact_evidence import freeze_artifact


LABELS = ("approach", "touch", "pick_up", "pass_by")


def _api():
    from neural_state_machine.r1_e3_dataset import (
        DatasetInvalid,
        load_registered_dataset,
    )

    return DatasetInvalid, load_registered_dataset


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


def _candidate_artifact(tmp_path: Path) -> Path:
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
    for prefix, count in (("train", 20), ("eval", 10)):
        for label in LABELS:
            for index in range(count):
                episodes.append(
                    {
                        "episode_id": f"{prefix}-{label}-{index:03d}",
                        "video_id": f"{prefix}-{index % 2}",
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


def _frozen_artifact(tmp_path: Path) -> Path:
    candidate = _candidate_artifact(tmp_path)
    frozen = tmp_path / "frozen"
    freeze_artifact(candidate, frozen)
    return frozen


def test_registered_dataset_uses_only_frozen_normalized_tensors(tmp_path: Path) -> None:
    _, load = _api()
    dataset = load(_frozen_artifact(tmp_path))

    assert len(dataset.training) == 80
    assert len(dataset.evaluation) == 40
    assert len(dataset.artifact_root_digest) == 64
    assert tuple(item.label for item in dataset.training[:4]) == (
        "approach",
        "approach",
        "approach",
        "approach",
    )
    assert all(
        item.tensor.shape == (20, 14)
        for item in (*dataset.training, *dataset.evaluation)
    )
    assert all(
        item.tensor.dtype == np.float64
        for item in (*dataset.training, *dataset.evaluation)
    )
    assert all(
        not item.tensor.flags.writeable
        for item in (*dataset.training, *dataset.evaluation)
    )


def test_registered_dataset_preserves_frozen_episode_identity_and_split(
    tmp_path: Path,
) -> None:
    _, load = _api()
    dataset = load(_frozen_artifact(tmp_path))

    assert dataset.training[0].episode_id == "train-approach-000"
    assert dataset.training[0].video_id == "train-0"
    assert dataset.training[0].split == "training"
    assert dataset.evaluation[0].episode_id == "eval-approach-000"
    assert dataset.evaluation[0].video_id == "eval-0"
    assert dataset.evaluation[0].split == "evaluation"
    assert set(item.video_id for item in dataset.training) == {"train-0", "train-1"}
    assert set(item.video_id for item in dataset.evaluation) == {"eval-0", "eval-1"}


def test_registered_dataset_preserves_tensor_digest_and_episode_order(
    tmp_path: Path,
) -> None:
    _, load = _api()
    root = _frozen_artifact(tmp_path)
    frozen_rows = [
        json.loads(line)
        for line in (root / "normalized-episodes.jsonl").read_text().splitlines()
    ]

    dataset = load(root)
    observed = [*dataset.training, *dataset.evaluation]

    assert [item.episode_id for item in observed] == [
        row["episode_id"] for row in frozen_rows
    ]
    assert [item.tensor_digest for item in observed] == [
        row["tensor_digest"] for row in frozen_rows
    ]


def test_registered_dataset_fails_closed_on_frozen_tensor_tamper(
    tmp_path: Path,
) -> None:
    DatasetInvalid, load = _api()
    root = _frozen_artifact(tmp_path)
    path = root / "normalized-episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["tensor"][0][0] = 0.999
    _write_signed_jsonl(root, "normalized-episodes.jsonl", rows)

    with pytest.raises(DatasetInvalid, match="frozen artifact"):
        load(root)


def test_registered_dataset_module_does_not_recompute_normalization() -> None:
    source = Path("src/neural_state_machine/r1_e3_dataset.py").read_text(
        encoding="utf-8"
    )

    assert "r1_e3_normalize" not in source
    assert "normalize_episode" not in source
    assert "normalize_registered_episodes" not in source
