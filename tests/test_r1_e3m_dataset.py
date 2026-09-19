from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from neural_state_machine.r1_e3m_artifact import (
    MechanismArtifact,
    MechanismVideoRecord,
    TrackWindow,
    track_window_id,
)
from neural_state_machine.r1_e3m_artifact_evidence import freeze_mechanism_artifact


def _api():
    from neural_state_machine.r1_e3m_dataset import (
        MechanismDataset,
        MechanismSample,
        load_mechanism_dataset,
    )

    return MechanismDataset, MechanismSample, load_mechanism_dataset


def _artifact() -> MechanismArtifact:
    videos = (
        MechanismVideoRecord(
            "train-video", "1" * 64, "train", 30.0, 300, 1280, 720, "train-c"
        ),
        MechanismVideoRecord(
            "eval-video", "2" * 64, "eval", 30.0, 300, 1280, 720, "eval-c"
        ),
    )
    windows = []
    for video_id, split, track_id in (
        ("train-video", "train", 7),
        ("eval-video", "eval", 9),
    ):
        tensor = np.zeros((20, 6), dtype=np.float64)
        tensor[:, :5] = (0.2, 0.3, 0.1, 0.2, 0.9)
        tensor[:, 5] = 1.0
        windows.append(
            TrackWindow(
                track_window_id(video_id, track_id, 0.0, 2.0),
                video_id,
                split,
                track_id,
                0.0,
                2.0,
                tensor,
            )
        )
    return MechanismArtifact(
        "a" * 64,
        "b" * 64,
        "c" * 64,
        videos,
        tuple(windows),
    )


def test_loader_returns_label_free_train_eval_samples(tmp_path: Path) -> None:
    MechanismDataset, MechanismSample, load = _api()
    root = tmp_path / "frozen"
    evidence = freeze_mechanism_artifact(_artifact(), root)

    dataset = load(root)

    assert isinstance(dataset, MechanismDataset)
    assert len(dataset.training) == 1
    assert len(dataset.evaluation) == 1
    assert isinstance(dataset.training[0], MechanismSample)
    assert dataset.training[0].split == "train"
    assert dataset.evaluation[0].split == "eval"
    assert dataset.artifact_root_digest == evidence.root_digest
    assert dataset.training[0].tensor.flags.writeable is False


def test_sample_schema_contains_no_semantic_fields() -> None:
    _, MechanismSample, _ = _api()

    names = {field.name for field in fields(MechanismSample)}

    assert "label" not in names
    assert "episode_id" not in names
    assert "actor_track_id" not in names
    assert "target_track_id" not in names


def test_train_eval_video_ids_remain_disjoint(tmp_path: Path) -> None:
    _, _, load = _api()
    root = tmp_path / "frozen"
    freeze_mechanism_artifact(_artifact(), root)

    dataset = load(root)

    train_videos = {sample.video_id for sample in dataset.training}
    eval_videos = {sample.video_id for sample in dataset.evaluation}
    assert train_videos.isdisjoint(eval_videos)
