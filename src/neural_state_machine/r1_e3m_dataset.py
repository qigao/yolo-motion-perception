from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_state_machine.r1_e3m_artifact_evidence import (
    FrozenMechanismArtifactInvalid,
    load_frozen_mechanism_artifact,
    verify_frozen_mechanism_artifact,
)


class MechanismDatasetInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class MechanismSample:
    window_id: str
    video_id: str
    split: str
    track_id: int
    source_start_seconds: float
    source_end_seconds: float
    tensor: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or len(self.window_id) != 64:
            raise MechanismDatasetInvalid(
                "window_id must be a SHA-256 digest"
            )
        if not isinstance(self.video_id, str) or not self.video_id:
            raise MechanismDatasetInvalid(
                "video_id must be a non-empty string"
            )
        if self.split not in ("train", "eval"):
            raise MechanismDatasetInvalid("split must be train or eval")
        if type(self.track_id) is not int or self.track_id < 0:
            raise MechanismDatasetInvalid(
                "track_id must be a non-negative integer"
            )

        values = np.asarray(self.tensor)
        if values.dtype != np.float64 or values.shape != (20, 6):
            raise MechanismDatasetInvalid(
                "tensor must have shape (20, 6) and dtype float64"
            )
        if not np.isfinite(values).all():
            raise MechanismDatasetInvalid(
                "tensor must contain only finite values"
            )
        copied = np.array(values, dtype=np.float64, copy=True, order="C")
        copied.setflags(write=False)
        object.__setattr__(self, "tensor", copied)


@dataclass(frozen=True)
class MechanismDataset:
    training: tuple[MechanismSample, ...]
    evaluation: tuple[MechanismSample, ...]
    artifact_root_digest: str

    def __post_init__(self) -> None:
        if not self.training or not self.evaluation:
            raise MechanismDatasetInvalid(
                "dataset must contain train and eval samples"
            )
        if any(sample.split != "train" for sample in self.training):
            raise MechanismDatasetInvalid(
                "training collection contains non-train sample"
            )
        if any(sample.split != "eval" for sample in self.evaluation):
            raise MechanismDatasetInvalid(
                "evaluation collection contains non-eval sample"
            )
        if (
            not isinstance(self.artifact_root_digest, str)
            or len(self.artifact_root_digest) != 64
        ):
            raise MechanismDatasetInvalid(
                "artifact_root_digest must be a SHA-256 digest"
            )

        train_videos = {sample.video_id for sample in self.training}
        eval_videos = {sample.video_id for sample in self.evaluation}
        if not train_videos.isdisjoint(eval_videos):
            raise MechanismDatasetInvalid(
                "source-video overlap between train and eval"
            )


def load_mechanism_dataset(
    frozen_root: Path | str,
) -> MechanismDataset:
    try:
        evidence = verify_frozen_mechanism_artifact(frozen_root)
        artifact = load_frozen_mechanism_artifact(frozen_root)
    except FrozenMechanismArtifactInvalid as exc:
        raise MechanismDatasetInvalid(
            f"frozen mechanism artifact invalid: {exc}"
        ) from exc

    samples = tuple(
        MechanismSample(
            window_id=window.window_id,
            video_id=window.video_id,
            split=window.split,
            track_id=window.track_id,
            source_start_seconds=window.source_start_seconds,
            source_end_seconds=window.source_end_seconds,
            tensor=window.tensor,
        )
        for window in artifact.windows
    )
    return MechanismDataset(
        training=tuple(
            sample for sample in samples if sample.split == "train"
        ),
        evaluation=tuple(
            sample for sample in samples if sample.split == "eval"
        ),
        artifact_root_digest=evidence.root_digest,
    )
