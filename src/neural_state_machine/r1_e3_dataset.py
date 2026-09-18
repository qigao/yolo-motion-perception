from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .r1_e3_artifact import ArtifactInvalid, load_artifact
from .r1_e3_artifact_evidence import verify_frozen_artifact


_REGISTERED_LABELS = ("approach", "touch", "pick_up", "pass_by")
_TENSOR_DIGEST_VERSION = b"r1-e3-normalized-episode-v1\0"


class DatasetInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class RegisteredEpisode:
    episode_id: str
    video_id: str
    split: str
    label: str
    tensor: np.ndarray
    tensor_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise DatasetInvalid("episode_id must be a non-empty string")
        if not isinstance(self.video_id, str) or not self.video_id:
            raise DatasetInvalid("video_id must be a non-empty string")
        if self.split not in ("training", "evaluation"):
            raise DatasetInvalid("split must be training or evaluation")
        if self.label not in _REGISTERED_LABELS:
            raise DatasetInvalid("label is not registered")
        if not _is_digest(self.tensor_digest):
            raise DatasetInvalid("tensor_digest must be a lowercase SHA-256 digest")

        values = np.asarray(self.tensor, dtype=np.float64)
        if values.shape != (20, 14):
            raise DatasetInvalid("registered tensor must have shape (20, 14)")
        if not np.all(np.isfinite(values)):
            raise DatasetInvalid("registered tensor must contain only finite values")
        copied = np.array(values, dtype=np.float64, copy=True, order="C")
        copied.flags.writeable = False
        object.__setattr__(self, "tensor", copied)


@dataclass(frozen=True)
class RegisteredDataset:
    training: tuple[RegisteredEpisode, ...]
    evaluation: tuple[RegisteredEpisode, ...]
    artifact_root_digest: str

    def __post_init__(self) -> None:
        if not self.training or not self.evaluation:
            raise DatasetInvalid("registered dataset must contain both splits")
        if any(item.split != "training" for item in self.training):
            raise DatasetInvalid("training collection contains non-training episode")
        if any(item.split != "evaluation" for item in self.evaluation):
            raise DatasetInvalid("evaluation collection contains non-evaluation episode")
        if not _is_digest(self.artifact_root_digest):
            raise DatasetInvalid(
                "artifact_root_digest must be a lowercase SHA-256 digest"
            )


def load_registered_dataset(root: Path | str) -> RegisteredDataset:
    root_path = Path(root)
    try:
        verified = verify_frozen_artifact(root_path)
        artifact = load_artifact(root_path)
    except ArtifactInvalid as exc:
        raise DatasetInvalid(f"frozen artifact invalid: {exc}") from exc

    root_digest = verified.get("artifact_root_digest")
    if not _is_digest(root_digest):
        raise DatasetInvalid("frozen artifact root digest is missing")

    rows = _read_rows(root_path / "normalized-episodes.jsonl")
    if len(rows) != len(artifact.episodes):
        raise DatasetInvalid("frozen normalized episode count mismatch")

    split_by_video = {video.video_id: video.split for video in artifact.videos}
    registered: list[RegisteredEpisode] = []
    for annotation, row in zip(artifact.episodes, rows, strict=True):
        if row.get("episode_id") != annotation.episode_id:
            raise DatasetInvalid("frozen episode order does not match annotations")
        if row.get("video_id") != annotation.video_id:
            raise DatasetInvalid("frozen episode video_id does not match annotations")
        if row.get("label") != annotation.label:
            raise DatasetInvalid("frozen episode label does not match annotations")

        tensor = _tensor(row.get("tensor"))
        expected_digest = row.get("tensor_digest")
        if not _is_digest(expected_digest):
            raise DatasetInvalid("frozen tensor digest is invalid")
        actual_digest = _tensor_digest(tensor)
        if actual_digest != expected_digest:
            raise DatasetInvalid("frozen tensor digest mismatch")

        split = split_by_video.get(annotation.video_id)
        if split not in ("training", "evaluation"):
            raise DatasetInvalid("frozen episode references invalid video split")
        registered.append(
            RegisteredEpisode(
                episode_id=annotation.episode_id,
                video_id=annotation.video_id,
                split=split,
                label=annotation.label,
                tensor=tensor,
                tensor_digest=expected_digest,
            )
        )

    training = tuple(item for item in registered if item.split == "training")
    evaluation = tuple(item for item in registered if item.split == "evaluation")
    return RegisteredDataset(
        training=training,
        evaluation=evaluation,
        artifact_root_digest=root_digest,
    )


def _read_rows(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetInvalid("cannot read frozen normalized episodes") from exc
    if not lines:
        raise DatasetInvalid("frozen normalized episodes must not be empty")

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise DatasetInvalid(
                f"frozen normalized episodes contain blank row at line {line_number}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetInvalid(
                f"invalid frozen normalized episode row at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise DatasetInvalid("frozen normalized episode row must be an object")
        rows.append(value)
    return rows


def _tensor(value: object) -> np.ndarray:
    try:
        tensor = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DatasetInvalid("frozen tensor is not float64-compatible") from exc
    if tensor.shape != (20, 14):
        raise DatasetInvalid("frozen tensor must have shape (20, 14)")
    if not np.all(np.isfinite(tensor)):
        raise DatasetInvalid("frozen tensor must contain only finite values")
    return np.ascontiguousarray(tensor, dtype=np.float64)


def _tensor_digest(tensor: np.ndarray) -> str:
    values = np.ascontiguousarray(tensor, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(_TENSOR_DIGEST_VERSION)
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
