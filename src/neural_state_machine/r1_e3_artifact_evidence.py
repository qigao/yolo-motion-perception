from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping

from .r1_e3_artifact import (
    ArtifactInvalid,
    FrozenTrackArtifact,
    artifact_root_digest as source_artifact_root_digest,
    load_artifact,
    validate_artifact_structure,
)
from .r1_e3_normalize import normalize_registered_episodes


FROZEN_ARTIFACT_SCHEMA = "r1-e3-frozen-artifact-v1"
REGISTERED_LABELS = ("approach", "touch", "pick_up", "pass_by")
_REQUIRED_TRAIN_PER_CLASS = 20
_REQUIRED_EVAL_PER_CLASS = 10
_REQUIRED_TRAIN_VIDEO_COUNT = 2
_REQUIRED_EVAL_VIDEO_COUNT = 2


def freeze_artifact(
    candidate_root: Path | str,
    frozen_root: Path | str,
) -> dict[str, object]:
    candidate = Path(candidate_root)
    destination = Path(frozen_root)
    artifact = load_artifact(candidate)
    validate_artifact_structure(candidate)
    counts = _registered_counts(artifact)
    _require_registration_gate(counts)

    _prepare_destination(destination)
    _copy_source_artifact(candidate, destination, artifact)

    normalized_rows = [
        {
            "episode_id": episode.episode_id,
            "video_id": episode.video_id,
            "label": episode.label,
            "tensor": episode.tensor.tolist(),
            "tensor_digest": episode.tensor_digest,
        }
        for episode in normalize_registered_episodes(artifact)
    ]
    normalized_digest = _write_signed_jsonl(
        destination,
        "normalized-episodes.jsonl",
        normalized_rows,
    )

    frozen_manifest = {
        "schema": FROZEN_ARTIFACT_SCHEMA,
        "source_artifact_root_digest": source_artifact_root_digest(candidate),
        "normalized_episodes_sha256": normalized_digest,
        "normalized_episode_count": len(normalized_rows),
        "training_episode_count": counts["training_episode_count"],
        "evaluation_episode_count": counts["evaluation_episode_count"],
        "training_video_count": counts["training_video_count"],
        "evaluation_video_count": counts["evaluation_video_count"],
        "class_counts": {
            "training": counts["training_class_counts"],
            "evaluation": counts["evaluation_class_counts"],
        },
    }
    root_digest = _write_signed_json(
        destination,
        "frozen-artifact.json",
        frozen_manifest,
    )
    return {
        "valid": True,
        "artifact_root_digest": root_digest,
        **counts,
    }


def verify_frozen_artifact(root: Path | str) -> dict[str, object]:
    root_path = Path(root)
    artifact = load_artifact(root_path)
    validate_artifact_structure(root_path)
    counts = _registered_counts(artifact)
    _require_registration_gate(counts)

    normalized_rows, normalized_digest = _read_verified_jsonl(
        root_path,
        "normalized-episodes.jsonl",
    )
    manifest, root_digest = _read_verified_json(
        root_path,
        "frozen-artifact.json",
    )
    if not isinstance(manifest, dict):
        raise ArtifactInvalid("frozen-artifact.json must be an object")
    if manifest.get("schema") != FROZEN_ARTIFACT_SCHEMA:
        raise ArtifactInvalid("frozen artifact schema mismatch")
    if manifest.get("source_artifact_root_digest") != source_artifact_root_digest(
        root_path
    ):
        raise ArtifactInvalid("source artifact root digest mismatch")
    if manifest.get("normalized_episodes_sha256") != normalized_digest:
        raise ArtifactInvalid("normalized-episodes digest mismatch")
    if manifest.get("normalized_episode_count") != len(normalized_rows):
        raise ArtifactInvalid("normalized-episodes count mismatch")

    expected_episode_ids = [episode.episode_id for episode in artifact.episodes]
    observed_episode_ids: list[str] = []
    for row in normalized_rows:
        if not isinstance(row, dict):
            raise ArtifactInvalid("normalized-episodes row must be an object")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str):
            raise ArtifactInvalid("normalized-episodes episode_id is invalid")
        observed_episode_ids.append(episode_id)
        label = row.get("label")
        if label not in REGISTERED_LABELS:
            raise ArtifactInvalid("normalized-episodes label is invalid")
        digest = row.get("tensor_digest")
        _require_digest(digest, "normalized-episodes tensor_digest")
        tensor = row.get("tensor")
        if (
            not isinstance(tensor, list)
            or len(tensor) != 20
            or any(not isinstance(item, list) or len(item) != 14 for item in tensor)
        ):
            raise ArtifactInvalid("normalized-episodes tensor must have shape (20, 14)")
    if observed_episode_ids != expected_episode_ids:
        raise ArtifactInvalid("normalized-episodes order does not match annotations")

    expected_manifest_fields = {
        "training_episode_count": counts["training_episode_count"],
        "evaluation_episode_count": counts["evaluation_episode_count"],
        "training_video_count": counts["training_video_count"],
        "evaluation_video_count": counts["evaluation_video_count"],
    }
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            raise ArtifactInvalid(f"frozen artifact {field} mismatch")

    expected_class_counts = {
        "training": counts["training_class_counts"],
        "evaluation": counts["evaluation_class_counts"],
    }
    if manifest.get("class_counts") != expected_class_counts:
        raise ArtifactInvalid("frozen artifact class counts mismatch")

    return {
        "valid": True,
        "artifact_root_digest": root_digest,
        **counts,
    }


def _registered_counts(artifact: FrozenTrackArtifact) -> dict[str, object]:
    split_by_video = {video.video_id: video.split for video in artifact.videos}
    training_counts: Counter[str] = Counter()
    evaluation_counts: Counter[str] = Counter()
    training_videos: set[str] = set()
    evaluation_videos: set[str] = set()

    for episode in artifact.episodes:
        split = split_by_video.get(episode.video_id)
        if split == "training":
            training_counts[episode.label] += 1
            training_videos.add(episode.video_id)
        elif split == "evaluation":
            evaluation_counts[episode.label] += 1
            evaluation_videos.add(episode.video_id)
        else:
            raise ArtifactInvalid("episode references video with invalid split")

    return {
        "training_episode_count": sum(training_counts.values()),
        "evaluation_episode_count": sum(evaluation_counts.values()),
        "training_video_count": len(training_videos),
        "evaluation_video_count": len(evaluation_videos),
        "training_class_counts": {
            label: training_counts[label] for label in REGISTERED_LABELS
        },
        "evaluation_class_counts": {
            label: evaluation_counts[label] for label in REGISTERED_LABELS
        },
    }


def _require_registration_gate(counts: Mapping[str, object]) -> None:
    training = counts["training_class_counts"]
    evaluation = counts["evaluation_class_counts"]
    assert isinstance(training, dict)
    assert isinstance(evaluation, dict)
    if any(training.get(label, 0) < _REQUIRED_TRAIN_PER_CLASS for label in REGISTERED_LABELS):
        raise ArtifactInvalid("artifact requires at least 20 training episodes per class")
    if any(evaluation.get(label, 0) < _REQUIRED_EVAL_PER_CLASS for label in REGISTERED_LABELS):
        raise ArtifactInvalid("artifact requires at least 10 evaluation episodes per class")
    if counts["training_video_count"] < _REQUIRED_TRAIN_VIDEO_COUNT:
        raise ArtifactInvalid("artifact requires two distinct training source videos")
    if counts["evaluation_video_count"] < _REQUIRED_EVAL_VIDEO_COUNT:
        raise ArtifactInvalid("artifact requires two distinct evaluation source videos")


def _prepare_destination(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)


def _copy_source_artifact(
    source: Path,
    destination: Path,
    artifact: FrozenTrackArtifact,
) -> None:
    files = artifact.manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactInvalid("artifact manifest files must be an object")
    names = ["artifact-manifest.json", "artifact-manifest.sha256"]
    for name in sorted(str(value) for value in files):
        names.extend((name, _sidecar_name(name)))
    for name in names:
        path = source / name
        if not path.is_file():
            raise ArtifactInvalid(f"missing artifact file: {name}")
        shutil.copy2(path, destination / name)


def _write_signed_json(root: Path, name: str, payload: object) -> str:
    data = _canonical_json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    (root / name).write_bytes(data)
    (root / _sidecar_name(name)).write_text(digest + "\n", encoding="utf-8")
    return digest


def _write_signed_jsonl(
    root: Path,
    name: str,
    rows: list[dict[str, object]],
) -> str:
    data = b"".join(_canonical_json_bytes(row) for row in rows)
    digest = hashlib.sha256(data).hexdigest()
    (root / name).write_bytes(data)
    (root / _sidecar_name(name)).write_text(digest + "\n", encoding="utf-8")
    return digest


def _read_verified_json(
    root: Path,
    name: str,
) -> tuple[object, str]:
    data, digest = _read_verified_bytes(root, name)
    try:
        return json.loads(data), digest
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactInvalid(f"invalid {name}") from exc


def _read_verified_jsonl(
    root: Path,
    name: str,
) -> tuple[list[object], str]:
    data, digest = _read_verified_bytes(root, name)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactInvalid(f"invalid {name}") from exc
    rows: list[object] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ArtifactInvalid(f"{name} contains blank row at line {line_number}")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ArtifactInvalid(f"invalid {name} row at line {line_number}") from exc
    if not rows:
        raise ArtifactInvalid(f"{name} must not be empty")
    return rows, digest


def _read_verified_bytes(root: Path, name: str) -> tuple[bytes, str]:
    path = root / name
    sidecar = root / _sidecar_name(name)
    if not path.is_file() or not sidecar.is_file():
        raise ArtifactInvalid(f"missing {name} or checksum")
    data = path.read_bytes()
    expected = sidecar.read_text(encoding="utf-8").strip()
    _require_digest(expected, _sidecar_name(name))
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ArtifactInvalid(f"{name} sha256 mismatch")
    return data, actual


def _canonical_json_bytes(payload: object) -> bytes:
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


def _sidecar_name(name: str) -> str:
    if name.endswith(".jsonl"):
        return f"{name[:-6]}.sha256"
    if name.endswith(".json"):
        return f"{name[:-5]}.sha256"
    return f"{name}.sha256"


def _require_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactInvalid(f"{name} must be a lowercase SHA-256 digest")
    return value
