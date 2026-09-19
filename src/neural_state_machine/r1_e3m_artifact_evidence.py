from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_state_machine.r1_e3m_artifact import (
    MechanismArtifact,
    MechanismArtifactInvalid,
    MechanismVideoRecord,
    TrackWindow,
    mechanism_artifact_digest,
    validate_mechanism_artifact,
    window_tensor_digest,
)


FROZEN_SCHEMA = "r1-e3m-frozen-artifact-v1"
CANDIDATE_SCHEMA = "r1-e3m-candidate-artifact-v1"
_PAYLOAD_FILES = (
    "videos.json",
    "windows.json",
    "window-tensors.npy",
)


class FrozenMechanismArtifactInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenMechanismArtifactEvidence:
    root_digest: str
    video_count: int
    window_count: int



def write_mechanism_candidate(
    artifact: MechanismArtifact,
    candidate_root: Path | str,
) -> Path:
    root = Path(candidate_root)
    if root.exists():
        raise FileExistsError(root)
    try:
        validate_mechanism_artifact(artifact)
    except MechanismArtifactInvalid as exc:
        raise FrozenMechanismArtifactInvalid(str(exc)) from exc

    root.mkdir(parents=True)
    tensors = np.stack(
        [window.tensor for window in artifact.windows],
        axis=0,
    ).astype(np.float64, copy=False)
    with (root / "window-tensors.npy").open("wb") as handle:
        np.save(handle, tensors, allow_pickle=False)
    payload = {
        "schema": CANDIDATE_SCHEMA,
        "source_manifest_sha256": artifact.source_manifest_sha256,
        "raw_track_sha256": artifact.raw_track_sha256,
        "extraction_provenance_sha256":
            artifact.extraction_provenance_sha256,
        "videos": [_video_payload(video) for video in artifact.videos],
        "windows": [
            _window_payload(window, index)
            for index, window in enumerate(artifact.windows)
        ],
    }
    _write_json(root / "candidate.json", payload)
    return root


def load_mechanism_candidate(
    candidate_root: Path | str,
) -> MechanismArtifact:
    root = Path(candidate_root)
    payload = _read_json(root / "candidate.json")
    if not isinstance(payload, dict):
        raise FrozenMechanismArtifactInvalid(
            "candidate.json must be an object"
        )
    if payload.get("schema") != CANDIDATE_SCHEMA:
        raise FrozenMechanismArtifactInvalid(
            "candidate schema mismatch"
        )
    videos_raw = payload.get("videos")
    windows_raw = payload.get("windows")
    if not isinstance(videos_raw, list) or not videos_raw:
        raise FrozenMechanismArtifactInvalid(
            "candidate videos must be a non-empty array"
        )
    if not isinstance(windows_raw, list) or not windows_raw:
        raise FrozenMechanismArtifactInvalid(
            "candidate windows must be a non-empty array"
        )
    tensors = _load_tensor_file(
        root / "window-tensors.npy",
        len(windows_raw),
    )
    try:
        artifact = MechanismArtifact(
            source_manifest_sha256=_required_digest(
                payload.get("source_manifest_sha256"),
                "source_manifest_sha256",
            ),
            raw_track_sha256=_required_digest(
                payload.get("raw_track_sha256"),
                "raw_track_sha256",
            ),
            extraction_provenance_sha256=_required_digest(
                payload.get("extraction_provenance_sha256"),
                "extraction_provenance_sha256",
            ),
            videos=tuple(
                _video_from_payload(item)
                for item in videos_raw
            ),
            windows=tuple(
                _window_from_payload(item, tensors, index)
                for index, item in enumerate(windows_raw)
            ),
        )
        validate_mechanism_artifact(artifact)
    except MechanismArtifactInvalid as exc:
        raise FrozenMechanismArtifactInvalid(str(exc)) from exc
    return artifact


def freeze_mechanism_artifact(
    artifact: MechanismArtifact,
    frozen_root: Path | str,
) -> FrozenMechanismArtifactEvidence:
    root = Path(frozen_root)
    if root.exists():
        raise FileExistsError(root)

    try:
        validate_mechanism_artifact(artifact)
    except MechanismArtifactInvalid as exc:
        raise FrozenMechanismArtifactInvalid(str(exc)) from exc

    temp = root.with_name(f"{root.name}.tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    try:
        videos_payload = [_video_payload(video) for video in artifact.videos]
        windows_payload = [
            _window_payload(window, index)
            for index, window in enumerate(artifact.windows)
        ]
        tensors = np.stack(
            [window.tensor for window in artifact.windows],
            axis=0,
        ).astype(np.float64, copy=False)

        _write_json(temp / "videos.json", videos_payload)
        _write_json(temp / "windows.json", windows_payload)
        with (temp / "window-tensors.npy").open("wb") as handle:
            np.save(handle, tensors, allow_pickle=False)

        file_digests = {
            name: _sha256_file(temp / name)
            for name in _PAYLOAD_FILES
        }
        root_digest = mechanism_artifact_digest(artifact)
        manifest = {
            "schema": FROZEN_SCHEMA,
            "root_digest": root_digest,
            "source_manifest_sha256": artifact.source_manifest_sha256,
            "raw_track_sha256": artifact.raw_track_sha256,
            "extraction_provenance_sha256":
                artifact.extraction_provenance_sha256,
            "files": file_digests,
        }
        manifest_path = temp / "artifact-manifest.json"
        _write_json(manifest_path, manifest)
        (temp / "artifact-manifest.sha256").write_text(
            _sha256_file(manifest_path) + "\n",
            encoding="utf-8",
        )

        temp.rename(root)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    return FrozenMechanismArtifactEvidence(
        root_digest=root_digest,
        video_count=len(artifact.videos),
        window_count=len(artifact.windows),
    )


def load_frozen_mechanism_artifact(
    frozen_root: Path | str,
) -> MechanismArtifact:
    root = Path(frozen_root)
    manifest = _load_verified_manifest(root)
    _verify_payloads(root, manifest)

    videos_raw = _read_json(root / "videos.json")
    windows_raw = _read_json(root / "windows.json")
    if not isinstance(videos_raw, list) or not videos_raw:
        raise FrozenMechanismArtifactInvalid(
            "videos.json must be a non-empty array"
        )
    if not isinstance(windows_raw, list) or not windows_raw:
        raise FrozenMechanismArtifactInvalid(
            "windows.json must be a non-empty array"
        )

    tensors = _load_tensor_file(
        root / "window-tensors.npy",
        len(windows_raw),
    )

    try:
        videos = tuple(_video_from_payload(item) for item in videos_raw)
        windows = tuple(
            _window_from_payload(item, tensors, index)
            for index, item in enumerate(windows_raw)
        )
        artifact = MechanismArtifact(
            source_manifest_sha256=_required_digest(
                manifest.get("source_manifest_sha256"),
                "source_manifest_sha256",
            ),
            raw_track_sha256=_required_digest(
                manifest.get("raw_track_sha256"),
                "raw_track_sha256",
            ),
            extraction_provenance_sha256=_required_digest(
                manifest.get("extraction_provenance_sha256"),
                "extraction_provenance_sha256",
            ),
            videos=videos,
            windows=windows,
        )
        validate_mechanism_artifact(artifact)
    except MechanismArtifactInvalid as exc:
        raise FrozenMechanismArtifactInvalid(str(exc)) from exc

    expected_root = _required_digest(
        manifest.get("root_digest"),
        "root_digest",
    )
    actual_root = mechanism_artifact_digest(artifact)
    if actual_root != expected_root:
        raise FrozenMechanismArtifactInvalid(
            "artifact root_digest mismatch"
        )
    return artifact


def verify_frozen_mechanism_artifact(
    frozen_root: Path | str,
) -> FrozenMechanismArtifactEvidence:
    artifact = load_frozen_mechanism_artifact(frozen_root)
    return FrozenMechanismArtifactEvidence(
        root_digest=mechanism_artifact_digest(artifact),
        video_count=len(artifact.videos),
        window_count=len(artifact.windows),
    )


def _video_payload(video: MechanismVideoRecord) -> dict[str, object]:
    return {
        "video_id": video.video_id,
        "sha256": video.sha256,
        "split": video.split,
        "fps": video.fps,
        "frame_count": video.frame_count,
        "width": video.width,
        "height": video.height,
        "source_window_component_id": video.source_window_component_id,
    }


def _window_payload(
    window: TrackWindow,
    tensor_index: int,
) -> dict[str, object]:
    return {
        "window_id": window.window_id,
        "video_id": window.video_id,
        "split": window.split,
        "track_id": window.track_id,
        "source_start_seconds": window.source_start_seconds,
        "source_end_seconds": window.source_end_seconds,
        "tensor_index": tensor_index,
        "tensor_sha256": window_tensor_digest(window),
    }


def _video_from_payload(value: object) -> MechanismVideoRecord:
    item = _object(value, "video")
    return MechanismVideoRecord(
        video_id=_required_text(item.get("video_id"), "video_id"),
        sha256=_required_digest(item.get("sha256"), "video sha256"),
        split=_required_text(item.get("split"), "video split"),
        fps=_number(item.get("fps"), "fps"),
        frame_count=_integer(item.get("frame_count"), "frame_count"),
        width=_integer(item.get("width"), "width"),
        height=_integer(item.get("height"), "height"),
        source_window_component_id=_required_text(
            item.get("source_window_component_id"),
            "source_window_component_id",
        ),
    )


def _window_from_payload(
    value: object,
    tensors: np.ndarray,
    expected_index: int,
) -> TrackWindow:
    item = _object(value, "window")
    tensor_index = _integer(item.get("tensor_index"), "tensor_index")
    if tensor_index != expected_index:
        raise FrozenMechanismArtifactInvalid(
            "window tensor_index must match frozen window order"
        )
    window = TrackWindow(
        window_id=_required_digest(
            item.get("window_id"),
            "window_id",
        ),
        video_id=_required_text(item.get("video_id"), "window video_id"),
        split=_required_text(item.get("split"), "window split"),
        track_id=_integer(item.get("track_id"), "track_id"),
        source_start_seconds=_number(
            item.get("source_start_seconds"),
            "source_start_seconds",
        ),
        source_end_seconds=_number(
            item.get("source_end_seconds"),
            "source_end_seconds",
        ),
        tensor=tensors[tensor_index],
    )
    expected_tensor_digest = _required_digest(
        item.get("tensor_sha256"),
        "tensor_sha256",
    )
    if window_tensor_digest(window) != expected_tensor_digest:
        raise FrozenMechanismArtifactInvalid(
            "window tensor_sha256 mismatch"
        )
    return window


def _load_verified_manifest(root: Path) -> dict[str, object]:
    path = root / "artifact-manifest.json"
    sidecar = root / "artifact-manifest.sha256"
    if not path.is_file() or not sidecar.is_file():
        raise FrozenMechanismArtifactInvalid(
            "missing artifact manifest or checksum"
        )
    expected = sidecar.read_text(encoding="utf-8").strip()
    _required_digest(expected, "artifact-manifest.sha256")
    if _sha256_file(path) != expected:
        raise FrozenMechanismArtifactInvalid(
            "artifact-manifest.json sha256 mismatch"
        )
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise FrozenMechanismArtifactInvalid(
            "artifact manifest must be an object"
        )
    if payload.get("schema") != FROZEN_SCHEMA:
        raise FrozenMechanismArtifactInvalid(
            "artifact manifest schema mismatch"
        )
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(_PAYLOAD_FILES):
        raise FrozenMechanismArtifactInvalid(
            "artifact manifest payload file set mismatch"
        )
    return payload


def _verify_payloads(
    root: Path,
    manifest: dict[str, object],
) -> None:
    files = manifest["files"]
    assert isinstance(files, dict)
    for name in _PAYLOAD_FILES:
        expected = _required_digest(files.get(name), f"{name} sha256")
        path = root / name
        if not path.is_file():
            raise FrozenMechanismArtifactInvalid(
                f"missing frozen payload: {name}"
            )
        if _sha256_file(path) != expected:
            raise FrozenMechanismArtifactInvalid(
                f"{name} sha256 mismatch"
            )



def _load_tensor_file(path: Path, expected_count: int) -> np.ndarray:
    try:
        with path.open("rb") as handle:
            tensors = np.load(handle, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FrozenMechanismArtifactInvalid(
            "invalid window-tensors.npy"
        ) from exc
    if (
        not isinstance(tensors, np.ndarray)
        or tensors.dtype != np.float64
        or tensors.ndim != 3
        or tensors.shape[1:] != (20, 6)
        or tensors.shape[0] != expected_count
        or not np.isfinite(tensors).all()
    ):
        raise FrozenMechanismArtifactInvalid(
            "window-tensors.npy must be finite float64 N x 20 x 6"
        )
    return tensors


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenMechanismArtifactInvalid(
            f"invalid JSON payload: {path.name}"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FrozenMechanismArtifactInvalid(
            f"{name} must be an object"
        )
    return value


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrozenMechanismArtifactInvalid(
            f"{name} must be a non-empty string"
        )
    return value


def _required_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise FrozenMechanismArtifactInvalid(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenMechanismArtifactInvalid(f"{name} must be numeric")
    return float(value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise FrozenMechanismArtifactInvalid(
            f"{name} must be an integer"
        )
    return value
