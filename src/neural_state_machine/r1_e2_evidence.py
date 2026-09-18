from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .r1_e2_protocol import (
    REGISTERED_ARCHITECTURES,
    REGISTERED_SEEDS,
    registered_manifest_payload,
)
from .r1_e2_yolo_episode import EPISODE_CORRUPTIONS


class EvidenceInvalid(RuntimeError):
    pass


def canonical_json_bytes(payload: object) -> bytes:
    text = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def prepare_prospective(
    root: Path | str,
    *,
    scientific_head: str,
) -> dict[str, object]:
    root_path = Path(root)
    _validate_head(scientific_head)
    if root_path.exists() and any(root_path.iterdir()):
        raise FileExistsError(f"destination is not empty: {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "r1-e2-manifest-v1",
        "scientific_head": scientific_head,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": registered_manifest_payload(),
    }
    manifest_sha = _write_pair(root_path, "manifest.json", manifest)
    provenance = {
        "schema": "r1-e2-prospective-provenance-v1",
        "scientific_head": scientific_head,
        "manifest_sha256": manifest_sha,
        "python": manifest["python"],
        "numpy": manifest["numpy"],
        "registered_measurement": False,
    }
    _write_pair(root_path, "prospective-provenance.json", provenance)
    return {
        "scientific_head": scientific_head,
        "manifest_sha256": manifest_sha,
    }


def write_measurement(
    root: Path | str,
    *,
    manifest_sha256: str,
    scientific_head: str,
    raw_result: object,
) -> dict[str, object]:
    root_path = Path(root)
    _validate_head(scientific_head)
    manifest = _mapping(_read_verified(root_path, "manifest.json"), "manifest")
    prospective = _mapping(
        _read_verified(root_path, "prospective-provenance.json"),
        "prospective provenance",
    )
    actual_manifest_sha = _read_digest(root_path, "manifest.sha256")
    if manifest_sha256 != actual_manifest_sha:
        raise EvidenceInvalid("manifest sha256 does not match sealed manifest")
    if manifest.get("scientific_head") != scientific_head:
        raise EvidenceInvalid("scientific head does not match sealed manifest")
    if prospective.get("scientific_head") != scientific_head:
        raise EvidenceInvalid("scientific head does not match prospective provenance")
    if prospective.get("manifest_sha256") != manifest_sha256:
        raise EvidenceInvalid("manifest sha256 does not match prospective provenance")
    _require_runtime_match(
        manifest,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
    )

    for name in (
        "result.json",
        "result.sha256",
        "provenance.json",
        "provenance.sha256",
        "trace-index.json",
        "trace-index.sha256",
    ):
        if (root_path / name).exists():
            raise FileExistsError("registered measurement evidence is write-once")

    result_payload = {
        "schema": "r1-e2-result-v1",
        "manifest_sha256": manifest_sha256,
        "measurement": _jsonable(raw_result),
    }
    result_sha = _write_pair(root_path, "result.json", result_payload)
    provenance = {
        "schema": "r1-e2-provenance-v1",
        "scientific_head": scientific_head,
        "manifest_sha256": manifest_sha256,
        "result_sha256": result_sha,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "registered_measurement": True,
    }
    provenance_sha = _write_pair(root_path, "provenance.json", provenance)
    trace = {
        "schema": "r1-e2-trace-index-v1",
        "manifest_sha256": manifest_sha256,
        "result_sha256": result_sha,
        "provenance_sha256": provenance_sha,
    }
    trace_sha = _write_pair(root_path, "trace-index.json", trace)
    return {
        "manifest_sha256": manifest_sha256,
        "result_sha256": result_sha,
        "provenance_sha256": provenance_sha,
        "trace_index_sha256": trace_sha,
    }


def verify_evidence(
    root: Path | str,
    *,
    no_result_ok: bool = False,
) -> dict[str, object]:
    root_path = Path(root)
    manifest = _mapping(_read_verified(root_path, "manifest.json"), "manifest")
    prospective = _mapping(
        _read_verified(root_path, "prospective-provenance.json"),
        "prospective provenance",
    )
    manifest_sha = _read_digest(root_path, "manifest.sha256")
    if manifest.get("schema") != "r1-e2-manifest-v1":
        raise EvidenceInvalid("manifest schema mismatch")
    scientific_head = _validate_head(manifest.get("scientific_head"))
    if prospective.get("scientific_head") != scientific_head:
        raise EvidenceInvalid("prospective scientific head mismatch")
    if prospective.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("prospective manifest sha256 mismatch")
    _require_runtime_match(
        manifest,
        python_version=prospective.get("python"),
        numpy_version=prospective.get("numpy"),
    )

    result_path = root_path / "result.json"
    if not result_path.exists():
        if no_result_ok:
            return {
                "valid": True,
                "prospective_only": True,
                "scientific_head": scientific_head,
                "manifest_sha256": manifest_sha,
            }
        raise EvidenceInvalid("registered result is missing")

    result = _mapping(_read_verified(root_path, "result.json"), "result")
    provenance = _mapping(_read_verified(root_path, "provenance.json"), "provenance")
    trace = _mapping(_read_verified(root_path, "trace-index.json"), "trace index")
    if result.get("schema") != "r1-e2-result-v1":
        raise EvidenceInvalid("result schema mismatch")
    if result.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("result manifest sha256 mismatch")
    if provenance.get("scientific_head") != scientific_head:
        raise EvidenceInvalid("provenance scientific head mismatch")
    if provenance.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("provenance manifest sha256 mismatch")
    _require_runtime_match(
        manifest,
        python_version=provenance.get("python"),
        numpy_version=provenance.get("numpy"),
    )

    result_sha = _read_digest(root_path, "result.sha256")
    provenance_sha = _read_digest(root_path, "provenance.sha256")
    if provenance.get("result_sha256") != result_sha:
        raise EvidenceInvalid("provenance result sha256 mismatch")
    if trace.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("trace manifest sha256 mismatch")
    if trace.get("result_sha256") != result_sha:
        raise EvidenceInvalid("trace result sha256 mismatch")
    if trace.get("provenance_sha256") != provenance_sha:
        raise EvidenceInvalid("trace provenance sha256 mismatch")

    arm_count = _validate_registered_measurement(result.get("measurement"))
    return {
        "valid": True,
        "prospective_only": False,
        "scientific_head": scientific_head,
        "manifest_sha256": manifest_sha,
        "registered_arm_count": arm_count,
    }


def _validate_registered_measurement(value: object) -> int:
    measurement = _mapping(value, "registered measurement")
    if measurement.get("registered_measurement") is not True:
        raise EvidenceInvalid("registered measurement flag is missing")
    arms = measurement.get("arms")
    if not isinstance(arms, list):
        raise EvidenceInvalid("registered arms are missing")

    expected_keys = {
        (seed, int(architecture))
        for seed in REGISTERED_SEEDS
        for architecture in REGISTERED_ARCHITECTURES
    }
    seen: set[tuple[object, object]] = set()
    for arm in arms:
        arm_map = _mapping(arm, "registered arm")
        key = (arm_map.get("seed"), arm_map.get("architecture"))
        if key not in expected_keys or key in seen:
            raise EvidenceInvalid("registered arm set is incomplete or duplicated")
        seen.add(key)
        _validate_memory(arm_map.get("memory"))
        _validate_composition(arm_map.get("composition"))
        _validate_yolo_episode(arm_map.get("yolo_episode"))
    if seen != expected_keys:
        raise EvidenceInvalid("registered arm count or identities do not match manifest")
    return len(arms)


def _validate_memory(value: object) -> None:
    memory = _mapping(value, "memory")
    core = _mapping(memory.get("core"), "memory core")
    if core.get("valid") is not True:
        raise EvidenceInvalid("memory validity gate failed")
    before = core.get("parameter_digest_before")
    after = core.get("parameter_digest_after")
    if not _digest(before) or before != after:
        raise EvidenceInvalid("memory reservoir parameter digest mismatch")
    for field in (
        "fixture_digest",
        "coefficient_digest",
        "prediction_digest",
        "reset_prediction_digest",
    ):
        if not _digest(core.get(field)):
            raise EvidenceInvalid(f"missing memory {field}")


def _validate_composition(value: object) -> None:
    composition = _mapping(value, "composition")
    histories = composition.get("histories")
    if not isinstance(histories, list) or len(histories) != 4:
        raise EvidenceInvalid("composition histories are incomplete")
    if [item.get("history") for item in histories if isinstance(item, dict)] != [
        5,
        10,
        20,
        40,
    ]:
        raise EvidenceInvalid("composition histories do not match registration")
    for history in histories:
        history_map = _mapping(history, "composition history")
        if history_map.get("reset_groups_equal") is not True:
            raise EvidenceInvalid("composition reset group gate failed")
        for field in (
            "train_fixture_digest",
            "evaluation_fixture_digest",
            "reservoir_parameter_digest",
        ):
            if not _digest(history_map.get(field)):
                raise EvidenceInvalid(f"missing composition {field}")
        for readout_name in ("instantaneous", "temporal_mean"):
            readout = _mapping(history_map.get(readout_name), "composition readout")
            if readout.get("reset_correct") != 25 or readout.get("reset_total") != 150:
                raise EvidenceInvalid("composition reset control mismatch")
            for field in (
                "coefficient_digest",
                "prediction_digest",
                "reset_prediction_digest",
            ):
                if not _digest(readout.get(field)):
                    raise EvidenceInvalid(f"missing composition {field}")
            _validate_metrics(readout.get("metrics"), class_count=6, total=150)
            geometry = _mapping(readout.get("geometry"), "composition geometry")
            centroids = geometry.get("centroid_distances")
            dispersions = geometry.get("within_class_dispersion")
            if not isinstance(centroids, list) or len(centroids) != 15:
                raise EvidenceInvalid("composition centroid geometry is incomplete")
            if not isinstance(dispersions, list) or len(dispersions) != 6:
                raise EvidenceInvalid("composition within-class geometry is incomplete")


def _validate_yolo_episode(value: object) -> None:
    yolo = _mapping(value, "yolo episode")
    if yolo.get("reset_groups_equal") is not True:
        raise EvidenceInvalid("yolo episode reset group gate failed")
    for field in (
        "train_fixture_digest",
        "evaluation_fixture_digest",
        "corruption_digest",
        "reservoir_parameter_digest",
    ):
        if not _digest(yolo.get(field)):
            raise EvidenceInvalid(f"missing yolo episode {field}")

    for baseline_name in (
        "frame_only",
        "reservoir_instantaneous",
        "reservoir_temporal_mean",
    ):
        baseline = _mapping(yolo.get(baseline_name), "yolo baseline")
        if not _digest(baseline.get("coefficient_digest")):
            raise EvidenceInvalid("missing yolo baseline coefficient digest")
        arm_metrics = baseline.get("arm_metrics")
        prediction_digests = baseline.get("prediction_digests")
        if not isinstance(arm_metrics, list) or not isinstance(prediction_digests, list):
            raise EvidenceInvalid("yolo episode corruption results are missing")
        metric_names = [
            row[0]
            for row in arm_metrics
            if isinstance(row, list) and len(row) == 2 and isinstance(row[0], str)
        ]
        digest_names = [
            row[0]
            for row in prediction_digests
            if (
                isinstance(row, list)
                and len(row) == 2
                and isinstance(row[0], str)
                and _digest(row[1])
            )
        ]
        if metric_names != list(EPISODE_CORRUPTIONS):
            raise EvidenceInvalid("yolo episode corruption names do not match registration")
        if digest_names != list(EPISODE_CORRUPTIONS):
            raise EvidenceInvalid("yolo episode corruption prediction digests mismatch")
        for _, metrics in arm_metrics:
            _validate_metrics(metrics, class_count=4, total=200)
        if baseline_name == "frame_only":
            if baseline.get("reset_correct") != 0 or baseline.get("reset_total") != 0:
                raise EvidenceInvalid("frame-only reset fields must be zero")
        elif baseline.get("reset_correct") != 50 or baseline.get("reset_total") != 200:
            raise EvidenceInvalid("yolo episode reset control mismatch")


def _validate_metrics(value: object, *, class_count: int, total: int) -> None:
    metrics = _mapping(value, "classification metrics")
    if metrics.get("total") != total:
        raise EvidenceInvalid("classification metric total mismatch")
    confusion = metrics.get("confusion_counts")
    if (
        not isinstance(confusion, list)
        or len(confusion) != class_count
        or any(not isinstance(row, list) or len(row) != class_count for row in confusion)
    ):
        raise EvidenceInvalid("classification confusion matrix shape mismatch")


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_pair(root: Path, name: str, payload: object) -> str:
    data = canonical_json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    (root / name).write_bytes(data)
    (root / f"{name.removesuffix('.json')}.sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )
    return digest


def _read_verified(root: Path, name: str) -> object:
    data_path = root / name
    sha_path = root / f"{name.removesuffix('.json')}.sha256"
    if not data_path.is_file() or not sha_path.is_file():
        raise EvidenceInvalid(f"missing {name} or checksum")
    data = data_path.read_bytes()
    expected = sha_path.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(data).hexdigest()
    if expected != actual:
        raise EvidenceInvalid(f"{name} sha256 mismatch")
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInvalid(f"invalid {name}") from exc


def _read_digest(root: Path, name: str) -> str:
    path = root / name
    if not path.is_file():
        raise EvidenceInvalid(f"missing {name}")
    value = path.read_text(encoding="utf-8").strip()
    if not _digest(value):
        raise EvidenceInvalid(f"invalid {name}")
    return value


def _validate_head(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise EvidenceInvalid("scientific head must be a 40-character lowercase git sha")
    return value


def _require_runtime_match(
    manifest: Mapping[str, object],
    *,
    python_version: object,
    numpy_version: object,
) -> None:
    if (
        manifest.get("python") != python_version
        or manifest.get("numpy") != numpy_version
    ):
        raise EvidenceInvalid("measurement runtime does not match sealed manifest")


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceInvalid(f"{name} must be an object")
    return value


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )
