from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .r1_e1_protocol import (
    CORRUPTION_ARMS,
    REGISTERED_ARCHITECTURES,
    REGISTERED_BUDGETS,
    REGISTERED_SEEDS,
    registered_manifest_payload,
)

_BASE_INTEGRATION_HEAD = "d3888b305e659e9f99f40f2c73b11a705622136a"
_AUTOESN_REFERENCE = "Ro6ertWcislo/AutoESN@b3d2e287716176fc3e1312b5be2a7a0b91ba538e"


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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_pair(root: Path, name: str, payload: object) -> str:
    data = canonical_json_bytes(payload)
    digest = _sha256(data)
    (root / name).write_bytes(data)
    (root / f"{name.removesuffix('.json')}.sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def _read_verified(root: Path, name: str) -> object:
    data_path = root / name
    sha_path = root / f"{name.removesuffix('.json')}.sha256"
    if not data_path.is_file() or not sha_path.is_file():
        raise EvidenceInvalid(f"missing {name} or checksum")
    data = data_path.read_bytes()
    expected = sha_path.read_text(encoding="utf-8").strip()
    actual = _sha256(data)
    if expected != actual:
        raise EvidenceInvalid(f"{name} sha256 mismatch")
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInvalid(f"invalid {name}") from exc


def _validate_head(value: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
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


def prepare_prospective(root: Path | str, *, scientific_head: str) -> dict[str, object]:
    root_path = Path(root)
    _validate_head(scientific_head)
    if root_path.exists() and any(root_path.iterdir()):
        raise FileExistsError(f"destination is not empty: {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "r1-e1-manifest-v1",
        "scientific_head": scientific_head,
        "base_integration_head": _BASE_INTEGRATION_HEAD,
        "autoesn_reference": _AUTOESN_REFERENCE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": registered_manifest_payload(),
    }
    manifest_sha = _write_pair(root_path, "manifest.json", manifest)
    prospective = {
        "schema": "r1-e1-prospective-provenance-v1",
        "scientific_head": scientific_head,
        "manifest_sha256": manifest_sha,
        "registered_measurement_executed": False,
    }
    _write_pair(root_path, "prospective-provenance.json", prospective)
    return {"scientific_head": scientific_head, "manifest_sha256": manifest_sha}


def write_measurement(
    root: Path | str,
    *,
    manifest_sha256: str,
    scientific_head: str,
    raw_result: object,
) -> dict[str, object]:
    root_path = Path(root)
    manifest = _read_verified(root_path, "manifest.json")
    if not isinstance(manifest, dict):
        raise EvidenceInvalid("manifest must be an object")
    actual_manifest_sha = _sha256((root_path / "manifest.json").read_bytes())
    if manifest_sha256 != actual_manifest_sha:
        raise EvidenceInvalid("manifest sha256 does not match sealed manifest")
    if scientific_head != manifest.get("scientific_head"):
        raise EvidenceInvalid("scientific head does not match sealed manifest")
    _validate_head(scientific_head)
    runtime_python = platform.python_version()
    runtime_numpy = np.__version__
    _require_runtime_match(
        manifest,
        python_version=runtime_python,
        numpy_version=runtime_numpy,
    )

    measurement_paths = (
        "result.json",
        "result.sha256",
        "provenance.json",
        "provenance.sha256",
        "trace-index.json",
        "trace-index.sha256",
    )
    if any((root_path / name).exists() for name in measurement_paths):
        raise FileExistsError("registered measurement bundle is write-once")

    result_payload = {
        "schema": "r1-e1-result-v1",
        "manifest_sha256": manifest_sha256,
        "measurement": _jsonable(raw_result),
    }
    result_sha = _write_pair(root_path, "result.json", result_payload)
    provenance = {
        "schema": "r1-e1-provenance-v1",
        "scientific_head": scientific_head,
        "manifest_sha256": manifest_sha256,
        "result_sha256": result_sha,
        "python": runtime_python,
        "numpy": runtime_numpy,
    }
    _write_pair(root_path, "provenance.json", provenance)
    _write_pair(root_path, "trace-index.json", {"schema": "r1-e1-trace-index-v1", "artifacts": []})
    return {"manifest_sha256": manifest_sha256, "result_sha256": result_sha}


def verify_evidence(root: Path | str, *, no_result_ok: bool = False) -> dict[str, object]:
    root_path = Path(root)
    manifest = _read_verified(root_path, "manifest.json")
    prospective = _read_verified(root_path, "prospective-provenance.json")
    if not isinstance(manifest, dict) or not isinstance(prospective, dict):
        raise EvidenceInvalid("manifest/prospective provenance must be objects")
    manifest_sha = _sha256((root_path / "manifest.json").read_bytes())
    if prospective.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("prospective provenance manifest mismatch")
    if prospective.get("scientific_head") != manifest.get("scientific_head"):
        raise EvidenceInvalid("prospective provenance scientific head mismatch")

    result_path = root_path / "result.json"
    if not result_path.exists():
        if no_result_ok:
            return {"valid": True, "registered_arm_count": 0, "prospective_only": True}
        raise EvidenceInvalid("registered result is missing")

    result = _read_verified(root_path, "result.json")
    provenance = _read_verified(root_path, "provenance.json")
    _read_verified(root_path, "trace-index.json")
    if not isinstance(result, dict) or not isinstance(provenance, dict):
        raise EvidenceInvalid("result/provenance must be objects")
    if result.get("manifest_sha256") != manifest_sha or provenance.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("measurement manifest linkage mismatch")
    if provenance.get("scientific_head") != manifest.get("scientific_head"):
        raise EvidenceInvalid("measurement scientific head mismatch")
    _require_runtime_match(
        manifest,
        python_version=provenance.get("python"),
        numpy_version=provenance.get("numpy"),
    )
    if provenance.get("result_sha256") != _sha256(result_path.read_bytes()):
        raise EvidenceInvalid("provenance result sha256 mismatch")

    measurement = result.get("measurement")
    arm_count = _validate_registered_measurement(measurement)
    return {"valid": True, "registered_arm_count": arm_count, "prospective_only": False}


def _validate_registered_measurement(measurement: object) -> int:
    if not isinstance(measurement, dict) or measurement.get("registered_measurement") is not True:
        raise EvidenceInvalid("result is not a registered measurement")
    compatibility = measurement.get("compatibility")
    if not isinstance(compatibility, list):
        raise EvidenceInvalid("compatibility records are missing")
    expected_seeds = set(REGISTERED_SEEDS)
    seen_compat = {
        item.get("seed")
        for item in compatibility
        if isinstance(item, dict) and item.get("passed") is True
    }
    if seen_compat != expected_seeds:
        raise EvidenceInvalid("compatibility records do not cover all registered seeds")

    arms = measurement.get("arms")
    if not isinstance(arms, list):
        raise EvidenceInvalid("registered arms are missing")
    expected_keys = {
        (seed, budget, int(architecture))
        for seed in REGISTERED_SEEDS
        for budget in REGISTERED_BUDGETS
        for architecture in REGISTERED_ARCHITECTURES
    }
    seen_keys: set[tuple[int, int, int]] = set()
    registered_corruptions = set(CORRUPTION_ARMS) - {"clean"}
    for arm in arms:
        if not isinstance(arm, dict):
            raise EvidenceInvalid("registered arm must be an object")
        key = (arm.get("seed"), arm.get("budget"), arm.get("architecture"))
        if key not in expected_keys or key in seen_keys:
            raise EvidenceInvalid("registered arm set is incomplete or duplicated")
        seen_keys.add(key)
        _validate_arm_digests(arm)
        yolo = arm.get("yolo_like")
        if not isinstance(yolo, dict):
            raise EvidenceInvalid("yolo_like result is missing")
        corrupted = yolo.get("corrupted")
        if not isinstance(corrupted, list):
            raise EvidenceInvalid("corruption results are missing")
        names = {
            row[0]
            for row in corrupted
            if isinstance(row, list) and len(row) == 2 and isinstance(row[0], str)
        }
        if names != registered_corruptions:
            raise EvidenceInvalid("corruption result names do not match registered arms")
    if seen_keys != expected_keys:
        raise EvidenceInvalid("registered arm count or identities do not match manifest")
    return len(arms)


def _validate_arm_digests(arm: dict[str, Any]) -> None:
    memory = arm.get("memory")
    if not isinstance(memory, dict) or memory.get("valid") is not True:
        raise EvidenceInvalid("memory validity gate failed")
    before = memory.get("parameter_digest_before")
    after = memory.get("parameter_digest_after")
    if not _digest(before) or before != after:
        raise EvidenceInvalid("reservoir parameter digest mismatch")
    for field in ("fixture_digest", "coefficient_digest", "prediction_digest", "reset_prediction_digest"):
        if not _digest(memory.get(field)):
            raise EvidenceInvalid(f"missing memory {field}")

    history = arm.get("history")
    if not isinstance(history, dict) or not isinstance(history.get("horizons"), list) or len(history["horizons"]) != 4:
        raise EvidenceInvalid("history horizon evidence is incomplete")
    for horizon in history["horizons"]:
        if not isinstance(horizon, dict):
            raise EvidenceInvalid("history horizon must be an object")
        for field in (
            "coefficient_digest",
            "prediction_digest",
            "reset_prediction_digest",
            "train_fixture_digest",
            "evaluation_fixture_digest",
            "reservoir_parameter_digest",
        ):
            if not _digest(horizon.get(field)):
                raise EvidenceInvalid(f"missing history {field}")

    yolo = arm.get("yolo_like")
    if not isinstance(yolo, dict):
        raise EvidenceInvalid("yolo_like evidence is missing")
    for field in (
        "train_fixture_digest",
        "evaluation_fixture_digest",
        "corruption_digest",
        "coefficient_digest",
        "reservoir_parameter_digest",
    ):
        if not _digest(yolo.get(field)):
            raise EvidenceInvalid(f"missing yolo_like {field}")
    prediction_digests = yolo.get("prediction_digests")
    if not isinstance(prediction_digests, list):
        raise EvidenceInvalid("yolo_like prediction digests are missing")
    names = {row[0] for row in prediction_digests if isinstance(row, list) and len(row) == 2 and _digest(row[1])}
    if names != set(CORRUPTION_ARMS):
        raise EvidenceInvalid("yolo_like prediction digest corruption set mismatch")


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)