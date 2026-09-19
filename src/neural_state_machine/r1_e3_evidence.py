from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import median
from typing import Mapping

import numpy as np

from .r1_e3_protocol import (
    ARM_COUNT,
    REGISTERED_ARCHITECTURES,
    REGISTERED_SEEDS,
    TEMPORAL_WINDOW_BINS,
    classify_registered_outcome,
    registered_manifest_payload,
)


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
    artifact_root_digest: object,
) -> dict[str, object]:
    root_path = Path(root)
    head = _validated_head(scientific_head)
    artifact_digest = _validated_digest(
        artifact_root_digest,
        "artifact_root_digest",
    )
    if root_path.exists() and any(root_path.iterdir()):
        raise FileExistsError(f"destination is not empty: {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "r1-e3-manifest-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": registered_manifest_payload(artifact_digest),
    }
    manifest_sha = _write_pair(root_path, "manifest.json", manifest)
    prospective = {
        "schema": "r1-e3-prospective-provenance-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
        "python": manifest["python"],
        "numpy": manifest["numpy"],
        "registered_measurement": False,
    }
    _write_pair(root_path, "prospective-provenance.json", prospective)
    return {
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
    }


def write_measurement(
    root: Path | str,
    *,
    manifest_sha256: str,
    scientific_head: str,
    artifact_root_digest: object,
    raw_result: object,
) -> dict[str, object]:
    root_path = Path(root)
    head = _validated_head(scientific_head)
    artifact_digest = _validated_digest(
        artifact_root_digest,
        "artifact_root_digest",
    )

    manifest = _mapping(
        _read_verified(root_path, "manifest.json"),
        "manifest",
    )
    prospective = _mapping(
        _read_verified(root_path, "prospective-provenance.json"),
        "prospective provenance",
    )
    actual_manifest_sha = _read_digest(root_path, "manifest.sha256")

    if manifest_sha256 != actual_manifest_sha:
        raise EvidenceInvalid("manifest sha256 does not match sealed manifest")
    if manifest.get("scientific_head") != head:
        raise EvidenceInvalid("scientific head does not match sealed manifest")
    if prospective.get("scientific_head") != head:
        raise EvidenceInvalid(
            "scientific head does not match prospective provenance"
        )
    if manifest.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid(
            "artifact root digest does not match sealed manifest"
        )
    if prospective.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid(
            "artifact root digest does not match prospective provenance"
        )
    if prospective.get("manifest_sha256") != actual_manifest_sha:
        raise EvidenceInvalid(
            "manifest sha256 does not match prospective provenance"
        )
    if manifest.get("protocol") != registered_manifest_payload(artifact_digest):
        raise EvidenceInvalid("sealed protocol does not match R1-E3 registration")

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
            raise FileExistsError(
                "registered measurement evidence is write-once"
            )

    measurement = _jsonable(raw_result)
    validation = _validate_registered_measurement(
        measurement,
        expected_artifact_digest=artifact_digest,
    )

    result_payload = {
        "schema": "r1-e3-result-v1",
        "manifest_sha256": actual_manifest_sha,
        "artifact_root_digest": artifact_digest,
        "measurement": measurement,
    }
    result_sha = _write_pair(root_path, "result.json", result_payload)

    provenance = {
        "schema": "r1-e3-provenance-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": actual_manifest_sha,
        "result_sha256": result_sha,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "registered_measurement": True,
        "registered_arm_count": validation["registered_arm_count"],
        "outcome": validation["outcome"],
    }
    provenance_sha = _write_pair(
        root_path,
        "provenance.json",
        provenance,
    )
    trace = {
        "schema": "r1-e3-trace-index-v1",
        "manifest_sha256": actual_manifest_sha,
        "result_sha256": result_sha,
        "provenance_sha256": provenance_sha,
        "artifact_root_digest": artifact_digest,
    }
    trace_sha = _write_pair(root_path, "trace-index.json", trace)

    return {
        "manifest_sha256": actual_manifest_sha,
        "result_sha256": result_sha,
        "provenance_sha256": provenance_sha,
        "trace_index_sha256": trace_sha,
        "registered_arm_count": validation["registered_arm_count"],
        "outcome": validation["outcome"],
    }


def verify_evidence(
    root: Path | str,
    *,
    no_result_ok: bool = False,
) -> dict[str, object]:
    root_path = Path(root)
    manifest = _mapping(
        _read_verified(root_path, "manifest.json"),
        "manifest",
    )
    prospective = _mapping(
        _read_verified(root_path, "prospective-provenance.json"),
        "prospective provenance",
    )
    manifest_sha = _read_digest(root_path, "manifest.sha256")

    if manifest.get("schema") != "r1-e3-manifest-v1":
        raise EvidenceInvalid("manifest schema mismatch")
    head = _validated_head(manifest.get("scientific_head"))
    artifact_digest = _validated_digest(
        manifest.get("artifact_root_digest"),
        "artifact_root_digest",
    )
    if manifest.get("protocol") != registered_manifest_payload(artifact_digest):
        raise EvidenceInvalid("manifest protocol does not match R1-E3 registration")

    if prospective.get("schema") != "r1-e3-prospective-provenance-v1":
        raise EvidenceInvalid("prospective provenance schema mismatch")
    if prospective.get("scientific_head") != head:
        raise EvidenceInvalid("prospective scientific head mismatch")
    if prospective.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid("prospective artifact root digest mismatch")
    if prospective.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("prospective manifest sha256 mismatch")
    if prospective.get("registered_measurement") is not False:
        raise EvidenceInvalid("prospective provenance measurement flag mismatch")
    _require_runtime_match(
        manifest,
        python_version=prospective.get("python"),
        numpy_version=prospective.get("numpy"),
    )

    if not (root_path / "result.json").exists():
        if no_result_ok:
            return {
                "valid": True,
                "prospective_only": True,
                "scientific_head": head,
                "artifact_root_digest": artifact_digest,
                "manifest_sha256": manifest_sha,
                "registered_arm_count": ARM_COUNT,
            }
        raise EvidenceInvalid("registered result is missing")

    result = _mapping(
        _read_verified(root_path, "result.json"),
        "result",
    )
    provenance = _mapping(
        _read_verified(root_path, "provenance.json"),
        "provenance",
    )
    trace = _mapping(
        _read_verified(root_path, "trace-index.json"),
        "trace index",
    )

    if result.get("schema") != "r1-e3-result-v1":
        raise EvidenceInvalid("result schema mismatch")
    if result.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("result manifest sha256 mismatch")
    if result.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid("result artifact root digest mismatch")

    result_sha = _read_digest(root_path, "result.sha256")
    provenance_sha = _read_digest(root_path, "provenance.sha256")

    if provenance.get("schema") != "r1-e3-provenance-v1":
        raise EvidenceInvalid("provenance schema mismatch")
    if provenance.get("scientific_head") != head:
        raise EvidenceInvalid("provenance scientific head mismatch")
    if provenance.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid("provenance artifact root digest mismatch")
    if provenance.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("provenance manifest sha256 mismatch")
    if provenance.get("result_sha256") != result_sha:
        raise EvidenceInvalid("provenance result sha256 mismatch")
    if provenance.get("registered_measurement") is not True:
        raise EvidenceInvalid("provenance measurement flag mismatch")
    _require_runtime_match(
        manifest,
        python_version=provenance.get("python"),
        numpy_version=provenance.get("numpy"),
    )

    if trace.get("schema") != "r1-e3-trace-index-v1":
        raise EvidenceInvalid("trace schema mismatch")
    if trace.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid("trace artifact root digest mismatch")
    if trace.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("trace manifest sha256 mismatch")
    if trace.get("result_sha256") != result_sha:
        raise EvidenceInvalid("trace result sha256 mismatch")
    if trace.get("provenance_sha256") != provenance_sha:
        raise EvidenceInvalid("trace provenance sha256 mismatch")

    validation = _validate_registered_measurement(
        result.get("measurement"),
        expected_artifact_digest=artifact_digest,
    )
    if provenance.get("registered_arm_count") != validation["registered_arm_count"]:
        raise EvidenceInvalid("provenance registered arm count mismatch")
    if provenance.get("outcome") != validation["outcome"]:
        raise EvidenceInvalid("provenance outcome mismatch")

    return {
        "valid": True,
        "prospective_only": False,
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
        "registered_arm_count": validation["registered_arm_count"],
        "outcome": validation["outcome"],
    }


def _validate_registered_measurement(
    value: object,
    *,
    expected_artifact_digest: str,
) -> dict[str, object]:
    measurement = _mapping(value, "registered measurement")
    if measurement.get("registered_measurement") is not True:
        raise EvidenceInvalid("registered measurement flag is missing")
    if measurement.get("manifest") != registered_manifest_payload(
        expected_artifact_digest
    ):
        raise EvidenceInvalid("registered measurement manifest mismatch")

    arms = measurement.get("arms")
    if not isinstance(arms, list) or len(arms) != ARM_COUNT:
        raise EvidenceInvalid("registered arm count mismatch")

    expected_keys = {
        (seed, int(architecture))
        for seed in REGISTERED_SEEDS
        for architecture in REGISTERED_ARCHITECTURES
    }
    seen: set[tuple[object, object]] = set()
    deltas: list[float] = []

    for arm in arms:
        arm_map = _mapping(arm, "registered arm")
        key = (arm_map.get("seed"), arm_map.get("architecture"))
        if key not in expected_keys or key in seen:
            raise EvidenceInvalid(
                "registered arm set is incomplete, duplicated, or unregistered"
            )
        seen.add(key)

        delta = _finite_float(
            arm_map.get("delta_macro_f1"),
            "arm delta_macro_f1",
        )
        result = _mapping(arm_map.get("result"), "registered arm result")
        if result.get("seed") != key[0] or result.get("architecture") != key[1]:
            raise EvidenceInvalid("registered arm result identity mismatch")
        if result.get("artifact_root_digest") != expected_artifact_digest:
            raise EvidenceInvalid("registered arm artifact root digest mismatch")
        result_delta = _finite_float(
            result.get("delta_macro_f1"),
            "benchmark delta_macro_f1",
        )
        if not math.isclose(
            delta,
            result_delta,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise EvidenceInvalid("registered arm delta mismatch")

        trajectory_b1 = result.get("trajectory_digest_b1")
        trajectory_b2 = result.get("trajectory_digest_b2")
        if (
            not _is_digest(trajectory_b1)
            or not _is_digest(trajectory_b2)
            or trajectory_b1 != trajectory_b2
        ):
            raise EvidenceInvalid(
                "B1/B2 trajectory digests must match exactly"
            )

        for field in (
            "reservoir_parameter_digest",
            "training_episode_digest",
            "evaluation_episode_digest",
        ):
            if not _is_digest(result.get(field)):
                raise EvidenceInvalid(f"missing or invalid {field}")

        _validate_readout(result.get("frame_only"), "frame_only")
        _validate_readout(
            result.get("reservoir_instantaneous"),
            "reservoir_instantaneous",
        )
        _validate_readout(
            result.get("reservoir_temporal_mean"),
            "reservoir_temporal_mean",
        )

        history = _mapping(
            result.get("history_destruction"),
            "history destruction",
        )
        if history.get("reset_before_bin") != 16:
            raise EvidenceInvalid(
                "history destruction reset must occur before bin 16"
            )
        if history.get("suffix_bins") != list(TEMPORAL_WINDOW_BINS):
            raise EvidenceInvalid(
                "history destruction suffix must match temporal window"
            )
        _validate_readout(
            history.get("instantaneous"),
            "history destruction instantaneous",
        )
        _validate_readout(
            history.get("temporal_mean"),
            "history destruction temporal_mean",
        )
        deltas.append(delta)

    if seen != expected_keys:
        raise EvidenceInvalid("registered arm identities do not match registration")

    median_delta = float(median(deltas))
    positive_count = sum(value > 0.0 for value in deltas)
    expected_outcome = classify_registered_outcome(deltas)

    measured_median = _finite_float(
        measurement.get("median_delta_macro_f1"),
        "median_delta_macro_f1",
    )
    if not math.isclose(
        measured_median,
        median_delta,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise EvidenceInvalid("registered median delta mismatch")
    if measurement.get("positive_arm_count") != positive_count:
        raise EvidenceInvalid("registered positive arm count mismatch")
    if measurement.get("outcome") != expected_outcome:
        raise EvidenceInvalid("registered outcome classification mismatch")

    return {
        "registered_arm_count": len(arms),
        "outcome": expected_outcome,
        "median_delta_macro_f1": median_delta,
        "positive_arm_count": positive_count,
    }


def _validate_readout(value: object, name: str) -> None:
    readout = _mapping(value, name)
    if not _is_digest(readout.get("coefficient_digest")):
        raise EvidenceInvalid(f"{name} coefficient digest is invalid")
    if not _is_digest(readout.get("prediction_digest")):
        raise EvidenceInvalid(f"{name} prediction digest is invalid")
    _validate_metrics(readout.get("metrics"), name)


def _validate_metrics(value: object, name: str) -> None:
    metrics = _mapping(value, f"{name} metrics")
    correct = metrics.get("correct")
    total = metrics.get("total")
    if type(correct) is not int or type(total) is not int:
        raise EvidenceInvalid(f"{name} metrics counts must be integers")
    if total <= 0 or not 0 <= correct <= total:
        raise EvidenceInvalid(f"{name} metrics counts are invalid")

    accuracy = _finite_float(metrics.get("accuracy"), f"{name} accuracy")
    macro_f1 = _finite_float(metrics.get("macro_f1"), f"{name} macro_f1")
    if not 0.0 <= accuracy <= 1.0 or not 0.0 <= macro_f1 <= 1.0:
        raise EvidenceInvalid(f"{name} accuracy/F1 must be in [0, 1]")

    vectors = (
        "precision_per_class",
        "recall_per_class",
        "f1_per_class",
    )
    for field in vectors:
        values = metrics.get(field)
        if not isinstance(values, list) or len(values) != 4:
            raise EvidenceInvalid(f"{name} {field} must contain four values")
        for index, item in enumerate(values):
            metric = _finite_float(
                item,
                f"{name} {field}[{index}]",
            )
            if not 0.0 <= metric <= 1.0:
                raise EvidenceInvalid(f"{name} {field} must be in [0, 1]")

    confusion = metrics.get("confusion_counts")
    if (
        not isinstance(confusion, list)
        or len(confusion) != 4
        or any(
            not isinstance(row, list) or len(row) != 4
            for row in confusion
        )
    ):
        raise EvidenceInvalid(f"{name} confusion matrix must be 4 x 4")

    confusion_total = 0
    diagonal = 0
    for row_index, row in enumerate(confusion):
        for col_index, item in enumerate(row):
            if type(item) is not int or item < 0:
                raise EvidenceInvalid(
                    f"{name} confusion counts must be non-negative integers"
                )
            confusion_total += item
            if row_index == col_index:
                diagonal += item

    if confusion_total != total:
        raise EvidenceInvalid(f"{name} confusion total mismatch")
    if diagonal != correct:
        raise EvidenceInvalid(f"{name} confusion correct-count mismatch")


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
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
    _validated_digest(expected, f"{name} sha256")
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
    return _validated_digest(
        path.read_text(encoding="utf-8").strip(),
        name,
    )


def _validated_head(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise EvidenceInvalid(
            "scientific head must be a 40-character lowercase git sha"
        )
    return value


def _validated_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise EvidenceInvalid(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceInvalid(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceInvalid(f"{name} must be finite")
    return result


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceInvalid(f"{name} must be an object")
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
        raise EvidenceInvalid(
            "measurement runtime does not match sealed manifest"
        )
