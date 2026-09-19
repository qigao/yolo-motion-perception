from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np

from .r1_e3m_protocol import (
    ARM_COUNT,
    ArmPrimaryResult,
    classify_outcome,
    registered_manifest_payload,
)


class EvidenceInvalid(RuntimeError):
    pass


def canonical_json_bytes(payload: object) -> bytes:
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


def prepare_prospective(
    root: Path | str,
    *,
    scientific_head: str,
    artifact_root_digest: object,
    train_sequence_count: int,
    eval_sequence_count: int,
) -> dict[str, object]:
    head = _validated_head(scientific_head)
    artifact_digest = _validated_digest(
        artifact_root_digest,
        "artifact_root_digest",
    )
    train_count = _positive_int(
        train_sequence_count,
        "train sequence count",
    )
    eval_count = _positive_int(
        eval_sequence_count,
        "eval sequence count",
    )

    root_path = Path(root)
    if root_path.exists() and any(root_path.iterdir()):
        raise FileExistsError(f"destination is not empty: {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

    protocol = registered_manifest_payload()
    manifest = {
        "schema": "r1-e3m-manifest-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "source_manifest_sha256": protocol["source_manifest_sha256"],
        "sequence_counts": {
            "train": train_count,
            "eval": eval_count,
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": protocol,
    }
    manifest_sha = _write_pair(root_path, "manifest.json", manifest)

    prospective = {
        "schema": "r1-e3m-prospective-provenance-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
        "python": manifest["python"],
        "numpy": manifest["numpy"],
        "registered_measurement": False,
        "human_approval_present": False,
    }
    _write_pair(
        root_path,
        "prospective-provenance.json",
        prospective,
    )
    return {
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
        "train_sequence_count": train_count,
        "eval_sequence_count": eval_count,
    }


def write_measurement(
    root: Path | str,
    *,
    manifest_sha256: str,
    scientific_head: str,
    artifact_root_digest: object,
    arms: object,
) -> dict[str, object]:
    root_path = Path(root)
    _assert_measurement_absent(root_path)

    head = _validated_head(scientific_head)
    artifact_digest = _validated_digest(
        artifact_root_digest,
        "artifact_root_digest",
    )
    supplied_manifest_sha = _validated_digest(
        manifest_sha256,
        "manifest_sha256",
    )

    manifest, actual_manifest_sha = _validated_prospective(
        root_path,
        require_current_runtime=True,
    )
    if supplied_manifest_sha != actual_manifest_sha:
        raise EvidenceInvalid("manifest sha256 does not match sealed manifest")
    if manifest.get("scientific_head") != head:
        raise EvidenceInvalid("scientific head does not match sealed manifest")
    if manifest.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid(
            "artifact root digest does not match sealed manifest"
        )

    _validated_approval(
        root_path,
        manifest=manifest,
        manifest_sha256=actual_manifest_sha,
    )
    normalized_arms, classification = _validated_arms(arms)

    measurement = {
        "registered_measurement": True,
        "arms": normalized_arms,
        "classification": asdict(classification),
    }
    result = {
        "schema": "r1-e3m-result-v1",
        "manifest_sha256": actual_manifest_sha,
        "artifact_root_digest": artifact_digest,
        "measurement": measurement,
    }
    result_sha = _write_pair(root_path, "result.json", result)

    provenance = {
        "schema": "r1-e3m-provenance-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": actual_manifest_sha,
        "result_sha256": result_sha,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "registered_measurement": True,
        "registered_arm_count": ARM_COUNT,
        "outcome": classification.outcome,
    }
    provenance_sha = _write_pair(
        root_path,
        "provenance.json",
        provenance,
    )
    trace = {
        "schema": "r1-e3m-trace-index-v1",
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": actual_manifest_sha,
        "result_sha256": result_sha,
        "provenance_sha256": provenance_sha,
    }
    trace_sha = _write_pair(root_path, "trace-index.json", trace)

    return {
        "manifest_sha256": actual_manifest_sha,
        "result_sha256": result_sha,
        "provenance_sha256": provenance_sha,
        "trace_index_sha256": trace_sha,
        "registered_arm_count": ARM_COUNT,
        "outcome": classification.outcome,
    }


def verify_evidence(
    root: Path | str,
    *,
    no_result_ok: bool = False,
) -> dict[str, object]:
    root_path = Path(root)
    manifest, manifest_sha = _validated_prospective(
        root_path,
        require_current_runtime=False,
    )
    head = _validated_head(manifest.get("scientific_head"))
    artifact_digest = _validated_digest(
        manifest.get("artifact_root_digest"),
        "artifact_root_digest",
    )
    counts = _validated_sequence_counts(manifest.get("sequence_counts"))

    result_path = root_path / "result.json"
    if not result_path.exists():
        if not no_result_ok:
            raise EvidenceInvalid("registered result is missing")
        return {
            "valid": True,
            "prospective_only": True,
            "scientific_head": head,
            "artifact_root_digest": artifact_digest,
            "manifest_sha256": manifest_sha,
            "train_sequence_count": counts["train"],
            "eval_sequence_count": counts["eval"],
            "approved": _approval_is_valid_if_present(
                root_path,
                manifest=manifest,
                manifest_sha256=manifest_sha,
            ),
        }

    _validated_approval(
        root_path,
        manifest=manifest,
        manifest_sha256=manifest_sha,
    )

    result = _mapping(
        _read_verified(root_path, "result.json"),
        "result",
    )
    if result.get("schema") != "r1-e3m-result-v1":
        raise EvidenceInvalid("result schema mismatch")
    if result.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("result manifest sha256 mismatch")
    if result.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid("result artifact root digest mismatch")

    measurement = _mapping(
        result.get("measurement"),
        "registered measurement",
    )
    if measurement.get("registered_measurement") is not True:
        raise EvidenceInvalid("registered measurement flag is missing")
    arms, classification = _validated_arms(measurement.get("arms"))
    stored_classification = _mapping(
        measurement.get("classification"),
        "classification",
    )
    expected_classification = asdict(classification)
    if stored_classification != expected_classification:
        raise EvidenceInvalid("registered outcome classification mismatch")

    result_sha = _read_digest(root_path, "result.sha256")
    provenance_sha = _read_digest(root_path, "provenance.sha256")
    provenance = _mapping(
        _read_verified(root_path, "provenance.json"),
        "provenance",
    )
    if provenance.get("schema") != "r1-e3m-provenance-v1":
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
    if provenance.get("registered_arm_count") != len(arms):
        raise EvidenceInvalid("provenance arm count mismatch")
    if provenance.get("outcome") != classification.outcome:
        raise EvidenceInvalid("provenance outcome mismatch")

    trace = _mapping(
        _read_verified(root_path, "trace-index.json"),
        "trace index",
    )
    if trace.get("schema") != "r1-e3m-trace-index-v1":
        raise EvidenceInvalid("trace schema mismatch")
    if trace.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid("trace artifact root digest mismatch")
    if trace.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("trace manifest sha256 mismatch")
    if trace.get("result_sha256") != result_sha:
        raise EvidenceInvalid("trace result sha256 mismatch")
    if trace.get("provenance_sha256") != provenance_sha:
        raise EvidenceInvalid("trace provenance sha256 mismatch")

    return {
        "valid": True,
        "prospective_only": False,
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
        "train_sequence_count": counts["train"],
        "eval_sequence_count": counts["eval"],
        "registered_arm_count": len(arms),
        "outcome": classification.outcome,
        "approved": True,
    }


def _validated_prospective(
    root: Path,
    *,
    require_current_runtime: bool,
) -> tuple[dict[str, object], str]:
    manifest = _mapping(
        _read_verified(root, "manifest.json"),
        "manifest",
    )
    if manifest.get("schema") != "r1-e3m-manifest-v1":
        raise EvidenceInvalid("manifest schema mismatch")
    head = _validated_head(manifest.get("scientific_head"))
    artifact_digest = _validated_digest(
        manifest.get("artifact_root_digest"),
        "artifact_root_digest",
    )
    protocol = registered_manifest_payload()
    if manifest.get("source_manifest_sha256") != protocol.get(
        "source_manifest_sha256"
    ):
        raise EvidenceInvalid("source manifest sha256 mismatch")
    if manifest.get("protocol") != protocol:
        raise EvidenceInvalid(
            "sealed protocol does not match R1-E3M registration"
        )
    _validated_sequence_counts(manifest.get("sequence_counts"))

    manifest_sha = _read_digest(root, "manifest.sha256")
    prospective = _mapping(
        _read_verified(root, "prospective-provenance.json"),
        "prospective provenance",
    )
    if prospective.get("schema") != "r1-e3m-prospective-provenance-v1":
        raise EvidenceInvalid("prospective provenance schema mismatch")
    if prospective.get("scientific_head") != head:
        raise EvidenceInvalid("prospective scientific head mismatch")
    if prospective.get("artifact_root_digest") != artifact_digest:
        raise EvidenceInvalid("prospective artifact root digest mismatch")
    if prospective.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid("prospective manifest sha256 mismatch")
    if prospective.get("registered_measurement") is not False:
        raise EvidenceInvalid("prospective measurement flag mismatch")
    if prospective.get("human_approval_present") is not False:
        raise EvidenceInvalid("prospective approval flag mismatch")

    if (
        manifest.get("python") != prospective.get("python")
        or manifest.get("numpy") != prospective.get("numpy")
    ):
        raise EvidenceInvalid("prospective runtime does not match manifest")
    if require_current_runtime and (
        manifest.get("python") != platform.python_version()
        or manifest.get("numpy") != np.__version__
    ):
        raise EvidenceInvalid(
            "measurement runtime does not match sealed manifest"
        )
    return manifest, manifest_sha


def _validated_approval(
    root: Path,
    *,
    manifest: Mapping[str, object],
    manifest_sha256: str,
) -> dict[str, object]:
    if not (root / "approval.json").is_file():
        raise EvidenceInvalid("explicit measurement approval is missing")
    approval = _mapping(
        _read_verified(root, "approval.json"),
        "approval",
    )
    if approval.get("schema") != "r1-e3m-measurement-approval-v1":
        raise EvidenceInvalid("approval schema mismatch")
    if approval.get("approved") is not True:
        raise EvidenceInvalid("approval flag must be true")
    if approval.get("manifest_sha256") != manifest_sha256:
        raise EvidenceInvalid("approval manifest sha256 mismatch")
    if approval.get("scientific_head") != manifest.get("scientific_head"):
        raise EvidenceInvalid("approval scientific head mismatch")
    if approval.get("artifact_root_digest") != manifest.get(
        "artifact_root_digest"
    ):
        raise EvidenceInvalid("approval artifact root digest mismatch")
    return approval


def _approval_is_valid_if_present(
    root: Path,
    *,
    manifest: Mapping[str, object],
    manifest_sha256: str,
) -> bool:
    if not (root / "approval.json").exists():
        return False
    _validated_approval(
        root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    return True


def _validated_arms(
    value: object,
) -> tuple[list[dict[str, object]], object]:
    if not isinstance(value, (list, tuple)) or len(value) != ARM_COUNT:
        raise EvidenceInvalid("registered measurement requires exactly 20 arms")

    normalized: list[dict[str, object]] = []
    primary: list[ArmPrimaryResult] = []
    for item in value:
        arm = _mapping(item, "registered arm")
        architecture = arm.get("architecture")
        seed = arm.get("seed")
        delta = _finite_float(arm.get("delta10"), "arm delta10")
        reset = _finite_float(arm.get("reset10"), "arm reset10")
        try:
            primary_arm = ArmPrimaryResult(
                architecture=architecture,
                seed=seed,
                delta10=delta,
                reset10=reset,
            )
        except ValueError as exc:
            raise EvidenceInvalid(str(exc)) from exc
        primary.append(primary_arm)
        normalized.append(
            {
                "architecture": primary_arm.architecture,
                "seed": primary_arm.seed,
                "delta10": primary_arm.delta10,
                "reset10": primary_arm.reset10,
            }
        )

    try:
        classification = classify_outcome(tuple(primary))
    except ValueError as exc:
        raise EvidenceInvalid(str(exc)) from exc
    return normalized, classification


def _validated_sequence_counts(value: object) -> dict[str, int]:
    counts = _mapping(value, "sequence counts")
    if set(counts) != {"train", "eval"}:
        raise EvidenceInvalid("sequence counts must contain train and eval")
    return {
        "train": _positive_int(counts.get("train"), "train sequence count"),
        "eval": _positive_int(counts.get("eval"), "eval sequence count"),
    }


def _assert_measurement_absent(root: Path) -> None:
    for name in (
        "result.json",
        "result.sha256",
        "provenance.json",
        "provenance.sha256",
        "trace-index.json",
        "trace-index.sha256",
    ):
        if (root / name).exists():
            raise FileExistsError("registered measurement evidence is write-once")


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
    expected = _validated_digest(
        sha_path.read_text(encoding="utf-8").strip(),
        f"{name} sha256",
    )
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
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


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceInvalid(f"{name} must be an object")
    return value


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


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise EvidenceInvalid(f"{name} must be a positive Python integer")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceInvalid(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceInvalid(f"{name} must be finite")
    return result
