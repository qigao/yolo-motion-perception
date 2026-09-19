from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping

import numpy as np

from neural_state_machine.r1_e2_reservoir import E2Architecture
from neural_state_machine.r1_e3m_benchmark import (
    ARM_COUNT,
    LONG_DELAYS,
    REGISTERED_ARCHITECTURES,
    REGISTERED_SEEDS,
)
from neural_state_machine.r1_e3m_probe import DELAYS, RIDGE_REGULARIZATION


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


def registered_manifest_payload(
    artifact_root_digest: object,
) -> dict[str, object]:
    digest = _validated_digest(
        artifact_root_digest,
        "artifact_root_digest",
    )
    return {
        "phase": "R1-E3M",
        "artifact_root_digest": digest,
        "architectures": {
            "flat": int(E2Architecture.FLAT),
            "grouped4": int(E2Architecture.GROUPED4),
            "hierarchical2": int(E2Architecture.HIERARCHICAL2),
            "hierarchical4": int(E2Architecture.HIERARCHICAL4),
        },
        "seeds": list(REGISTERED_SEEDS),
        "input_size": 6,
        "neuron_budget": 256,
        "arm_count": ARM_COUNT,
        "delays": list(DELAYS),
        "long_delays": list(LONG_DELAYS),
        "ridge_regularization": RIDGE_REGULARIZATION,
        "window": {
            "duration_seconds": 2.0,
            "bin_count": 20,
            "min_present_bins": 16,
            "max_observation_age_seconds": 0.5,
            "strictly_causal": True,
        },
        "controls": {
            "h1_reset_before_bin": 16,
            "h1_suffix_bins": [16, 17, 18, 19],
            "h2_permuted_prefix_bins": list(range(16)),
            "h2_unchanged_suffix_bins": [16, 17, 18, 19],
            "control_refit_allowed": False,
        },
        "outcome": {
            "robust_positive_arm_min": 16,
            "median_long_delay_delta_gt": 0.0,
            "median_h1_long_delay_drop_gt": 0.0,
            "m_b_requires_positive_median_delta": True,
            "m_c_median_delta_lte": 0.0,
        },
    }


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
        raise FileExistsError(
            f"destination is not empty: {root_path}"
        )
    root_path.mkdir(parents=True, exist_ok=True)

    protocol = registered_manifest_payload(artifact_digest)
    manifest = {
        "schema": "r1-e3m-manifest-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": protocol,
    }
    manifest_sha = _write_pair(
        root_path,
        "manifest.json",
        manifest,
    )
    prospective = {
        "schema": "r1-e3m-prospective-provenance-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
        "python": manifest["python"],
        "numpy": manifest["numpy"],
        "registered_measurement": False,
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
        _read_verified(
            root_path,
            "prospective-provenance.json",
        ),
        "prospective provenance",
    )
    manifest_sha = _read_digest(
        root_path,
        "manifest.sha256",
    )

    if manifest.get("schema") != "r1-e3m-manifest-v1":
        raise EvidenceInvalid("manifest schema mismatch")
    head = _validated_head(manifest.get("scientific_head"))
    artifact_digest = _validated_digest(
        manifest.get("artifact_root_digest"),
        "artifact_root_digest",
    )
    if (
        manifest.get("protocol")
        != registered_manifest_payload(artifact_digest)
    ):
        raise EvidenceInvalid(
            "manifest protocol does not match R1-E3M registration"
        )

    if (
        prospective.get("schema")
        != "r1-e3m-prospective-provenance-v1"
    ):
        raise EvidenceInvalid(
            "prospective provenance schema mismatch"
        )
    if prospective.get("scientific_head") != head:
        raise EvidenceInvalid(
            "prospective scientific head mismatch"
        )
    if (
        prospective.get("artifact_root_digest")
        != artifact_digest
    ):
        raise EvidenceInvalid(
            "prospective artifact root digest mismatch"
        )
    if prospective.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid(
            "prospective manifest sha256 mismatch"
        )
    if prospective.get("registered_measurement") is not False:
        raise EvidenceInvalid(
            "prospective measurement flag mismatch"
        )
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
                "delays": list(DELAYS),
            }
        raise EvidenceInvalid(
            "registered result is missing"
        )

    raise EvidenceInvalid(
        "registered result verification is not implemented yet"
    )


def _write_pair(
    root: Path,
    name: str,
    payload: object,
) -> str:
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
        raise EvidenceInvalid(
            f"missing {name} or checksum"
        )
    data = data_path.read_bytes()
    expected = sha_path.read_text(
        encoding="utf-8"
    ).strip()
    _validated_digest(expected, f"{name} sha256")
    actual = hashlib.sha256(data).hexdigest()
    if expected != actual:
        raise EvidenceInvalid(
            f"{name} sha256 mismatch"
        )
    try:
        return json.loads(data)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise EvidenceInvalid(
            f"invalid {name}"
        ) from exc


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
        or any(
            char not in "0123456789abcdef"
            for char in value
        )
    ):
        raise EvidenceInvalid(
            "scientific head must be a 40-character lowercase git sha"
        )
    return value


def _validated_digest(
    value: object,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            char not in "0123456789abcdef"
            for char in value
        )
    ):
        raise EvidenceInvalid(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _mapping(
    value: object,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceInvalid(
            f"{name} must be an object"
        )
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
            "runtime does not match sealed manifest"
        )
