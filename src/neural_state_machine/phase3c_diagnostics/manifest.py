from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .contracts import REGISTERED_DELAY_SUPPORT, registered_config, validate_registered_config

FROZEN_PHASE3C_EVIDENCE = "docs/experiments/phase-3c-anonymous-temporal-credit.json"
FROZEN_PHASE3C_SHA256 = "53b52fb6716daaceeb68b4e5c78f33333c0076b0d727462aa7265b0887798263"
FORMAL_COMMIT = "3de297cee2a94a7fc309531334f720f1b34467c9"
SCIENTIFIC_REFERENCE_SHA = "55a7ba59ac301975f3df996636f9a596e90bc730"
SPEC_COMMIT = "b510be9969be13b0ccc87b7323ceac9cb663d6e3"
PLAN_COMMIT = "097afe5ba1df380aef7472ae87d157c219c61f8b"

IMMUTABLE_INPUT_PATHS = (
    "docs/experiments/phase-2b-failure.json",
    "docs/experiments/phase-2c-diagnostics.json",
    "docs/experiments/phase-3a-action-value.json",
    "docs/experiments/phase-3a-credit-comparison.json",
    "docs/experiments/phase-3a-failure-analysis.md",
    "docs/experiments/phase-3a-failure-attribution.json",
    "docs/experiments/phase-3b-delayed-credit-report.md",
    "docs/experiments/phase-3b-delayed-credit.json",
    "docs/experiments/phase-3b-harness-audit.md",
    "docs/experiments/phase-3c-anonymous-temporal-credit-report.md",
    FROZEN_PHASE3C_EVIDENCE,
    "docs/experiments/phase-3c-formal-contract.json",
    "docs/superpowers/specs/2026-09-16-phase-3c-anonymous-temporal-credit-design.md",
    "docs/superpowers/specs/2026-09-16-phase-3c-failure-attribution-diagnostics-design.md",
    "docs/superpowers/plans/2026-09-16-phase-3c-failure-attribution-diagnostics.md",
    "scripts/benchmark_phase3c_anonymous_credit.py",
    "scripts/verify_phase3c_anonymous_credit.py",
    "src/neural_state_machine/action_value.py",
    "src/neural_state_machine/action_value_benchmark.py",
    "src/neural_state_machine/memory_task.py",
    "src/neural_state_machine/policy.py",
    "src/neural_state_machine/reward_learning.py",
    "src/neural_state_machine/phase3c_benchmark.py",
    "src/neural_state_machine/phase3c_controls.py",
    "src/neural_state_machine/phase3c_formal_contract.py",
    "src/neural_state_machine/phase3c_learners.py",
    "src/neural_state_machine/phase3c_schedule.py",
)


def canonical_json_bytes(payload: object) -> bytes:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(root, Path):
        raise ValueError("root must be a Path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("path containment requires a non-empty relative path")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("path containment violation") from exc
    return candidate


def sha256_file(path: Path) -> str:
    if not isinstance(path, Path):
        raise ValueError("path must be a Path")
    if path.is_symlink() or not path.is_file():
        raise ValueError("path must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_relative_files(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    paths = tuple(relative_paths)
    if len(paths) != len(set(paths)):
        raise ValueError("immutable input paths must not contain duplicates")
    return {relative: sha256_file(safe_path(root, relative)) for relative in paths}


def assert_hashes(root: Path, expected: Mapping[str, str]) -> None:
    current = hash_relative_files(root, expected.keys())
    if current != dict(expected):
        differing = sorted(key for key in set(current) | set(expected) if current.get(key) != expected.get(key))
        raise RuntimeError(f"immutable input hash mismatch: {differing[0] if differing else 'unknown'}")


def write_new_bytes(root: Path, relative: str, content: bytes) -> Path:
    path = safe_path(root, relative)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def write_new_json(root: Path, relative: str, payload: object) -> Path:
    return write_new_bytes(root, relative, canonical_json_bytes(payload))


def load_json_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("JSON path must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def environment_snapshot() -> dict[str, object]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    thread_keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "blas_configuration": stream.getvalue(),
        "thread_environment": {key: os.environ.get(key) for key in thread_keys},
    }


@dataclass(frozen=True, slots=True)
class InputManifest:
    implementation_sha: str
    scientific_reference_sha: str
    formal_commit: str
    frozen_evidence_sha256: str
    registered_configuration: dict[str, object]
    input_hashes: dict[str, str]

    def __post_init__(self) -> None:
        for name in ("implementation_sha", "scientific_reference_sha", "formal_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 40:
                raise ValueError(f"{name} must be a 40-character commit SHA")
        if self.frozen_evidence_sha256 != FROZEN_PHASE3C_SHA256:
            raise ValueError("frozen evidence SHA-256 differs from the registered value")
        validate_registered_config(self.registered_configuration)
        if set(self.input_hashes) != set(IMMUTABLE_INPUT_PATHS):
            raise ValueError("input hash coverage differs from the immutable input list")
        if any(not isinstance(value, str) or len(value) != 64 for value in self.input_hashes.values()):
            raise ValueError("input hashes must be SHA-256 hex digests")

    def to_payload(self) -> dict[str, object]:
        return {
            "implementation_sha": self.implementation_sha,
            "scientific_reference_sha": self.scientific_reference_sha,
            "formal_commit": self.formal_commit,
            "frozen_evidence_sha256": self.frozen_evidence_sha256,
            "registered_configuration": self.registered_configuration,
            "input_hashes": self.input_hashes,
        }


def build_input_manifest(root: Path, *, implementation_sha: str, scientific_reference_sha: str = SCIENTIFIC_REFERENCE_SHA) -> InputManifest:
    input_hashes = hash_relative_files(root, IMMUTABLE_INPUT_PATHS)
    if input_hashes[FROZEN_PHASE3C_EVIDENCE] != FROZEN_PHASE3C_SHA256:
        raise RuntimeError("frozen Phase 3C evidence bytes do not match the registered SHA-256")
    formal_payload = load_json_object(safe_path(root, "docs/experiments/phase-3c-formal-contract.json"))
    if formal_payload.get("commit") != FORMAL_COMMIT:
        raise RuntimeError("formal contract binding differs from the registered commit")
    evidence_payload = load_json_object(safe_path(root, FROZEN_PHASE3C_EVIDENCE))
    if evidence_payload.get("config") != registered_config():
        raise RuntimeError("frozen Phase 3C configuration differs from the registered configuration")
    if evidence_payload.get("delay_support") != list(REGISTERED_DELAY_SUPPORT):
        raise RuntimeError("frozen Phase 3C delay support differs from the registered support")
    return InputManifest(
        implementation_sha=implementation_sha,
        scientific_reference_sha=scientific_reference_sha,
        formal_commit=FORMAL_COMMIT,
        frozen_evidence_sha256=FROZEN_PHASE3C_SHA256,
        registered_configuration=registered_config(),
        input_hashes=input_hashes,
    )
