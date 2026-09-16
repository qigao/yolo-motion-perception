"""Canonical, fail-closed evidence sealing for Phase C4-A and C4-B."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
from pathlib import Path

from .phase_c4_benchmark import PhaseC4Config, secondary_evaluation_manifest
from .phase_c4_controls import frozen_input_hashes, load_phase_c4_formal_contract

_FORMAL_HEAD = "8b2180ed24b6ff03db4b927ec29cdd9903b0ccac"
_REGISTERED_SEEDS = (7, 17, 29)
_DELAY_SUPPORT = (1, 3, 5)
_DELAY_PROBABILITIES = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
_THRESHOLDS = {
    "normal_min": 180,
    "per_delay_min": 34,
    "reset_exact": 100,
    "reset_per_delay_exact": 20,
    "shuffled_max_exclusive": 150,
}
_SCIENCE_PATHS = (
    "docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json",
    "src/neural_state_machine/phase_c4_delay_model.py",
    "src/neural_state_machine/phase_c4_batch_probe.py",
    "src/neural_state_machine/phase_c4_learner.py",
    "src/neural_state_machine/phase_c4_controls.py",
    "src/neural_state_machine/phase_c4_benchmark.py",
    "src/neural_state_machine/phase_c4_measurement.py",
    "src/neural_state_machine/phase_c4_evidence.py",
)
_C4A_RESULT_KEYS = (
    "schema_version",
    "stage",
    "formal_valid",
    "protocol_valid",
    "operator_passed",
    "results",
)
_C4B_RESULT_KEYS = (
    "schema_version",
    "stage",
    "formal_valid",
    "protocol_valid",
    "operator_passed",
    "behavior_passed",
    "all_passed",
    "results",
)
_COMMON_MANIFEST_KEYS = {
    "schema_version",
    "stage",
    "scientific_head",
    "scientific_hashes",
    "formal",
    "frozen_c3_hashes",
    "environment",
    "seeds",
    "delay_law",
    "config",
    "lineages",
    "public_drain_feedback_count",
    "expected_scalar_design_rows",
    "thresholds",
    "evaluation_manifest",
    "expected_result_keys",
    "c4a_binding",
}


def canonical_json_bytes(payload: object) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload is not canonical-JSON serializable") from exc
    return (text + "\n").encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing or unsafe: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_paths(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    resolved_root = root.resolve()
    result: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"scientific path must not be a symlink: {relative}")
        resolved = path.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"scientific path escapes repository: {relative}") from exc
        result[relative] = _sha256_file(path)
    return result


def _git_head(root: Path) -> str:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve current scientific Git head") from exc
    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
        raise RuntimeError("scientific Git head must be lowercase 40-hex")
    return head


def _environment_snapshot() -> dict[str, object]:
    package_names = (
        "numpy",
        "pytest",
        "ruff",
        "iniconfig",
        "packaging",
        "pluggy",
        "Pygments",
        "pip",
        "setuptools",
        "wheel",
    )
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _config_payload(config: PhaseC4Config) -> dict[str, object]:
    return {
        "hidden_size": config.hidden_size,
        "recurrent_radius": config.recurrent_radius,
        "step_size": config.step_size,
        "training_decisions": config.training_decisions,
        "evaluation_blocks": config.evaluation_blocks,
        "checkpoint_interval": config.checkpoint_interval,
        "ridge_penalty": config.ridge_penalty,
    }


def _formal_payload(root: Path) -> dict[str, object]:
    contract = load_phase_c4_formal_contract(root)
    if contract.commit != _FORMAL_HEAD:
        raise RuntimeError("C4 formal contract is not bound to the approved Lean head")
    path = root / "docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json"
    return {
        "commit": contract.commit,
        "theorems": list(contract.theorems),
        "sha256": _sha256_file(path),
    }


def _registered_manifest(root: Path, stage: str) -> dict[str, object]:
    config = PhaseC4Config()
    result_keys = _C4A_RESULT_KEYS if stage == "c4a" else _C4B_RESULT_KEYS
    return {
        "schema_version": 1,
        "stage": stage,
        "scientific_head": _git_head(root),
        "scientific_hashes": _hash_paths(root, _SCIENCE_PATHS),
        "formal": _formal_payload(root),
        "frozen_c3_hashes": dict(frozen_input_hashes(root)),
        "environment": _environment_snapshot(),
        "seeds": list(_REGISTERED_SEEDS),
        "delay_law": {
            "support": list(_DELAY_SUPPORT),
            "probabilities": list(_DELAY_PROBABILITIES),
        },
        "config": _config_payload(config),
        "lineages": {
            "behavior_action": ["seed", 0x33414354],
            "hidden_delay": ["seed", 0x3343444C],
            "reward_shuffle": ["seed", 0x33534846],
            "secondary_evaluation": ["seed", 0x33434641, 2, "evaluation_id"],
        },
        "public_drain_feedback_count": 5,
        "expected_scalar_design_rows": 2_005,
        "thresholds": dict(_THRESHOLDS),
        "evaluation_manifest": list(secondary_evaluation_manifest(config)),
        "expected_result_keys": list(result_keys),
        "c4a_binding": None,
    }


def _write_new_json(output_dir: Path, name: str, payload: object) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing evidence: {name}")
    path.write_bytes(canonical_json_bytes(payload))
    return path


def prepare_c4a_manifest(root: Path, output_dir: Path) -> Path:
    root = Path(root)
    output_dir = Path(output_dir)
    payload = _registered_manifest(root, "c4a")
    return _write_new_json(output_dir, "c4a-manifest.json", payload)


def _load_canonical_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence path must be a regular file: {path.name}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"evidence JSON must be an object: {path.name}")
    if raw != canonical_json_bytes(payload):
        raise ValueError(f"evidence JSON is not canonical: {path.name}")
    return payload


def prepare_c4b_manifest(root: Path, c4a_result_path: Path, output_dir: Path) -> Path:
    root = Path(root)
    result_path = Path(c4a_result_path)
    result = _load_canonical_object(result_path)
    provenance_path = result_path.with_name("c4a-provenance.json")
    provenance = _load_canonical_object(provenance_path)
    if (
        result.get("schema_version") != 1
        or result.get("stage") != "c4a"
        or result.get("operator_passed") is not True
    ):
        raise RuntimeError("C4-B requires a verified passing C4-A result")
    if provenance.get("stage") != "c4a":
        raise RuntimeError("C4-B requires C4-A provenance")
    payload = _registered_manifest(root, "c4b")
    payload["c4a_binding"] = {
        "result_sha256": _sha256_file(result_path),
        "provenance_sha256": _sha256_file(provenance_path),
        "operator_passed": True,
    }
    return _write_new_json(Path(output_dir), "c4b-manifest.json", payload)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _validate_manifest(payload: dict[str, object], stage: str) -> None:
    if set(payload) != _COMMON_MANIFEST_KEYS:
        raise ValueError("manifest key set differs from the registered schema")
    if payload.get("schema_version") != 1 or payload.get("stage") != stage:
        raise ValueError("manifest schema or stage mismatch")
    if payload.get("seeds") != list(_REGISTERED_SEEDS):
        raise ValueError("manifest seeds mismatch")
    if payload.get("delay_law") != {
        "support": list(_DELAY_SUPPORT),
        "probabilities": list(_DELAY_PROBABILITIES),
    }:
        raise ValueError("manifest delay law mismatch")
    config = payload.get("config")
    if not isinstance(config, dict) or config != _config_payload(PhaseC4Config()):
        raise ValueError("manifest C4 configuration mismatch")
    if payload.get("public_drain_feedback_count") != 5:
        raise ValueError("manifest public drain horizon mismatch")
    if payload.get("expected_scalar_design_rows") != 2_005:
        raise ValueError("manifest scalar/design row count mismatch")
    if payload.get("thresholds") != _THRESHOLDS:
        raise ValueError("manifest thresholds mismatch")
    formal = payload.get("formal")
    if not isinstance(formal, dict) or formal.get("commit") != _FORMAL_HEAD:
        raise ValueError("manifest formal binding mismatch")
    if not _is_sha256(formal.get("sha256")):
        raise ValueError("manifest formal hash is invalid")
    head = payload.get("scientific_head")
    if not isinstance(head, str) or len(head) != 40:
        raise ValueError("manifest scientific head is invalid")
    scientific_hashes = payload.get("scientific_hashes")
    frozen_hashes = payload.get("frozen_c3_hashes")
    if not isinstance(scientific_hashes, dict) or set(scientific_hashes) != set(_SCIENCE_PATHS):
        raise ValueError("manifest scientific hash coverage mismatch")
    if not isinstance(frozen_hashes, dict) or not frozen_hashes:
        raise ValueError("manifest frozen C3 hashes are missing")
    if any(not _is_sha256(value) for value in scientific_hashes.values()):
        raise ValueError("manifest scientific hash is invalid")
    if any(not _is_sha256(value) for value in frozen_hashes.values()):
        raise ValueError("manifest frozen C3 hash is invalid")
    evaluation = payload.get("evaluation_manifest")
    if not isinstance(evaluation, list) or len(evaluation) != 27:
        raise ValueError("manifest evaluation lineage is incomplete")
    expected_keys = list(_C4A_RESULT_KEYS if stage == "c4a" else _C4B_RESULT_KEYS)
    if payload.get("expected_result_keys") != expected_keys:
        raise ValueError("manifest expected result schema mismatch")
    binding = payload.get("c4a_binding")
    if stage == "c4a":
        if binding is not None:
            raise ValueError("C4-A manifest must not contain a C4-A result binding")
    else:
        if not isinstance(binding, dict) or binding.get("operator_passed") is not True:
            raise ValueError("C4-B manifest requires a passing C4-A binding")
        if not _is_sha256(binding.get("result_sha256")) or not _is_sha256(
            binding.get("provenance_sha256")
        ):
            raise ValueError("C4-B C4-A binding hashes are invalid")


def verify_c4_stage(root: Path, stage: str, allow_missing_result: bool) -> None:
    if stage not in {"c4a", "c4b"}:
        raise ValueError("stage must be c4a or c4b")
    if type(allow_missing_result) is not bool:
        raise ValueError("allow_missing_result must be a boolean")
    evidence_root = Path(root)
    manifest_path = evidence_root / f"{stage}-manifest.json"
    manifest = _load_canonical_object(manifest_path)
    _validate_manifest(manifest, stage)

    result_path = evidence_root / f"{stage}-result.json"
    if allow_missing_result:
        if result_path.exists() or result_path.is_symlink():
            raise RuntimeError("no-result verification found a result artifact")
        return
    result = _load_canonical_object(result_path)
    expected_keys = set(_C4A_RESULT_KEYS if stage == "c4a" else _C4B_RESULT_KEYS)
    if set(result) != expected_keys:
        raise ValueError("result key set differs from the sealed manifest")
    if result.get("schema_version") != 1 or result.get("stage") != stage:
        raise ValueError("result schema or stage mismatch")
