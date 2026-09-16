"""Canonical, fail-closed evidence sealing for Phase C4-A and C4-B."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import math
import platform
import subprocess
from pathlib import Path

from .memory_benchmark import AccuracyCount
from .phase_c4_benchmark import (
    PhaseC4Config,
    registered_c4_gate,
    secondary_evaluation_manifest,
)
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
_LINEAGES = {
    "behavior_action": ["seed", 0x33414354],
    "hidden_delay": ["seed", 0x3343444C],
    "reward_shuffle": ["seed", 0x33534846],
    "secondary_evaluation": ["seed", 0x33434641, 2, "evaluation_id"],
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
    "scripts/benchmark_phase_c4_delay_marginalized_credit.py",
    "scripts/verify_phase_c4_delay_marginalized_credit.py",
    "requirements/phase-c4.in",
    "requirements/phase-c4-python312.lock",
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
_C4A_SEED_KEYS = {
    "seed",
    "design_row_count",
    "normal_fit",
    "shuffled_fit",
    "post_training",
    "per_delay",
    "state_reset",
    "reset_per_delay",
    "shuffled_control",
    "shuffled_per_delay",
    "secondary_scores",
    "operator_passed",
}
_C4B_SEED_KEYS = {
    "seed",
    "post_training",
    "per_delay",
    "state_reset",
    "reset_per_delay",
    "shuffled_control",
    "shuffled_per_delay",
    "secondary_scores",
    "normal_drain_feedback_count",
    "shuffled_drain_feedback_count",
    "action_lineage_equal",
    "schedule_lineage_equal",
    "behavior_passed",
}
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
_PROVENANCE_KEYS = {
    "schema_version",
    "stage",
    "manifest_sha256",
    "result_sha256",
    "scientific_head",
    "formal_commit",
    "execution_head",
    "environment",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        raise RuntimeError("cannot resolve current Git head") from exc
    if not _is_commit(head):
        raise RuntimeError("Git head must be lowercase 40-hex")
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
        "lineages": dict(_LINEAGES),
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
    payload = _registered_manifest(Path(root), "c4a")
    return _write_new_json(Path(output_dir), "c4a-manifest.json", payload)


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _validate_manifest_schema(payload: dict[str, object], stage: str) -> None:
    if set(payload) != _COMMON_MANIFEST_KEYS:
        raise ValueError("manifest key set differs from the registered schema")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("manifest schema_version must be integer 1")
    if payload.get("stage") != stage:
        raise ValueError("manifest stage mismatch")
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
    if type(payload.get("public_drain_feedback_count")) is not int:
        raise ValueError("manifest public drain count must be an integer")
    if payload["public_drain_feedback_count"] != 5:
        raise ValueError("manifest public drain horizon mismatch")
    if type(payload.get("expected_scalar_design_rows")) is not int:
        raise ValueError("manifest scalar/design row count must be an integer")
    if payload["expected_scalar_design_rows"] != 2_005:
        raise ValueError("manifest scalar/design row count mismatch")
    if payload.get("thresholds") != _THRESHOLDS:
        raise ValueError("manifest thresholds mismatch")
    if payload.get("lineages") != _LINEAGES:
        raise ValueError("manifest RNG lineage mismatch")
    formal = payload.get("formal")
    if not isinstance(formal, dict) or formal.get("commit") != _FORMAL_HEAD:
        raise ValueError("manifest formal binding mismatch")
    if not _is_sha256(formal.get("sha256")):
        raise ValueError("manifest formal hash is invalid")
    if not _is_commit(payload.get("scientific_head")):
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
        if not isinstance(binding, dict) or set(binding) != {
            "result_sha256",
            "provenance_sha256",
            "operator_passed",
        }:
            raise ValueError("C4-B manifest requires the exact C4-A binding schema")
        if binding.get("operator_passed") is not True:
            raise ValueError("C4-B manifest requires operator_passed=true")
        if not _is_sha256(binding.get("result_sha256")) or not _is_sha256(
            binding.get("provenance_sha256")
        ):
            raise ValueError("C4-B C4-A binding hashes are invalid")


def _validate_manifest_against_repository(
    payload: dict[str, object],
    stage: str,
    repository_root: Path,
    *,
    require_environment: bool,
) -> None:
    _validate_manifest_schema(payload, stage)
    if payload["scientific_hashes"] != _hash_paths(repository_root, _SCIENCE_PATHS):
        raise RuntimeError("sealed C4 scientific bytes differ from the current repository")
    if payload["frozen_c3_hashes"] != dict(frozen_input_hashes(repository_root)):
        raise RuntimeError("sealed frozen C3 bytes differ from the current repository")
    if payload["formal"] != _formal_payload(repository_root):
        raise RuntimeError("sealed formal contract differs from the current repository")
    expected_evaluations = list(secondary_evaluation_manifest(PhaseC4Config()))
    if payload["evaluation_manifest"] != expected_evaluations:
        raise RuntimeError("sealed evaluation lineage differs from the registered lineage")
    if require_environment and payload["environment"] != _environment_snapshot():
        raise RuntimeError("sealed Python environment differs from the current environment")


def validate_measurement_manifest(
    repository_root: Path,
    manifest_path: Path,
    stage: str,
) -> dict[str, object]:
    repository_root = Path(repository_root)
    manifest_path = Path(manifest_path)
    payload = _load_canonical_object(manifest_path)
    _validate_manifest_against_repository(
        payload,
        stage,
        repository_root,
        require_environment=True,
    )
    if stage == "c4b":
        binding = payload["c4a_binding"]
        if not isinstance(binding, dict):
            raise RuntimeError("C4-B manifest is missing C4-A binding")
        result_path = manifest_path.with_name("c4a-result.json")
        provenance_path = manifest_path.with_name("c4a-provenance.json")
        if _sha256_file(result_path) != binding["result_sha256"]:
            raise RuntimeError("sealed C4-A result hash mismatch")
        if _sha256_file(provenance_path) != binding["provenance_sha256"]:
            raise RuntimeError("sealed C4-A provenance hash mismatch")
        verify_c4_stage(manifest_path.parent, "c4a", False)
    return payload


def prepare_c4b_manifest(root: Path, c4a_result_path: Path, output_dir: Path) -> Path:
    root = Path(root)
    result_path = Path(c4a_result_path)
    c4a_root = result_path.parent
    verify_c4_stage(c4a_root, "c4a", False)
    result = _load_canonical_object(result_path)
    provenance_path = result_path.with_name("c4a-provenance.json")
    if result.get("operator_passed") is not True:
        raise RuntimeError("C4-B requires verifier-confirmed operator_passed=true")
    c4a_manifest = _load_canonical_object(c4a_root / "c4a-manifest.json")
    _validate_manifest_against_repository(
        c4a_manifest,
        "c4a",
        root,
        require_environment=False,
    )
    payload = _registered_manifest(root, "c4b")
    payload["scientific_head"] = c4a_manifest["scientific_head"]
    payload["scientific_hashes"] = c4a_manifest["scientific_hashes"]
    payload["formal"] = c4a_manifest["formal"]
    payload["frozen_c3_hashes"] = c4a_manifest["frozen_c3_hashes"]
    payload["c4a_binding"] = {
        "result_sha256": _sha256_file(result_path),
        "provenance_sha256": _sha256_file(provenance_path),
        "operator_passed": True,
    }
    return _write_new_json(Path(output_dir), "c4b-manifest.json", payload)


def _count_payload(value: AccuracyCount) -> dict[str, int]:
    return {"correct": int(value.correct), "total": int(value.total)}


def _per_delay_payload(values: tuple[tuple[int, AccuracyCount], ...]) -> list[dict[str, int]]:
    return [
        {"delay": int(delay), "correct": int(count.correct), "total": int(count.total)}
        for delay, count in values
    ]


def _fit_payload(fit: object) -> dict[str, object]:
    singular = getattr(fit, "singular_values")
    return {
        "row_count": int(getattr(fit, "row_count")),
        "column_count": int(getattr(fit, "column_count")),
        "augmented_rank": int(getattr(fit, "augmented_rank")),
        "residual_norm": float(getattr(fit, "residual_norm")),
        "singular_values": [float(value) for value in singular],
        "penalty": float(getattr(fit, "penalty")),
    }


def _secondary_payload(scores: object) -> list[dict[str, object]]:
    return [
        {
            "evaluation_id": int(score.evaluation_id),
            "fixture_digest": str(score.fixture_digest),
            "overall": _count_payload(score.overall),
            "per_delay": _per_delay_payload(score.per_delay),
        }
        for score in scores
    ]


def build_c4a_result_payload(results: object) -> dict[str, object]:
    rows = tuple(results)
    if tuple(result.seed for result in rows) != _REGISTERED_SEEDS:
        raise ValueError("C4-A results must contain registered seeds in canonical order")
    serialized = [
        {
            "seed": int(result.seed),
            "design_row_count": int(result.design_row_count),
            "normal_fit": _fit_payload(result.normal_fit),
            "shuffled_fit": _fit_payload(result.shuffled_fit),
            "post_training": _count_payload(result.post_training),
            "per_delay": _per_delay_payload(result.per_delay),
            "state_reset": _count_payload(result.state_reset),
            "reset_per_delay": _per_delay_payload(result.reset_per_delay),
            "shuffled_control": _count_payload(result.shuffled_control),
            "shuffled_per_delay": _per_delay_payload(result.shuffled_per_delay),
            "secondary_scores": _secondary_payload(result.secondary_scores),
            "operator_passed": bool(result.operator_passed),
        }
        for result in rows
    ]
    operator_passed = all(row["operator_passed"] is True for row in serialized)
    return {
        "schema_version": 1,
        "stage": "c4a",
        "formal_valid": True,
        "protocol_valid": True,
        "operator_passed": operator_passed,
        "results": serialized,
    }


def build_c4b_result_payload(
    results: object,
    *,
    operator_passed: bool,
) -> dict[str, object]:
    if operator_passed is not True:
        raise ValueError("C4-B result construction requires operator_passed=true")
    rows = tuple(results)
    if tuple(result.seed for result in rows) != _REGISTERED_SEEDS:
        raise ValueError("C4-B results must contain registered seeds in canonical order")
    serialized = [
        {
            "seed": int(result.seed),
            "post_training": _count_payload(result.post_training),
            "per_delay": _per_delay_payload(result.per_delay),
            "state_reset": _count_payload(result.state_reset),
            "reset_per_delay": _per_delay_payload(result.reset_per_delay),
            "shuffled_control": _count_payload(result.shuffled_control),
            "shuffled_per_delay": _per_delay_payload(result.shuffled_per_delay),
            "secondary_scores": _secondary_payload(result.secondary_scores),
            "normal_drain_feedback_count": int(result.normal_drain_feedback_count),
            "shuffled_drain_feedback_count": int(result.shuffled_drain_feedback_count),
            "action_lineage_equal": bool(result.action_lineage_equal),
            "schedule_lineage_equal": bool(result.schedule_lineage_equal),
            "behavior_passed": bool(result.behavior_passed),
        }
        for result in rows
    ]
    behavior_passed = all(row["behavior_passed"] is True for row in serialized)
    return {
        "schema_version": 1,
        "stage": "c4b",
        "formal_valid": True,
        "protocol_valid": True,
        "operator_passed": True,
        "behavior_passed": behavior_passed,
        "all_passed": behavior_passed,
        "results": serialized,
    }


def write_measurement_bundle(
    repository_root: Path,
    manifest_path: Path,
    output_dir: Path,
    result_payload: dict[str, object],
) -> tuple[Path, Path, Path]:
    stage = result_payload.get("stage")
    if stage not in {"c4a", "c4b"}:
        raise ValueError("measurement result stage must be c4a or c4b")
    manifest = validate_measurement_manifest(repository_root, manifest_path, stage)
    expected_keys = set(_C4A_RESULT_KEYS if stage == "c4a" else _C4B_RESULT_KEYS)
    if set(result_payload) != expected_keys:
        raise ValueError("measurement result top-level schema mismatch")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination_manifest = output_dir / f"{stage}-manifest.json"
    source_bytes = Path(manifest_path).read_bytes()
    if destination_manifest.resolve() != Path(manifest_path).resolve():
        if destination_manifest.exists() or destination_manifest.is_symlink():
            raise FileExistsError("refusing to overwrite sealed manifest")
        destination_manifest.write_bytes(source_bytes)
    result_path = _write_new_json(output_dir, f"{stage}-result.json", result_payload)
    provenance_payload = {
        "schema_version": 1,
        "stage": stage,
        "manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "result_sha256": _sha256_file(result_path),
        "scientific_head": manifest["scientific_head"],
        "formal_commit": _FORMAL_HEAD,
        "execution_head": _git_head(Path(repository_root)),
        "environment": _environment_snapshot(),
    }
    provenance_path = _write_new_json(
        output_dir,
        f"{stage}-provenance.json",
        provenance_payload,
    )
    return destination_manifest, result_path, provenance_path


def _parse_count(payload: object, name: str) -> AccuracyCount:
    if not isinstance(payload, dict) or set(payload) != {"correct", "total"}:
        raise ValueError(f"{name} must contain exact correct/total keys")
    correct = payload.get("correct")
    total = payload.get("total")
    if type(correct) is not int or type(total) is not int or correct < 0 or total <= 0:
        raise ValueError(f"{name} counts must be positive integer totals and non-negative correct")
    if correct > total:
        raise ValueError(f"{name} correct count exceeds total")
    return AccuracyCount(correct, total)


def _parse_per_delay(payload: object, name: str) -> tuple[tuple[int, AccuracyCount], ...]:
    if not isinstance(payload, list) or len(payload) != 5:
        raise ValueError(f"{name} must contain five delay rows")
    rows: list[tuple[int, AccuracyCount]] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"delay", "correct", "total"}:
            raise ValueError(f"{name} delay row schema mismatch")
        delay = item.get("delay")
        if type(delay) is not int:
            raise ValueError(f"{name} delay must be an integer")
        rows.append((delay, _parse_count({"correct": item["correct"], "total": item["total"]}, name)))
    if tuple(delay for delay, _ in rows) != (1, 2, 3, 4, 5):
        raise ValueError(f"{name} delays must be canonical 1..5")
    return tuple(rows)


def _validate_secondary(payload: object, seed: int, manifest: dict[str, object]) -> None:
    if not isinstance(payload, list) or len(payload) != 8:
        raise ValueError("secondary_scores must contain exactly eight rows")
    evaluation = manifest["evaluation_manifest"]
    if not isinstance(evaluation, list):
        raise ValueError("manifest evaluation lineage is invalid")
    expected = {
        int(row["evaluation_id"]): str(row["digest"])
        for row in evaluation
        if isinstance(row, dict)
        and row.get("seed") == seed
        and row.get("evaluation_id") != -1
    }
    if set(expected) != set(range(8)):
        raise ValueError("manifest secondary evaluation lineage is incomplete for seed")
    seen: set[int] = set()
    for item in payload:
        if not isinstance(item, dict) or set(item) != {
            "evaluation_id",
            "fixture_digest",
            "overall",
            "per_delay",
        }:
            raise ValueError("secondary score schema mismatch")
        evaluation_id = item.get("evaluation_id")
        if type(evaluation_id) is not int or evaluation_id not in expected or evaluation_id in seen:
            raise ValueError("secondary evaluation_id is invalid or duplicated")
        if item.get("fixture_digest") != expected[evaluation_id]:
            raise ValueError("secondary fixture digest mismatch")
        _parse_count(item.get("overall"), "secondary overall")
        _parse_per_delay(item.get("per_delay"), "secondary per_delay")
        seen.add(evaluation_id)
    if seen != set(range(8)):
        raise ValueError("secondary evaluations are incomplete")


def _validate_fit(payload: object) -> None:
    keys = {
        "row_count",
        "column_count",
        "augmented_rank",
        "residual_norm",
        "singular_values",
        "penalty",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("fit diagnostic schema mismatch")
    for name in ("row_count", "column_count", "augmented_rank"):
        if type(payload.get(name)) is not int or int(payload[name]) <= 0:
            raise ValueError(f"fit {name} must be a positive integer")
    if payload["row_count"] != 2_005:
        raise ValueError("fit row_count must equal registered N+5")
    if payload["penalty"] != 1e-6:
        raise ValueError("fit penalty must equal registered 1e-6")
    residual = payload.get("residual_norm")
    if isinstance(residual, bool) or not isinstance(residual, (int, float)):
        raise ValueError("fit residual_norm must be finite")
    if not math.isfinite(float(residual)) or float(residual) < 0.0:
        raise ValueError("fit residual_norm must be finite and non-negative")
    singular = payload.get("singular_values")
    if not isinstance(singular, list) or len(singular) != payload["column_count"]:
        raise ValueError("fit singular_values length mismatch")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in singular
    ):
        raise ValueError("fit singular_values must be finite and non-negative")


def _recompute_c4a_result(result: dict[str, object], manifest: dict[str, object]) -> bool:
    if set(result) != set(_C4A_RESULT_KEYS):
        raise ValueError("C4-A result key set differs from the sealed schema")
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise ValueError("C4-A result schema_version must be integer 1")
    if result.get("stage") != "c4a":
        raise ValueError("C4-A result stage mismatch")
    if result.get("formal_valid") is not True or result.get("protocol_valid") is not True:
        raise ValueError("C4-A formal/protocol status must be true")
    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("C4-A result must contain exactly three seed rows")
    passes: list[bool] = []
    for expected_seed, row in zip(_REGISTERED_SEEDS, rows, strict=True):
        if not isinstance(row, dict) or set(row) != _C4A_SEED_KEYS:
            raise ValueError("C4-A seed result schema mismatch")
        if type(row.get("seed")) is not int or row["seed"] != expected_seed:
            raise ValueError("C4-A seed order mismatch")
        if type(row.get("design_row_count")) is not int or row["design_row_count"] != 2_005:
            raise ValueError("C4-A design row count mismatch")
        _validate_fit(row.get("normal_fit"))
        _validate_fit(row.get("shuffled_fit"))
        post = _parse_count(row.get("post_training"), "post_training")
        per_delay = _parse_per_delay(row.get("per_delay"), "per_delay")
        reset = _parse_count(row.get("state_reset"), "state_reset")
        reset_per_delay = _parse_per_delay(row.get("reset_per_delay"), "reset_per_delay")
        shuffled = _parse_count(row.get("shuffled_control"), "shuffled_control")
        _parse_per_delay(row.get("shuffled_per_delay"), "shuffled_per_delay")
        _validate_secondary(row.get("secondary_scores"), expected_seed, manifest)
        passed = registered_c4_gate(post, per_delay, reset, reset_per_delay, shuffled)
        if row.get("operator_passed") is not passed:
            raise ValueError("serialized C4-A seed gate disagrees with raw counts")
        passes.append(passed)
    recomputed = all(passes)
    if result.get("operator_passed") is not recomputed:
        raise ValueError("serialized C4-A operator gate disagrees with raw counts")
    return recomputed


def _recompute_c4b_result(result: dict[str, object], manifest: dict[str, object]) -> bool:
    if set(result) != set(_C4B_RESULT_KEYS):
        raise ValueError("C4-B result key set differs from the sealed schema")
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise ValueError("C4-B result schema_version must be integer 1")
    if result.get("stage") != "c4b":
        raise ValueError("C4-B result stage mismatch")
    if result.get("formal_valid") is not True or result.get("protocol_valid") is not True:
        raise ValueError("C4-B formal/protocol status must be true")
    binding = manifest.get("c4a_binding")
    if not isinstance(binding, dict) or binding.get("operator_passed") is not True:
        raise ValueError("C4-B manifest lacks verified C4-A pass")
    if result.get("operator_passed") is not True:
        raise ValueError("C4-B operator_passed must inherit verified C4-A pass")
    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("C4-B result must contain exactly three seed rows")
    passes: list[bool] = []
    for expected_seed, row in zip(_REGISTERED_SEEDS, rows, strict=True):
        if not isinstance(row, dict) or set(row) != _C4B_SEED_KEYS:
            raise ValueError("C4-B seed result schema mismatch")
        if type(row.get("seed")) is not int or row["seed"] != expected_seed:
            raise ValueError("C4-B seed order mismatch")
        for count_name in ("normal_drain_feedback_count", "shuffled_drain_feedback_count"):
            if type(row.get(count_name)) is not int or row[count_name] != 5:
                raise ValueError("C4-B drain count must be exactly five")
        if row.get("action_lineage_equal") is not True:
            raise ValueError("C4-B action lineage must remain equal")
        if row.get("schedule_lineage_equal") is not True:
            raise ValueError("C4-B schedule lineage must remain equal")
        post = _parse_count(row.get("post_training"), "post_training")
        per_delay = _parse_per_delay(row.get("per_delay"), "per_delay")
        reset = _parse_count(row.get("state_reset"), "state_reset")
        reset_per_delay = _parse_per_delay(row.get("reset_per_delay"), "reset_per_delay")
        shuffled = _parse_count(row.get("shuffled_control"), "shuffled_control")
        _parse_per_delay(row.get("shuffled_per_delay"), "shuffled_per_delay")
        _validate_secondary(row.get("secondary_scores"), expected_seed, manifest)
        passed = registered_c4_gate(post, per_delay, reset, reset_per_delay, shuffled)
        if row.get("behavior_passed") is not passed:
            raise ValueError("serialized C4-B seed gate disagrees with raw counts")
        passes.append(passed)
    recomputed = all(passes)
    if result.get("behavior_passed") is not recomputed:
        raise ValueError("serialized C4-B behavior gate disagrees with raw counts")
    expected_all = recomputed
    if result.get("all_passed") is not expected_all:
        raise ValueError("serialized C4-B all_passed disagrees with recomputed status")
    return recomputed


def verify_c4_stage(root: Path, stage: str, allow_missing_result: bool) -> None:
    if stage not in {"c4a", "c4b"}:
        raise ValueError("stage must be c4a or c4b")
    if type(allow_missing_result) is not bool:
        raise ValueError("allow_missing_result must be a boolean")
    evidence_root = Path(root)
    manifest_path = evidence_root / f"{stage}-manifest.json"
    manifest = _load_canonical_object(manifest_path)
    _validate_manifest_against_repository(
        manifest,
        stage,
        _repository_root(),
        require_environment=False,
    )
    result_path = evidence_root / f"{stage}-result.json"
    provenance_path = evidence_root / f"{stage}-provenance.json"
    if allow_missing_result:
        if result_path.exists() or result_path.is_symlink():
            raise RuntimeError("no-result verification found a result artifact")
        if provenance_path.exists() or provenance_path.is_symlink():
            raise RuntimeError("no-result verification found a provenance artifact")
        return
    result = _load_canonical_object(result_path)
    if stage == "c4a":
        _recompute_c4a_result(result, manifest)
    else:
        _recompute_c4b_result(result, manifest)
    provenance = _load_canonical_object(provenance_path)
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("provenance key set differs from the registered schema")
    if type(provenance.get("schema_version")) is not int or provenance["schema_version"] != 1:
        raise ValueError("provenance schema_version must be integer 1")
    if provenance.get("stage") != stage:
        raise ValueError("provenance stage mismatch")
    if provenance.get("manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("provenance manifest hash mismatch")
    if provenance.get("result_sha256") != _sha256_file(result_path):
        raise ValueError("provenance result hash mismatch")
    if provenance.get("scientific_head") != manifest["scientific_head"]:
        raise ValueError("provenance scientific head mismatch")
    if provenance.get("formal_commit") != _FORMAL_HEAD:
        raise ValueError("provenance formal commit mismatch")
    if not _is_commit(provenance.get("execution_head")):
        raise ValueError("provenance execution head is invalid")
    environment = provenance.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("provenance environment must be an object")
