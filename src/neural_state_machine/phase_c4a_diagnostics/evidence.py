"""Canonical, fail-closed evidence sealing for C4-A failure attribution."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path

from ..phase_c4_benchmark import PhaseC4Config, secondary_evaluation_manifest
from .model import DiagnosticConfig, FrozenC4AIdentity, sha256_file

_STAGE = "c4a-failure-attribution"
_MANIFEST_KEYS = {
    "schema_version",
    "stage",
    "scientific_head",
    "scientific_hashes",
    "frozen_c4a",
    "formal_head",
    "environment",
    "registered_seeds",
    "registered_scores",
    "diagnostic_config",
    "permutation_lineage",
    "secondary_evaluation_manifest",
    "expected_result_keys",
}
_RESULT_KEYS = (
    "schema_version",
    "stage",
    "integrity_valid",
    "geometry",
    "registered_attribution",
    "permutation_rows",
    "permutation_summaries",
    "mechanism_table",
)
_MECHANISMS = (
    "design_geometry",
    "task_relevant_target_projection",
    "bias_action_shortcut",
    "local_block_structure",
    "evaluation_fixture_sensitivity",
    "unresolved_multiple_mechanisms",
)
_REGISTERED_SCORES = (
    {"seed": 7, "normal": 200, "reset": 100, "shuffled": 147},
    {"seed": 17, "normal": 200, "reset": 100, "shuffled": 170},
    {"seed": 29, "normal": 200, "reset": 100, "shuffled": 160},
)
_SCIENCE_PATHS = (
    "docs/superpowers/specs/2026-09-16-phase-c4a-failure-attribution-diagnostics-design.md",
    "scripts/diagnose_phase_c4a_failure.py",
    "scripts/verify_phase_c4a_failure_diagnostics.py",
    "requirements/phase-c4a-diagnostics.in",
    "requirements/phase-c4a-diagnostics-python312.lock",
    "src/neural_state_machine/action_value.py",
    "src/neural_state_machine/action_value_benchmark.py",
    "src/neural_state_machine/memory_benchmark.py",
    "src/neural_state_machine/memory_task.py",
    "src/neural_state_machine/policy.py",
    "src/neural_state_machine/reward_learning.py",
    "src/neural_state_machine/phase3c_schedule.py",
    "src/neural_state_machine/phase_c4_batch_probe.py",
    "src/neural_state_machine/phase_c4_benchmark.py",
    "src/neural_state_machine/phase_c4_delay_model.py",
    "src/neural_state_machine/phase_c4_measurement.py",
    "src/neural_state_machine/phase_c4a_diagnostics/model.py",
    "src/neural_state_machine/phase_c4a_diagnostics/integrity.py",
    "src/neural_state_machine/phase_c4a_diagnostics/geometry.py",
    "src/neural_state_machine/phase_c4a_diagnostics/attribution.py",
    "src/neural_state_machine/phase_c4a_diagnostics/decomposition.py",
    "src/neural_state_machine/phase_c4a_diagnostics/permutations.py",
    "src/neural_state_machine/phase_c4a_diagnostics/report.py",
    "src/neural_state_machine/phase_c4a_diagnostics/evidence.py",
)
_PROVENANCE_KEYS = {
    "schema_version",
    "stage",
    "manifest_sha256",
    "result_sha256",
    "scientific_head",
    "formal_head",
    "execution_head",
    "environment",
}
_TRACE_INDEX_KEYS = {"schema_version", "stage", "files"}
_TRACE_ROW_KEYS = {"filename", "size", "sha256", "semantic_role"}
_CANONICAL_FILES = {"manifest.json", "result.json", "provenance.json", "trace-index.json"}
_FORBIDDEN_RESULT_KEYS = {
    "outcome",
    "operator_passed",
    "behavior_passed",
    "all_passed",
    "c4b",
}


def _canonical_json_bytes(payload: object) -> bytes:
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


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def _hash_science(root: Path) -> dict[str, str]:
    resolved_root = root.resolve()
    hashes: dict[str, str] = {}
    for relative in _SCIENCE_PATHS:
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"scientific path must not be a symlink: {relative}")
        resolved = path.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"scientific path escapes repository: {relative}") from exc
        hashes[relative] = sha256_file(path)
    return hashes


def _frozen_payload(identity: FrozenC4AIdentity) -> dict[str, str]:
    return {
        "scientific_head": identity.scientific_head,
        "manifest_sha256": identity.manifest_sha256,
        "result_sha256": identity.result_sha256,
        "provenance_sha256": identity.provenance_sha256,
    }


def _diagnostic_payload(config: DiagnosticConfig) -> dict[str, object]:
    return {
        "permutation_replicates": config.permutation_replicates,
        "modes": list(config.modes),
        "near_zero_margin": config.near_zero_margin,
    }


def _manifest_payload(root: Path) -> dict[str, object]:
    identity = FrozenC4AIdentity.registered()
    diagnostic = DiagnosticConfig.registered()
    return {
        "schema_version": 1,
        "stage": _STAGE,
        "scientific_head": _git_head(root),
        "scientific_hashes": _hash_science(root),
        "frozen_c4a": _frozen_payload(identity),
        "formal_head": identity.formal_head,
        "environment": _environment_snapshot(),
        "registered_seeds": list(identity.seeds),
        "registered_scores": [dict(row) for row in _REGISTERED_SCORES],
        "diagnostic_config": _diagnostic_payload(diagnostic),
        "permutation_lineage": diagnostic.permutation_lineage,
        "secondary_evaluation_manifest": list(secondary_evaluation_manifest(PhaseC4Config())),
        "expected_result_keys": list(_RESULT_KEYS),
    }


def _write_new_json(output_dir: Path, name: str, payload: object) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing evidence: {name}")
    path.write_bytes(_canonical_json_bytes(payload))
    return path


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
    if raw != _canonical_json_bytes(payload):
        raise ValueError(f"evidence JSON is not canonical: {path.name}")
    return payload


def _validate_secondary_manifest(value: object) -> None:
    if not isinstance(value, list) or len(value) != 27:
        raise ValueError("secondary evaluation manifest must contain exactly 27 rows")
    seen: set[tuple[int, int]] = set()
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("secondary evaluation row must be an object")
        if set(row) != {"seed", "evaluation_id", "kind", "fixture_count", "digest"}:
            raise ValueError("secondary evaluation row schema mismatch")
        seed = row["seed"]
        evaluation_id = row["evaluation_id"]
        if seed not in (7, 17, 29) or evaluation_id not in range(-1, 8):
            raise ValueError("secondary evaluation lineage is invalid")
        if (seed, evaluation_id) in seen:
            raise ValueError("secondary evaluation lineage contains duplicates")
        seen.add((seed, evaluation_id))
        if row["kind"] != ("original" if evaluation_id == -1 else "additional"):
            raise ValueError("secondary evaluation kind mismatch")
        if type(row["fixture_count"]) is not int or row["fixture_count"] <= 0:
            raise ValueError("secondary evaluation fixture count is invalid")
        if not _is_sha256(row["digest"]):
            raise ValueError("secondary evaluation digest is invalid")


def _validate_manifest_static(payload: dict[str, object]) -> None:
    if set(payload) != _MANIFEST_KEYS:
        raise ValueError("manifest key set differs from the registered schema")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("manifest schema_version must be integer 1")
    if payload.get("stage") != _STAGE:
        raise ValueError("manifest stage mismatch")
    if not _is_commit(payload.get("scientific_head")):
        raise ValueError("manifest scientific head is invalid")
    scientific_hashes = payload.get("scientific_hashes")
    if not isinstance(scientific_hashes, dict) or set(scientific_hashes) != set(_SCIENCE_PATHS):
        raise ValueError("manifest scientific hash coverage mismatch")
    if any(not _is_sha256(value) for value in scientific_hashes.values()):
        raise ValueError("manifest scientific hash is invalid")
    identity = FrozenC4AIdentity.registered()
    if payload.get("frozen_c4a") != _frozen_payload(identity):
        raise ValueError("manifest frozen C4-A binding mismatch")
    if payload.get("formal_head") != identity.formal_head:
        raise ValueError("manifest formal head mismatch")
    if payload.get("registered_seeds") != list(identity.seeds):
        raise ValueError("manifest registered seeds mismatch")
    if payload.get("registered_scores") != [dict(row) for row in _REGISTERED_SCORES]:
        raise ValueError("manifest registered scores mismatch")
    diagnostic = DiagnosticConfig.registered()
    if payload.get("diagnostic_config") != _diagnostic_payload(diagnostic):
        raise ValueError("manifest diagnostic configuration mismatch")
    if payload.get("permutation_lineage") != diagnostic.permutation_lineage:
        raise ValueError("manifest permutation lineage mismatch")
    if payload.get("expected_result_keys") != list(_RESULT_KEYS):
        raise ValueError("manifest expected result schema mismatch")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or environment != _environment_snapshot():
        raise ValueError("manifest environment mismatch")
    _validate_secondary_manifest(payload.get("secondary_evaluation_manifest"))


def prepare_attribution_manifest(repository_root: Path, output_dir: Path) -> Path:
    """Write the registered prospective manifest and no result files."""
    root = Path(repository_root)
    payload = _manifest_payload(root)
    return _write_new_json(Path(output_dir), "manifest.json", payload)


def validate_attribution_manifest(
    repository_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Validate canonical schema and recompute all current-repository bindings."""
    root = Path(repository_root)
    payload = _load_canonical_object(Path(manifest_path))
    _validate_manifest_static(payload)
    expected = _manifest_payload(root)
    for key in _MANIFEST_KEYS:
        if payload[key] != expected[key]:
            if key == "environment":
                raise ValueError("manifest environment mismatch")
            raise ValueError(f"manifest {key} mismatch")
    return payload


def _iter_mapping_keys(value: object):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _iter_mapping_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_mapping_keys(nested)


def _validate_mechanism_table(value: object, integrity_valid: bool) -> None:
    if not isinstance(value, list) or len(value) != len(_MECHANISMS):
        raise ValueError("result mechanism table must contain exactly six rows")
    for expected, row in zip(_MECHANISMS, value, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "mechanism",
            "integrity_valid",
            "observed_association",
            "matched_counterfactual_evidence",
            "interpretation",
        }:
            raise ValueError("result mechanism row schema mismatch")
        if row["mechanism"] != expected or row["integrity_valid"] is not integrity_valid:
            raise ValueError("result mechanism row binding mismatch")
        for key in (
            "observed_association",
            "matched_counterfactual_evidence",
            "interpretation",
        ):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("result mechanism text must be non-empty")


def _validate_result_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != set(_RESULT_KEYS):
        raise ValueError("result key set differs from the registered raw schema")
    if any(key in _FORBIDDEN_RESULT_KEYS for key in _iter_mapping_keys(payload)):
        raise ValueError("result contains a forbidden outcome/pass/C4-B field")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("result schema_version must be integer 1")
    if payload.get("stage") != _STAGE:
        raise ValueError("result stage mismatch")
    if type(payload.get("integrity_valid")) is not bool:
        raise ValueError("result integrity_valid must be a boolean")
    for key in (
        "geometry",
        "registered_attribution",
        "permutation_rows",
        "permutation_summaries",
    ):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"result {key} must be a list")
    _validate_mechanism_table(payload["mechanism_table"], payload["integrity_valid"])
    _canonical_json_bytes(payload)
    return payload


def _trace_rows(root: Path, trace_index: Mapping[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filename in sorted(trace_index):
        role = trace_index[filename]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in _CANONICAL_FILES
        ):
            raise ValueError("trace filename must be a plain non-canonical basename")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("trace semantic role must be a non-empty string")
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"trace must be an existing regular file: {filename}")
        raw = path.read_bytes()
        rows.append(
            {
                "filename": filename,
                "size": len(raw),
                "sha256": _sha256_bytes(raw),
                "semantic_role": role.strip(),
            }
        )
    return rows


def write_attribution_bundle(
    repository_root: Path,
    manifest_path: Path,
    output_dir: Path,
    result_payload: object,
    trace_index: Mapping[str, str],
) -> None:
    """Write result/provenance/trace-index without refitting or interpretation."""
    root = Path(output_dir)
    manifest = Path(manifest_path)
    manifest_payload = validate_attribution_manifest(Path(repository_root), manifest)
    result = _validate_result_payload(result_payload)
    if not isinstance(trace_index, Mapping):
        raise ValueError("trace_index must be a mapping")

    for name in ("result.json", "provenance.json", "trace-index.json"):
        path = root / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing evidence: {name}")

    trace_rows = _trace_rows(root, trace_index)
    existing_noncanonical = {
        path.name
        for path in root.iterdir()
        if path.name not in _CANONICAL_FILES
    }
    indexed = {row["filename"] for row in trace_rows}
    if existing_noncanonical != indexed:
        raise ValueError("output directory contains unindexed trace files")

    result_raw = _canonical_json_bytes(result)
    manifest_raw = manifest.read_bytes()
    provenance = {
        "schema_version": 1,
        "stage": _STAGE,
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "result_sha256": _sha256_bytes(result_raw),
        "scientific_head": manifest_payload["scientific_head"],
        "formal_head": manifest_payload["formal_head"],
        "execution_head": _git_head(Path(repository_root)),
        "environment": _environment_snapshot(),
    }
    trace_payload = {
        "schema_version": 1,
        "stage": _STAGE,
        "files": trace_rows,
    }
    provenance_raw = _canonical_json_bytes(provenance)
    trace_raw = _canonical_json_bytes(trace_payload)

    (root / "result.json").write_bytes(result_raw)
    (root / "provenance.json").write_bytes(provenance_raw)
    (root / "trace-index.json").write_bytes(trace_raw)


def _validate_provenance(
    payload: dict[str, object],
    manifest_raw: bytes,
    result_raw: bytes,
    manifest_payload: dict[str, object],
) -> None:
    if set(payload) != _PROVENANCE_KEYS:
        raise ValueError("provenance key set mismatch")
    if payload.get("schema_version") != 1 or payload.get("stage") != _STAGE:
        raise ValueError("provenance schema/stage mismatch")
    if payload.get("manifest_sha256") != _sha256_bytes(manifest_raw):
        raise ValueError("provenance manifest hash mismatch")
    if payload.get("result_sha256") != _sha256_bytes(result_raw):
        raise ValueError("provenance result hash mismatch")
    if payload.get("scientific_head") != manifest_payload.get("scientific_head"):
        raise ValueError("provenance scientific head mismatch")
    if payload.get("formal_head") != manifest_payload.get("formal_head"):
        raise ValueError("provenance formal head mismatch")
    if not _is_commit(payload.get("execution_head")):
        raise ValueError("provenance execution head is invalid")
    if payload.get("environment") != _environment_snapshot():
        raise ValueError("provenance environment mismatch")


def _validate_trace_index(root: Path, payload: dict[str, object]) -> set[str]:
    if set(payload) != _TRACE_INDEX_KEYS:
        raise ValueError("trace index key set mismatch")
    if payload.get("schema_version") != 1 or payload.get("stage") != _STAGE:
        raise ValueError("trace index schema/stage mismatch")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("trace index files must be a list")
    indexed: set[str] = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != _TRACE_ROW_KEYS:
            raise ValueError("trace index row schema mismatch")
        filename = row["filename"]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in _CANONICAL_FILES
            or filename in indexed
        ):
            raise ValueError("trace index filename is invalid")
        indexed.add(filename)
        if type(row["size"]) is not int or row["size"] < 0 or not _is_sha256(row["sha256"]):
            raise ValueError("trace index size/hash is invalid")
        if not isinstance(row["semantic_role"], str) or not row["semantic_role"].strip():
            raise ValueError("trace index semantic role is invalid")
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"trace file is missing or unsafe: {filename}")
        raw = path.read_bytes()
        if len(raw) != row["size"] or _sha256_bytes(raw) != row["sha256"]:
            raise ValueError(f"trace file hash/size mismatch: {filename}")
    return indexed


def verify_attribution_stage(root: Path, allow_missing_result: bool) -> None:
    """Strictly verify a prospective manifest or a complete raw evidence bundle."""
    if type(allow_missing_result) is not bool:
        raise ValueError("allow_missing_result must be a boolean")
    stage_root = Path(root)
    manifest_path = stage_root / "manifest.json"
    manifest_payload = _load_canonical_object(manifest_path)
    _validate_manifest_static(manifest_payload)
    manifest_raw = manifest_path.read_bytes()

    result_path = stage_root / "result.json"
    provenance_path = stage_root / "provenance.json"
    trace_index_path = stage_root / "trace-index.json"
    present = tuple(path.exists() or path.is_symlink() for path in (result_path, provenance_path, trace_index_path))
    if any(present) and not all(present):
        raise ValueError("partial attribution result bundle")
    if not any(present):
        extras = {path.name for path in stage_root.iterdir() if path.name != "manifest.json"}
        if extras:
            raise ValueError("prospective stage contains unindexed files")
        if allow_missing_result:
            return
        raise ValueError("result bundle is missing")

    result_payload = _load_canonical_object(result_path)
    _validate_result_payload(result_payload)
    provenance_payload = _load_canonical_object(provenance_path)
    trace_payload = _load_canonical_object(trace_index_path)
    _validate_provenance(
        provenance_payload,
        manifest_raw,
        result_path.read_bytes(),
        manifest_payload,
    )
    indexed = _validate_trace_index(stage_root, trace_payload)
    existing_noncanonical = {
        path.name
        for path in stage_root.iterdir()
        if path.name not in _CANONICAL_FILES
    }
    if existing_noncanonical != indexed:
        raise ValueError("stage contains unindexed trace files")
