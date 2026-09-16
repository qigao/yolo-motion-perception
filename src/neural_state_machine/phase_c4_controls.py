"""Fail-closed Phase C4 formal-contract loading and protocol controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


_FORMAL_COMMIT = "8b2180ed24b6ff03db4b927ec29cdd9903b0ccac"
_FORMAL_REPOSITORY = "qigao/lean"
_FORMAL_MODULE = "NarrativeDynamics/Core/MarginalizedTemporalCredit.lean"
_FORMAL_THEOREMS = (
    "NarrativeDynamics.MarginalizedTemporalCredit.registered_law_valid",
    "NarrativeDynamics.MarginalizedTemporalCredit.expected_aggregate_decomposition",
    "NarrativeDynamics.MarginalizedTemporalCredit.candidate_support_bounded",
    "NarrativeDynamics.MarginalizedTemporalCredit.current_weight_observation",
    "NarrativeDynamics.MarginalizedTemporalCredit.immediate_reduction_to_phase3a",
    "NarrativeDynamics.MarginalizedTemporalCredit.invalid_candidates_zero",
)
_FORMAL_PATH = Path(
    "docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json"
)
_FORMAL_KEYS = {
    "schema_version",
    "repository",
    "module",
    "commit",
    "lean_version",
    "mathlib_version",
    "exact_head_ci_passed",
    "axiom_audit_reviewed",
    "sorry_ax_present",
    "custom_axiom_present",
    "theorems",
}

# Frozen Phase 3C scientific sources and evidence inherited by C4.
FROZEN_C3_INPUT_PATHS = (
    "src/neural_state_machine/action_value.py",
    "src/neural_state_machine/phase3c_schedule.py",
    "src/neural_state_machine/phase3c_learners.py",
    "src/neural_state_machine/phase3c_controls.py",
    "src/neural_state_machine/phase3c_benchmark.py",
    "src/neural_state_machine/reward_learning.py",
    "src/neural_state_machine/action_value_benchmark.py",
    "scripts/verify_phase3c_anonymous_credit.py",
    "docs/experiments/phase-3c-formal-contract.json",
    "docs/experiments/phase-3c-anonymous-temporal-credit.json",
    "docs/experiments/phase-3c-anonymous-temporal-credit-report.md",
    "docs/experiments/phase-3c-failure-attribution-v1/provenance.json",
    "docs/experiments/phase-3c-failure-attribution-v1/registered-result.md",
)


@dataclass(frozen=True, slots=True)
class PhaseC4FormalContract:
    commit: str
    theorems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseC4ProtocolAudit:
    decision_count: int
    latent_reward_count: int
    delivered_reward_count: int
    real_feedback_count: int
    drain_feedback_count: int
    pending_final: int
    action_digest: str
    schedule_digest: str
    call_digest: str
    candidate_digest: str
    parameter_digest: str
    source_relabel_invariant: bool
    hidden_multiplicity_invariant: bool
    current_weight_probe_passed: bool
    bounded_history_passed: bool
    immediate_continuity_passed: bool
    repeatable: bool


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_contract_path(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError("formal contract root must not be a symlink")
    resolved_root = root.resolve()
    path = root / _FORMAL_PATH
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError("formal contract path must not contain symlinks")
        current = current.parent
    if not path.exists() or not path.is_file():
        raise ValueError("formal contract file is missing")
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("formal contract must remain within the repository root") from exc
    return path


def _require_bool(payload: dict[str, object], key: str, expected: bool) -> None:
    if payload.get(key) is not expected:
        raise ValueError(f"formal contract {key} must be {expected}")


def load_phase_c4_formal_contract(root: Path | None = None) -> PhaseC4FormalContract:
    resolved_root = _repository_root() if root is None else Path(root)
    path = _validate_contract_path(resolved_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("formal contract must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _FORMAL_KEYS:
        raise ValueError("formal contract must contain exactly the registered keys")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("formal contract schema_version must be 1")
    if payload.get("repository") != _FORMAL_REPOSITORY:
        raise ValueError("formal contract repository mismatch")
    if payload.get("module") != _FORMAL_MODULE:
        raise ValueError("formal contract module mismatch")
    commit = payload.get("commit")
    if not isinstance(commit, str) or commit != _FORMAL_COMMIT:
        raise ValueError("formal contract commit mismatch")
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("formal contract commit must be lowercase 40-hex")
    if payload.get("lean_version") != "4.32.0":
        raise ValueError("formal contract Lean version mismatch")
    if payload.get("mathlib_version") != "v4.32.0":
        raise ValueError("formal contract Mathlib version mismatch")
    _require_bool(payload, "exact_head_ci_passed", True)
    _require_bool(payload, "axiom_audit_reviewed", True)
    _require_bool(payload, "sorry_ax_present", False)
    _require_bool(payload, "custom_axiom_present", False)
    theorems = payload.get("theorems")
    if not isinstance(theorems, list) or any(not isinstance(item, str) for item in theorems):
        raise ValueError("formal contract theorems must be a list of strings")
    theorem_tuple = tuple(theorems)
    if theorem_tuple != _FORMAL_THEOREMS:
        raise ValueError("formal contract theorem set mismatch")
    return PhaseC4FormalContract(commit=commit, theorems=theorem_tuple)


def _is_lower_hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer_sequence_digest(values: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(values).to_bytes(8, "big", signed=False))
    for value in values:
        digest.update(value.to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def validate_phase_c4_protocol(
    audit: PhaseC4ProtocolAudit,
    expected_actions: tuple[int, ...],
) -> None:
    if not isinstance(audit, PhaseC4ProtocolAudit):
        raise ValueError("audit must be a PhaseC4ProtocolAudit")
    if not isinstance(expected_actions, tuple) or not expected_actions:
        raise ValueError("expected_actions must be a non-empty tuple")
    if any(type(action) is not int or action < 0 for action in expected_actions):
        raise ValueError("expected_actions must contain non-negative integers")

    expected_count = len(expected_actions)
    count_fields = (
        audit.decision_count,
        audit.latent_reward_count,
        audit.delivered_reward_count,
        audit.real_feedback_count,
    )
    if any(type(value) is not int or value != expected_count for value in count_fields):
        raise ValueError("registered real-step counts must match expected actions")
    if type(audit.drain_feedback_count) is not int or audit.drain_feedback_count != 5:
        raise ValueError("registered drain feedback count must be five")
    if type(audit.pending_final) is not int or audit.pending_final != 0:
        raise ValueError("registered protocol must end with no pending feedback")

    if audit.action_digest != _integer_sequence_digest(expected_actions):
        raise ValueError("action digest does not match expected actions")
    for name, value in (
        ("schedule_digest", audit.schedule_digest),
        ("call_digest", audit.call_digest),
        ("candidate_digest", audit.candidate_digest),
        ("parameter_digest", audit.parameter_digest),
    ):
        if not _is_lower_hex_digest(value):
            raise ValueError(f"{name} must be lowercase 64-hex")

    for name, value in (
        ("source_relabel_invariant", audit.source_relabel_invariant),
        ("hidden_multiplicity_invariant", audit.hidden_multiplicity_invariant),
        ("current_weight_probe_passed", audit.current_weight_probe_passed),
        ("bounded_history_passed", audit.bounded_history_passed),
        ("immediate_continuity_passed", audit.immediate_continuity_passed),
        ("repeatable", audit.repeatable),
    ):
        if value is not True:
            raise ValueError(f"{name} must be true")


def _validated_frozen_path(root: Path, relative: str) -> Path:
    if root.is_symlink():
        raise ValueError("frozen input root must not be a symlink")
    resolved_root = root.resolve()
    path = root / relative
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"frozen input path contains symlink: {relative}")
        current = current.parent
    if not path.exists() or not path.is_file():
        raise ValueError(f"frozen input is missing: {relative}")
    try:
        path.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"frozen input escapes repository root: {relative}") from exc
    return path


def frozen_input_hashes(root: Path | None = None) -> tuple[tuple[str, str], ...]:
    resolved_root = _repository_root() if root is None else Path(root)
    hashes: list[tuple[str, str]] = []
    for relative in FROZEN_C3_INPUT_PATHS:
        path = _validated_frozen_path(resolved_root, relative)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"failed to read frozen input: {relative}") from exc
        hashes.append((relative, hashlib.sha256(payload).hexdigest()))
    return tuple(hashes)
