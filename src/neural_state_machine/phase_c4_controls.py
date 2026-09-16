"""Fail-closed Phase C4 formal-contract loading and protocol helpers."""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class PhaseC4FormalContract:
    commit: str
    theorems: tuple[str, ...]


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
