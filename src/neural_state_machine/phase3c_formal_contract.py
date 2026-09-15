"""Fail-closed binding from Phase 3C Python evidence to the approved Lean Gate F head."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


APPROVED_FORMAL_COMMIT = "3de297cee2a94a7fc309531334f720f1b34467c9"
REQUIRED_THEOREMS = (
    "NarrativeDynamics.AnonymousTemporalCredit.aggregate_view_source_noninterference",
    "NarrativeDynamics.AnonymousTemporalCredit.aggregate_conservation",
    "NarrativeDynamics.AnonymousTemporalCredit.learner_view_source_relabel_invariant",
    "NarrativeDynamics.AnonymousTemporalCredit.eligibility_historical_coefficient",
    "NarrativeDynamics.AnonymousTemporalCredit.immediate_reduction_to_phase3a",
)

_CONTRACT_PATH = Path("docs/experiments/phase-3c-formal-contract.json")
_EXPECTED_KEYS = {
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


@dataclass(frozen=True)
class Phase3CFormalContract:
    commit: str
    theorems: tuple[str, ...]
    exact_head_ci_passed: bool
    axiom_audit_reviewed: bool


def _contract_path(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError("formal contract root must not be a symlink")
    current = root
    for part in _CONTRACT_PATH.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("formal contract path must not contain symlinks")
    return current


def load_phase3c_formal_contract(root: Path | None = None) -> Phase3CFormalContract:
    repository_root = Path(__file__).resolve().parents[2] if root is None else Path(root)
    path = _contract_path(repository_root)
    if not path.is_file():
        raise ValueError("formal contract must exist at the approved repository path")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("formal contract must be valid UTF-8 JSON") from exc

    if not isinstance(payload, dict) or set(payload) != _EXPECTED_KEYS:
        raise ValueError("formal contract schema keys do not match the approved schema")
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise ValueError("formal contract schema_version must be 1")
    if payload["repository"] != "qigao/lean":
        raise ValueError("formal contract repository is not approved")
    if payload["module"] != "NarrativeDynamics/Core/AnonymousTemporalCredit.lean":
        raise ValueError("formal contract module is not approved")
    if payload["lean_version"] != "4.32.0":
        raise ValueError("formal contract Lean version is not approved")
    if payload["mathlib_version"] != "v4.32.0":
        raise ValueError("formal contract Mathlib version is not approved")

    commit = payload["commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("formal contract commit must be a lowercase 40-hex SHA")
    if commit != APPROVED_FORMAL_COMMIT:
        raise ValueError("formal contract commit does not match the approved Gate F head")

    if payload["exact_head_ci_passed"] is not True:
        raise ValueError("formal contract requires exact-head CI success")
    if payload["axiom_audit_reviewed"] is not True:
        raise ValueError("formal contract requires reviewed axiom output")
    if payload["sorry_ax_present"] is not False:
        raise ValueError("formal contract rejects sorryAx")
    if payload["custom_axiom_present"] is not False:
        raise ValueError("formal contract rejects custom axioms")

    theorems = payload["theorems"]
    if not isinstance(theorems, list) or tuple(theorems) != REQUIRED_THEOREMS:
        raise ValueError("formal contract theorem set or order does not match the approved gate")
    if any(not isinstance(name, str) for name in theorems):
        raise ValueError("formal contract theorem names must be strings")

    return Phase3CFormalContract(
        commit=commit,
        theorems=tuple(theorems),
        exact_head_ci_passed=True,
        axiom_audit_reviewed=True,
    )
